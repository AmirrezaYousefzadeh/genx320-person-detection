/*
 * FatFs disk I/O for the microSD card (SDMMC1, 4-bit, DMA) through ST's Discovery BSP driver.
 *
 * Transfers use DMA and wait on a semaphore for the completion callback. The buffers FatFs hands over are its
 * own objects (placed in the uncached SDRAM by sdcard.c) or the recording ring (also uncached), so no cache
 * maintenance is needed. The SD clock is 25 MHz (see disk_initialize; CubeMX had left the SDMMC kernel
 * clock at 200 MHz, which gave a 50 MHz card clock and random failures).
 *
 * Copyright (c) 2026 Amirreza Yousefzadeh, University of Twente. Licensed under the Apache License 2.0.
 */
#include "app.h"
#include "ff.h"
#include "diskio.h"
#include "stm32746g_discovery_sd.h"

#define SD_TIMEOUT_MS 5000
#define BLOCK_SIZE 512

static volatile DSTATUS status = STA_NOINIT;
extern SD_HandleTypeDef uSdHandle;
/* first failure: op (1 read, 2 write), BSP result, HAL error code, card state, sector, count; then the failure count */
__attribute__((used)) volatile uint32_t sd_diag[8];

static void diag(uint32_t op, uint32_t bsp, uint32_t sector, uint32_t count) {
  if (sd_diag[6]++ == 0) {
    sd_diag[0] = op, sd_diag[1] = bsp, sd_diag[2] = HAL_SD_GetError(&uSdHandle), sd_diag[3] = BSP_SD_GetCardState();
    sd_diag[4] = sector, sd_diag[5] = count;
  }
}

static SemaphoreHandle_t done_sem;
static StaticSemaphore_t done_sem_buf;

/* DMA transfer: wait for the completion callback, then until the card has finished programming */
static int wait_done(void) {
  if (xSemaphoreTake(done_sem, pdMS_TO_TICKS(SD_TIMEOUT_MS)) != pdTRUE) return 0;
  uint32_t t0 = HAL_GetTick();
  while (BSP_SD_GetCardState() != SD_TRANSFER_OK)
    if (HAL_GetTick() - t0 > SD_TIMEOUT_MS) return 0;
    else vTaskDelay(1);
  return 1;
}

void BSP_SD_WriteCpltCallback(void) {
  BaseType_t woken = pdFALSE;
  xSemaphoreGiveFromISR(done_sem, &woken);
  portYIELD_FROM_ISR(woken);
}

void BSP_SD_ReadCpltCallback(void) {
  BaseType_t woken = pdFALSE;
  xSemaphoreGiveFromISR(done_sem, &woken);
  portYIELD_FROM_ISR(woken);
}

DSTATUS disk_status(BYTE pdrv) {
  (void)pdrv;
  return status;
}

DSTATUS disk_initialize(BYTE pdrv) {
  (void)pdrv;
  if (done_sem == NULL) done_sem = xSemaphoreCreateBinaryStatic(&done_sem_buf);
  status = STA_NOINIT;
  if (BSP_SD_IsDetected() != SD_PRESENT) return status | STA_NODISK;
  if (BSP_SD_Init() != MSD_OK) return status;
  /* SDMMC_CK = 50 MHz kernel clock / (ClockDiv + 2): 25 MHz on the card, 4 data lines */
  uSdHandle.Init.ClockDiv = 0;
  if (HAL_SD_ConfigWideBusOperation(&uSdHandle, SDMMC_BUS_WIDE_4B) != HAL_OK) return status;
  status &= ~STA_NOINIT;
  return status;
}

/* Writes: the SDMMC's DMA (memory to peripheral) fails with a transfer error when its source is the SDRAM
   (measured; reading into the SDRAM works), so every write goes through this SRAM bounce buffer, up to
   BOUNCE_SECTORS sectors per DMA transfer, with the cache cleaned first. Reads into an aligned uncached
   SDRAM buffer go directly; other reads use the bounce buffer one sector at a time. */
#define BOUNCE_SECTORS 64 /* 32 KB, the size of the recorder's chunks */
static uint8_t bounce[BOUNCE_SECTORS * BLOCK_SIZE] __attribute__((aligned(32)));

static int direct_ok(const void *p) {
  uint32_t a = (uint32_t)p;
  return (a & 15u) == 0 && a >= 0xC0000000u && a < 0xC0400000u;
}

static DRESULT read_blocks(uint8_t *dst, uint32_t sector, uint32_t count) {
  uint8_t r = BSP_SD_ReadBlocks_DMA((uint32_t *)dst, sector, count);
  if (r != MSD_OK || !wait_done()) {
    diag(1, r, sector, count);
    return RES_ERROR;
  }
  return RES_OK;
}

static DRESULT write_blocks(const uint8_t *src, uint32_t sector, uint32_t count) {
  uint8_t r = BSP_SD_WriteBlocks_DMA((uint32_t *)src, sector, count);
  if (r != MSD_OK || !wait_done()) {
    diag(2, r, sector, count);
    return RES_ERROR;
  }
  return RES_OK;
}

DRESULT disk_read(BYTE pdrv, BYTE *buff, LBA_t sector, UINT count) {
  (void)pdrv;
  if (status & STA_NOINIT) return RES_NOTRDY;
  if (direct_ok(buff)) return read_blocks(buff, (uint32_t)sector, count);
  for (UINT i = 0; i < count; i++) {
    DRESULT r = read_blocks(bounce, (uint32_t)sector + i, 1);
    if (r != RES_OK) return r;
    SCB_InvalidateDCache_by_Addr((uint32_t *)bounce, BLOCK_SIZE);
    memcpy(buff + i * BLOCK_SIZE, bounce, BLOCK_SIZE);
  }
  return RES_OK;
}

DRESULT disk_write(BYTE pdrv, const BYTE *buff, LBA_t sector, UINT count) {
  (void)pdrv;
  if (status & STA_NOINIT) return RES_NOTRDY;
  for (UINT done = 0; done < count;) {
    UINT n = count - done < BOUNCE_SECTORS ? count - done : BOUNCE_SECTORS;
    memcpy(bounce, buff + done * BLOCK_SIZE, n * BLOCK_SIZE);
    SCB_CleanDCache_by_Addr((uint32_t *)bounce, (int32_t)(n * BLOCK_SIZE));
    DRESULT r = write_blocks(bounce, (uint32_t)sector + done, n);
    if (r != RES_OK) return r;
    done += n;
  }
  return RES_OK;
}

DRESULT disk_ioctl(BYTE pdrv, BYTE cmd, void *buff) {
  (void)pdrv;
  if (status & STA_NOINIT) return RES_NOTRDY;
  HAL_SD_CardInfoTypeDef info;
  switch (cmd) {
    case CTRL_SYNC:
      return RES_OK;
    case GET_SECTOR_COUNT:
      BSP_SD_GetCardInfo(&info);
      *(LBA_t *)buff = info.LogBlockNbr;
      return RES_OK;
    case GET_SECTOR_SIZE:
      *(WORD *)buff = BLOCK_SIZE;
      return RES_OK;
    case GET_BLOCK_SIZE:
      BSP_SD_GetCardInfo(&info);
      *(DWORD *)buff = info.LogBlockSize / BLOCK_SIZE;
      return RES_OK;
    default:
      return RES_PARERR;
  }
}
