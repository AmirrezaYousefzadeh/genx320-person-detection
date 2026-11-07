/*
 * Frame task (make HISTO=1), the replacement of events.c: the sensor's Event Histogram Computer (EHC) counts the
 * events of every pixel (two 4-bit counters, OFF low nibble, ON high nibble) and sends the 320x320 frame over the
 * CPI at the end of every accumulation period (DET_WINDOW_MS). The DCMI's DMA fills two frame buffers in turn
 * (circular, the DMA half/complete interrupts mark a finished frame); this task turns each finished frame into
 * the detector's picture (nn_pic) and the event view. The processor decodes no events at all and its load no
 * longer depends on the event rate. The microSD recorder gets the raw frames (16-byte header + 102400 bytes each)
 * instead of EVT2.0 words; tools/histo_raw.py reads them.
 *
 * Measured: frames come every ~119 ms at a 50 ms period and every ~312 ms at 200 ms, i.e.
 * the sensor needs ~55 ms to drain the histogram and the period runs ~1.3x slow; the CPI sends VSYNC once per
 * two frames and HSYNC per 80 bytes with the sequences used here. The DMA counters are therefore the frame clock.
 *
 * Copyright (c) 2026 Amirreza Yousefzadeh, University of Twente. Licensed under the Apache License 2.0.
 */
#include "app.h"

#include "genx320.h"

#define FRAME_WORDS (SENSOR_SIZE * SENSOR_SIZE / 4)
#define FRAME_MAGIC 0x4D524648u /* "HFRM": header of a recorded frame */
#define HISTO_HSYNC_BLANKING 880 /* CPI clock cycles between lines, as in the OpenMV driver */
#define HISTO_VSYNC_LINES 8

void psee_sensor_write(uint16_t reg, uint32_t data);

extern DCMI_HandleTypeDef hdcmi;
extern DMA_HandleTypeDef hdma_dcmi;

const struct issd *current_issd = NULL; /* used by genx320.c */

/* the DMA writes the frames: uncached SDRAM, line sy at frames[k][sy * 320], byte sx = pixel (sx, sy) */
static uint8_t frames[2][SENSOR_SIZE * SENSOR_SIZE] __attribute__((section("Uncached_SDRAM"), aligned(32)));
uint8_t nn_pic[PIC_SIZE * PIC_SIZE] __attribute__((section("NN_Arena"), aligned(32)));
volatile uint32_t ev_rate_kevps = 0; /* sum of the counts of the last frame per ms: about the event rate */
volatile uint32_t ev_overruns = 0;   /* frames the task was too late for (overwritten before it got to them) */
uint32_t record_buf[REC_WORDS] __attribute__((section("Record_Buffer"), aligned(32)));

static uint8_t *volatile write_fb = NULL;
static volatile int painted = 0;
static volatile uint32_t frame_seq = 0;
static volatile unsigned done_frame = 0; /* which buffer the DMA finished last */
static void (*hal_dma_cplt)(DMA_HandleTypeDef *h) = NULL;
static SemaphoreHandle_t frame_sem, pic_mutex;
static StaticSemaphore_t frame_sem_buf, pic_mutex_buf;

void events_set_write_buffer(uint8_t *fb) { write_fb = fb; }
int events_take_painted(void) { int p = painted; painted = 0; return p; }
uint32_t events_frame_seq(void) { return frame_seq; }
void events_lock_pic(void) { xSemaphoreTake(pic_mutex, portMAX_DELAY); }
void events_unlock_pic(void) { xSemaphoreGive(pic_mutex); }

static void frame_done(unsigned which) {
  BaseType_t woken = pdFALSE;
  done_frame = which;
  if (frame_sem != NULL) xSemaphoreGiveFromISR(frame_sem, &woken);
  portYIELD_FROM_ISR(woken);
}

static void dma_half(DMA_HandleTypeDef *h) {
  (void)h;
  frame_done(0);
}

static void dma_cplt(DMA_HandleTypeDef *h) {
  if (hal_dma_cplt != NULL) hal_dma_cplt(h); /* the HAL's own bookkeeping */
  frame_done(1);
}

static void sensor_start(void) {
  current_issd = psee_open_histo(); /* power sequence, clock, reset, register init with the EHC in the path */
  psee_sensor_set_biases(&genx320_default_biases);
  psee_sensor_start(current_issd);
  psee_sensor_set_flip(0, 0);
  psee_disable_afk();
  /* CPI pipeline for frames (psee_PM3C_Histo_config of the OpenMV driver), one 320-byte line per packet */
  psee_sensor_write(CPI_PIPELINE_CONTROL, CPI_PIPELINE_CONTROL_ENABLE | CPI_PIPELINE_CONTROL_OUTPUT_IF_MODE |
                                              CPI_PIPELINE_CONTROL_PACKET_FIXED_SIZE_ENABLE |
                                              CPI_PIPELINE_CONTROL_FRAME_FIXED_SIZE_ENABLE |
                                              CPI_PIPELINE_CONTROL_CLK_OUT_EN | CPI_PIPELINE_CONTROL_CLK_INVERSION |
                                              CPI_PIPELINE_CONTROL_CLK_OUT_GATING_ENABLE |
                                              (0x1Fu << CPI_PIPELINE_CONTROL_CLK_TIMEOUT_Pos));
  psee_sensor_write(CPI_PACKET_SIZE_CONTROL, SENSOR_SIZE);
  psee_sensor_write(CPI_PACKET_TIME_CONTROL, (SENSOR_SIZE << CPI_PACKET_TIME_CONTROL_PERIOD_Pos) |
                                                 (HISTO_HSYNC_BLANKING << CPI_PACKET_TIME_CONTROL_BLANKING_Pos));
  psee_sensor_write(CPI_FRAME_SIZE_CONTROL, SENSOR_SIZE);
  psee_sensor_write(CPI_FRAME_TIME_CONTROL, HISTO_VSYNC_LINES);
  sensor_erc_set(ERC_DEFAULT_KEVPS); /* same rate cap as in event mode: the pictures match the training data */
  sensor_ehc_start(DET_WINDOW_MS * 1000);
}

/* One finished frame -> detector picture and event view. Picture coordinates (app.h): X = 319 - sy, Y = sx. */
static uint32_t paint(const uint8_t *f) {
  uint8_t *fb = write_fb;
  uint32_t count = 0;
  events_lock_pic();
  memset(nn_pic, PIC_NONE, sizeof(nn_pic));
  for (uint32_t sy = 0; sy < SENSOR_SIZE; sy++) {
    const uint32_t *line = (const uint32_t *)(f + sy * SENSOR_SIZE);
    uint32_t X = (SENSOR_SIZE - 1) - sy;
    for (uint32_t w = 0; w < SENSOR_SIZE / 4; w++) {
      uint32_t word = line[w];
      if (word == 0) continue; /* most of the frame: no events in these four pixels */
      for (uint32_t b = 0; b < 4; b++) {
        uint32_t v = (word >> (8 * b)) & 0xFF;
        if (v == 0) continue;
        uint32_t off = v & 0xF, on = v >> 4; /* more ON than OFF events: white; more OFF: black; a tie: either,
                                                 by pixel parity (event mode shows the last event, about 50/50) */
        count += on + off;
        uint32_t Y = 4 * w + b, Xp = X;
        uint32_t white = on != off ? on > off : ((Xp ^ Y) & 1);
#if SENSOR_ROT180
        Xp = (PIC_SIZE - 1) - Xp, Y = (PIC_SIZE - 1) - Y;
#endif
        nn_pic[Y * PIC_SIZE + Xp] = white ? PIC_ON : PIC_OFF;
        uint32_t row = Y - VIEW_CROP;
        if (fb != NULL && row < EV_VIEW_H) {
#if VIEW_ROT180
          fb[(EV_VIEW_H - 1 - row) * EV_VIEW_W + (EV_VIEW_W - 1 - Xp)] = white ? EV_ON : EV_OFF;
#else
          fb[row * EV_VIEW_W + Xp] = white ? EV_ON : EV_OFF;
#endif
        }
      }
    }
  }
  frame_seq++;
  events_unlock_pic();
  painted = 1;
  return count;
}

void events_task(void *argument) {
  (void)argument;
  frame_sem = xSemaphoreCreateBinaryStatic(&frame_sem_buf);
  pic_mutex = xSemaphoreCreateMutexStatic(&pic_mutex_buf);
  memset(nn_pic, PIC_NONE, sizeof(nn_pic));
  sensor_start();
  hdcmi.Init.JPEGMode = DCMI_JPEG_DISABLE; /* frames with lines now, not a byte stream */
  HAL_DCMI_Init(&hdcmi);
  hdma_dcmi.XferHalfCpltCallback = dma_half; /* frame 0 written */
  while (HAL_DCMI_Start_DMA(&hdcmi, DCMI_MODE_CONTINUOUS, (uint32_t)frames, 2 * FRAME_WORDS) != HAL_OK) vTaskDelay(10);
  hal_dma_cplt = hdma_dcmi.XferCpltCallback; /* frame 1 written: wrap the HAL's callback */
  hdma_dcmi.XferCpltCallback = dma_cplt;

  unsigned expected = 0;
  uint32_t start_tick = HAL_GetTick();
  for (;;) {
    if (xSemaphoreTake(frame_sem, pdMS_TO_TICKS(500)) != pdTRUE) continue;
    unsigned done = done_frame;
    if (done != expected && HAL_GetTick() - start_tick > 1000) ev_overruns++; /* a frame was missed */
    expected = done ^ 1;
    ev_rate_kevps = paint(frames[done]) / DET_WINDOW_MS;
    /* to the card recorder (sdcard.c): a 16-byte header, then the raw frame */
    uint32_t hdr[4] = {FRAME_MAGIC, HAL_GetTick(), frame_seq, 0};
    sd_push(hdr, 4);
    sd_push((const uint32_t *)frames[done], FRAME_WORDS);
  }
}
