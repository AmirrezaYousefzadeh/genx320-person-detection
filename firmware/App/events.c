/*
 * Event task: GenX320 start-up, EVT2.0 decoding, event view, detector picture; raw words to the SD recorder.
 *
 * The sensor streams EVT2.0 words over its parallel (CPI) interface into the DCMI, whose DMA fills a ring
 * buffer continuously. The ring lives in DTCM, which the DMA reaches through the AHBS port and the core does
 * not cache, so no cache maintenance is needed. The task follows the DMA's write position (NDTR).
 *
 * Copyright (c) 2026 Amirreza Yousefzadeh, University of Twente. Licensed under the Apache License 2.0.
 */
#include "app.h"

#include "evt_2_0.h"
#include "genx320.h"

#define RING_WORDS (15 * 1024) /* 60 KB: 15 ms of events at 1 Mevents/s before the DMA laps the reader */

extern DCMI_HandleTypeDef hdcmi;
extern DMA_HandleTypeDef hdma_dcmi;

const struct issd *current_issd = NULL;  /* used by genx320.c */

static uint32_t ring[RING_WORDS] __attribute__((section(".dma_ring"), aligned(32)));
uint8_t nn_pic[PIC_SIZE * PIC_SIZE] __attribute__((section("NN_Arena"), aligned(32)));
volatile uint32_t ev_rate_kevps = 0;
volatile uint32_t ev_overruns = 0; /* times the DMA lapped the reader (events lost, stream corrupted) */

static uint8_t *volatile write_fb = NULL;
static uint32_t ev_count = 0;

uint32_t record_buf[REC_WORDS] __attribute__((section("Record_Buffer"), aligned(32)));

void events_set_write_buffer(uint8_t *fb) { write_fb = fb; }

static inline void decode(uint32_t word) {
  uint32_t type = __EVT20_TYPE(word);
  if (type > TD_HIGH) return;  /* time high, triggers, ... */
  uint32_t sx = __EVT20_X(word), sy = __EVT20_Y(word);
  if (sx >= SENSOR_SIZE || sy >= SENSOR_SIZE) return;
  /* Picture coordinates (see app.h): X to the world's right, Y down */
  uint32_t X = (SENSOR_SIZE - 1) - sy, Y = sx;
#if SENSOR_ROT180
  X = (PIC_SIZE - 1) - X, Y = (PIC_SIZE - 1) - Y;
#endif
  nn_pic[Y * PIC_SIZE + X] = type == TD_HIGH ? PIC_ON : PIC_OFF;
  /* Event view: screen x = 160 + X, y = Y - VIEW_CROP, i.e. layer column = X, row = Y - VIEW_CROP */
  uint8_t *fb = write_fb;
  uint32_t row = Y - VIEW_CROP;
  if (fb != NULL && row < EV_VIEW_H) {
#if VIEW_ROT180
    fb[(EV_VIEW_H - 1 - row) * EV_VIEW_W + (EV_VIEW_W - 1 - X)] = type == TD_HIGH ? EV_ON : EV_OFF;
#else
    fb[row * EV_VIEW_W + X] = type == TD_HIGH ? EV_ON : EV_OFF;
#endif
  }
  ev_count++;
}

/* Replay (sdcard.c, SD_CMD_PLAY): the recorded words are painted when their time has come, counted from the
   first event of the item's window; the card task starts reading about 0.2 s before the window, events before it
   are skipped at once, the first one after it ends the item. Times are relative to the file's first time-high
   word; a CD event counts only after a time-high word has been seen (a file starts with a few before one). */
static void replay_step(void) {
  static uint32_t seq, th, started, tick0;
  static int th_valid;
  static int64_t t0;
  if (sd_play_item_seq() != seq) seq = sd_play_item_seq(), th_valid = 0, t0 = sd_play_t0(), started = 0;
  uint32_t from, to;
  sd_play_window(&from, &to);
  const uint32_t *w;
  uint32_t n;
  while ((n = sd_play_peek(&w)) > 0) {
    uint32_t i = 0;
    for (; i < n; i++) {
      uint32_t word = w[i], type = word >> 28;
      if (type == 0x8) {
        th = word & 0x0FFFFFFFu, th_valid = 1;
        continue;
      }
      if (type > TD_HIGH || !th_valid) continue;
      int64_t t = ((int64_t)th << 6) | ((word >> 22) & 0x3F);
      if (t0 < 0) t0 = t;
      uint32_t rel = (uint32_t)((t - t0) / 1000); /* ms into the recording */
      if (rel < from) continue;
      if (to && rel >= to) {
        sd_play_item_done();
        continue;
      }
      if (!started) started = 1, tick0 = HAL_GetTick();
      if ((int32_t)(HAL_GetTick() - (tick0 + (rel - from))) < 0) break; /* not due yet */
      decode(word);
    }
    sd_play_consume(i);
    if (i < n) break;
  }
}

static void sensor_start(void) {
  current_issd = psee_open_evt();                  /* power sequence, clock, reset, register init */
  psee_sensor_set_biases(&genx320_default_biases);
  psee_sensor_start(current_issd);
  psee_sensor_set_flip(0, 0);
#if SENSOR_AFK
  psee_enable_afk(AFK_MIN_HZ, AFK_MAX_HZ);         /* the sensor drops flickering pixels (lamps, screens) */
#else
  psee_disable_afk();
#endif
#if SENSOR_STC
  sensor_stc_start(STC_THRESHOLD_MS);              /* and events without a second one on the same pixel (noise) */
#endif
  psee_PM3C_config();
  sensor_erc_set(ERC_DEFAULT_KEVPS);               /* cap the event rate on the sensor */
}

void events_task(void *argument) {
  (void)argument;
  memset(nn_pic, PIC_NONE, sizeof(nn_pic));
  sensor_start();
  while (HAL_DCMI_Start_DMA(&hdcmi, DCMI_MODE_CONTINUOUS, (uint32_t)ring, RING_WORDS) != HAL_OK) vTaskDelay(10);

  uint32_t read_idx = 0, last_tick = HAL_GetTick(), last_count = 0, start_tick = last_tick;
  for (;;) {
    uint32_t now = HAL_GetTick();
    if (now - start_tick < 1000) ev_overruns = 0; /* the first batches after start are not real overruns */
    if (now - last_tick >= 50) {
      ev_rate_kevps = (ev_count - last_count) / (now - last_tick);
      last_count = ev_count;
      last_tick = now;
    }
    /* The DMA counts a word in NDTR when it leaves the DCMI, but writes memory in bursts of 4 words from its
       FIFO: up to 4 words behind NDTR may not be in the ring yet (reading them gives the previous lap's words
       with timestamps one lap old). Stay 8 words behind, on a burst boundary. */
    uint32_t write_idx = ((RING_WORDS - hdma_dcmi.Instance->NDTR) & ~3u) + RING_WORDS - 8;
    write_idx %= RING_WORDS;
    if (sd_playing) { /* a recording is replayed: the sensor's words are dropped, the card's painted */
      read_idx = write_idx;
      replay_step();
      vTaskDelay(1);
      continue;
    }
    if (read_idx == write_idx) {
      vTaskDelay(2);
      continue;
    }
    uint32_t start_idx = read_idx;
    if (read_idx > write_idx) {
      sd_push(ring + read_idx, RING_WORDS - read_idx);
      while (read_idx < RING_WORDS) decode(ring[read_idx++]);
      read_idx = 0;
    }
    sd_push(ring + read_idx, write_idx - read_idx);
    while (read_idx < write_idx) decode(ring[read_idx++]);
    /* Overrun check: if the DMA has meanwhile written past where we started reading, part of what we read
       was already overwritten (stale or mixed words). */
    uint32_t now_idx = ((RING_WORDS - hdma_dcmi.Instance->NDTR) & ~3u) + RING_WORDS - 8;
    now_idx %= RING_WORDS;
    uint32_t written = (now_idx + RING_WORDS - start_idx) % RING_WORDS;
    uint32_t consumed = (write_idx + RING_WORDS - start_idx) % RING_WORDS;
    if (written < consumed) ev_overruns++;
  }
}
