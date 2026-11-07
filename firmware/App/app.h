/*
 * Live person detection on the GenX320 + STM32F746G-DISCO: shared definitions of the three application tasks.
 *
 *   events_task    reads the sensor's EVT2.0 stream (DCMI + DMA) and paints every event into the event view
 *                  (L8 frame buffer shown by the LTDC) and into the detector's picture (nn_pic)
 *   detector_task  every few hundred ms: stretches nn_pic to 192x192, runs the network, decodes the boxes
 *   display_task   swaps the event-view buffers at about 30 Hz, draws the boxes and the text panel
 *
 * Geometry. The board stands in landscape, lens on top: the text panel is the left 160 px of the LCD, the
 * event view (320x272) the rest. The sensor is 320x320. Measured with a monitor in front of it
 * the world's right is -sy and the world's down is +sx. The detector's picture
 * P[Y][X] is therefore X = 319 - sy, Y = sx, and the event view shows its rows VIEW_CROP..VIEW_CROP+271
 * (screen x = 160 + X, y = Y - VIEW_CROP).
 * SENSOR_ROT180 = 1 turns the picture by 180 degrees (board clamped the other way up), VIEW_ROT180 = 1 turns
 * everything on the screen by 180 degrees (panel at the right); `make ROT180=1` sets both.
 *
 * Copyright (c) 2026 Amirreza Yousefzadeh, University of Twente. Licensed under the Apache License 2.0.
 */
#ifndef APP_H
#define APP_H

#include <stdint.h>
#include <string.h>

#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "stm32f7xx_hal.h"

#define SENSOR_SIZE 320          /* GenX320 pixels per side */
#define PIC_SIZE 320             /* detector picture (nn_pic) per side */
#define LCD_W 480                /* LCD in landscape */
#define LCD_H 272
#define EV_VIEW_W 320            /* event view layer (L8), landscape */
#define EV_VIEW_H 272
#define EV_VIEW_X 160            /* its x offset on the LCD */
#define PANEL_W 160              /* text panel width */
#define VIEW_CROP ((PIC_SIZE - EV_VIEW_H) / 2) /* picture rows above the event view */
#ifndef VIEW_ROT180
#define VIEW_ROT180 0
#endif
#ifndef SENSOR_ROT180
#define SENSOR_ROT180 0
#endif
#ifndef HISTO_MODE
#define HISTO_MODE 0             /* 1 (make HISTO=1): the sensor sends 320x320 count frames (EHC), frames.c instead of events.c */
#endif

/* Colour indices of the event view layer (CLUT in display.c) */
#define EV_BG 0
#define EV_ON 1
#define EV_OFF 2
#define EV_BOX 3

/* Grey levels of the detector picture */
#define PIC_NONE 128
#define PIC_ON 255
#define PIC_OFF 0

#define DET_WINDOW_MS 50         /* events collected into one detector picture */
#ifndef DET_THRESHOLD
#define DET_THRESHOLD 0.7f       /* a box needs this score; 0.6 gave boxes without a person in the live test */
#endif
#define DET_NMS_IOU 0.5f
#define DET_MAX_BOXES 8

typedef struct {
  float x0, y0, x1, y1;          /* in 0..1 of the detector picture, x = u, y = v */
  float score;
  uint8_t held;                  /* 0: found in this run; else runs since its person was last seen (drawn thin) */
} det_box_t;

typedef struct {
  uint32_t seq;                  /* +1 per detector run */
  int32_t status;                /* 0, or the error code of person_nn_init()/person_nn_run() */
  int32_t n;                     /* boxes to draw */
  int32_t count;                 /* people counted (median of the last runs, see detector.c) */
  float count_raw;               /* the counting head's estimate of this run (not rounded) */
  det_box_t box[DET_MAX_BOXES];
  uint32_t ms_nn;                /* network time of the last run */
  uint32_t ms_total;             /* window + network + decode */
} det_result_t;

/* microSD recorder (sdcard.c, PC side tools/sdrec.py): the event task pushes the raw EVT2.0 words into a
   3.5 MB SDRAM ring, the SD task writes them to /evt/recNNNN.raw. The PC drives it over SWD through sd_ctrl.
   LIST and READ leave their result in the ring (buf_addr, bytes). */
#if HISTO_MODE
#define REC_WORDS (820 * 1024) /* the two 100 KB sensor frames take their share */
#else
#define REC_WORDS (880 * 1024) /* 3.44 MB: what is left of the lower SDRAM half after the frame buffers */
#endif
#define SD_MAGIC 0x31444453 /* "SDD1" */
enum { SD_CMD_NONE = 0, SD_CMD_START = 1, SD_CMD_STOP = 2, SD_CMD_LIST = 3, SD_CMD_CLEAN = 4, SD_CMD_READ = 5,
       SD_CMD_FORMAT = 6, SD_CMD_PLAY = 7 };
enum { SD_NOCARD = 0, SD_IDLE = 1, SD_RECORDING = 2, SD_BUSY = 3, SD_DONE = 4, SD_ERROR = 5, SD_PLAYING = 6 };
#define PLAY_MAX 16
typedef struct {
  uint32_t magic;
  volatile uint32_t cmd;         /* written by the PC */
  volatile uint32_t state;       /* SD_NOCARD ... SD_ERROR */
  volatile int32_t result;       /* FatFs result of the last operation */
  volatile uint32_t file_no;     /* current / last recording number */
  volatile uint32_t bytes;       /* bytes written to the recording, or bytes of the LIST/READ result */
  volatile uint32_t dropped;     /* words dropped because the card could not keep up */
  volatile uint32_t start_tick;  /* HAL ticks (ms) of the recording */
  volatile uint32_t stop_tick;
  uint32_t arg0, arg1;           /* READ: file number, byte offset */
  uint32_t buf_addr, buf_len;    /* the SDRAM ring */
  volatile uint32_t card_mb, free_mb;
  /* Replay (SD_CMD_PLAY, ended by SD_CMD_STOP): play_n items, item i = recording play_file[i] from play_from_ms[i]
     to play_to_ms[i] (0: to its end), the list looped while play_loop; play_item is the item playing now. The
     event task then paints the recorded events at their own pace instead of the sensor's (events.c). */
  uint32_t play_n, play_loop;
  uint32_t play_file[PLAY_MAX], play_from_ms[PLAY_MAX], play_to_ms[PLAY_MAX];
  volatile uint32_t play_item;
} sd_ctrl_t;
extern sd_ctrl_t sd_ctrl;
extern uint32_t record_buf[REC_WORDS];
void sd_task(void *argument);
void sd_push(const uint32_t *words, uint32_t n);
extern volatile uint32_t sd_playing;            /* 1 while a recording is replayed */
uint32_t sd_play_peek(const uint32_t **words);  /* replayed words ready (contiguous), 0 if none */
void sd_play_consume(uint32_t n);
uint32_t sd_play_item_seq(void);                /* +1 when the next item starts */
int64_t sd_play_t0(void);                       /* the item's time zero, us (its file's first time-high word) */
void sd_play_window(uint32_t *from_ms, uint32_t *to_ms);
void sd_play_item_done(void);                   /* the event task passed the item's end: the card task moves on */

/* cpu_load.c: processor load over the last second, from the run-time statistics */
#define CPU_TASKS 4
extern volatile uint32_t cpu_busy_percent;
extern volatile uint32_t cpu_task_percent[CPU_TASKS]; /* events, detector, display, sd */
void cpu_load_update(void);

/* sensor_filters.c: the sensor's own event rate controller (0 = off) */
#ifndef ERC_DEFAULT_KEVPS
#define ERC_DEFAULT_KEVPS 500
#endif
void sensor_erc_set(uint32_t kevps);
uint32_t sensor_erc_kevps(void);
void sensor_ehc_start(uint32_t period_us);     /* histogram mode: count frames every period_us */
void sensor_stc_start(uint32_t threshold_ms);  /* event mode: pass an event only if its pixel fired within threshold_ms before */
#ifndef SENSOR_AFK
#define SENSOR_AFK 1       /* anti-flicker filter, 50-520 Hz (make AFK=0) */
#endif
#ifndef SENSOR_STC
#define SENSOR_STC 0       /* spatio-temporal contrast filter (make STC=1): off by default: a monitor bench cannot evaluate it */
#endif
#ifndef STC_THRESHOLD_MS
#define STC_THRESHOLD_MS 10 /* a pixel's event passes if the pixel fired within this time before (make STC_MS=) */
#endif
#define AFK_MIN_HZ 50
#define AFK_MAX_HZ 520

/* events.c */
extern uint8_t nn_pic[PIC_SIZE * PIC_SIZE];
extern volatile uint32_t ev_rate_kevps;        /* events per ms = kev/s, updated every 50 ms */
extern volatile uint32_t ev_overruns;          /* DMA ring overruns since start (events lost) */
void events_task(void *argument);
void events_set_write_buffer(uint8_t *fb);     /* L8 EV_VIEW_W x EV_VIEW_H buffer to paint events into */
#if HISTO_MODE                                 /* frames.c */
int events_take_painted(void);                 /* 1 once after each frame painted into the write buffer */
uint32_t events_frame_seq(void);               /* +1 per frame in nn_pic */
void events_lock_pic(void);                    /* nn_pic is replaced per frame: hold this while reading it */
void events_unlock_pic(void);
#endif

/* detector.c */
void detector_task(void *argument);
void detector_get_result(det_result_t *out);

/* display.c */
void display_task(void *argument);
void display_reload_callback(void);            /* from HAL_LTDC_ReloadEventCallback */

#endif
