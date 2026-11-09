/*
 * Display task: LTDC layers, event view buffers, boxes and the text panel (see app.h for the geometry).
 * Landscape: panel left (160 px), event view right (320x272).
 *
 * Layer 0 (RGB565, whole LCD) holds the panel, drawn by the CPU. Layer 1 (L8 + colour table, the 320x272
 * event view) shows one of three buffers: one is painted by the event task, one is on the LCD, one is being
 * cleared by DMA. Every frame the three rotate; the boxes of the last detection (with a "person 87%" tag) and,
 * while the card records, a blinking REC badge are drawn into the buffer just before it goes on the LCD. All
 * buffers are in SDRAM, which is device memory (not cached), so the LTDC and the DMA always see what the CPU wrote.
 * The bottom row of the panel is a record button: a touch there, or the blue USER button, starts a recording to the
 * microSD card and stops it again, so that the board can record on its own (on a power bank, without a PC).
 *
 * Copyright (c) 2026 Amirreza Yousefzadeh, University of Twente. Licensed under the Apache License 2.0.
 */
#include "app.h"

#include <stdio.h>

#include "drv_dma.h"
#include "font.h"

extern LTDC_HandleTypeDef hltdc;
extern DrvDmaHandle_t drv_dma;

#define RGB565(r, g, b) ((uint16_t)((((r) >> 3) << 11) | (((g) >> 2) << 5) | ((b) >> 3)))
#define RGB888(r, g, b) (((uint32_t)(r) << 16) | ((uint32_t)(g) << 8) | (uint32_t)(b))

/* Panel: a light card with a blue title band next to the dark event view (the earlier dark panel, after
   Prophesee's demo, looked black on the LCD and in phone videos). */
static const uint16_t kPanelBg = RGB565(244, 246, 250);
static const uint16_t kHeader = RGB565(28, 94, 200);
static const uint16_t kHeaderSub = RGB565(190, 210, 245);
static const uint16_t kWhite = RGB565(255, 255, 255);
static const uint16_t kLabel = RGB565(112, 124, 148);
static const uint16_t kValue = RGB565(30, 37, 52);
static const uint16_t kIdle = RGB565(150, 160, 178);
static const uint16_t kAccent = RGB565(255, 140, 0); /* the boxes' orange */
static const uint16_t kRec = RGB565(225, 35, 35);
static const uint16_t kSpark = RGB565(232, 84, 90);
static const uint16_t kSparkFill = RGB565(250, 210, 212);
static const uint16_t kDivider = RGB565(222, 227, 236);

/* Event view colour table (L8): the event colours of app.h, then the REC badge's red and white */
#define EV_REC 4
#define EV_WHITE 5
static const uint32_t kClut[] = {RGB888(30, 37, 52),  RGB888(216, 223, 236), RGB888(64, 126, 201),
                                 RGB888(255, 140, 0), RGB888(225, 35, 35),    RGB888(255, 255, 255)};

static uint16_t gui_fb[LCD_W * LCD_H] __attribute__((section("Display_Buffers"), aligned(32)));
static uint8_t ev_fb[3][EV_VIEW_W * EV_VIEW_H] __attribute__((section("Display_Buffers"), aligned(32)));

static SemaphoreHandle_t reload_sem;
static StaticSemaphore_t reload_sem_buf;

void display_reload_callback(void) {
  BaseType_t woken = pdFALSE;
  if (reload_sem != NULL) xSemaphoreGiveFromISR(reload_sem, &woken);
  portYIELD_FROM_ISR(woken);
}

/* ---- text, for both layers ---------------------------------------------------------------------------- */

typedef void (*plot_fn)(void *ctx, int x, int y, uint32_t c);

static int render_text(plot_fn plot, void *ctx, int px, int py, const font_t *font, uint32_t c, const char *s) {
  for (; *s; s++) {
    unsigned ch = (unsigned char)*s;
    if (ch < font->first || ch > font->last) ch = '?';
    const font_glyph_t *g = &font->glyphs[ch - font->first];
    const uint8_t *bits = font->bitmap + g->bitmap_offset;
    int stride = (g->width + 7) / 8;
    for (int y = 0; y < g->height; y++, bits += stride)
      for (int x = 0; x < g->width; x++)
        if (bits[x >> 3] & (0x80 >> (x & 7))) plot(ctx, px + g->x_off + x, py + g->y_off + y, c);
    px += g->advance;
  }
  return px;
}

static int text_width(const font_t *font, const char *s) {
  int w = 0;
  for (; *s; s++) {
    unsigned ch = (unsigned char)*s;
    if (ch < font->first || ch > font->last) ch = '?';
    w += font->glyphs[ch - font->first].advance;
  }
  return w;
}

/* ---- panel drawing (screen coordinates, landscape) ----------------------------------------------------- */

static inline void put_pixel(int x, int y, uint16_t c) {
#if VIEW_ROT180
  x = LCD_W - 1 - x, y = LCD_H - 1 - y;
#endif
  if ((unsigned)x < LCD_W && (unsigned)y < LCD_H) gui_fb[y * LCD_W + x] = c;
}

static void plot_panel(void *ctx, int x, int y, uint32_t c) {
  (void)ctx;
  put_pixel(x, y, (uint16_t)c);
}

static void fill_rect(int x0, int y0, int w, int h, uint16_t c) {
  for (int y = y0; y < y0 + h; y++)
    for (int x = x0; x < x0 + w; x++) put_pixel(x, y, c);
}

static int draw_text(int px, int py, const font_t *font, uint16_t c, const char *s) {
  return render_text(plot_panel, NULL, px, py, font, c, s);
}

static void fill_circle(int cx, int cy, int r, uint16_t c) {
  for (int y = -r; y <= r; y++)
    for (int x = -r; x <= r; x++)
      if (x * x + y * y <= r * r + r) put_pixel(cx + x, cy + y, c);
}

/* Layout of the panel (y of each block); PX is the left margin */
#define PX 10
#define HEADER_H 54
#define Y_PEOPLE 58
#define Y_EVENTS 136
#define Y_NET 204
#define Y_BTN 249 /* the record button row, to the bottom of the panel */
#define BTN_W 64
#define BTN_H 21
#define SPARK_N (PANEL_W - 2 * PX) /* event-rate history, one sample per panel update (~0.2 s): ~28 s */
#define SPARK_H 18

static uint16_t spark[SPARK_N];
static int spark_head;

/* A small person figure, 13 x 38 px, top-left at (x, y) */
static void draw_person(int x, int y, uint16_t c) {
  fill_circle(x + 6, y + 5, 5, c);
  fill_rect(x + 1, y + 12, 11, 15, c);
  fill_rect(x + 2, y + 27, 4, 11, c);
  fill_rect(x + 7, y + 27, 4, 11, c);
}

/* The non-default sensor/detector configuration, or NULL for the demo's default (event mode, AFK, ERC 500k) */
static const char *mode_string(void) {
  static char line[32];
#if HISTO_MODE
  snprintf(line, sizeof(line), "histogram mode, %d ms", DET_WINDOW_MS);
  return line;
#else
  if (SENSOR_AFK && !SENSOR_STC && ERC_DEFAULT_KEVPS == 500) return NULL;
  snprintf(line, sizeof(line), "ERC %luk%s%s", (unsigned long)ERC_DEFAULT_KEVPS, SENSOR_AFK ? " AFK" : " no AFK",
           SENSOR_STC ? " STC" : "");
  return line;
#endif
}

static void draw_panel_static(void) {
  fill_rect(0, 0, PANEL_W, LCD_H, kPanelBg);
  fill_rect(0, 0, PANEL_W, HEADER_H, kHeader);
  draw_text(PX, 5, &font_tiny, kHeaderSub, "UNIVERSITY OF TWENTE");
  draw_text(PX, 19, &font_title, kWhite, "PERSON DETECTION");
  const char *mode = mode_string();
  draw_text(PX, 37, &font_tiny, kHeaderSub, mode ? mode : "GenX320 + STM32F746");
  draw_text(PX, Y_PEOPLE + 4, &font_label, kLabel, "PEOPLE IN VIEW");
  fill_rect(PX, Y_EVENTS, PANEL_W - 2 * PX, 1, kDivider);
  draw_text(PX, Y_EVENTS + 5, &font_label, kLabel, "EVENTS PER SECOND");
  fill_rect(PX, Y_NET, PANEL_W - 2 * PX, 1, kDivider);
  draw_text(PX, Y_NET + 5, &font_label, kLabel, "NEURAL NETWORK");
}

static void draw_people(int n) {
  fill_rect(0, Y_PEOPLE + 16, PANEL_W, Y_EVENTS - Y_PEOPLE - 17, kPanelBg);
  char line[8];
  snprintf(line, sizeof(line), "%d", n);
  uint16_t c = n > 0 ? kAccent : kValue;
  int x = draw_text(PX, Y_PEOPLE + 16, &font_huge, c, line);
  for (int i = 0; i < n && i < 4; i++) draw_person(x + 8 + i * 17, Y_PEOPLE + 27, c); /* one figure per person */
}

static void draw_events(uint32_t kevps) {
  char line[24];
  if (kevps >= 1000)
    snprintf(line, sizeof(line), "%lu.%lu M", (unsigned long)(kevps / 1000), (unsigned long)(kevps / 100 % 10));
  else
    snprintf(line, sizeof(line), "%lu k", (unsigned long)kevps);
  fill_rect(0, Y_EVENTS + 18, PANEL_W, font_large.height, kPanelBg);
  draw_text(PX, Y_EVENTS + 18, &font_large, kValue, line);
  /* history graph, newest sample at the right, linear scale from 0 to the largest value shown (at least
     200 kevents/s, so that a quiet room does not fill the graph with noise) */
  spark[spark_head] = kevps > 65535 ? 65535 : (uint16_t)kevps;
  spark_head = (spark_head + 1) % SPARK_N;
  uint32_t top = 200;
  for (int i = 0; i < SPARK_N; i++)
    if (spark[i] > top) top = spark[i];
  int y0 = Y_EVENTS + 46;
  for (int i = 0; i < SPARK_N; i++) {
    int h = (int)(spark[(spark_head + i) % SPARK_N] * SPARK_H / top);
    for (int y = 0; y < SPARK_H; y++)
      put_pixel(PX + i, y0 + SPARK_H - 1 - y, y >= h ? kPanelBg : (y >= h - 2 ? kSpark : kSparkFill));
  }
  fill_rect(PX, y0 + SPARK_H, SPARK_N, 1, kDivider);
}

static void draw_network(const det_result_t *r) {
  char value[24], sub[24];
  uint16_t c = kValue;
  if (r->status != 0) {
    snprintf(value, sizeof(value), "error");
    snprintf(sub, sizeof(sub), "%ld", (long)r->status);
  } else if (r->seq == 0) {
    snprintf(value, sizeof(value), "starting");
    sub[0] = 0;
  } else if (r->ms_nn == 0) { /* the frame gate skipped the network */
    snprintf(value, sizeof(value), "idle");
    snprintf(sub, sizeof(sub), "nothing moves");
    c = kIdle;
  } else {
    uint32_t t = r->ms_total ? r->ms_total : 1;
    snprintf(value, sizeof(value), "%lu ms", (unsigned long)r->ms_nn);
    snprintf(sub, sizeof(sub), "%lu.%lu /s", (unsigned long)(1000 / t), (unsigned long)(10000 / t % 10));
  }
  fill_rect(0, Y_NET + 18, PANEL_W, font_large.height, kPanelBg);
  int x = draw_text(PX, Y_NET + 18, &font_large, c, value);
  draw_text(x + 8, Y_NET + 18 + font_large.ascent - font_tiny.ascent, &font_tiny, kLabel, sub);
}

/* A rectangle with its corners cut by one pixel, filled with c */
static void round_rect(int x, int y, int w, int h, uint16_t c) {
  fill_rect(x + 1, y, w - 2, h, c);
  fill_rect(x, y + 1, w, h - 2, c);
}

/* The record button and the recorder's state next to it */
static void draw_rec_row(void) {
  char line[24];
  uint16_t c = kLabel;
  int rec = sd_ctrl.state == SD_RECORDING, nocard = sd_ctrl.state == SD_NOCARD;
  fill_rect(0, Y_BTN, PANEL_W, LCD_H - Y_BTN, kPanelBg);
  round_rect(PX, Y_BTN, BTN_W, BTN_H, rec ? kRec : (nocard ? kIdle : kDivider));
  if (rec) {
    fill_rect(PX + 10, Y_BTN + 6, 9, 9, kWhite); /* stop: a square */
    draw_text(PX + 25, Y_BTN + 5, &font_label, kWhite, "STOP");
  } else {
    round_rect(PX + 1, Y_BTN + 1, BTN_W - 2, BTN_H - 2, kWhite);
    fill_circle(PX + 14, Y_BTN + BTN_H / 2, 4, nocard ? kIdle : kRec); /* record: a red dot */
    draw_text(PX + 25, Y_BTN + 5, &font_label, nocard ? kIdle : kValue, nocard ? "NO SD" : "REC");
  }
  if (rec) {
    uint32_t s = (HAL_GetTick() - sd_ctrl.start_tick) / 1000;
    snprintf(line, sizeof(line), "%lu:%02lu %luM%s", (unsigned long)(s / 60), (unsigned long)(s % 60),
             (unsigned long)(sd_ctrl.bytes >> 20), sd_ctrl.dropped ? "!" : "");
    c = kRec;
  } else if (sd_ctrl.state == SD_PLAYING) {
    snprintf(line, sizeof(line), "replay");
  } else if (sd_ctrl.state == SD_ERROR) {
    snprintf(line, sizeof(line), "card error");
    c = kRec;
  } else {
    snprintf(line, sizeof(line), "cpu %lu%%%s", (unsigned long)cpu_busy_percent, ev_overruns ? " !" : "");
  }
  draw_text(PX + BTN_W + 7, Y_BTN + 5, &font_tiny, c, line);
}

/* ---- record button input: a touch on the bottom of the panel or the blue USER button (B1, PI11) -------------
   The touch controller (FT5336 on I2C3, the bus CubeMX set up for it) is polled; its X axis is the LCD's y. A
   press toggles the recorder (sdcard.c, the same commands the PC sends over SWD), at most once a second. */
extern I2C_HandleTypeDef hi2c3;
#define TS_ADDR 0x70 /* FT5336 */
#define REC_LOCKOUT_MS 1000

/* For a test over SWD: touches seen, the last one's panel coordinates, failed reads in a row */
volatile uint32_t touch_count, touch_errors;
volatile int touch_x = -1, touch_y = -1;

static void short_delay(void) {
  for (volatile int i = 0; i < 400; i++) {
  }
}

/* Free the touch bus and reset I2C3: a read cut short (a reset of the processor, an error) can leave the touch
   controller holding SDA low; up to nine clock pulses and a STOP release it, then I2C3 gets its pins back. */
static void touch_bus_recover(void) {
  GPIO_InitTypeDef g = {GPIO_PIN_7 | GPIO_PIN_8, GPIO_MODE_OUTPUT_OD, GPIO_PULLUP, GPIO_SPEED_FREQ_LOW, 0};
  __HAL_I2C_DISABLE(&hi2c3);
  HAL_GPIO_WritePin(GPIOH, GPIO_PIN_7 | GPIO_PIN_8, GPIO_PIN_SET);
  HAL_GPIO_Init(GPIOH, &g); /* PH7 = SCL, PH8 = SDA */
  for (int i = 0; i < 9 && HAL_GPIO_ReadPin(GPIOH, GPIO_PIN_8) == GPIO_PIN_RESET; i++) {
    HAL_GPIO_WritePin(GPIOH, GPIO_PIN_7, GPIO_PIN_RESET), short_delay();
    HAL_GPIO_WritePin(GPIOH, GPIO_PIN_7, GPIO_PIN_SET), short_delay();
  }
  HAL_GPIO_WritePin(GPIOH, GPIO_PIN_8, GPIO_PIN_RESET), short_delay(); /* STOP: SDA rises while SCL is high */
  HAL_GPIO_WritePin(GPIOH, GPIO_PIN_8, GPIO_PIN_SET), short_delay();
  g.Mode = GPIO_MODE_AF_OD, g.Speed = GPIO_SPEED_FREQ_VERY_HIGH, g.Alternate = GPIO_AF4_I2C3;
  HAL_GPIO_Init(GPIOH, &g);
  __HAL_I2C_ENABLE(&hi2c3);
  hi2c3.State = HAL_I2C_STATE_READY;
}

static int touch_point(int *x, int *y) {
  static uint32_t calls;
  uint8_t b[5]; /* TD_STATUS, P1_XH, P1_XL, P1_YH, P1_YL */
  if (calls++ == 0) touch_bus_recover();
  if (touch_errors >= 5 && calls % 64) return 0; /* no answer: try again every few seconds */
  /* No task switch during the read: a pre-emption by the event task would run the HAL into its timeout in the
     middle of a transfer and leave the bus busy. The read takes about 0.7 ms. */
  vTaskSuspendAll();
  HAL_StatusTypeDef st = HAL_I2C_Mem_Read(&hi2c3, TS_ADDR, 0x02, I2C_MEMADD_SIZE_8BIT, b, sizeof(b), 5);
  xTaskResumeAll();
  if (st != HAL_OK) {
    touch_errors++;
    touch_bus_recover();
    return 0;
  }
  touch_errors = 0;
  int n = b[0] & 0x0F;
  if (n < 1 || n > 5) return 0;
  *y = ((b[1] & 0x0F) << 8) | b[2];
  *x = ((b[3] & 0x0F) << 8) | b[4];
#if VIEW_ROT180
  *x = LCD_W - 1 - *x, *y = LCD_H - 1 - *y;
#endif
  touch_count++, touch_x = *x, touch_y = *y;
  return 1;
}

static void rec_toggle(void) {
  if (sd_ctrl.cmd != SD_CMD_NONE) return; /* a command (from the PC or a press) is still being served */
  if (sd_ctrl.state == SD_RECORDING)
    sd_ctrl.cmd = SD_CMD_STOP;
  else if (sd_ctrl.state != SD_BUSY && sd_ctrl.state != SD_PLAYING)
    sd_ctrl.cmd = SD_CMD_START; /* from idle, done, no card or error: the recorder mounts the card again */
}

static void rec_input(void) {
  static int was_down = 1; /* a finger or the button already down at start-up does not count */
  static uint32_t last;
  int x, y;
  int down = HAL_GPIO_ReadPin(GPIOI, GPIO_PIN_11) == GPIO_PIN_SET ||
             (touch_point(&x, &y) && x < PANEL_W && y >= Y_BTN - 8);
  if (down && !was_down && HAL_GetTick() - last >= REC_LOCKOUT_MS) {
    rec_toggle();
    last = HAL_GetTick();
  }
  was_down = down;
}

static void draw_panel(const det_result_t *r) {
  draw_people((int)r->count);
  draw_events(ev_rate_kevps);
  draw_network(r);
  draw_rec_row();
}

/* ---- event view (viewer coordinates: column X of the picture, row Y - VIEW_CROP; VIEW_ROT180 turns them) --- */

static inline void put_ev(uint8_t *fb, int x, int y, uint8_t c) {
  if ((unsigned)x >= EV_VIEW_W || (unsigned)y >= EV_VIEW_H) return;
#if VIEW_ROT180
  x = EV_VIEW_W - 1 - x, y = EV_VIEW_H - 1 - y;
#endif
  fb[y * EV_VIEW_W + x] = c;
}

static void plot_view(void *ctx, int x, int y, uint32_t c) { put_ev((uint8_t *)ctx, x, y, (uint8_t)c); }

static void fill_ev(uint8_t *fb, int x0, int y0, int w, int h, uint8_t c) {
  for (int y = y0; y < y0 + h; y++)
    for (int x = x0; x < x0 + w; x++) put_ev(fb, x, y, c);
}

typedef struct {
  int x, y, w, h;
} rect_t;

static int overlaps(const rect_t *a, const rect_t *b) {
  return a->x < b->x + b->w && b->x < a->x + a->w && a->y < b->y + b->h && b->y < a->y + a->h;
}

/* The boxes of one detection: a fresh detection gets a 2 px frame and a "person 87%" tag (above the box, or
   inside its top when there is no room above; shortened to "87%" or left out where it would cover another
   tag); a box held over from an earlier run (only with DET_COUNT_RUNS > 1, see detector.c) gets a thin frame and
   no tag. */
static void draw_boxes(uint8_t *fb, const det_result_t *r) {
  rect_t tags[DET_MAX_BOXES];
  int n_tags = 0;
  int order[DET_MAX_BOXES];
  for (int i = 0; i < r->n; i++) order[i] = i;
  for (int i = 1; i < r->n; i++) /* highest score first: it gets its tag */
    for (int j = i; j > 0 && r->box[order[j]].score > r->box[order[j - 1]].score; j--) {
      int t = order[j];
      order[j] = order[j - 1], order[j - 1] = t;
    }
  for (int k = 0; k < r->n; k++) {
    const det_box_t *b = &r->box[order[k]];
    int fresh = !b->held;
    int c0 = (int)(b->x0 * (PIC_SIZE - 1)), c1 = (int)(b->x1 * (PIC_SIZE - 1));
    int r0 = (int)(b->y0 * (PIC_SIZE - 1)) - VIEW_CROP, r1 = (int)(b->y1 * (PIC_SIZE - 1)) - VIEW_CROP;
    for (int t = 0; t < (fresh ? 2 : 1); t++) {
      for (int c = c0; c <= c1; c++) put_ev(fb, c, r0 + t, EV_BOX), put_ev(fb, c, r1 - t, EV_BOX);
      for (int y = r0; y <= r1; y++) put_ev(fb, c0 + t, y, EV_BOX), put_ev(fb, c1 - t, y, EV_BOX);
    }
    if (!fresh) continue;
    int pct = (int)(b->score * 100.f + 0.5f);
    char tag[24];
    for (int shortened = 0; shortened < 2; shortened++) {
      snprintf(tag, sizeof(tag), shortened ? "%d%%" : "person %d%%", pct > 99 ? 99 : pct);
      rect_t t = {0, 0, text_width(&font_label, tag) + 6, font_label.height + 1};
      t.y = r0 - t.h >= 0 ? r0 - t.h : (r0 > 0 ? r0 : 0);
      t.x = c0 + t.w <= EV_VIEW_W ? (c0 > 0 ? c0 : 0) : EV_VIEW_W - t.w;
      int free = 1;
      for (int i = 0; i < n_tags && free; i++) free = !overlaps(&t, &tags[i]);
      if (!free) continue;
      fill_ev(fb, t.x, t.y, t.w, t.h, EV_BOX);
      render_text(plot_view, fb, t.x + 3, t.y, &font_label, EV_BG, tag);
      tags[n_tags++] = t;
      break;
    }
  }
}

/* Blinking REC badge in the top-left corner while the card records */
static void draw_rec(uint8_t *fb, unsigned frame) {
  if (sd_ctrl.state != SD_RECORDING || (frame / 16) % 2) return;
  fill_ev(fb, 6, 6, 44, 16, EV_REC);
  for (int y = -3; y <= 3; y++)
    for (int x = -3; x <= 3; x++)
      if (x * x + y * y <= 10) put_ev(fb, 14 + x, 14 + y, EV_WHITE);
  render_text(plot_view, fb, 21, 8, &font_label, EV_WHITE, "REC");
}

static void config_layers(void) {
  LTDC_LayerCfgTypeDef cfg = {0};
  cfg.WindowX0 = 0;
  cfg.WindowX1 = LCD_W;
  cfg.WindowY0 = 0;
  cfg.WindowY1 = LCD_H;
  cfg.PixelFormat = LTDC_PIXEL_FORMAT_RGB565;
  cfg.Alpha = 255;
  cfg.Alpha0 = 0;
  cfg.BlendingFactor1 = LTDC_BLENDING_FACTOR1_CA;
  cfg.BlendingFactor2 = LTDC_BLENDING_FACTOR2_CA;
  cfg.FBStartAdress = (uint32_t)gui_fb;
  cfg.ImageWidth = LCD_W;
  cfg.ImageHeight = LCD_H;
  HAL_LTDC_ConfigLayer(&hltdc, &cfg, 0);

  /* The event view covers only the right part. Outside its window a layer shows its default colour (black,
     alpha 0), and with constant-alpha blending that black is opaque and hid the panel (2026-11-05: the panel
     was never visible on the glass). Blending with pixel alpha x constant alpha makes it transparent there; L8
     pixels have alpha 1, so the view itself stays opaque. */
  cfg.BlendingFactor1 = LTDC_BLENDING_FACTOR1_PAxCA;
  cfg.BlendingFactor2 = LTDC_BLENDING_FACTOR2_PAxCA;
  cfg.WindowX0 = EV_VIEW_X;
  cfg.WindowX1 = EV_VIEW_X + EV_VIEW_W;
  cfg.PixelFormat = LTDC_PIXEL_FORMAT_L8;
  cfg.FBStartAdress = (uint32_t)ev_fb[1];
  cfg.ImageWidth = EV_VIEW_W;
  cfg.ImageHeight = EV_VIEW_H;
  HAL_LTDC_ConfigLayer(&hltdc, &cfg, 1);
  HAL_LTDC_ConfigCLUT(&hltdc, (uint32_t *)kClut, sizeof(kClut) / sizeof(kClut[0]), 1);
  HAL_LTDC_EnableCLUT(&hltdc, 1);
}

void display_task(void *argument) {
  (void)argument;
  reload_sem = xSemaphoreCreateBinaryStatic(&reload_sem_buf);
  for (int i = 0; i < LCD_W * LCD_H; i++) gui_fb[i] = kPanelBg;
  memset(ev_fb, EV_BG, sizeof(ev_fb));
  draw_panel_static();
  config_layers();
  GPIO_InitTypeDef button = {GPIO_PIN_11, GPIO_MODE_INPUT, GPIO_NOPULL, GPIO_SPEED_FREQ_LOW, 0};
  HAL_GPIO_Init(GPIOI, &button); /* the blue USER button: high while pressed (pull-down on the board) */

  uint8_t *write = ev_fb[0], *shown = ev_fb[1], *spare = ev_fb[2];
  events_set_write_buffer(write);
  det_result_t r;
  for (unsigned frame = 0;; frame++) {
    detector_get_result(&r);
#if HISTO_MODE
    if (events_take_painted()) { /* a new sensor frame is in the write buffer: show it, else keep the last one */
#endif
      /* Rotate: the event task paints into the cleared buffer from now on */
      uint8_t *done = write;
      write = spare;
      events_set_write_buffer(write);

      draw_boxes(done, &r);
      draw_rec(done, frame);

      LTDC_LAYER(&hltdc, 1)->CFBAR = (uint32_t)done;
      HAL_LTDC_Reload(&hltdc, LTDC_RELOAD_VERTICAL_BLANKING);
      xSemaphoreTake(reload_sem, pdMS_TO_TICKS(50));
      spare = shown;
      shown = done;
      ucDrvDmaMemsetTo0(drv_dma, spare, EV_VIEW_W * EV_VIEW_H);
#if HISTO_MODE
    }
#endif

    if (frame % 32 == 0) cpu_load_update(); /* about once a second */
    if (frame % 4 == 0) rec_input();        /* about every 0.1 s */
    if (frame % 8 == 0) draw_panel(&r);
    vTaskDelay(pdMS_TO_TICKS(12));
  }
}
