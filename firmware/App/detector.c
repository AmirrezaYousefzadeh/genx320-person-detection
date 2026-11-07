/*
 * Detector task: collects events for DET_WINDOW_MS into the picture, stretches it to the network's 192x192
 * RGB input (the grey picture copied to all three channels), runs st_yololcv1_192 and decodes the boxes.
 *
 * Copyright (c) 2026 Amirreza Yousefzadeh, University of Twente. Licensed under the Apache License 2.0.
 */
#include "app.h"

#include <math.h>
#include <stdio.h>

#include "count_head.h"
#include "person_nn.h"
#include "yolo_decode.h"

static uint8_t nn_input[PERSON_NN_INPUT_LEN] __attribute__((section("NN_Arena"), aligned(32)));
static float nn_output[PERSON_NN_OUTPUT_LEN] __attribute__((section("NN_Arena"), aligned(32)));

/* Mirrored second pass (make MIRROR=0 turns it off): the network runs on the picture and on its mirror image; the
   mirrored output is turned back (columns reversed, x offset negated) and the two are averaged as logits before
   decoding, so a box must be found in both passes. Validation pictures, threshold 0.7 (tools/eval_tta.py): false
   boxes -11 % (Eizo), -15 % (Samsung), -29 % (dimmed), recall +0.01..0.02; twice the network time. */
#ifndef DET_MIRROR_PASS
#define DET_MIRROR_PASS 0 /* off since the room-trained model: in the room it no longer adds accuracy, it doubles
                             the delay (tools/room_sim.py, 2026-11-05) */
#endif
#if DET_MIRROR_PASS
static float nn_output2[PERSON_NN_OUTPUT_LEN] __attribute__((section("NN_Arena"), aligned(32)));

static void mirror_input(uint8_t *img) {
  const int s = PERSON_NN_INPUT_SIZE, ch = SPARSE_L1 ? 1 : 3;
  for (int y = 0; y < s; y++) {
    uint8_t *row = img + y * s * ch;
    for (int a = 0, b = s - 1; a < b; a++, b--)
      for (int c = 0; c < ch; c++) {
        uint8_t t = row[a * ch + c];
        row[a * ch + c] = row[b * ch + c], row[b * ch + c] = t;
      }
  }
}

static void fuse_mirrored(float *out, const float *mir) {
  const int g = PERSON_NN_GRID, na = 5;
  for (int r = 0; r < g; r++)
    for (int c = 0; c < g; c++)
      for (int a = 0; a < na; a++) {
        float *o = out + ((r * g + c) * na + a) * 6;
        const float *m = mir + ((r * g + (g - 1 - c)) * na + a) * 6;
        o[0] = 0.5f * (o[0] - m[0]); /* x offset: sigmoid(-t) = 1 - sigmoid(t) */
        for (int k = 1; k < 6; k++) o[k] = 0.5f * (o[k] + m[k]);
      }
}
#endif
static det_result_t result;

/* What the screen shows and counts (tools/count_sim.py replays logged runs through it). The network often puts
   two boxes on one person (a part and the whole, overlapping too little for the NMS), and at 0.6 s per run a
   walking person moves by more than a box width, so a tracker by overlap kept the old box as a second person:
   with one person in front of the board, 2 or more were shown in 24 % of the runs (2026-11-05). Now:
   1. part boxes are suppressed: a box overlapping a higher-scored one by IoU > 0.3, or lying more than 60 %
      inside it (or it inside the box), goes;
   2. the count is the median of the last DET_COUNT_RUNS runs' box counts (a run the frame gate skipped counts 0):
      one person was then shown as 2 or more in 1 % of the runs, about 1.8 s of delay before the count changes;
   3. the screen shows exactly that many boxes: this run's best ones, and when this run found fewer, the boxes of
      the previous run that do not overlap them (drawn thin: their score is halved below the threshold).
   The serial D line keeps the raw boxes: the bench scores the network, not this. make RAW_BOXES=1 shows them. */
/* The count from the counting head on the detector's output grid (count_head.c; make COUNT_HEAD=0: count the
   boxes instead). Monitor validation, true count 0..5+: exact 0.41 against 0.32 for the boxes (board setting before),
   2 people counted as 1: 6 % against 40 % (tools/count_bench.py); but for 0..3 people, the demo's range, it is worse
   (right in 0.34 of the samples against 0.47..0.49 for the boxes): off by default. The boxes to draw come from candidates down to
   DET_CAND_THRESHOLD, so that a person the head counts can still get a box. */
#ifndef DET_COUNT_HEAD
#define DET_COUNT_HEAD 0 /* off: on the monitor data the head is worse than counting boxes for 0..3 people */
#endif
#define DET_CAND_THRESHOLD 0.5f
/* Counting the boxes (tools/room_sim.py: room recordings overlaid to 2 people, the board's logic replayed). With the
   room-trained model (person_room_v7) on the room validation scenarios: threshold 0.7 with part removal, mirrored
   pass, median 3: right for 0 / 1 / 2 people in 0.65 / 0.79 / 0.80 of the runs (v4 with the same logic: 0.72 / 0.51 /
   0.60). The
   threshold and whether part boxes are pruned first. make COUNT_THR=0.6 COUNT_PARTS=0 */
#ifndef DET_COUNT_THRESHOLD
#define DET_COUNT_THRESHOLD 0.7f
#endif
#ifndef DET_COUNT_PARTS
#define DET_COUNT_PARTS 1
#endif
#ifndef DET_COUNT_RUNS
#define DET_COUNT_RUNS 1 /* no smoothing: with the room-trained model a median over runs cost more at every entry
                            and exit than it saved (room replay, 160 px, one pass: 0/1/2 people right in
                            0.84/0.88/0.92 of the runs, median delay to the right count 0.60 s; median 3:
                            0.77/0.87/0.67) */
#endif
#define DET_PART_IOU 0.3f
#define DET_PART_COVER 0.6f

static float box_overlap(const det_box_t *a, const det_box_t *b, float *cover) {
  float ix = fminf(a->x1, b->x1) - fmaxf(a->x0, b->x0), iy = fminf(a->y1, b->y1) - fmaxf(a->y0, b->y0);
  if (ix <= 0 || iy <= 0) {
    *cover = 0;
    return 0;
  }
  float inter = ix * iy, area_a = (a->x1 - a->x0) * (a->y1 - a->y0), area_b = (b->x1 - b->x0) * (b->y1 - b->y0);
  *cover = inter / fminf(area_a, area_b);
  return inter / (area_a + area_b - inter);
}

/* Sorts by score (highest first) and drops part boxes in place; returns how many are left. */
__attribute__((unused)) static int suppress_parts(det_box_t *b, int n) {
  for (int i = 1; i < n; i++)
    for (int j = i; j > 0 && b[j].score > b[j - 1].score; j--) {
      det_box_t t = b[j];
      b[j] = b[j - 1], b[j - 1] = t;
    }
  int k = 0;
  for (int i = 0; i < n; i++) {
    int keep = 1;
    for (int j = 0; j < k && keep; j++) {
      float cover, iou = box_overlap(&b[i], &b[j], &cover);
      keep = iou <= DET_PART_IOU && cover <= DET_PART_COVER;
    }
    if (keep) b[k++] = b[i];
  }
  return k;
}

static int median_count(int n) {
  static int hist[DET_COUNT_RUNS], filled, head;
  hist[head] = n, head = (head + 1) % DET_COUNT_RUNS;
  if (filled < DET_COUNT_RUNS) filled++;
  int s[DET_COUNT_RUNS];
  memcpy(s, hist, sizeof(s));
  for (int i = 1; i < filled; i++)
    for (int j = i; j > 0 && s[j] < s[j - 1]; j--) {
      int t = s[j];
      s[j] = s[j - 1], s[j - 1] = t;
    }
  return s[filled / 2];
}

/* The boxes to show for `count` people: this run's best ones, then boxes of the previous run that do not overlap
   them, held at most DET_HOLD_RUNS runs (drawn thin) */
#define DET_HOLD_RUNS 2
static int select_boxes(const det_box_t *fresh, int n_fresh, int count, det_box_t *out) {
  static det_box_t prev[DET_MAX_BOXES];
  static int n_prev;
  int m = 0;
  for (int i = 0; i < n_fresh && m < count; i++) {
    out[m] = fresh[i];
    out[m++].held = 0;
  }
  for (int i = 0; i < n_prev && m < count; i++) {
    int free = 1;
    for (int j = 0; j < m && free; j++) {
      float cover, iou = box_overlap(&prev[i], &out[j], &cover);
      free = iou <= DET_PART_IOU && cover <= DET_PART_COVER;
    }
    if (free && prev[i].held < DET_HOLD_RUNS) {
      out[m] = prev[i];
      out[m++].held++;
    }
  }
  memcpy(prev, out, m * sizeof(*out));
  n_prev = m;
  return m;
}

void detector_get_result(det_result_t *out) {
  taskENTER_CRITICAL();
  *out = result;
  taskEXIT_CRITICAL();
}

static void publish(const det_result_t *r) {
  taskENTER_CRITICAL();
  result = *r;
  taskEXIT_CRITICAL();
}

/* Nearest-neighbour stretch of the 320x320 picture to the network's input: grey copied to R, G, B, or the grey
   picture alone with SPARSE_L1. Returns the number of active (non-grey) input pixels. */
static uint32_t make_input(void) {
  static uint16_t src_idx[PERSON_NN_INPUT_SIZE];
  if (src_idx[1] == 0)
    for (int i = 0; i < PERSON_NN_INPUT_SIZE; i++) src_idx[i] = (uint16_t)(i * PIC_SIZE / PERSON_NN_INPUT_SIZE);
  uint8_t *dst = nn_input;
  uint32_t active = 0;
  for (int y = 0; y < PERSON_NN_INPUT_SIZE; y++) {
    const uint8_t *row = nn_pic + src_idx[y] * PIC_SIZE;
    for (int x = 0; x < PERSON_NN_INPUT_SIZE; x++) {
      uint8_t g = row[src_idx[x]];
      active += g != PIC_NONE;
#if SPARSE_L1
      *dst++ = g;
#else
      *dst++ = g;
      *dst++ = g;
      *dst++ = g;
#endif
    }
  }
  return active;
}

/* Frame gate: with fewer active input pixels than this, nothing moves and the network is not run (the boxes
   fade through the tracker). 0 turns the gate off. make GATE=N. */
#ifndef DET_GATE_ACTIVE
#define DET_GATE_ACTIVE 150
#endif

void detector_task(void *argument) {
  (void)argument;
  det_result_t r = {0};
  r.status = person_nn_init();
  publish(&r);
  if (r.status != 0) vTaskSuspend(NULL);
  /* Why did the board (re)start? RCC_CSR: bit 25 brown-out, 26 NRST pin, 27 power-on, 28 software,
     29/30 watchdogs, 31 low-power. Kept and printed in every E line, the flags cleared so that the value names
     this start only (the board restarted by itself once on 2026-11-04, cause unknown). */
  uint32_t reset_csr = RCC->CSR;
  __HAL_RCC_CLEAR_RESET_FLAGS();

  for (;;) {
#if HISTO_MODE /* the sensor made the picture: wait for the next frame */
    static uint32_t seq_used = 0;
    while (events_frame_seq() == seq_used) vTaskDelay(1); /* usually a new frame is already there */
    seq_used = events_frame_seq();
    uint32_t t0 = HAL_GetTick() - DET_WINDOW_MS; /* the frame's accumulation period counts as the window */
    uint32_t c0 = DWT->CYCCNT;
    events_lock_pic();
    uint32_t active = make_input();
    events_unlock_pic();
    uint32_t cyc_input = DWT->CYCCNT - c0;
#else
    uint32_t t0 = HAL_GetTick();
    memset(nn_pic, PIC_NONE, PIC_SIZE * PIC_SIZE);
    vTaskDelay(DET_WINDOW_MS);
    uint32_t c0 = DWT->CYCCNT;
    uint32_t active = make_input();
    uint32_t cyc_input = DWT->CYCCNT - c0;
#endif
    uint32_t t1 = HAL_GetTick();
    int gated = DET_GATE_ACTIVE > 0 && active < DET_GATE_ACTIVE; /* nothing moves: skip the network */
#ifdef DET_TEST_IMAGE /* make TEST_IMAGE=1: detect on the built-in skier photo instead, to check the box drawing */
    extern const uint8_t nn_test_image[];
    gated = 0;
    r.status = person_nn_run(nn_test_image, nn_output);
#else
    if (!gated) {
      r.status = person_nn_run(nn_input, nn_output);
#if DET_MIRROR_PASS
      if (r.status == 0) {
        mirror_input(nn_input);
        r.status = person_nn_run(nn_input, nn_output2);
        if (r.status == 0) fuse_mirrored(nn_output, nn_output2);
      }
#endif
    }
#endif
    r.ms_nn = gated ? 0 : HAL_GetTick() - t1;
    uint32_t c1 = DWT->CYCCNT;
    det_box_t cand[DET_MAX_BOXES], raw[DET_MAX_BOXES];
    int n_cand = (r.status == 0 && !gated) ? yolo_decode(nn_output, DET_CAND_THRESHOLD, DET_NMS_IOU, cand, DET_MAX_BOXES) : 0;
    int n_raw = 0; /* the boxes at the detection threshold: the serial log and the bench (NMS keeps them best first) */
    for (int i = 0; i < n_cand; i++)
      if (cand[i].score >= DET_THRESHOLD) raw[n_raw++] = cand[i];
    if (gated) vTaskDelay(DET_WINDOW_MS); /* keep the cadence; the display shows "idle" */
#if RAW_BOXES
    memcpy(r.box, raw, n_raw * sizeof(raw[0]));
    r.n = r.count = n_raw;
#else
    /* the boxes that count: candidates from DET_COUNT_THRESHOLD, optionally pruned of part boxes; the screen
       shows exactly these (the best `count` of them, then held ones) */
    det_box_t cnt[DET_MAX_BOXES];
    int n_cnt = 0;
    for (int i = 0; i < n_cand; i++)
      if (cand[i].score >= DET_COUNT_THRESHOLD) cnt[n_cnt++] = cand[i];
#if DET_COUNT_PARTS
    n_cnt = suppress_parts(cnt, n_cnt);
#endif
#if DET_COUNT_HEAD
    r.count_raw = (r.status == 0 && !gated) ? count_head(nn_output) : 0.f;
    r.count = median_count((int)lroundf(r.count_raw));
#else
    r.count_raw = (float)n_cnt;
    r.count = median_count(n_cnt);
#endif
    r.n = select_boxes(cnt, n_cnt, r.count, r.box);
#endif
    uint32_t cyc_decode = DWT->CYCCNT - c1;
    r.ms_total = HAL_GetTick() - t0;
    r.seq++;
    publish(&r);
    /* One line per run for the PC (tools/det_log.py): D <tick ms> <seq> <network ms> <n> [x0 y0 x1 y1 score]... C <cpu %> */
    printf("D %lu %lu %lu %d", (unsigned long)HAL_GetTick(), (unsigned long)r.seq, (unsigned long)r.ms_nn, n_raw);
    for (int i = 0; i < n_raw; i++) /* the raw boxes: the bench scores the network, not the smoothing */
      printf(" %.3f %.3f %.3f %.3f %.2f", raw[i].x0, raw[i].y0, raw[i].x1, raw[i].y1, raw[i].score);
    printf(" C %lu N %ld %.2f\r\n", (unsigned long)cpu_busy_percent, (long)r.count, r.count_raw); /* CPU %, people */
    /* Profile line: P <make_input us> <decode+smoothing us> <network ms> <total ms> <cpu %> <events %> <net %> <display %> <sd %> */
    uint32_t cyc_per_us = SystemCoreClock / 1000000;
    printf("P %lu %lu %lu %lu %lu %lu %lu %lu %lu\r\n", (unsigned long)(cyc_input / cyc_per_us),
           (unsigned long)(cyc_decode / cyc_per_us), (unsigned long)r.ms_nn, (unsigned long)r.ms_total,
           (unsigned long)cpu_busy_percent, (unsigned long)cpu_task_percent[0], (unsigned long)cpu_task_percent[1],
           (unsigned long)cpu_task_percent[2], (unsigned long)cpu_task_percent[3]);
    printf("A %lu %lu\r\n", (unsigned long)active, (unsigned long)gated); /* active input pixels, gated */
    /* Event rate line: E <kev/s after the sensor's filters and ERC> <DMA ring overruns> <reset flags>: the bench
       logs the rate as the reference for the light and the monitor (tools/det_log.py keeps the last value per run) */
    printf("E %lu %lu 0x%08lx\r\n", (unsigned long)ev_rate_kevps, (unsigned long)ev_overruns,
           (unsigned long)reset_csr);
  }
}
