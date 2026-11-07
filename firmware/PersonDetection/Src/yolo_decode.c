/* See yolo_decode.h. */
#include "yolo_decode.h"

#include <math.h>

#include "person_nn.h"

#define GRID PERSON_NN_GRID
#define N_ANCHORS 5
#define N_CANDIDATES (GRID * GRID * N_ANCHORS)

/* Anchor (w, h) pairs in grid cells, from ST's st_yololcv1_192_config.yaml */
static const float anchors[N_ANCHORS][2] = {
    {0.076023f, 0.258508f}, {0.163031f, 0.413531f}, {0.234769f, 0.702585f},
    {0.427054f, 0.715892f}, {0.748154f, 0.857092f}};

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float clip01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

static float iou(const det_box_t *a, const det_box_t *b) {
  float ix = fminf(a->x1, b->x1) - fmaxf(a->x0, b->x0);
  float iy = fminf(a->y1, b->y1) - fmaxf(a->y0, b->y0);
  if (ix <= 0.f || iy <= 0.f) return 0.f;
  float inter = ix * iy;
  return inter / ((a->x1 - a->x0) * (a->y1 - a->y0) + (b->x1 - b->x0) * (b->y1 - b->y0) - inter + 1e-9f);
}

int yolo_decode(const float *output, float threshold, float nms_iou, det_box_t *boxes, int max_boxes) {
  static det_box_t cand[N_CANDIDATES] __attribute__((section("NN_Arena"))); /* 14 KB, in SDRAM */
  int n_cand = 0;
  for (int row = 0; row < GRID; row++)
    for (int col = 0; col < GRID; col++)
      for (int a = 0; a < N_ANCHORS; a++) {
        const float *f = output + ((row * GRID + col) * N_ANCHORS + a) * 6;
        float score = sigmoid(f[4]);  /* softmax over the single class is 1 */
        if (score < threshold) continue;
        float x = (sigmoid(f[0]) + col) / GRID, y = (sigmoid(f[1]) + row) / GRID;
        float w = expf(f[2]) * anchors[a][0] / GRID, h = expf(f[3]) * anchors[a][1] / GRID;
        det_box_t *c = &cand[n_cand++];
        c->x0 = clip01(x - w / 2), c->y0 = clip01(y - h / 2);
        c->x1 = clip01(x + w / 2), c->y1 = clip01(y + h / 2);
        c->score = score;
      }

  /* Greedy NMS: repeatedly take the best remaining candidate, drop the ones overlapping it. */
  int n = 0;
  while (n < max_boxes) {
    int best = -1;
    for (int i = 0; i < n_cand; i++)
      if (cand[i].score >= 0.f && (best < 0 || cand[i].score > cand[best].score)) best = i;
    if (best < 0) break;
    boxes[n++] = cand[best];
    for (int i = 0; i < n_cand; i++)
      if (cand[i].score >= 0.f && iou(&cand[i], &cand[best]) >= nms_iou) cand[i].score = -1.f;
  }
  return n;
}
