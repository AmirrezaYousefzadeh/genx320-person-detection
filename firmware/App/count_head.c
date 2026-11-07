/*
 * Counting head: the number of people from the detector's output grid (12x12 cells x 5 anchors x 6 logits), by a
 * tiny network trained on the people-counting dataset (tools/count_head.py) instead of counting thresholded boxes.
 * Standardise, 1x1 conv 30->16 ReLU, 3x3 conv 16->16 ReLU (zero padding), 1x1 conv 16->1 softplus, sum.
 * About 0.4 M float multiply-adds. Weights: count_head_weights.h (tools/count_head_c.py).
 *
 * Copyright (c) 2026 Amirreza Yousefzadeh, University of Twente. Licensed under the Apache License 2.0.
 */
#include <math.h>

#include "count_head.h"
#include "person_nn.h"
#ifndef DET_MIRROR_PASS
#define DET_MIRROR_PASS 0
#endif
#if DET_MIRROR_PASS /* the head was trained on the grid of the same pass setting */
#include "count_head_weights.h"
#else
#include "count_head_weights_single.h"
#endif

#define G PERSON_NN_GRID

float count_head(const float *grid) {
  static float h1[G][G][CH_H], h2[G][G][CH_H];
  for (int r = 0; r < G; r++)
    for (int c = 0; c < G; c++) {
      const float *x = grid + (r * G + c) * CH_IN;
      float xs[CH_IN];
      for (int i = 0; i < CH_IN; i++) xs[i] = (x[i] - CH_MU[i]) / CH_SD[i];
      for (int o = 0; o < CH_H; o++) {
        float acc = CH_B1[o];
        for (int i = 0; i < CH_IN; i++) acc += xs[i] * CH_W1[i][o];
        h1[r][c][o] = acc > 0 ? acc : 0;
      }
    }
  float total = 0;
  for (int r = 0; r < G; r++)
    for (int c = 0; c < G; c++) {
      for (int o = 0; o < CH_H; o++) {
        float acc = CH_B2[o];
        for (int ky = 0; ky < 3; ky++) {
          int rr = r + ky - 1;
          if (rr < 0 || rr >= G) continue;
          for (int kx = 0; kx < 3; kx++) {
            int cc = c + kx - 1;
            if (cc < 0 || cc >= G) continue;
            for (int i = 0; i < CH_H; i++) acc += h1[rr][cc][i] * CH_W2[ky][kx][i][o];
          }
        }
        h2[r][c][o] = acc > 0 ? acc : 0;
      }
      float d = CH_B3[0];
      for (int i = 0; i < CH_H; i++) d += h2[r][c][i] * CH_W3[i];
      total += d > 20.f ? d : log1pf(expf(d)); /* softplus */
    }
  return total;
}
