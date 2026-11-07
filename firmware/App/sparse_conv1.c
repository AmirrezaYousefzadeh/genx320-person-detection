/*
 * Event-driven first convolution (3x3, stride 2, SAME, ReLU, int8) of st_yololcv1, bit-exact with TFLite.
 *
 * The network's input is the event picture: 128 where nothing happened, 255 or 0 where an event was. After
 * TFLite's quantisation the convolution multiplies the weights with (int8 - zero point) = the uint8 pixel itself,
 * so a window with no event gives bias + 128 * (sum of the taps), the same 16 values everywhere (conv1_const),
 * and only the 2-20 % active pixels change anything. This kernel therefore starts every output row from that
 * constant and scatters the contributions of the active pixels (each reaches at most 2x2 outputs with stride 2),
 * then requantises only the touched columns. Dense CMSIS-NN needs 192*192*9*16 multiplies whatever the picture;
 * this needs 64 per active pixel. The three identical RGB channels are folded into one (conv1_w).
 * Constants from tools/sparse_conv1.py (which also checks the exactness on the PC).
 *
 * Copyright (c) 2026 Amirreza Yousefzadeh, University of Twente. Licensed under the Apache License 2.0.
 */
#include "sparse_conv1.h"

#include <string.h>

#include "sparse_conv1_params.h"

#define IN 192
#define OUT 96
#define CH 16
#define NONE 0x80808080u /* four "nothing" pixels */

/* TFLite's MultiplyByQuantizedMultiplier: doubling high multiply with rounding, then a rounding right shift */
static inline int32_t requant(int32_t acc, int32_t mult, int32_t shift) {
  int32_t left = shift > 0 ? shift : 0, right = shift > 0 ? 0 : -shift;
  int64_t ab = (int64_t)(acc << left) * (int64_t)mult;
  int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  int32_t hi = (int32_t)((ab + nudge) >> 31);
  if (right > 0) {
    int32_t mask = (1 << right) - 1, rem = hi & mask, thr = (mask >> 1) + (hi < 0 ? 1 : 0);
    hi = (hi >> right) + (rem > thr ? 1 : 0);
  }
  return hi;
}

static inline void scatter(int32_t *acc, uint8_t *touched, int ox, int32_t d, const int16_t *w) {
  int32_t *a = acc + ox * CH;
  for (int c = 0; c < CH; c++) a[c] += d * w[c];
  touched[ox] = 1;
}

uint32_t sparse_conv1(const uint8_t *grey, int8_t *out) {
  static int32_t acc[OUT * CH] __attribute__((section("NN_Arena"), aligned(16))); /* one output row, cached SDRAM: SRAM is full */
  static uint8_t touched[OUT];
  uint32_t active = 0;
  int32_t edge_x[CH], edge_y[CH]; /* what the padding tap (kx = 2 / ky = 2) would have contributed */
  for (int c = 0; c < CH; c++) {
    edge_x[c] = 128 * (conv1_w[0][2][c] + conv1_w[1][2][c] + conv1_w[2][2][c]);
    edge_y[c] = 128 * (conv1_w[2][0][c] + conv1_w[2][1][c] + conv1_w[2][2][c]);
  }
  for (int oy = 0; oy < OUT; oy++) {
    memset(touched, 0, sizeof(touched));
    for (int ox = 0; ox < OUT; ox++)
      for (int c = 0; c < CH; c++) acc[ox * CH + c] = conv1_acc0[c];
    for (int c = 0; c < CH; c++) acc[(OUT - 1) * CH + c] -= edge_x[c];
    if (oy == OUT - 1) {
      for (int ox = 0; ox < OUT; ox++)
        for (int c = 0; c < CH; c++) acc[ox * CH + c] -= edge_y[c];
      for (int c = 0; c < CH; c++) acc[(OUT - 1) * CH + c] += 128 * conv1_w[2][2][c]; /* removed twice */
    }
    for (int ky = 0; ky < 3; ky++) {
      int y = 2 * oy + ky;
      if (y >= IN) break;
      const uint32_t *row = (const uint32_t *)(grey + y * IN);
      for (int xw = 0; xw < IN / 4; xw++) {
        uint32_t word = row[xw];
        if (word == NONE) continue;
        for (int b = 0; b < 4; b++) {
          int32_t d = (int32_t)((word >> (8 * b)) & 0xFF) - 128;
          if (d == 0) continue;
          if (ky != 2) active++; /* an even input row is seen twice (ky 0 and 2) */
          int x = 4 * xw + b;
          if (x & 1) {
            scatter(acc, touched, (x - 1) >> 1, d, conv1_w[ky][1]);
          } else {
            scatter(acc, touched, x >> 1, d, conv1_w[ky][0]);
            if (x >= 2) scatter(acc, touched, (x - 2) >> 1, d, conv1_w[ky][2]);
          }
        }
      }
    }
    int8_t *o = out + oy * OUT * CH;
    for (int ox = 0; ox < OUT; ox++, o += CH) {
      if (!touched[ox] && ox != OUT - 1 && oy != OUT - 1) {
        memcpy(o, conv1_const, CH);
        continue;
      }
      const int32_t *a = acc + ox * CH;
      for (int c = 0; c < CH; c++) {
        int32_t v = requant(a[c], conv1_mult[c], conv1_shift[c]) + CONV1_OUT_ZP;
        o[c] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v)); /* ReLU: the zero point is -128 */
      }
    }
  }
  return active;
}
