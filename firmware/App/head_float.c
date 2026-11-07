/*
 * Float detection head (make HEAD_FLOAT=1): the network's last 1x1 convolution (batch normalisation folded in) in
 * float after the int8 body, because a single int8 scale for the head's 30 outputs costs box precision (room
 * recordings: AP50 0.836 -> 0.880, AP50:95 0.319 -> 0.399 with this head; tools/headfloat.py writes the weights).
 * out[cell][k] = b[k] + sum_c x[cell][c] W[c][k], x = (q - zero_point) * scale; zero inputs (ReLU) are skipped.
 */
#include "head_float.h"

#if HEAD_FLOAT
#ifdef HEAD_WEIGHTS /* the model's own head: PersonDetection/models/<MODEL>_head.h (Makefile) */
#define HF_STR_(x) #x
#define HF_STR(x) HF_STR_(x)
#include HF_STR(HEAD_WEIGHTS)
#else
#include "head_float_weights.h"
#endif
#include "person_nn.h"

int head_float_channels(void) { return HEAD_C; }

void head_float(const int8_t *q, float scale, int32_t zero_point, float *out) {
  for (int cell = 0; cell < PERSON_NN_GRID * PERSON_NN_GRID; cell++) {
    const int8_t *x = q + cell * HEAD_C;
    float *o = out + cell * HEAD_K;
    for (int k = 0; k < HEAD_K; k++) o[k] = head_b[k];
    for (int c = 0; c < HEAD_C; c++) {
      int32_t v = (int32_t)x[c] - zero_point;
      if (v == 0) continue;
      const float xf = (float)v * scale;
      const float *w = head_W[c];
      for (int k = 0; k < HEAD_K; k++) o[k] += xf * w[k];
    }
  }
}
#endif
