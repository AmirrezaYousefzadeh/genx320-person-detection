/* Event-driven first convolution of the person network (make SPARSE_L1=1), see sparse_conv1.c.
   Copyright (c) 2026 Amirreza Yousefzadeh, University of Twente. Licensed under the Apache License 2.0. */
#ifndef SPARSE_CONV1_H
#define SPARSE_CONV1_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* grey: the NxN picture (N = PERSON_NN_INPUT_SIZE; 128 nothing, 255 ON, 0 OFF); out: N/2 x N/2 x 16 int8 NHWC,
   bit-exact with TFLite's conv; returns the number of active (non-128) pixels */
uint32_t sparse_conv1(const uint8_t *grey, int8_t *out);
#ifdef __cplusplus
}
#endif
#endif
