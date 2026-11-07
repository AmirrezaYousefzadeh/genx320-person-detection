/* Counting head on the detector's output grid (count_head.c). Copyright (c) 2026 Amirreza Yousefzadeh,
   University of Twente. Licensed under the Apache License 2.0. */
#ifndef COUNT_HEAD_H
#define COUNT_HEAD_H
#ifdef __cplusplus
extern "C" {
#endif
float count_head(const float *grid); /* PERSON_NN_GRID^2 x 30 logits -> people (not rounded) */
#ifdef __cplusplus
}
#endif
#endif
