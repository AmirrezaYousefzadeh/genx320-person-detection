/*
 * Person detector: ST's YOLO-LC v1 (st_yololcv1, fine-tuned on event pictures; models/README.md) on TensorFlow Lite
 * Micro + CMSIS-NN.
 *
 * Input:  NxN RGB, uint8, row-major (r, g, b per pixel); N = PERSON_NN_INPUT_SIZE (160 in the demo, make NN_SIZE=).
 * Output: N/16 x N/16 grid cells x 5 anchors x (x, y, w, h, objectness, person), float, as the raw YOLOv2 head.
 */
#ifndef PERSON_NN_H
#define PERSON_NN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PERSON_NN_INPUT_SIZE
#define PERSON_NN_INPUT_SIZE 192 /* make NN_SIZE= */
#endif
#ifndef SPARSE_L1
#define SPARSE_L1 0 /* 1 (make SPARSE_L1=1): the first convolution runs event-driven on the grey picture (App/sparse_conv1.c) */
#endif
#define PERSON_NN_INPUT_LEN (PERSON_NN_INPUT_SIZE * PERSON_NN_INPUT_SIZE * (SPARSE_L1 ? 1 : 3))
#ifndef PERSON_NN_GRID
#define PERSON_NN_GRID 12
#endif
#define PERSON_NN_OUTPUT_LEN (PERSON_NN_GRID * PERSON_NN_GRID * 5 * 6)

/* Load the model and plan the tensor arena. Returns 0 on success. */
int person_nn_init(void);

/* Run the network on one image (RGB, or the grey picture with SPARSE_L1). Returns 0 on success. */
int person_nn_run(const uint8_t *rgb, float *output);

/* SPARSE_L1: active pixels of the last picture */
uint32_t person_nn_last_active(void);

/* Run the built-in test image a few times and keep the results in nn_bench (read it over SWD). */
void person_nn_bench(void);

#ifdef __cplusplus
}
#endif

#endif /* PERSON_NN_H */
