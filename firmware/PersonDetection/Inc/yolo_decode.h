/*
 * Decoding of the st_yololcv1_192 output (YOLOv2 head: 12x12 cells, 5 anchors, 1 class) into boxes, with
 * non-maximum suppression. Same arithmetic as ST's postprocess.py and tools/yolo_ref.py.
 */
#ifndef YOLO_DECODE_H
#define YOLO_DECODE_H

#include "app.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fills up to max_boxes boxes (best first) with score >= threshold after NMS; returns their number. */
int yolo_decode(const float *output, float threshold, float nms_iou, det_box_t *boxes, int max_boxes);

#ifdef __cplusplus
}
#endif

#endif
