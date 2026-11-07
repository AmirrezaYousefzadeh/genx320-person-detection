#ifndef HEAD_FLOAT_H
#define HEAD_FLOAT_H
#include <stdint.h>
#ifndef HEAD_FLOAT
#define HEAD_FLOAT 0
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* q: the int8 body output (grid x grid x HEAD_C), out: grid x grid x 30 floats as the raw YOLOv2 head */
void head_float(const int8_t *q, float scale, int32_t zero_point, float *out);
int head_float_channels(void);
#ifdef __cplusplus
}
#endif
#endif
