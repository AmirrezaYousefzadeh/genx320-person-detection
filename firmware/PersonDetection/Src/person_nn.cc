/*
 * Person detector on TensorFlow Lite Micro + CMSIS-NN (see person_nn.h).
 *
 * The model is split in two (models/README.md) because its first layers need a 258 KiB tensor arena, which
 * only fits in SDRAM, while the rest needs about 110 KiB, which fits in fast internal SRAM:
 *   part A: 192x192x3 uint8 -> 48x48x32 int8, arena in SDRAM (own cacheable MPU region, MPU_Config() in main.c)
 *   part B: 48x48x32 int8 -> 12x12x30 float, arena in internal SRAM
 * Both parts are in internal flash. Times are counted in CPU cycles with the DWT cycle counter.
 */
#include "person_nn.h"
#include "head_float.h"
#if HEAD_FLOAT && SPARSE_L1
#error "HEAD_FLOAT with SPARSE_L1 is not supported (its bench compares float outputs)"
#endif

#include <stdio.h>
#include <string.h>

#include <new>

#include "stm32f7xx.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

extern "C" const uint8_t nn_model_a[];    /* nn_data.s */
extern "C" const uint8_t nn_model_b[];    /* nn_data.s */
extern "C" const uint8_t nn_test_image[]; /* nn_data.s */
#if SPARSE_L1
extern "C" const uint8_t nn_model_a2[];   /* nn_data.s: the second convolution alone (ops 2 of the full model) */
#include "sparse_conv1.h"
#endif

namespace {

constexpr size_t kArenaSizeA = 320 * 1024;
constexpr size_t kArenaSizeB = 128 * 1024;
constexpr size_t kMiddleLen = (PERSON_NN_INPUT_SIZE / 4) * (PERSON_NN_INPUT_SIZE / 4) * 32; /* 48x48x32 at 192 */
constexpr int kMaxOps = 32;
constexpr int kBenchRuns = 3;

uint8_t arena_a[kArenaSizeA] __attribute__((section("NN_Arena"), aligned(16)));
uint8_t arena_b[kArenaSizeB] __attribute__((aligned(16)));
#if SPARSE_L1
/* SPARSE_L1: the first convolution is App/sparse_conv1.c (event-driven, on the grey picture); TFLM runs the second
   convolution alone (part A2, arena in SDRAM) and then part B as before. Part A (full) stays for the exactness check
   of the benchmark build. */
constexpr size_t kConv1Len = (PERSON_NN_INPUT_SIZE / 2) * (PERSON_NN_INPUT_SIZE / 2) * 16;
uint8_t arena_a2[kArenaSizeA] __attribute__((section("NN_Arena"), aligned(16)));
int8_t conv1_out[kConv1Len] __attribute__((section("NN_Arena"), aligned(16)));
uint8_t grey_test[PERSON_NN_INPUT_SIZE * PERSON_NN_INPUT_SIZE] __attribute__((section("NN_Arena"), aligned(16)));
alignas(tflite::MicroInterpreter) uint8_t interpreter_buffer_a2[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter *part_a2 = nullptr;
uint32_t last_active = 0;
#endif

/* Cycles of each operator of the last run (part A, then part B). */
class OpProfiler : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char *) override {
    uint32_t handle = n_ < kMaxOps ? n_++ : kMaxOps - 1;
    start_[handle] = DWT->CYCCNT;
    return handle;
  }
  void EndEvent(uint32_t handle) override { cycles_[handle] = DWT->CYCCNT - start_[handle]; }
  void Reset() { n_ = 0; }

  uint32_t n_ = 0;
  uint32_t start_[kMaxOps] = {};
  uint32_t cycles_[kMaxOps] = {};
};

OpProfiler profiler;
tflite::MicroMutableOpResolver<5> resolver;
alignas(tflite::MicroInterpreter) uint8_t interpreter_buffer_a[sizeof(tflite::MicroInterpreter)];
alignas(tflite::MicroInterpreter) uint8_t interpreter_buffer_b[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter *part_a = nullptr;
tflite::MicroInterpreter *part_b = nullptr;

/* TFLM messages go to printf, i.e. USART1 = the ST-LINK virtual COM port (115200 baud). */
void log_to_uart(const char *s) { printf("%s", s); }

void cycle_counter_start() {
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->LAR = 0xC5ACCE55;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

tflite::MicroInterpreter *load(const uint8_t *data, uint8_t *buffer, uint8_t *arena, size_t arena_size) {
  const tflite::Model *model = tflite::GetModel(data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return nullptr;
  auto *interpreter = new (buffer) tflite::MicroInterpreter(model, resolver, arena, arena_size, nullptr, &profiler);
  return interpreter->AllocateTensors() == kTfLiteOk ? interpreter : nullptr;
}

}  // namespace

extern "C" {

/* Results of person_nn_bench(), read by tools/nn_bench.py over SWD. */
struct nn_bench_t {
  uint32_t magic; /* 0x32424E4E ("NNB2") once the bench has finished */
  int32_t status; /* 0 = ok, otherwise the error code of person_nn_init() or person_nn_run() */
  uint32_t cpu_hz;
  uint32_t arena_used[2]; /* part A (SDRAM), part B (SRAM) */
  uint32_t run_cycles[kBenchRuns];
  uint32_t n_ops;
  uint32_t l1_mismatch; /* SPARSE_L1: values of the second convolution's output that differ from TFLM's own path */
  uint32_t out_diff_milli; /* SPARSE_L1: max |difference| of the final outputs of the two paths, x1000 */
  uint32_t op_cycles[kMaxOps];
  float output[PERSON_NN_OUTPUT_LEN];
};
__attribute__((used)) struct nn_bench_t nn_bench;

int person_nn_init(void) {
  if (part_b != nullptr) return 0;
  cycle_counter_start();
  RegisterDebugLogCallback(log_to_uart);

  if (resolver.AddQuantize() != kTfLiteOk || resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddDepthwiseConv2D() != kTfLiteOk || resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddDequantize() != kTfLiteOk)
    return -1;
#if SPARSE_L1
#ifdef NN_BENCH
  part_a = load(nn_model_a, interpreter_buffer_a, arena_a, kArenaSizeA); /* only for the exactness check */
  if (part_a == nullptr) return -2;
#endif
  part_a2 = load(nn_model_a2, interpreter_buffer_a2, arena_a2, kArenaSizeA);
  if (part_a2 == nullptr) return -7;
  part_b = load(nn_model_b, interpreter_buffer_b, arena_b, kArenaSizeB);
  if (part_b == nullptr) return -3;
  const TfLiteTensor *in2 = part_a2->input(0), *mid_a = part_a2->output(0);
  const TfLiteTensor *mid_b = part_b->input(0), *out = part_b->output(0);
  if (in2->type != kTfLiteInt8 || in2->bytes != kConv1Len || mid_a->type != kTfLiteInt8 || mid_a->bytes != kMiddleLen ||
      mid_b->type != kTfLiteInt8 || mid_b->bytes != kMiddleLen || out->type != kTfLiteFloat32 ||
      out->bytes != PERSON_NN_OUTPUT_LEN * sizeof(float))
    return -4;
  return 0;
#else
  part_a = load(nn_model_a, interpreter_buffer_a, arena_a, kArenaSizeA);
  if (part_a == nullptr) return -2;
  part_b = load(nn_model_b, interpreter_buffer_b, arena_b, kArenaSizeB);
  if (part_b == nullptr) return -3;

  const TfLiteTensor *in = part_a->input(0);
  const TfLiteTensor *mid_a = part_a->output(0);
  const TfLiteTensor *mid_b = part_b->input(0);
  const TfLiteTensor *out = part_b->output(0);
  if (in->type != kTfLiteUInt8 || in->bytes != PERSON_NN_INPUT_LEN || mid_a->type != kTfLiteInt8 ||
      mid_a->bytes != kMiddleLen || mid_b->type != kTfLiteInt8 || mid_b->bytes != kMiddleLen)
    return -4;
#if HEAD_FLOAT
  /* the split model ends before the 1x1 head: int8 features, grid x grid x 256 (App/head_float.c computes the head) */
  if (out->type != kTfLiteInt8 || out->bytes != (size_t)PERSON_NN_GRID * PERSON_NN_GRID * 256) return -4;
#else
  if (out->type != kTfLiteFloat32 || out->bytes != PERSON_NN_OUTPUT_LEN * sizeof(float)) return -4;
#endif
  return 0;
#endif
}

uint32_t person_nn_last_active(void) {
#if SPARSE_L1
  return last_active;
#else
  return 0;
#endif
}

int person_nn_run(const uint8_t *rgb, float *output) {
  profiler.Reset();
#if SPARSE_L1
  uint32_t h = profiler.BeginEvent("sparse_conv1"); /* counted as the first operator */
  last_active = sparse_conv1(rgb, conv1_out);
  profiler.EndEvent(h);
  memcpy(part_a2->input(0)->data.int8, conv1_out, kConv1Len);
  if (part_a2->Invoke() != kTfLiteOk) return -5;
  memcpy(part_b->input(0)->data.int8, part_a2->output(0)->data.int8, kMiddleLen);
#else
  memcpy(part_a->input(0)->data.uint8, rgb, PERSON_NN_INPUT_LEN);
  if (part_a->Invoke() != kTfLiteOk) return -5;
  memcpy(part_b->input(0)->data.int8, part_a->output(0)->data.int8, kMiddleLen);
#endif
  if (part_b->Invoke() != kTfLiteOk) return -6;
#if HEAD_FLOAT
  const TfLiteTensor *o = part_b->output(0); /* int8 features; the head runs in float (App/head_float.c) */
  head_float(o->data.int8, o->params.scale, o->params.zero_point, output);
#else
  memcpy(output, part_b->output(0)->data.f, PERSON_NN_OUTPUT_LEN * sizeof(float));
#endif
  return 0;
}

void person_nn_bench(void) {
  memset(&nn_bench, 0, sizeof(nn_bench));
  nn_bench.cpu_hz = SystemCoreClock;
  nn_bench.status = person_nn_init();
  if (nn_bench.status == 0) {
#if SPARSE_L1
    nn_bench.arena_used[0] = part_a2->arena_used_bytes();
    for (size_t i = 0; i < sizeof(grey_test); i++) grey_test[i] = nn_test_image[3 * i]; /* one channel of the RGB test image */
    const uint8_t *input = grey_test;
#else
    nn_bench.arena_used[0] = part_a->arena_used_bytes();
    const uint8_t *input = nn_test_image;
#endif
    nn_bench.arena_used[1] = part_b->arena_used_bytes();
    for (int i = 0; i < kBenchRuns && nn_bench.status == 0; i++) {
      uint32_t start = DWT->CYCCNT;
      nn_bench.status = person_nn_run(input, nn_bench.output);
      nn_bench.run_cycles[i] = DWT->CYCCNT - start;
    }
#if SPARSE_L1
    if (nn_bench.status == 0 && part_a != nullptr) { /* TFLM's own first two ops on the same picture, compared after conv 2 */
      uint8_t *rgb = part_a->input(0)->data.uint8;
      for (size_t i = 0; i < sizeof(grey_test); i++) rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = grey_test[i];
      if (part_a->Invoke() == kTfLiteOk) {
        const int8_t *ref = part_a->output(0)->data.int8, *ours = part_a2->output(0)->data.int8;
        for (size_t i = 0; i < kMiddleLen; i++) nn_bench.l1_mismatch += ref[i] != ours[i];
        /* and the whole network through TFLM's path, compared with the sparse path's final output */
        memcpy(part_b->input(0)->data.int8, ref, kMiddleLen);
        if (part_b->Invoke() == kTfLiteOk) {
          const float *o = part_b->output(0)->data.f;
          float worst = 0;
          for (size_t i = 0; i < PERSON_NN_OUTPUT_LEN; i++) {
            float d = o[i] - nn_bench.output[i];
            if (d < 0) d = -d;
            if (d > worst) worst = d;
          }
          nn_bench.out_diff_milli = (uint32_t)(worst * 1000);
        }
      } else {
        nn_bench.l1_mismatch = 0xFFFFFFFF;
      }
    }
#endif
    nn_bench.n_ops = profiler.n_;
    memcpy(nn_bench.op_cycles, profiler.cycles_, sizeof(nn_bench.op_cycles));
  }
  nn_bench.magic = 0x32424E4E;
  printf("person_nn bench: status %ld, %lu cycles per run\r\n", (long)nn_bench.status,
         (unsigned long)nn_bench.run_cycles[kBenchRuns - 1]);
}

}  // extern "C"
