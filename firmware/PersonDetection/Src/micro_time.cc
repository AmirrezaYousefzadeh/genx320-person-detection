/* TFLM timing hooks on the Cortex-M7 cycle counter (DWT), which person_nn_init() starts. */
#include "tensorflow/lite/micro/micro_time.h"

#include "stm32f7xx.h"

namespace tflite {
namespace micro {

uint32_t ticks_per_second() { return SystemCoreClock; }

uint32_t GetCurrentTimeTicks() { return DWT->CYCCNT; }

}  // namespace micro
}  // namespace tflite
