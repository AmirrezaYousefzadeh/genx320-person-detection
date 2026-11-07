# TensorFlow Lite Micro + CMSIS-NN as a static library (build/libtflm.a), with the flags of TFLM's own
# cortex_m_generic build (kernels at -O2). Included by the Makefile.

TFLM_DIR := Middlewares/Third_Party/tflite-micro

TFLM_SRCS := $(filter-out \
	$(TFLM_DIR)/tensorflow/lite/micro/cortex_m_generic/micro_time.cc \
	$(TFLM_DIR)/tensorflow/lite/micro/test_helper%.cc, \
	$(shell find $(TFLM_DIR)/tensorflow $(TFLM_DIR)/signal -name '*.cc') \
	$(shell find $(TFLM_DIR)/third_party/cmsis_nn/Source -name '*.c') \
	$(TFLM_DIR)/third_party/kissfft/kiss_fft.c $(TFLM_DIR)/third_party/kissfft/tools/kiss_fftr.c)

TFLM_INC := \
	$(TFLM_DIR) \
	$(TFLM_DIR)/third_party/flatbuffers/include \
	$(TFLM_DIR)/third_party/cmsis_nn \
	$(TFLM_DIR)/third_party/cmsis_nn/Include \
	$(TFLM_DIR)/third_party/kissfft

TFLM_DEFS := -DCMSIS_NN -DKERNELS_OPTIMIZED_FOR_SPEED -DTF_LITE_MCU_DEBUG_LOG -DTFLM_REMOVE_VIRTUAL_DELETE \
             -DTF_LITE_DISABLE_X86_NEON -DARM_NN_ENABLE_F16=0 -DARM_NN_ENABLE_F32=0

TFLM_COMMON := $(MCU) $(TFLM_DEFS) -O2 -g $(addprefix -I,$(TFLM_INC)) -ffunction-sections -fdata-sections \
               -fno-unwind-tables -fno-asynchronous-unwind-tables -fomit-frame-pointer -MMD -MP
TFLM_CXXFLAGS := $(TFLM_COMMON) -std=c++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics
TFLM_CFLAGS := $(TFLM_COMMON) -std=c17 -I$(TFLM_DIR)/third_party/cmsis/CMSIS/Core/Include

TFLM_OBJS := $(addprefix $(BUILD)/tflm/,$(addsuffix .o,$(basename $(TFLM_SRCS))))

$(BUILD)/tflm/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(TFLM_CXXFLAGS) -c $< -o $@
$(BUILD)/tflm/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(TFLM_CFLAGS) -c $< -o $@

$(BUILD)/libtflm.a: $(TFLM_OBJS)
	rm -f $@
	$(PREFIX)ar rcs $@ $^

-include $(TFLM_OBJS:.o=.d)
