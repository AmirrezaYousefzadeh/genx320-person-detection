# Firmware of the live person detector: GenX320 + STM32F746G-DISCO. Needs arm-none-eabi-gcc and openocd.
# The network (PersonDetection/) runs on TensorFlow Lite Micro + CMSIS-NN, built by tflm.mk.
#   make              build build/person_detector.elf (+ size report)
#   make flash        write it to the board with openocd
#   make NN_BENCH=1   also time the network on its test image at start-up (read with tools/nn_bench.py)
#   make TEST_IMAGE=1 detect on the built-in test photo instead of the events (checks the box drawing)
#   make ERC=500       event rate cap on the sensor in kevents/s (default 500, 0 = off)
#   make THRESHOLD=0.7 detection score threshold (default 0.7)
#   make NOFILTERS=1  sensor anti-flicker and STC filters off (default on in event mode); AFK=0/1 STC=0/1 one by one
#   make GATE=N       frame gate: skip the network below N active input pixels (150; 0 = off)
#   make TEST_IMG=PersonDetection/test/evt_192x192.rgb   another built-in test picture (bench, TEST_IMAGE)
#   make SPARSE_L1=1  event-driven first convolution (App/sparse_conv1.c) instead of TFLM's; needs <MODEL>_part_a2.tflite
#   make NN_SIZE=192  network input side (160 default); the MODEL must match (tools/resize_model.py, finetune.py --size)
#   make RAW_BOXES=1  show the raw detections instead of the counted ones (see detector.c)
#   make HEAD_FLOAT=0 the whole network in int8 (default 1: int8 body + float head, App/head_float.c;
#                     MODEL = tools/headfloat.py's body, split; tools/build_hf.sh does all of it)
#   make MIRROR=1     two network passes (picture + mirror image, averaged; default one, see detector.c)
#   make COUNT_HEAD=0 count the boxes instead of using the counting head (see detector.c, count_head.c)
#   make HISTO=1      sensor histogram mode: the sensor sends 320x320 count frames, no event decoding (build_histo/)
#   make MODEL=st_yololcv1_192_int8   build with ST's original COCO model instead of the fine-tuned one
#   make ROT180=1     board clamped the other way up (picture and screen turned by 180 degrees)
#   make DISPLAY_ROT180=1  only the screen turned (panel at the right)
#   make clean

TARGET   := person_detector
BUILD    := build$(if $(HISTO),_histo,)
PREFIX   := arm-none-eabi-
CC       := $(PREFIX)gcc
CXX      := $(PREFIX)g++
SIZE     := $(PREFIX)size

include sources.mk

# C++ sources and binary data of the network
APP_SRCS := \
	PersonDetection/Src/person_nn.cc \
	PersonDetection/Src/micro_time.cc \
	PersonDetection/Src/nn_data.s

INCDIRS := \
	Core/Inc \
	App \
	PersonDetection/Inc \
	PersonDetection/models \
	Drivers/Prophesee \
	Drivers/Prophesee/genx320 \
	Drivers/Prophesee/rtos \
	Drivers/STM32F7xx_HAL_Driver/Inc \
	Drivers/STM32F7xx_HAL_Driver/Inc/Legacy \
	Drivers/CMSIS/Device/ST/STM32F7xx/Include \
	Drivers/CMSIS/Include \
	Drivers/BSP/STM32746G-Discovery \
	Drivers/BSP/Components/ft5336 \
	Middlewares/Third_Party/FatFs \
	Middlewares/Third_Party/FreeRTOS/Source/include \
	Middlewares/Third_Party/FreeRTOS/Source/portable/GCC/ARM_CM7/r0p1 \
	Middlewares/Third_Party/FreeRTOS/Source/CMSIS_RTOS_V2

MCU      := -mcpu=cortex-m7 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb
DEFS     := -DUSE_HAL_DRIVER -DSTM32F746xx
ifdef NN_BENCH
DEFS     += -DNN_BENCH
endif
ifdef TEST_IMAGE
DEFS     += -DDET_TEST_IMAGE
endif
ifdef HISTO
DEFS     += -DHISTO_MODE=1
endif
ifdef NOFILTERS
AFK := 0
STC := 0
endif
ifdef AFK
DEFS     += -DSENSOR_AFK=$(AFK)
endif
ifdef STC
DEFS     += -DSENSOR_STC=$(STC)
endif
ifdef STC_MS
DEFS     += -DSTC_THRESHOLD_MS=$(STC_MS)
endif
ifdef GATE
DEFS     += -DDET_GATE_ACTIVE=$(GATE)
endif
NN_SIZE  ?= 160
ifdef NN_SIZE
DEFS     += -DPERSON_NN_INPUT_SIZE=$(NN_SIZE) -DPERSON_NN_GRID=$$(($(NN_SIZE)/16))
endif
# float detection head on by default since 2026-11-06 (App/head_float.c): the counting threshold follows
# its scores (0.5); models of older builds need HEAD_FLOAT=0 (e.g. make MODEL=person_evt_v4_int8 NN_SIZE=192 HEAD_FLOAT=0)
HEAD_FLOAT ?= 1
DEFS     += -DHEAD_FLOAT=$(HEAD_FLOAT)
ifeq ($(HEAD_FLOAT),1)
COUNT_THR ?= 0.5
endif
ifdef RAW_BOXES
DEFS     += -DRAW_BOXES=1
endif
ifdef MIRROR
DEFS     += -DDET_MIRROR_PASS=$(MIRROR)
endif
ifdef COUNT_HEAD
DEFS     += -DDET_COUNT_HEAD=$(COUNT_HEAD)
endif
ifdef COUNT_THR
DEFS     += -DDET_COUNT_THRESHOLD=$(COUNT_THR)f
endif
ifdef COUNT_PARTS
DEFS     += -DDET_COUNT_PARTS=$(COUNT_PARTS)
endif
ifdef ERC
DEFS     += -DERC_DEFAULT_KEVPS=$(ERC)
endif
ifdef THRESHOLD
DEFS     += -DDET_THRESHOLD=$(THRESHOLD)f
endif
ifdef ROT180
DEFS     += -DVIEW_ROT180=1 -DSENSOR_ROT180=1
endif
ifdef DISPLAY_ROT180
DEFS     += -DVIEW_ROT180=1
endif
# Network (default: the demo): person_room_v7_160_hf_int8 at NN_SIZE=160, the 160 px detector fine-tuned with the room
# recordings (tools/room_sim.py: faster and more accurate in the room than 192 px);
# person_evt_v4_int8 (monitor passes only; the paper's monitor benches: make MODEL=person_evt_v4_int8 NN_SIZE=192)
# or st_yololcv1_192_int8 (ST's COCO model)
ifeq ($(HEAD_FLOAT),1)
MODEL    ?= person_room_v7_160_hf_int8
else
MODEL    ?= person_room_v7_160_int8
endif
MODEL_A  := PersonDetection/models/$(MODEL)_part_a.tflite
MODEL_B  := PersonDetection/models/$(MODEL)_part_b.tflite
# a float-head model brings its head weights: PersonDetection/models/<MODEL>_head.h (tools/headfloat.py, build_hf.sh)
ifeq ($(HEAD_FLOAT),1)
DEFS     += -DHEAD_WEIGHTS=$(MODEL)_head.h
endif
OPT      := -O2 -g3
# The objects depend on this stamp, which changes whenever the -D flags change (make FOO=1 then make again).
FLAGS_FILE := $(BUILD)/flags
$(shell mkdir -p $(BUILD); [ "$$(cat $(FLAGS_FILE) 2>/dev/null)" = "$(DEFS)" ] || echo "$(DEFS)" > $(FLAGS_FILE))
COMMON   := $(MCU) $(DEFS) $(OPT) $(addprefix -I,$(INCDIRS)) -ffunction-sections -fdata-sections -Wall \
            --specs=nano.specs -MMD -MP
CFLAGS   := $(COMMON) -std=gnu11
ifdef SPARSE_L1
DEFS     += -DSPARSE_L1=1
MODEL_A2 := PersonDetection/models/$(MODEL)_part_a2.tflite
endif
ifdef TEST_IMG
ASFLAGS_TEST := -DNN_TEST_IMAGE='"$(TEST_IMG)"'
endif
ASFLAGS  := $(MCU) -g3 -x assembler-with-cpp -DNN_MODEL_A='"$(MODEL_A)"' -DNN_MODEL_B='"$(MODEL_B)"' $(if $(SPARSE_L1),-DNN_MODEL_A2='"$(MODEL_A2)"',) $(ASFLAGS_TEST)
LDSCRIPT := Core/STM32F746NGHX_FLASH.ld
LDFLAGS  := $(MCU) -T$(LDSCRIPT) --specs=nosys.specs --specs=nano.specs -Wl,-Map=$(BUILD)/$(TARGET).map \
            -Wl,--gc-sections -static -u _printf_float
LIBS     := $(BUILD)/libtflm.a -Wl,--start-group -lc -lm -lstdc++ -lsupc++ -Wl,--end-group

.DEFAULT_GOAL := all
include tflm.mk
APP_CXXFLAGS := $(MCU) $(DEFS) $(OPT) $(addprefix -I,$(INCDIRS) $(TFLM_INC)) $(TFLM_DEFS) -ffunction-sections \
                -fdata-sections -Wall -MMD -MP -std=gnu++17 -fno-exceptions -fno-rtti -fno-threadsafe-statics \
                -fno-use-cxa-atexit

OBJS := $(addprefix $(BUILD)/,$(addsuffix .o,$(basename $(SRCS) $(APP_SRCS))))

.PHONY: all flash clean
all: $(BUILD)/$(TARGET).elf

$(BUILD)/%.o: %.c $(FLAGS_FILE)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@
$(BUILD)/%.o: %.cc $(FLAGS_FILE)
	@mkdir -p $(dir $@)
	$(CXX) $(APP_CXXFLAGS) -c $< -o $@
$(BUILD)/%.o: %.s $(FLAGS_FILE)
	@mkdir -p $(dir $@)
	$(CC) $(ASFLAGS) -c $< -o $@
$(BUILD)/PersonDetection/Src/nn_data.o: $(MODEL_A) $(MODEL_B) $(MODEL_A2) PersonDetection/test/skier_192x192.rgb Makefile $(FLAGS_FILE)

$(BUILD)/$(TARGET).elf: $(OBJS) $(BUILD)/libtflm.a $(LDSCRIPT)
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)
	$(SIZE) $@

flash: $(BUILD)/$(TARGET).elf
	openocd -f board/stm32f746g-disco.cfg -c "program $< verify reset exit"

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d)
