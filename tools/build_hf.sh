#!/bin/sh
# Firmware with an int8 body and a float head from a fine-tuning run (tools/headfloat.py, make HEAD_FLOAT=1).
# Usage: [EXTRA="NN_BENCH=1"] tools/build_hf.sh <run> <model name> <count threshold> <build dir> [size]
cd "$(dirname "$0")/.." || exit 1
R=work/finetune/$1; M=firmware/PersonDetection/models/$2; THR=$3; B=$4; S=${5:-160}
export CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3
if [ -f $R/hf_body_int8.tflite ] && [ -f $R/head_float_weights.h ]; then  # converted before: reuse it (the evaluated model)
    cp $R/head_float_weights.h firmware/App/head_float_weights.h
else
    work/venv/bin/python -I tools/headfloat.py $R/best.keras $R/hf --size $S --data work/dataset_sam1,work/dataset_room_train | tail -2 || exit 1
    cp firmware/App/head_float_weights.h $R/head_float_weights.h
fi
cp $R/hf_body_int8.tflite $M.tflite
cp firmware/App/head_float_weights.h ${M}_head.h  # the model's own head weights (Makefile: HEAD_WEIGHTS)
work/venv/bin/python -I -c "
import cv2, numpy as np
p = cv2.resize(cv2.imread('work/dataset_real1/02_walk_3m/pictures/pic_0040.png', 0), ($S, $S), interpolation=cv2.INTER_NEAREST)
np.repeat(p[..., None], 3, -1).astype(np.uint8).tofile('$R/test_input.rgb')"
work/venv/bin/python -I tools/split_model.py $M.tflite 3 ${M}_part_a.tflite ${M}_part_b.tflite $R/test_input.rgb || exit 1
make -C firmware BUILD=$B MODEL=$2 NN_SIZE=$S HEAD_FLOAT=1 COUNT_THR=$THR $EXTRA -j4 2>&1 | tail -3
arm-none-eabi-size firmware/$B/person_detector.elf
