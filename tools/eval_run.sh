#!/bin/sh
# Float-head version of a fine-tuning run and its row in the accuracy table. Usage: tools/eval_run.sh <run name> [sets]
cd "$(dirname "$0")/.." || exit 1
R=work/finetune/$1
SETS=${2:-mon30,mon30z,mon60z,pp22test,room,pedro}
SIZE=${3:-160}
export CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3
[ -f $R/hf_head.npz ] || work/venv/bin/python -I tools/headfloat.py $R/best.keras $R/hf --size $SIZE --data work/dataset_sam1,work/dataset_room_train > $R/headfloat.log 2>&1
work/venv/bin/python -I tools/eval_suite.py work/results/ap_table.json "$R/int8.tflite=$1 int8" "$R/hf_body_int8.tflite+$R/hf_head.npz=$1 int8+float head" --sets $SETS --size $SIZE
