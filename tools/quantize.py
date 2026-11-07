"""Post-training int8 quantisation of a fine-tuned Keras model with a chosen number of calibration pictures, and
the precision/recall of float and int8 on the validation pictures (same scoring as finetune.py).

Usage: quantize.py BEST.keras OUT.tflite --data a,b,c [--n 1000] [--val 09,11] [--eval-n 600] [--seed 0]
Why: finetune.py calibrates on 200 pictures; the int8 model lost 0.05-0.12 precision against float in v1/v2.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import finetune as ft  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ROOT = Path(__file__).resolve().parent.parent
ap.add_argument("--data", default=",".join(str(ROOT / f"work/dataset_v{i}") for i in (1, 2, 3)))
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--val", default="09,11")
ap.add_argument("--eval-n", type=int, default=600)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-float", action="store_true")
args = ap.parse_args()

import tensorflow as tf  # noqa: E402

train, val = ft.list_samples(args.data, args.val.split(","))
rng = np.random.default_rng(args.seed)
val_eval = [val[i] for i in rng.choice(len(val), min(args.eval_n, len(val)), replace=False)]
model = tf.keras.models.load_model(args.model, compile=False)
metrics = {"calibration_pictures": args.n}
if not args.no_float:
    metrics["float"] = ft.evaluate(ft.keras_predictor(model), val_eval)
    print("float:", metrics["float"], flush=True)
rep = [train[i][0] for i in rng.choice(len(train), min(args.n, len(train)), replace=False)]


def representative():
    for p in rep:
        yield [ft.load_picture(p)[None]]


conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.uint8
conv.inference_output_type = tf.float32
Path(args.out).write_bytes(conv.convert())
metrics["int8"] = ft.evaluate(ft.tflite_predictor(args.out), val_eval)
print("int8:", metrics["int8"], flush=True)
Path(args.out).with_suffix(".json").write_text(json.dumps(metrics, indent=1))
