"""Split a detector into an int8 body and a float head for the board (make HEAD_FLOAT=1).

Quantising the whole network costs box precision: the head's 30 outputs (x, y, w, h, objectness per anchor) share one
int8 scale (deployed model, room recordings: AP50 0.891 / AP50:95 0.426 in float, 0.836 / 0.319 in int8, 0.880 / 0.399
with the head in float). The body (everything up to the head's input, 10x10x256 at 160 px) stays int8 under TFLM +
CMSIS-NN; the head, a 1x1 convolution with its batch normalisation folded in (256 -> 30, 0.77 M multiply-adds), runs in
float in App/head_float.c (on the Cortex-M7's FPU: 8 / 13 / 15 ms at 128 / 160 / 192 px).
Writes OUT_body_int8.tflite (uint8 input, int8 output), OUT_head.npz (W 256x30, b 30) and
OUT_head.h (the firmware header; tools/build_hf.sh copies it to firmware/PersonDetection/models/<MODEL>_head.h);
prints how far body+head is from the float model.
Usage: headfloat.py MODEL.keras OUT --data ROOT[,ROOT] [--size 160] [--n 300]
"""
import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))


def split_head(model):
    """(body model, W (C, 30), b (30,)) with a batch normalisation after the last convolution folded in"""
    convs = [i for i, l in enumerate(model.layers) if isinstance(l, tf.keras.layers.Conv2D)]
    k = convs[-1]
    conv = model.layers[k]
    w = conv.get_weights()
    W = w[0][0, 0].astype(np.float64)  # (C, K)
    b = w[1].astype(np.float64) if len(w) > 1 else np.zeros(W.shape[1])
    for l in model.layers[k + 1:]:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            gamma, beta, mean, var = (v.astype(np.float64) for v in l.get_weights())
            s = gamma / np.sqrt(var + l.epsilon)
            W, b = W * s, (b - mean) * s + beta
    body = tf.keras.Model(model.input, conv.input)
    return body, W.astype(np.float32), b.astype(np.float32)


def pictures(data, size, n):
    pics = sorted(str(p) for r in data.split(",") for p in Path(r).glob("*_f*/pictures/pic_*.png"))
    random.Random(0).shuffle(pics)
    import evpic
    return [evpic.network_input(p, size).astype(np.float32) for p in pics[:n]]


def run_body(path, img):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(path)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], img.astype(np.uint8)[None])
    it.invoke()
    q = it.get_tensor(o["index"])[0].astype(np.float32)
    s, z = o["quantization"]
    return (q - z) * s


def write_header(W, b, path, source=""):
    """the firmware header of a float head (W: 256 x 30, b: 30)"""
    rows = ",\n".join("  {" + ", ".join(f"{v:.8e}f" for v in W[c]) + "}" for c in range(W.shape[0]))
    h = Path(path)
    h.write_text(f"/* Float detection head (tools/headfloat.py{' from ' + source if source else ''}): out[k] = b[k] + sum_c x[c] W[c][k],\n"
                 f"   x = the int8 body output dequantised. Generated, do not edit. */\n"
                 f"#define HEAD_C {W.shape[0]}\n#define HEAD_K {W.shape[1]}\n"
                 f"static const float head_W[HEAD_C][HEAD_K] = {{\n{rows}\n}};\n"
                 f"static const float head_b[HEAD_K] = {{" + ", ".join(f"{v:.8e}f" for v in b) + "};\n")
    return h


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--header-from":  # headfloat.py --header-from HEAD.npz OUT.h
        hw = np.load(sys.argv[2])
        print("wrote", write_header(hw["W"], hw["b"], sys.argv[3], sys.argv[2]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--data", required=True)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    model = tf.keras.models.load_model(a.model, compile=False)
    body, W, b = split_head(model)
    rep_pics = pictures(a.data, a.size, a.n)
    conv = tf.lite.TFLiteConverter.from_keras_model(body)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([p[None]] for p in rep_pics)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.int8
    body_path = Path(a.out + "_body_int8.tflite")
    body_path.write_bytes(conv.convert())
    np.savez(a.out + "_head.npz", W=W, b=b)
    # how far is int8 body + float head from the float model?
    diffs = []
    for img in rep_pics[:20]:
        ref = model(img[None], training=False).numpy()[0]
        ours = run_body(str(body_path), img) @ W + b
        diffs.append(np.abs(ours - ref).max())
    print(f"body {body_path} ({body_path.stat().st_size} bytes), head {W.shape[0]}x{W.shape[1]}; max |body+head - float| "
          f"over 20 pictures: median {np.median(diffs):.3f}, max {np.max(diffs):.3f}")
    h = write_header(W, b, f"{a.out}_head.h", Path(a.model).as_posix())
    print("wrote", h)


if __name__ == "__main__":
    main()
