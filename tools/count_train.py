"""Train a small people-counting network on the counting dataset (count_data.py), as density-map regression: the
network turns a 128x128 event picture into a 16x16 map of person density whose sum is the number of people
(each labelled person contributes a mass of 1 spread over its box). About 38 MMAC, int8 for the board.
Augmentation: mirror, event dropout (fewer events, as in dim light), noise events.
Writes OUT/best.keras, OUT/int8.tflite, OUT/metrics.json (per true count 0..5+: share counted right after
rounding; exact accuracy with classes weighted equally; mean absolute error; for the float and int8 models).
Usage: count_train.py --index work/count/count_v1.jsonl --out work/count/run1 [--epochs 40] [--size 128]
       [--width 1.0] [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
MAXC = 5


def load(rows, size, grid):
    """pictures (N, size, size, 1) uint8 and density targets (N, grid, grid, 1) float32"""
    x = np.zeros((len(rows), size, size, 1), np.uint8)
    y = np.zeros((len(rows), grid, grid, 1), np.float32)
    cache = {}
    for i, r in enumerate(rows):
        if r["png"] not in cache:
            cache.clear()
            cache[r["png"]] = cv2.imread(r["png"], cv2.IMREAD_GRAYSCALE)
        img = cache[r["png"]]
        h, w = img.shape
        x0, y0, x1, y1 = r["region"]
        crop = img[int(round(y0 * h)):int(round(y1 * h)), int(round(x0 * w)):int(round(x1 * w))]
        x[i, ..., 0] = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
        for bx0, by0, bx1, by1 in r["boxes"]:
            c0, c1 = bx0 * grid, max(bx1 * grid, bx0 * grid + 1e-3)
            r0, r1 = by0 * grid, max(by1 * grid, by0 * grid + 1e-3)
            cols = np.arange(grid)
            fx = np.clip(np.minimum(cols + 1, c1) - np.maximum(cols, c0), 0, None)
            fy = np.clip(np.minimum(cols + 1, r1) - np.maximum(cols, r0), 0, None)
            m = np.outer(fy, fx)
            y[i, ..., 0] += m / m.sum()
    return x, y


def build(size, grid, width, head):
    """head "density": a grid x grid map whose sum is the count; head "class": scores of 0..MAXC people from
    global pooling over a 4x4 map (the whole picture at once)"""
    import tensorflow as tf
    L = tf.keras.layers
    c = lambda n: max(8, int(n * width))  # noqa: E731
    inp = L.Input((size, size, 1), dtype="float32")
    h = L.Rescaling(1 / 127.5, offset=-128 / 127.5)(inp)  # nothing 0, ON +1, OFF -1
    for n, s in ((16, 2), (32, 2), (32, 1), (64, 2), (64, 1), (64, 1)):
        h = L.Conv2D(c(n), 3, strides=s, padding="same", use_bias=False)(h)
        h = L.BatchNormalization()(h)
        h = L.ReLU()(h)
    if head == "class":
        for n in (128, 128):
            h = L.Conv2D(c(n), 3, strides=2, padding="same", use_bias=False)(h)
            h = L.BatchNormalization()(h)
            h = L.ReLU()(h)
        h = L.GlobalAveragePooling2D()(h)
        return tf.keras.Model(inp, L.Dense(MAXC + 1, name="count_class")(h))
    while h.shape[1] > grid:
        h = L.Conv2D(c(64), 3, strides=2, padding="same", activation="relu")(h)
    d = L.Conv2D(1, 1, activation="relu", name="density")(h)
    return tf.keras.Model(inp, d)


def augment(x, y, rng):
    """mirror (picture and density map together), event dropout and noise events"""
    x, y = x.copy(), y.copy()
    flip = rng.random(len(x)) < 0.5
    x[flip] = x[flip, :, ::-1]
    y[flip] = y[flip, :, ::-1]
    for i in range(len(x)):  # event dropout and noise events
        act = x[i] != 128
        keep = rng.random(act.shape) > rng.uniform(0, 0.4)
        x[i][act & ~keep] = 128
        noise = rng.random(act.shape) < rng.uniform(0, 0.01)
        x[i][noise] = np.where(rng.random(noise.sum()) < 0.5, 0, 255).astype(np.uint8)
    return x, y


def scores(pred_counts, true_counts):
    p = np.minimum(np.rint(pred_counts).astype(int), MAXC)
    t = np.minimum(true_counts, MAXC)
    per = [float((p[t == c] == c).mean()) if (t == c).any() else float("nan") for c in range(MAXC + 1)]
    return {"per_class": [round(v, 3) for v in per], "exact": round(float(np.nanmean(per)), 3),
            "mae": round(float(np.abs(pred_counts - true_counts).mean()), 3),
            "two_as_one": round(float((p[t == 2] == 1).mean()), 3) if (t == 2).any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--head", choices=("density", "class"), default="density")
    args = ap.parse_args()
    import tensorflow as tf
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def say(*a):
        print(*a, flush=True)
        print(*a, file=log, flush=True)

    rows = [json.loads(l) for l in open(args.index)]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if args.limit:
        train, val = train[:args.limit], val[:args.limit // 4]
    xt, yt = load(train, args.size, args.grid)
    xv, yv = load(val, args.size, args.grid)
    ct, cv_ = yt.sum((1, 2, 3)), yv.sum((1, 2, 3))
    say(f"train {len(xt)}, val {len(xv)}; GPUs {tf.config.list_physical_devices('GPU')}")
    model = build(args.size, args.grid, args.width, args.head)
    say(f"parameters {model.count_params()}")
    opt = tf.keras.optimizers.Adam(1e-3)

    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def step(x, y):
        with tf.GradientTape() as tape:
            d = model(tf.cast(x, tf.float32), training=True)
            if args.head == "class":
                loss = ce(tf.minimum(tf.cast(tf.round(tf.reduce_sum(y, [1, 2, 3])), tf.int32), MAXC), d)
            else:
                loss = 100 * tf.reduce_mean(tf.square(d - y)) + tf.reduce_mean(
                    tf.abs(tf.reduce_sum(d, [1, 2, 3]) - tf.reduce_sum(y, [1, 2, 3])))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    def to_count(out):
        return out.argmax(-1).astype(np.float32) if args.head == "class" else out.sum((1, 2, 3))

    def predict(m, x):
        return np.concatenate([to_count(m(tf.cast(x[i:i + 256], tf.float32), training=False).numpy())
                               for i in range(0, len(x), 256)])

    rng = np.random.default_rng(0)
    best, wait = 1e9, 0
    for ep in range(args.epochs):
        opt.learning_rate.assign(1e-3 * 0.5 * (1 + np.cos(np.pi * ep / args.epochs)))
        idx = rng.permutation(len(xt))
        losses = [float(step(*augment(xt[idx[i:i + args.batch]], yt[idx[i:i + args.batch]], rng)))
                  for i in range(0, len(idx), args.batch)]
        s = scores(predict(model, xv), cv_)
        say(f"epoch {ep + 1:3d}  loss {np.mean(losses):.4f}  val exact {s['exact']:.3f} MAE {s['mae']:.3f} "
            f"per class {s['per_class']} 2->1 {s['two_as_one']}")
        if s["mae"] < best:
            best, wait = s["mae"], 0
            model.save(out / "best.keras")
        else:
            wait += 1
            if wait >= 8:
                say("no improvement for 8 epochs, stopping")
                break
    model = tf.keras.models.load_model(out / "best.keras", compile=False)
    metrics = {"head": args.head, "float": scores(predict(model, xv), cv_), "val_samples": len(xv), "train_samples": len(xt),
               "parameters": model.count_params(), "size": args.size, "grid": args.grid}
    rep = xt[rng.choice(len(xt), 300, replace=False)]

    def rep_data():
        for i in range(len(rep)):
            yield [rep[i:i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.float32
    (out / "int8.tflite").write_bytes(conv.convert())
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(str(out / "int8.tflite"))
    it.allocate_tensors()
    i_in, i_out = it.get_input_details()[0], it.get_output_details()[0]
    pc = []
    for k in range(len(xv)):
        it.set_tensor(i_in["index"], xv[k:k + 1])
        it.invoke()
        pc.append(to_count(it.get_tensor(i_out["index"]))[0])
    metrics["int8"] = scores(np.array(pc), cv_)
    metrics["int8_bytes"] = (out / "int8.tflite").stat().st_size
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    say(json.dumps(metrics))


if __name__ == "__main__":
    main()
