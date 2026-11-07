"""Does a second, mirrored network pass make the detections more reliable? Runs the int8 model on each picture and
on its mirror image, turns the mirrored output back (columns reversed, x offset negated) and averages the two
outputs before decoding: in logit space (a box must be found in both passes) or as probabilities (softer).
Scores single pass vs both fusions on the validation pictures, per group of passes.
Usage: eval_tta.py --model M.tflite --groups name=dir,dir;name=dir [--n 400] [--thresholds 0.6,0.7]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from finetune import MIN_PERSON_H, evaluate, list_samples, load_picture  # noqa: E402

GRID, NA = 12, 5


def make_predictors(model):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]

    def run(img):
        x = img.astype(np.uint8) if inp["dtype"] == np.uint8 else img.astype(np.float32)
        it.set_tensor(inp["index"], x[None])
        it.invoke()
        return it.get_tensor(outp["index"])[0].reshape(GRID, GRID, NA, 6).copy()

    cache = {}

    def both(path):
        if path not in cache:
            img = load_picture(path).numpy()
            a, b = run(img), run(img[:, ::-1].copy())
            b = b[:, ::-1].copy()
            b[..., 0] = -b[..., 0]  # x offset: sigmoid(-t) = 1 - sigmoid(t)
            cache.clear()
            cache[path] = (a, b)
        return cache[path]

    def single(path):
        return both(path)[0].reshape(GRID, GRID, NA * 6)

    def logit_mean(path):
        a, b = both(path)
        return ((a + b) / 2).reshape(GRID, GRID, NA * 6)

    def prob_mean(path):
        a, b = both(path)
        f = (a + b) / 2
        p = (1 / (1 + np.exp(-a[..., 4])) + 1 / (1 + np.exp(-b[..., 4]))) / 2
        f[..., 4] = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        return f.reshape(GRID, GRID, NA * 6)

    return {"single pass": single, "mirror, logit mean": logit_mean, "mirror, probability mean": prob_mean}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--thresholds", default="0.6,0.7")
    args = ap.parse_args()
    preds = make_predictors(args.model)
    for g in args.groups.split(";"):
        name, dirs = g.split("=", 1)
        _, val = list_samples(dirs, ["09", "11"], MIN_PERSON_H)
        rng = np.random.default_rng(0)
        sample = [val[i] for i in sorted(rng.choice(len(val), min(args.n, len(val)), replace=False))]
        for thr in (float(t) for t in args.thresholds.split(",")):
            for pname, p in preds.items():
                r = evaluate(p, sample, threshold=thr)
                print(f"{name:8s} thr {thr:.1f}  {pname:26s} P {r['precision']:.3f} R {r['recall']:.3f} "
                      f"F1 {r['f1']:.3f}  fp {r['fp']}", flush=True)


if __name__ == "__main__":
    main()
