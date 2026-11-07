"""The people-counting benchmark: every counting method on the same samples of a counting index (count_data.py).
  boxes      the detector's boxes counted (single or mirrored pass, threshold, part-box suppression)
  head       a tiny head on the detector's output grid (count_head.py)
  cnn        a dedicated counting network (count_train.py, int8)
Per true count 0..5+: share counted right; exact accuracy (classes present, weighted equally); mean absolute error;
2 counted as 1. Usage: count_bench.py INDEX.jsonl [--split val|all] [--per-class 400] [--detector M.tflite]
    [--head DIR] [--cnn NAME=M.tflite ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from count_eval import GRID, NA, region_input, suppress  # noqa: E402
from count_train import MAXC  # noqa: E402
from yolo_ref import decode  # noqa: E402


def interp(path):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(str(path))
    it.allocate_tensors()
    return it, it.get_input_details()[0], it.get_output_details()[0]


def run(m, x):
    it, i, o = m
    it.set_tensor(i["index"], x[None].astype(np.uint8))
    it.invoke()
    return it.get_tensor(o["index"])[0]


def main():
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    rows = [json.loads(l) for l in open(a[0])]
    split = opt("--split", "val")
    rows = [r for r in rows if split == "all" or r["split"] == split]
    rng = np.random.default_rng(0)
    per = int(opt("--per-class", 400))
    sample = []
    for c in range(MAXC + 1):
        cls = [r for r in rows if r["class"] == c]
        sample += [cls[i] for i in rng.choice(len(cls), min(per, len(cls)), replace=False)] if cls else []
    det = interp(opt("--detector", "work/finetune/finetune_v4/int8.tflite"))
    from count_head import head_numpy
    head_dir = Path(opt("--head", "work/count/heads_v4"))
    heads = {n: json.loads((head_dir / f"head_{n}.json").read_text()) for n in ("single", "fused")
             if (head_dir / f"head_{n}.json").exists()}
    cnns = {}
    for k, v in enumerate(a):
        if v == "--cnn":
            name, path = a[k + 1].split("=", 1)
            kind = json.loads((Path(path).parent / "metrics.json").read_text()).get("head", "density")
            cnns[name] = (interp(path), kind)
    methods = ["boxes: mirror, thr .7, parts (board)", "boxes: 1 pass, thr .6, loose parts",
               "boxes: mirror, thr .6, NMS only"] + [f"head on detector ({n})" for n in heads] + \
        [f"counting network {n}" for n in cnns]
    conf = {m: np.zeros((MAXC + 1, MAXC + 1), int) for m in methods}
    preds = {m: [] for m in methods}
    for r in sample:
        x = region_input(r["png"], r["region"])
        g1 = run(det, x).reshape(GRID, GRID, NA, 6)
        g2 = run(det, x[:, ::-1].copy()).reshape(GRID, GRID, NA, 6)[:, ::-1].copy()
        g2[..., 0] = -g2[..., 0]
        gf = (g1 + g2) / 2
        flat = lambda g: g.reshape(GRID, GRID, NA * 6)  # noqa: E731
        got = {methods[0]: len(suppress(decode(flat(gf), threshold=0.7), 0.3, 0.6)),
               methods[1]: len(suppress(decode(flat(g1), threshold=0.6), 0.45, 0.8)),
               methods[2]: len(decode(flat(gf), threshold=0.6))}
        for n, meta in heads.items():
            g = (g1 if n == "single" else gf).reshape(1, GRID, GRID, NA * 6).astype(np.float32)
            got[f"head on detector ({n})"] = float(head_numpy(meta, g)[0])
        x128 = region_input(r["png"], r["region"], 128)[..., :1]
        for n, (m, kind) in cnns.items():
            out = run(m, x128)
            got[f"counting network {n}"] = float(np.argmax(out)) if kind == "class" else float(out.sum())
        for m, v in got.items():
            preds[m].append(v)
            conf[m][min(r["class"], MAXC), min(int(round(v)), MAXC)] += 1
    true = np.array([r["count"] for r in sample], float)
    present = [c for c in range(MAXC + 1) if conf[methods[0]][c].sum()]
    print(f"{len(sample)} {split} samples; true counts present: {present}")
    print(f"{'method':38s} " + " ".join(f"{'ok@' + str(c):>6s}" for c in present) + f" {'exact':>6s} {'MAE':>5s} {'2->1':>5s}")
    for m in methods:
        c = conf[m]
        acc = [c[k, k] / c[k].sum() for k in present]
        mae = float(np.abs(np.array(preds[m]) - true).mean())
        two = c[2, 1] / c[2].sum() if c[2].sum() else float("nan")
        print(f"{m:38s} " + " ".join(f"{v:6.2f}" for v in acc) + f" {np.mean(acc):6.3f} {mae:5.2f} {two:5.2f}", flush=True)


if __name__ == "__main__":
    main()
