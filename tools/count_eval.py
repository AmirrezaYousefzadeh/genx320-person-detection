"""Counting from the detector's boxes, scored on the counting dataset (count_data.py): for each validation region,
the int8 detector runs on it (cut and scaled to 192x192 like the firmware's input), boxes are decoded, optionally
pruned of part boxes, and counted. Variants: single pass or mirrored pass (logits averaged), threshold, suppression.
Prints, per true count 0..5+, the share counted right, and overall: exact accuracy (classes weighted equally),
mean absolute error, and how often 2 people were counted as 1.
Usage: count_eval.py INDEX.jsonl MODEL.tflite [--per-class 400]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from yolo_ref import decode  # noqa: E402

SIZE, GRID, NA, MAXC = 192, 12, 5, 5


def region_input(png, region, size=SIZE):
    img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    x0, y0, x1, y1 = region
    crop = img[int(round(y0 * h)):int(round(y1 * h)), int(round(x0 * w)):int(round(x1 * w))]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.repeat(crop[..., None], 3, axis=-1)


def iou_cover(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9), inter / (min(aa, ab) + 1e-9)


def suppress(boxes, iou_thr, cover_thr):
    kept = []
    for b in boxes:  # decode returns best first
        if all(iou_cover(b[1:], k[1:])[0] <= iou_thr and iou_cover(b[1:], k[1:])[1] <= cover_thr for k in kept):
            kept.append(b)
    return kept


def main(index, model, per_class=400):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]

    def run(x):
        it.set_tensor(inp["index"], x[None].astype(np.uint8))
        it.invoke()
        return it.get_tensor(outp["index"])[0].reshape(GRID, GRID, NA, 6).copy()

    rows = [json.loads(l) for l in open(index)]
    rng = np.random.default_rng(0)
    val = [r for r in rows if r["split"] == "val"]
    sample = []
    for c in range(MAXC + 1):
        cls = [r for r in val if r["class"] == c]
        sample += [cls[i] for i in rng.choice(len(cls), min(per_class, len(cls)), replace=False)]
    variants = {  # name: (mirror, threshold, iou, cover)
        "1 pass, thr .7, NMS only": (False, 0.7, 1.0, 1.01),
        "1 pass, thr .7, parts": (False, 0.7, 0.3, 0.6),
        "mirror, thr .7, parts (board now)": (True, 0.7, 0.3, 0.6),
        "mirror, thr .6, parts": (True, 0.6, 0.3, 0.6),
        "mirror, thr .6, NMS only": (True, 0.6, 1.0, 1.01),
        "mirror, thr .6, loose parts": (True, 0.6, 0.45, 0.8),
        "1 pass, thr .6, loose parts": (False, 0.6, 0.45, 0.8),
    }
    res = {v: np.zeros((MAXC + 1, MAXC + 1), int) for v in variants}  # [true, predicted]
    for r in sample:
        x = region_input(r["png"], r["region"])
        a = run(x)
        b = run(x[:, ::-1].copy())[:, ::-1].copy()
        b[..., 0] = -b[..., 0]
        outs = {False: a.reshape(GRID, GRID, NA * 6), True: ((a + b) / 2).reshape(GRID, GRID, NA * 6)}
        for v, (mir, thr, iou_t, cov_t) in variants.items():
            boxes = suppress(decode(outs[mir], threshold=thr, nms_iou=0.5), iou_t, cov_t)
            res[v][r["class"], min(len(boxes), MAXC)] += 1
    print(f"{len(sample)} validation regions, up to {per_class} per true count")
    print(f"{'variant':36s} " + " ".join(f"{'ok@' + str(c) + ('+' if c == MAXC else ''):>6s}" for c in range(MAXC + 1))
          + f" {'exact':>6s} {'MAE':>5s} {'2->1':>5s}")
    for v, m in res.items():
        per = m.diagonal() / m.sum(1).clip(1)
        pred = np.arange(MAXC + 1)
        mae = (np.abs(pred[None, :] - pred[:, None]) * m).sum() / m.sum()
        print(f"{v:36s} " + " ".join(f"{p:6.2f}" for p in per) + f" {per.mean():6.3f} {mae:5.2f} "
              f"{m[2, 1] / max(m[2].sum(), 1):5.2f}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[a.index("--per-class") + 1]) if "--per-class" in a else 400)
