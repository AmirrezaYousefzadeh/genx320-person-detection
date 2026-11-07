"""Average precision of a detector on labelled event pictures (the paper's accuracy numbers).

AP50 and AP50:95 (COCO style: 101-point interpolated precision, IoU 0.50, 0.55 .. 0.95), plus precision, recall and F1
at one score threshold. A person below --min-h of the picture height (12 %: 38 px of 320, the dataset's rule) and every
class-1 label (reflection, crowd, hardly visible person) is a region to ignore: a detection on it is neither right
nor wrong, and missing it costs nothing.

  eval_ap.py --model M.keras|M.tflite --data ROOT[,ROOT...] [--select S1,S2] [--size 160] [--stride 1] [--thr 0.5]
             [--json OUT.json] [--name NAME]
--select keeps the segments whose name contains one of the strings (MOT17 numbers like 09 become "MOT17-09-").
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from yolo_ref import ANCHORS  # noqa: E402
import evpic  # noqa: E402

N_ANCHORS = 5


def samples(roots, select, stride, min_h):
    keys = [f"MOT17-{s}-" if s.isdigit() else s for s in select if s]
    out = []
    for seg in sorted(seg for root in roots for seg in Path(root).glob("*_f*")):
        if keys and not any(k in seg.name for k in keys):
            continue
        for png in sorted((seg / "pictures").glob("pic_*.png"))[::stride]:
            gt, ign = [], []
            txt = seg / "labels" / (png.stem + ".txt")
            if txt.exists():
                for line in txt.read_text().splitlines():
                    c, cx, cy, w, h = (float(v) for v in line.split()[:5])
                    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                    (gt if int(c) == 0 and h >= min_h else ign).append(box)
            out.append((str(png), np.array(gt, np.float32).reshape(-1, 4), np.array(ign, np.float32).reshape(-1, 4)))
    return out


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def decode(out, threshold=0.01, nms_iou=0.5, top=100):
    """(scores (K,), boxes (K, 4)) after NMS, best first, from the raw YOLOv2 head"""
    g = int(round((out.size / (N_ANCHORS * 6)) ** 0.5))
    f = out.reshape(g, g, N_ANCHORS, 6).astype(np.float64)
    col, row = np.meshgrid(np.arange(g), np.arange(g))
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    x, y = (sig(f[..., 0]) + col[..., None]) / g, (sig(f[..., 1]) + row[..., None]) / g
    w = np.exp(np.clip(f[..., 2], -10, 10)) * ANCHORS[:, 0] / g
    h = np.exp(np.clip(f[..., 3], -10, 10)) * ANCHORS[:, 1] / g
    s = sig(f[..., 4]).reshape(-1)
    b = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], -1).reshape(-1, 4).clip(0, 1)
    keep = np.where(s >= threshold)[0]
    keep = keep[np.argsort(-s[keep])][:top]
    s, b = s[keep], b[keep]
    chosen = []
    alive = np.ones(len(s), bool)
    m = iou_matrix(b, b)
    for i in range(len(s)):
        if alive[i]:
            chosen.append(i)
            alive &= m[i] < nms_iou
    return s[chosen].astype(np.float32), b[chosen].astype(np.float32)


def predictor(model_path, size):
    if "+" in model_path:  # BODY_int8.tflite+HEAD.npz: the board's HEAD_FLOAT build (tools/headfloat.py)
        from ai_edge_litert.interpreter import Interpreter
        body, head = model_path.split("+")
        hw = np.load(head)
        W, b = hw["W"], hw["b"]
        it = Interpreter(body, num_threads=int(os.environ.get("NUM_THREADS", "2")))
        it.allocate_tensors()
        inp, outp = it.get_input_details()[0], it.get_output_details()[0]
        s, z = outp["quantization"]

        def run_bh(batch):
            outs = []
            for img in batch:
                it.set_tensor(inp["index"], img.astype(np.uint8)[None])
                it.invoke()
                x = (it.get_tensor(outp["index"])[0].astype(np.float32) - z) * s
                outs.append(x @ W + b)
            return outs
        return run_bh
    if model_path.endswith(".tflite"):
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path, num_threads=int(os.environ.get("NUM_THREADS", "2")))
        it.allocate_tensors()
        inp, outp = it.get_input_details()[0], it.get_output_details()[0]

        def run(batch):
            outs = []
            for img in batch:
                x = img.astype(np.uint8) if inp["dtype"] == np.uint8 else img.astype(np.float32)
                it.set_tensor(inp["index"], x[None])
                it.invoke()
                outs.append(it.get_tensor(outp["index"])[0])
            return outs
        return run
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    return lambda batch: list(model.predict(np.stack(batch).astype(np.float32), verbose=0, batch_size=256))


def load(path, size):
    return evpic.network_input(path, size)  # as the firmware and finetune.py (GENX_TEMPORAL=3: with the pictures before)


def ap_from(dets, n_gt):
    """COCO 101-point AP from [(score, is_tp)] (ignored detections already removed)"""
    if n_gt == 0:
        return float("nan")
    if not dets:
        return 0.0
    d = np.array(sorted(dets, key=lambda v: -v[0]))
    tp = np.cumsum(d[:, 1])
    fp = np.cumsum(1 - d[:, 1])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, pts, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def evaluate(results, thr=0.5, groups=None, n_boot=200):
    """results: [(scores, boxes, gt, ign)] -> metrics. groups: the segment of each result; with it, AP50_std is the
    standard deviation of AP50 over n_boot resamplings of the segments (a bootstrap over segments, not pictures:
    consecutive pictures of one segment are not independent)."""
    ious = np.round(np.arange(0.5, 0.96, 0.05), 2)
    per_t = {t: [] for t in ious}
    n_gt = sum(len(g) for _, _, g, _ in results)
    tp_thr = fp_thr = 0
    per_pic50 = []  # (detections at IoU 0.5, people) of every picture, for the bootstrap
    for scores, boxes, gt, ign in results:
        m = iou_matrix(boxes, gt)
        mi = iou_matrix(boxes, ign)
        for t in ious:
            used = np.zeros(len(gt), bool)
            dets_t = []
            for k in range(len(scores)):
                j = -1
                if len(gt):
                    cand = np.where(~used & (m[k] >= t))[0]
                    if len(cand):
                        j = cand[np.argmax(m[k, cand])]
                if j >= 0:
                    used[j] = True
                    dets_t.append((scores[k], 1))
                    if t == 0.5 and scores[k] >= thr:
                        tp_thr += 1
                elif len(ign) and mi[k].max() >= 0.5 or (len(ign) and _inside(boxes[k], ign)):
                    continue  # on a region to ignore
                else:
                    dets_t.append((scores[k], 0))
                    if t == 0.5 and scores[k] >= thr:
                        fp_thr += 1
            per_t[t].extend(dets_t)
            if t == 0.5:
                per_pic50.append((dets_t, len(gt)))
    aps = {float(t): ap_from(per_t[t], n_gt) for t in ious}
    ap50_std = None
    if groups is not None and len(set(groups)) > 1:
        by_seg = {}
        for g, (dets, n) in zip(groups, per_pic50):
            by_seg.setdefault(g, [[], 0])
            by_seg[g][0].extend(dets)
            by_seg[g][1] += n
        segs = list(by_seg.values())
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(segs), len(segs))
            dets = [d for i in pick for d in segs[i][0]]
            boots.append(ap_from(dets, sum(segs[i][1] for i in pick)))
        ap50_std = float(np.nanstd(boots))
    fn = n_gt - tp_thr
    p = tp_thr / max(tp_thr + fp_thr, 1)
    r = tp_thr / max(n_gt, 1)
    return {"AP50": round(aps[0.5], 4), "AP50_95": round(float(np.mean(list(aps.values()))), 4),
            "AP50_std": None if ap50_std is None else round(ap50_std, 4), "segments": len(set(groups)) if groups else None,
            "AP75": round(aps[0.75], 4), "people": int(n_gt), "pictures": len(results),
            f"P@{thr}": round(p, 4), f"R@{thr}": round(r, 4), f"F1@{thr}": round(2 * p * r / max(p + r, 1e-9), 4),
            "tp": tp_thr, "fp": fp_thr, "fn": fn}


def _inside(box, ign):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return bool(((ign[:, 0] <= cx) & (cx <= ign[:, 2]) & (ign[:, 1] <= cy) & (cy <= ign[:, 3])).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--select", default="")
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-h", type=float, default=0.12)
    ap.add_argument("--json", default="")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    t0 = time.time()
    data = samples(a.data.split(","), a.select.split(","), a.stride, a.min_h)
    run = predictor(a.model, a.size)
    results = []
    for i in range(0, len(data), 256):
        chunk = data[i:i + 256]
        outs = run([load(p, a.size) for p, _, _ in chunk])
        for (p, gt, ign), out in zip(chunk, outs):
            s, b = decode(np.asarray(out))
            results.append((s, b, gt, ign))
    res = evaluate(results, a.thr, groups=[Path(p).parent.parent.name for p, _, _ in data])
    res.update({"model": a.model, "data": a.data, "select": a.select, "size": a.size, "stride": a.stride,
                "min_h": a.min_h, "seconds": round(time.time() - t0, 1)})
    print(json.dumps(res))
    if a.json:
        p = Path(a.json)
        allres = json.loads(p.read_text()) if p.exists() else {}
        allres[a.name or f"{Path(a.model).parent.name}/{Path(a.model).name} on {a.select or a.data}"] = res
        p.write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
