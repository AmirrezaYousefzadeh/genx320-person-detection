"""Data curation of the recorded event dataset (Amir, 2026-11-06: find the data that is still wrong, inspect it).

  align  ROOT[,ROOT]  per segment: does the label geometry and timing fit the events? The label offset (dx, dy) and the
                      time shift (pictures) that put the most events inside the boxes, and the contrast (events per
                      pixel inside the boxes / outside). A good segment: offset within 8 px, shift 0, contrast >= 1.6.
                      (This found the 52-56 px error of the sam3 passes.)
  suspects ROOT[,ROOT] --model BODY.tflite+HEAD.npz   per box and picture, the disagreement between the labels and a
                      detector that did not train on these segments: labelled people the detector does not see at all
                      (score < 0.05: invisible, cut, misplaced?) and confident detections (>= 0.7) on no label
                      (unlabelled people, reflections, or detector errors). Written per segment, ranked.
  montage CSV OUT.png [N]   the N worst pictures of a suspects list, labels orange, detections cyan, for inspection.
Outputs go to work/curation/.
"""
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work/curation"


def segments(roots):
    return sorted(s for r in roots.split(",") for s in Path(r).resolve().glob("*_f*") if (s / "labels").is_dir())


def read_labels(txt, min_h=0.12):
    boxes = []
    if txt.exists():
        for line in txt.read_text().splitlines():
            c, cx, cy, w, h = (float(v) for v in line.split()[:5])
            if int(c) == 0 and h >= min_h:
                boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def align(roots):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seg in segments(roots):
        pics = sorted((seg / "pictures").glob("pic_*.png"))[::2]
        if len(pics) < 10:
            continue
        integ, labels = [], []
        for p in pics:
            a = (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) != 128).astype(np.float32)
            integ.append(cv2.integral(a))
            labels.append(read_labels(seg / "labels" / (p.stem + ".txt")))
        H, W = integ[0].shape[0] - 1, integ[0].shape[1] - 1

        def density(dx, dy, dt):
            tot = area = 0.0
            for k, I in enumerate(integ):
                j = k + dt
                if not 0 <= j < len(labels):
                    continue
                for x0, y0, x1, y1 in labels[j]:
                    X0, Y0 = int(np.clip(x0 * W + dx, 0, W)), int(np.clip(y0 * H + dy, 0, H))
                    X1, Y1 = int(np.clip(x1 * W + dx, 0, W)), int(np.clip(y1 * H + dy, 0, H))
                    if X1 > X0 and Y1 > Y0:
                        tot += I[Y1, X1] - I[Y0, X1] - I[Y1, X0] + I[Y0, X0]
                        area += (X1 - X0) * (Y1 - Y0)
            return tot / area if area else 0.0

        if not any(labels):
            continue
        base = density(0, 0, 0)
        best = max(((density(dx, dy, 0), dx, dy) for dy in range(-48, 49, 8) for dx in range(-32, 33, 8)))
        bt = max(((density(0, 0, dt), dt) for dt in (-3, -2, -1, 0, 1, 2, 3)))
        allact = np.mean([I[-1, -1] / (W * H) for I in integ])
        boxfrac = np.mean([sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in l) for l in labels])
        outside = (allact - base * min(boxfrac, 0.95)) / max(1 - min(boxfrac, 0.95), 1e-3)
        contrast = base / max(outside, 1e-4)
        sync = json.loads((seg / "events.json").read_text()).get("sync", {}).get("method", "?") if (seg / "events.json").exists() else "?"
        flag = []
        if abs(best[1]) > 8 or abs(best[2]) > 8:
            flag.append(f"offset {best[1]:+d},{best[2]:+d} px")
        if bt[1] != 0 and bt[0] > base * 1.05:
            flag.append(f"time {bt[1]:+d} pictures")
        if contrast < 1.6:
            flag.append(f"contrast {contrast:.2f}")
        if "marker" not in sync:
            flag.append("no marker sync")
        rows.append({"segment": str(seg.relative_to(ROOT)), "density": round(base, 4), "contrast": round(contrast, 2),
                     "best_dx": best[1], "best_dy": best[2], "best_dt": bt[1], "sync": sync, "flags": "; ".join(flag)})
    out = OUT / "align.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["flags"]]
    print(f"{len(rows)} segments checked, {len(bad)} flagged -> {out}")
    by = {}
    for r in bad:
        for fl in r["flags"].split("; "):
            by[fl.split()[0]] = by.get(fl.split()[0], 0) + 1
    print("flags:", by)


def suspects(roots, model):
    from eval_ap import decode, iou_matrix, load, predictor
    OUT.mkdir(parents=True, exist_ok=True)
    run = predictor(model, 160)
    rows = []
    for seg in segments(roots):
        pics = sorted((seg / "pictures").glob("pic_*.png"))
        for i in range(0, len(pics), 128):
            chunk = pics[i:i + 128]
            outs = run([load(str(p), 160) for p in chunk])
            for p, o in zip(chunk, outs):
                s, b = decode(np.asarray(o), threshold=0.05)
                gt = np.array(read_labels(seg / "labels" / (p.stem + ".txt")), np.float32).reshape(-1, 4)
                allgt = np.array(read_labels(seg / "labels_all" / (p.stem + ".txt"), 0.0), np.float32).reshape(-1, 4) \
                    if (seg / "labels_all").is_dir() else gt
                m = iou_matrix(gt, b)
                missed = int(sum(1 for k in range(len(gt)) if not (m[k] >= 0.3).any())) if len(gt) else 0
                ma = iou_matrix(b, allgt)
                extra = int(sum(1 for k in range(len(b)) if s[k] >= 0.7 and not (len(allgt) and (ma[k] >= 0.2).any())))
                if missed or extra:
                    rows.append({"picture": str(p.relative_to(ROOT)), "labelled": len(gt), "unseen_labels": missed,
                                 "unlabelled_detections": extra})
    out = OUT / f"suspects_{Path(roots.split(',')[0]).name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["picture", "labelled", "unseen_labels", "unlabelled_detections"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} suspect pictures -> {out}")


def montage(csv_path, out_png, n=24, model=None):
    from eval_ap import decode, load, predictor
    rows = list(csv.DictReader(open(csv_path)))
    rows.sort(key=lambda r: -(int(r["unseen_labels"]) + int(r["unlabelled_detections"])))
    run = predictor(model, 160) if model else None
    tiles = []
    for r in rows[:n]:
        p = ROOT / r["picture"]
        img = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        S = img.shape[0] - 1
        for x0, y0, x1, y1 in read_labels(p.parent.parent / "labels" / (p.stem + ".txt"), 0.0):
            cv2.rectangle(img, (int(x0 * S), int(y0 * S)), (int(x1 * S), int(y1 * S)), (0, 140, 255), 1)
        if run:
            s, b = decode(np.asarray(run([load(str(p), 160)])[0]), threshold=0.5)
            for sc, (x0, y0, x1, y1) in zip(s, b):
                cv2.rectangle(img, (int(x0 * S), int(y0 * S)), (int(x1 * S), int(y1 * S)), (255, 255, 0), 1)
        cv2.putText(img, f"{p.parent.parent.parent.name[8:30]} {p.parent.parent.name[:18]} {p.stem[4:]}", (2, 10),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
        tiles.append(cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows_img = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(out_png, np.vstack(rows_img))
    print("wrote", out_png)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "align":
        align(sys.argv[2])
    elif cmd == "suspects":
        suspects(sys.argv[2], sys.argv[sys.argv.index("--model") + 1])
    elif cmd == "montage":
        montage(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 24,
                sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None)
    else:
        sys.exit(__doc__)
