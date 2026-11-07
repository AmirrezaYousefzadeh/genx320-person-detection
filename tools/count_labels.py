"""Complete person labels for counting. The pass labels (record_dataset.py) keep MOT17 pedestrians with visibility
>= 0.25 only; but reflections of walking people (a shop window in MOT17-09), distractors (person-like objects) and
crowds also give events that look like people. This maps every person-like ground-truth entry of the frame shown
into each event picture, with the homography the pass was recorded with (refitted from the pass's own labels: the
pictures whose labelled boxes correspond one to one, in order, to the frame's pedestrians), and counts the events
in each box. Writes <segment>/labels_count/pic_NNNN.txt: "class cx cy w h visibility events" (MOT17 classes:
1 pedestrian, 2 person on vehicle, 7 static person, 8 distractor, 12 reflection, 13 crowd).
Usage: count_labels.py PASS_DIR [PASS_DIR ...]
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_mot import seq_info  # noqa: E402

PERSON, LOOKALIKE = {1, 2, 7}, {8, 12, 13}
PIC = 320


def gt_all(seq_dir):
    out = {}
    for r in csv.reader(open(Path(seq_dir) / "gt" / "gt.txt")):
        frame, x, y, w, h, conf, cls, vis = int(r[0]) - 1, *(float(v) for v in r[2:6]), int(r[6]), int(r[7]), float(r[8])
        if cls in PERSON | LOOKALIKE:
            out.setdefault(frame, []).append((x, y, x + w, y + h, cls, vis, conf))
    return out


def prepare(seg):
    """the segment's frame per picture, ground truth, monitor mapping, and its one-to-one box correspondences"""
    meta = json.loads((seg / "segment.json").read_text())
    ev = json.loads((seg / "events.json").read_text())
    lines = [l for l in (seg / "frames.log").read_text().split("\n") if l]
    rx, ry, rw, rh = (int(v) for v in [l for l in lines if l.startswith("rect")][-1].split()[1:5])
    flipped = any(l == "flip 1" for l in lines)
    shown = np.array([[float(a), int(b)] for a, b in (l.split()[:2] for l in lines if l[0].isdigit())])
    _, img_w, img_h, _ = seq_info(meta["seq_dir"])
    sx, sy = rw / img_w, rh / img_h
    sync = ev.get("sync", {})
    if sync.get("method") == "blink marker":
        icpt, slope = meta["pc_start"] + sync["offset_s"], 1 + sync["drift_ppm"] / 1e6
    else:
        icpt, slope = meta["pc_start"], 1.0
    win = ev.get("window_ms", 50) / 1000
    gt = gt_all(meta["seq_dir"])

    def mon(b):
        x0, y0, x1, y1 = b[:4]
        if flipped:
            x0, x1 = img_w - x1, img_w - x0
        return np.array([[rx + x0 * sx, ry + y0 * sy], [rx + x1 * sx, ry + y0 * sy], [rx + x1 * sx, ry + y1 * sy],
                         [rx + x0 * sx, ry + y1 * sy]], np.float64)

    frames = []
    for k in range(ev["pictures"]):
        t = icpt + slope * (k + 0.5) * win
        i = np.searchsorted(shown[:, 0], t) - 1
        frames.append(int(shown[i, 1]) if 0 <= i < len(shown) and t - shown[i, 0] < 0.5 else None)
    # the pass's homography: monitor px -> picture px, from the pictures labelled one to one in order
    src, dst = [], []
    for k, f in enumerate(frames):
        if f is None:
            continue
        lab = seg / "labels_all" / f"pic_{k:04d}.txt"
        if not lab.exists():
            continue
        boxes = [[float(v) for v in l.split()[1:5]] for l in lab.read_text().splitlines()]
        peds = [b for b in gt.get(f, []) if b[4] in PERSON and b[6] > 0 and b[5] >= 0.25]
        if not boxes or len(boxes) != len(peds):
            continue
        for (cx, cy, w, h), p in zip(boxes, peds):
            if 0.02 < cx - w / 2 and cx + w / 2 < 0.98 and 0.02 < cy - h / 2 and cy + h / 2 < 0.98:  # not clipped
                src.append(mon(p))
                dst.append(np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2],
                                     [cx - w / 2, cy + h / 2]]) * (PIC - 1))
    return frames, gt, mon, src, dst


def write(seg, prep, H):
    frames, gt, mon, _, _ = prep
    (seg / "labels_count").mkdir(exist_ok=True)
    for k, f in enumerate(frames):
        out = []
        if f is not None:
            pic = cv2.imread(str(seg / "pictures" / f"pic_{k:04d}.png"), cv2.IMREAD_GRAYSCALE)
            active = pic != 128
            for b in gt.get(f, []):
                p = cv2.perspectiveTransform(mon(b)[:, None], H).reshape(-1, 2) / (PIC - 1)
                x0, y0, x1, y1 = np.clip([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], 0, 1)
                if x1 - x0 > 0.02 and y1 - y0 > 0.02:
                    n_ev = int(active[int(y0 * 319):int(y1 * 319) + 1, int(x0 * 319):int(x1 * 319) + 1].sum())
                    out.append(f"{b[4]} {(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {x1 - x0:.4f} {y1 - y0:.4f} "
                               f"{b[5]:.2f} {n_ev}")
        (seg / "labels_count" / f"pic_{k:04d}.txt").write_text("\n".join(out) + ("\n" if out else ""))


def main(dirs):
    """one homography per pass (one calibration per pass), fitted on all its segments"""
    for d in dirs:
        segs = sorted(p for p in Path(d).glob("*_f*") if (p / "events.json").exists())
        preps = {seg: prepare(seg) for seg in segs}
        src = [a for p in preps.values() for a in p[3]]
        dst = [b for p in preps.values() for b in p[4]]
        if len(src) < 20:
            print(f"{Path(d).name}: too few one-to-one pictures to fit the homography, skipped", flush=True)
            continue
        H, inl = cv2.findHomography(np.concatenate(src), np.concatenate(dst), cv2.RANSAC, 3.0)
        err = np.linalg.norm(cv2.perspectiveTransform(np.concatenate(src)[:, None], H).reshape(-1, 2)
                             - np.concatenate(dst), axis=1)
        print(f"{Path(d).name}: homography from {len(src)} boxes, inliers {inl.mean():.0%}, median corner error "
              f"{np.median(err):.1f} px", flush=True)
        for seg, prep in preps.items():
            write(seg, prep, H)


if __name__ == "__main__":
    main(sys.argv[1:])
