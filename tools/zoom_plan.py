"""Zoomed replay crops: for a segment of a MOT17 sequence, a fixed square crop of the video in which the people
passing through are large in the sensor's picture (monitor.py --warp --crop shows the crop filling the sensor's
view). The sensor has 320x320 pixels: a person below ~12 % of the picture height gives no usable picture (the
dataset's minimum-size rule), so small people are zoomed to TARGET of the view height instead.
  size      side = the segment's median person height / TARGET, between MIN_SIDE source pixels (the
            square the sensor sees is ~276 pixels, so the video is enlarged at most ~1.7x; a quarter of
            the frame height before 2026-11-05 kept people below ~0.45 of the view) and the frame height
  position  the crop holding the most (person, frame) pairs whole and at least 15 % of the view high, minus half
            the pairs it cuts at its border (a cut person is a partial silhouette); RANK > 0 takes the next best
            position at least half a side away, for variety
The crop is fixed for the segment: a moving crop would make the whole scene give events, unlike a static camera.
Usage (test): zoom_plan.py SEQ_DIR START_FRAME N_FRAMES [TARGET 0.4] [RANK 0]
"""
import configparser
import csv
import sys
from pathlib import Path

import numpy as np

MIN_SIDE = 160  # source pixels


def seq_meta(seq_dir):
    c = configparser.ConfigParser()
    c.read(Path(seq_dir) / "seqinfo.ini")
    s = c["Sequence"]
    return int(s["imWidth"]), int(s["imHeight"]), float(s["frameRate"]), int(s["seqLength"])


def people(seq_dir, start, n):
    """(frame, x0, y0, x1, y1) of the pedestrians with visibility >= 0.25 in frames [start, start + n)"""
    out = []
    for r in csv.reader(open(Path(seq_dir) / "gt" / "gt.txt")):
        f, x, y, w, h, conf, cls, vis = int(r[0]) - 1, *(float(v) for v in r[2:6]), int(r[6]), int(r[7]), float(r[8])
        if start <= f < start + n and cls in (1, 2, 7) and conf > 0 and vis >= 0.25:
            out.append((f, x, y, x + w, y + h))
    return np.array(out, np.float64).reshape(-1, 5)


def plan(seq_dir, start, n, target=0.4, rank=0):
    """(cx0, cy0, side) of the crop in frame pixels, and its expected (whole people per frame, median relative height)"""
    W, H, _, _ = seq_meta(seq_dir)
    return plan_boxes(people(seq_dir, start, n), W, H, n, target, rank)


def plan_boxes(p, W, H, n, target=0.4, rank=0):
    """plan() for an array of (frame, x0, y0, x1, y1) people over n frames of a W x H video"""
    if len(p) == 0:
        side = min(W, H)
        return ((W - side) / 2, (H - side) / 2, side), (0.0, 0.0)
    hs = p[:, 4] - p[:, 2]
    side = float(np.clip(np.median(hs) / target, max(MIN_SIDE, 0.12 * H), min(W, H)))
    xs = np.linspace(0, W - side, max(2, int((W - side) / (0.05 * W)) + 1))
    ys = np.linspace(0, H - side, max(2, int((H - side) / (0.05 * H)) + 1))
    area = (p[:, 3] - p[:, 1]) * (p[:, 4] - p[:, 2])
    big = hs >= 0.15 * side
    cands = []
    for cy0 in ys:
        for cx0 in xs:
            ix = np.clip(np.minimum(p[:, 3], cx0 + side) - np.maximum(p[:, 1], cx0), 0, None)
            iy = np.clip(np.minimum(p[:, 4], cy0 + side) - np.maximum(p[:, 2], cy0), 0, None)
            frac = ix * iy / area
            whole = (frac >= 0.9) & big
            cut = (frac > 0.1) & (frac < 0.9)
            cands.append((whole.sum() - 0.5 * cut.sum(), cx0, cy0, whole))
    cands.sort(key=lambda c: -c[0])
    chosen = []
    for c in cands:
        if all(np.hypot(c[1] - o[1], c[2] - o[2]) >= side / 2 for o in chosen):
            chosen.append(c)
        if len(chosen) > rank:
            break
    score, cx0, cy0, whole = chosen[min(rank, len(chosen) - 1)]
    per_frame = whole.sum() / max(n, 1)
    rel = float(np.median(hs[whole] / side)) if whole.any() else 0.0
    return (float(cx0), float(cy0), side), (float(per_frame), rel)


if __name__ == "__main__":
    a = sys.argv[1:]
    crop, stats = plan(a[0], int(a[1]), int(a[2]), float(a[3]) if len(a) > 3 else 0.4, int(a[4]) if len(a) > 4 else 0)
    W, H, _, _ = seq_meta(a[0])
    print(f"crop x {crop[0]:.0f} y {crop[1]:.0f} side {crop[2]:.0f} (zoom {H / crop[2]:.1f}x of the frame height): "
          f"{stats[0]:.1f} whole people per frame, median height {stats[1]:.2f} of the view")
