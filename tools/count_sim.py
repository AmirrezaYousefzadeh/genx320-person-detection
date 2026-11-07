"""Replay the board's logged raw boxes (detections.csv of tools/self_record.py or tools/bench_mot.py) through the
detector's box post-processing, in the firmware's version and in candidate versions, and report how many people
each version shows per run. For a recording with one person, a count above 1 is always wrong.
Usage: count_sim.py DIR [DIR ...]      (every */detections.csv below each DIR)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def cover(a, b):
    """intersection over the smaller box"""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / (min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) + 1e-9)


class Firmware:
    """detector.c as flashed: IoU 0.3 matching, EMA 0.5, hold 2 runs, containment dedup 0.7; count = all tracks"""
    HOLD, IOU, ALPHA = 2, 0.3, 0.5

    def __init__(self):
        self.tracks = []  # [box(4), score, age]

    def step(self, dets):
        used = [False] * len(dets)
        for t in self.tracks:
            best, best_v = -1, self.IOU
            for d, b in enumerate(dets):
                v = 0 if used[d] else iou(t[0], b)
                if v > best_v:
                    best, best_v = d, v
            if best >= 0:
                used[best] = True
                t[0] = [t[0][i] + self.ALPHA * (dets[best][i] - t[0][i]) for i in range(4)]
                t[1], t[2] = dets[best][4], 0
            else:
                t[2] += 1
        for d, b in enumerate(dets):
            if not used[d]:
                self.tracks.append([list(b[:4]), b[4], 0])
        keep = []
        for i, t in enumerate(self.tracks):
            dup = any(j != i and cover(t[0], u[0]) > 0.7 and (u[2] < t[2] or (u[2] == t[2] and (
                (u[0][2] - u[0][0]) * (u[0][3] - u[0][1]) > (t[0][2] - t[0][0]) * (t[0][3] - t[0][1]) or j < i)))
                for j, u in enumerate(self.tracks))
            if t[2] <= self.HOLD and not dup:
                keep.append(t)
        self.tracks = keep
        return len(self.tracks)


def suppress(dets, iou_thr, cover_thr):
    """greedy, highest score first: drop a box overlapping a kept one by IoU > iou_thr or cover > cover_thr"""
    kept = []
    for b in sorted(dets, key=lambda b: -b[4]):
        if all(iou(b, k) <= iou_thr and cover(b, k) <= cover_thr for k in kept):
            kept.append(b)
    return kept


class Candidate:
    """suppression of part boxes, looser matching, a track counts only once it was seen in two runs"""

    def __init__(self, iou_thr=0.3, cover_thr=0.6, hold=2, confirm=2, match_iou=0.2, match_cover=0.5):
        self.p = dict(iou_thr=iou_thr, cover_thr=cover_thr)
        self.hold, self.confirm, self.match_iou, self.match_cover = hold, confirm, match_iou, match_cover
        self.tracks = []  # dict(box, score, age, hits)

    def step(self, dets):
        dets = suppress(dets, **self.p)
        pairs = sorted(((max(iou(t["box"], d), 0.8 * cover(t["box"], d)), ti, di)
                        for ti, t in enumerate(self.tracks) for di, d in enumerate(dets)), reverse=True)
        t_used, d_used = set(), set()
        for v, ti, di in pairs:
            if ti in t_used or di in d_used:
                continue
            t, d = self.tracks[ti], dets[di]
            if iou(t["box"], d) < self.match_iou and cover(t["box"], d) < self.match_cover:
                continue
            t_used.add(ti), d_used.add(di)
            t["box"] = [t["box"][i] + 0.5 * (d[i] - t["box"][i]) for i in range(4)]
            t["score"], t["age"], t["hits"] = d[4], 0, t["hits"] + 1
        for ti, t in enumerate(self.tracks):
            if ti not in t_used:
                t["age"] += 1
        for di, d in enumerate(dets):
            if di not in d_used:
                self.tracks.append(dict(box=list(d[:4]), score=d[4], age=0, hits=1))
        # drop: unconfirmed after one missed run, confirmed after `hold` missed runs; merge overlapping tracks
        self.tracks = [t for t in self.tracks if t["age"] <= (self.hold if t["hits"] >= self.confirm else 1)]
        self.tracks.sort(key=lambda t: (-t["hits"], t["age"]))
        kept = []
        for t in self.tracks:
            if all(iou(t["box"], k["box"]) <= 0.3 and cover(t["box"], k["box"]) <= 0.6 for k in kept):
                kept.append(t)
        self.tracks = kept
        return sum(t["hits"] >= self.confirm for t in self.tracks)


def runs(path):
    with open(path) as f:
        return [json.loads(r["boxes"]) for r in csv.DictReader(f)]


def main(dirs):
    files = sorted(p for d in dirs for p in Path(d).glob("*/detections.csv"))
    variants = {"firmware now": Firmware, "suppress only": lambda: Candidate(confirm=1),
                "suppress + confirm 2": Candidate, "suppress + confirm 2, hold 3": lambda: Candidate(hold=3)}
    print(f"{'segment':16s}" + "".join(f"{v:>30s}" for v in variants))
    print(f"{'':16s}" + "".join(f"{'count 0 / 1 / 2+ (% of runs)':>30s}" for _ in variants))
    totals = {v: np.zeros(3) for v in variants}
    for f in files:
        rs = runs(f)
        line = f"{f.parent.name:16s}"
        for v, make in variants.items():
            sim = make()
            c = np.array([sim.step(r) for r in rs])
            h = np.array([(c == 0).sum(), (c == 1).sum(), (c >= 2).sum()])
            totals[v] += h
            line += f"{'%3d / %3d / %3d' % tuple((100 * h / len(c)).round()):>30s}"
        print(line)
    print(f"{'all':16s}" + "".join(f"{'%3d / %3d / %3d' % tuple((100 * t / t.sum()).round()):>30s}"
                                   for t in totals.values()))


if __name__ == "__main__":
    main(sys.argv[1:])


class PerRun:
    """no tracker: the count of this run's boxes after suppression, optionally the median of the last n runs"""

    def __init__(self, n=1, iou_thr=0.3, cover_thr=0.6):
        self.n, self.hist, self.p = n, [], dict(iou_thr=iou_thr, cover_thr=cover_thr)

    def step(self, dets):
        self.hist = (self.hist + [len(suppress(dets, **self.p))])[-self.n:]
        return int(np.median(self.hist)) if len(self.hist) == self.n else self.hist[-1]


class Gated:
    """tracks matched to detections by nearest centre within a gate (a fraction of the picture), so that a person
    who moved far between two runs keeps his track; a track counts once seen in `confirm` runs"""

    def __init__(self, gate=0.35, hold=1, confirm=2):
        self.gate, self.hold, self.confirm, self.tracks = gate, hold, confirm, []

    def step(self, dets):
        dets = suppress(dets, 0.3, 0.6)
        c = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)  # noqa: E731
        pairs = sorted((np.hypot(c(t["box"])[0] - c(d)[0], c(t["box"])[1] - c(d)[1]), ti, di)
                       for ti, t in enumerate(self.tracks) for di, d in enumerate(dets))
        t_used, d_used = set(), set()
        for dist, ti, di in pairs:
            if dist > self.gate or ti in t_used or di in d_used:
                continue
            t_used.add(ti), d_used.add(di)
            t = self.tracks[ti]
            t["box"], t["age"], t["hits"] = list(dets[di][:4]), 0, t["hits"] + 1
        for ti, t in enumerate(self.tracks):
            if ti not in t_used:
                t["age"] += 1
        for di, d in enumerate(dets):
            if di not in d_used:
                self.tracks.append(dict(box=list(d[:4]), age=0, hits=1))
        self.tracks = [t for t in self.tracks if t["age"] <= (self.hold if t["hits"] >= self.confirm else 1)]
        return sum(t["hits"] >= self.confirm for t in self.tracks)


def compare(dirs, variants):
    files = sorted(p for d in dirs for p in Path(d).glob("*/detections.csv"))
    totals = {v: np.zeros(3) for v in variants}
    rows = []
    for f in files:
        rs = runs(f)
        line = f"{f.parent.name:16s}"
        for v, make in variants.items():
            sim = make()
            c = np.array([sim.step(r) for r in rs])
            h = np.array([(c == 0).sum(), (c == 1).sum(), (c >= 2).sum()])
            totals[v] += h
            line += f"{'%3d/%3d/%3d' % tuple((100 * h / len(c)).round()):>16s}"
        rows.append(line)
    print(f"{'count 0/1/2+ %':16s}" + "".join(f"{v:>16s}" for v in variants))
    print("\n".join(rows))
    print(f"{'all':16s}" + "".join(f"{'%3d/%3d/%3d' % tuple((100 * t / t.sum()).round()):>16s}" for t in totals.values()))
