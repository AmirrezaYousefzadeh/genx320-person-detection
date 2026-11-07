"""A MOT-style sequence at 60 frames per second for the monitor (the Samsung refreshes at 59.95 Hz): every refresh shows
a new picture, so the sensor sees motion in 16.7 ms steps instead of 33 ms (30 fps), 40 ms (25 fps) or 71 ms (14 fps)
jumps. At 30 fps the event rate pulsed at the frame rate (45-52 % of its fluctuation power at 30 Hz; 68 % at 14 Hz for
MOT17-05; under 1 % in the room), which the LCD showed as flicker (Amir, 2026-11-05).

Missing frames come from tools/interp_flow.py (DIS optical flow, both directions, occlusion-aware blend). Output frame
k shows source position p = min(k * fps / 60, n - 1) (0-based source frames, n of them; round(n * 60 / fps) frames); the labels (gt/gt.txt) are interpolated linearly
per track id at the same positions, so tools/record_dataset.py, zoom_plan.py and dataset_run.py use the new folder like
any MOT sequence (frameRate=60 in its seqinfo.ini). interp.json records how it was made.
Usage: make60.py OUT_ROOT SEQ_DIR [SEQ_DIR ...]   (one process per sequence, NUM_THREADS of them at a time)
"""
import configparser
import csv
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import interp_flow  # noqa: E402

FPS_OUT = 60.0
JPEG_Q = 95


def seq_meta(seq_dir):
    c = configparser.ConfigParser()
    c.optionxform = str
    c.read(Path(seq_dir) / "seqinfo.ini")
    return c


def load_tracks(seq_dir):
    """{frame (0-based): {id: row}} of every gt row (all classes; the readers filter)"""
    tr = {}
    for r in csv.reader(open(Path(seq_dir) / "gt" / "gt.txt")):
        f, tid = int(r[0]) - 1, int(r[1])
        tr.setdefault(f, {})[tid] = [float(v) for v in r[2:6]] + [int(float(r[6])), int(float(r[7])), float(r[8])]
    return tr


def interp_rows(tracks, p):
    """gt rows (id, x, y, w, h, conf, cls, vis) at source position p"""
    j, t = int(math.floor(p)), p - math.floor(p)
    a, b = tracks.get(j, {}), tracks.get(j + 1, {})
    rows = []
    for tid in sorted(set(a) | set(b)):
        if tid in a and tid in b:
            ra, rb = a[tid], b[tid]
            box = [(1 - t) * u + t * v for u, v in zip(ra[:4], rb[:4])]
            near = ra if t < 0.5 else rb
            rows.append((tid, *box, near[4], near[5], (1 - t) * ra[6] + t * rb[6]))
        elif (tid in a and t < 0.5) or (tid in b and t >= 0.5):  # the track starts or ends here: nearest frame
            r = a.get(tid) or b.get(tid)
            rows.append((tid, *r))
    return rows


def make(args):
    out_root, seq_dir = args
    seq_dir = Path(seq_dir)
    cv2.setNumThreads(1)
    c = seq_meta(seq_dir)
    s = c["Sequence"]
    fps, n, ext = float(s["frameRate"]), int(s["seqLength"]), s.get("imExt", ".jpg")
    files = sorted((seq_dir / s.get("imDir", "img1")).glob("*" + ext))[:n]
    out = Path(out_root) / seq_dir.name
    (out / "img1").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(exist_ok=True)
    step = fps / FPS_OUT
    n_out = int(round(n / step))  # the source's whole duration n / fps; positions past the last frame hold it
    tracks = load_tracks(seq_dir)
    by_pair = {}
    pos = [min(k * step, n - 1) for k in range(n_out)]
    for k, p in enumerate(pos):
        by_pair.setdefault(int(math.floor(p + 1e-9)), []).append((k, p - math.floor(p + 1e-9)))
    with open(out / "gt" / "gt.txt", "w") as g:
        for k in range(n_out):
            for tid, x, y, w, h, conf, cls, vis in interp_rows(tracks, pos[k]):
                g.write(f"{k + 1},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf},{cls},{vis:.4f}\n")
    done = 0
    for j in sorted(by_pair):
        todo = [(k, t) for k, t in by_pair[j] if not (out / "img1" / f"{k + 1:06d}.jpg").exists()]
        if not todo:
            continue
        i0 = cv2.imread(str(files[j]))
        exact = [k for k, t in todo if t < 1e-6]
        for k in exact:
            cv2.imwrite(str(out / "img1" / f"{k + 1:06d}.jpg"), i0, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        between = [(k, t) for k, t in todo if t >= 1e-6]
        if between:
            i1 = cv2.imread(str(files[j + 1]))
            frames = interp_flow.interpolate(i0, i1, [t for _, t in between])
            for (k, _), fr in zip(between, frames):
                cv2.imwrite(str(out / "img1" / f"{k + 1:06d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        done += len(todo)
    s["frameRate"], s["seqLength"] = f"{FPS_OUT:g}", str(n_out)
    with open(out / "seqinfo.ini", "w") as f:
        c.write(f, space_around_delimiters=False)
    (out / "interp.json").write_text(json.dumps({
        "source": str(seq_dir), "source_fps": fps, "source_frames": n, "fps": FPS_OUT, "frames": n_out,
        "source_position_of_frame_k": f"min(k * {step:.6f}, {n - 1}) (0-based)", "method": "tools/interp_flow.py (OpenCV DIS optical "
        "flow at half resolution, both directions, linear-motion intermediate flows, occlusion-aware blend)",
        "labels": "gt.txt rows interpolated linearly per track id between the two source frames"}, indent=1))
    return f"{seq_dir.name}: {n} frames at {fps:g} fps -> {n_out} at {FPS_OUT:g} ({done} written)"


if __name__ == "__main__":
    out_root, seqs = sys.argv[1], sys.argv[2:]
    with Pool(int(os.environ.get("NUM_THREADS", "2"))) as pool:
        for line in pool.imap_unordered(make, [(out_root, s) for s in seqs]):
            print(line, flush=True)
