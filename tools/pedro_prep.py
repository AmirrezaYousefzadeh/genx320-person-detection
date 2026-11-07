"""PEDRo (Boretti et al., CVPRW 2023; CC BY 4.0) as event pictures in the board's format, for the paper's benchmark.

PEDRo: DAVIS346 (346x260) on a moving robot, 119 recordings, each sample = the events of 40 ms with its person boxes
(YOLO format), split by recording into train / val / test (info.txt). Every sample becomes one picture like the
board's (128 nothing, 255 ON, 0 OFF, last event wins), padded to 346x346 (43 grey rows above and below, so people
keep their shape when the network squeezes the picture to its square input); labels move with the padding.
Output: work/pedro/pedro_<split>_<rec>_f0000/pictures/pic_NNNN.png + labels/pic_NNNN.txt (class 0 = person).
Usage: pedro_prep.py [RAW_DIR work/datasets/pedro/raw] [OUT work/pedro]
"""
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
W, H = 346, 260
PAD = (W - H) // 2  # 43


def splits(info):
    """{split: [(rec, first, last)]} from info.txt"""
    out, cur = {}, None
    for line in Path(info).read_text().splitlines():
        m = re.match(r"(TRAINING|VALIDATION|TEST) SET", line.strip())
        if m:
            cur = {"TRAINING": "train", "VALIDATION": "val", "TEST": "test"}[m.group(1)]
            out[cur] = []
            continue
        m = re.match(r"(rec\d+) from index (\d+) to index (\d+)", line.strip())
        if m and cur:
            out[cur].append((m.group(1), int(m.group(2)), int(m.group(3))))
    return out


def events(path):
    a = np.load(path, allow_pickle=False)
    if a.dtype.names:
        t, x, y, p = (a[k] for k in ("t", "x", "y", "p"))
    else:
        t, x, y, p = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    o = np.argsort(t, kind="stable")
    return x[o].astype(np.int32), y[o].astype(np.int32), p[o].astype(np.int32)


def picture(path):
    pic = np.full((W, W), 128, np.uint8)
    x, y, p = events(path)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    pic[y[ok] + PAD, x[ok]] = np.where(p[ok] > 0, 255, 0).astype(np.uint8)  # later events overwrite: last wins
    return pic


def convert(args):
    split, rec, first, last, npy_dir, yolo_dir, out = args
    d = Path(out) / f"pedro_{split}_{rec}_f0000"
    (d / "pictures").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(exist_ok=True)
    n = 0
    for k, idx in enumerate(range(first, last + 1)):
        src = Path(npy_dir) / f"frame_{idx:07d}.npy"
        if not src.exists():
            src = Path(npy_dir) / f"frame{idx:07d}.npy"
        if not src.exists():
            continue
        cv2.imwrite(str(d / "pictures" / f"pic_{k:04d}.png"), picture(src))
        lab = Path(yolo_dir) / (src.stem + ".txt")
        lines = []
        if lab.exists():
            for line in lab.read_text().splitlines():
                v = line.split()
                if len(v) < 5:
                    continue
                c, cx, cy, w, h = int(float(v[0])), *(float(u) for u in v[1:5])
                lines.append(f"{0 if c == 0 else 1} {cx:.5f} {(cy * H + PAD) / W:.5f} {w:.5f} {h * H / W:.5f}")
        (d / "labels" / f"pic_{k:04d}.txt").write_text("".join(l + "\n" for l in lines))
        n += 1
    return f"{d.name}: {n} pictures"


def find_dirs(raw, split):
    """the numpy and yolo folders of a split (the zips' layout is not documented)"""
    npy = [p for p in raw.rglob("*") if p.is_dir() and p.name == split and any(p.glob("*.npy"))]
    yolo = [p for p in raw.rglob("*") if p.is_dir() and p.name == split and any(p.glob("*.txt"))]
    return (npy[0] if npy else None), (yolo[0] if yolo else None)


if __name__ == "__main__":
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work/datasets/pedro/raw"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "work/pedro"
    sp = splits(raw.parent / "info.txt")
    tasks = []
    for split, recs in sp.items():
        npy_dir, yolo_dir = find_dirs(raw, split)
        print(split, len(recs), "recordings;", npy_dir, yolo_dir, flush=True)
        tasks += [(split, rec, a, b, npy_dir, yolo_dir, out) for rec, a, b in recs]
    with Pool(int(os.environ.get("NUM_THREADS", "2"))) as pool:
        for line in pool.imap_unordered(convert, tasks):
            print(line, flush=True)
