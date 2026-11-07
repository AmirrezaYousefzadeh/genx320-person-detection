"""People-counting dataset from the labelled event pictures: how many people does a 50 ms event picture show?

Each sample is a square region of a picture (the whole picture or a crop) and the number of visible people in it.
A person counts when it is visible, lies at least 80 % inside the region and is at least MIN_H of the region's
height. Visible: at least VISIBLE of its box's pixels hold an event (labels_all/ gives the events per box; the
sensor's noise alone fills 3..7 % of the pixels); below INVISIBLE it is not counted, as a camera would not see it.
A region is left out when a person is cut by its border (3..80 % inside: a visible fragment is ambiguous), is visible but smaller than MIN_H, or is
between invisible and visible: all ambiguous. Real-scene passes (real_label.py) have motion boxes only: visible. Crops
enlarge the people to the scale of a person a few metres from the camera and give the low counts (0..3) that the
monitor passes, mostly crowds, lack. Classes 0, 1, 2, 3, 4 and 5 (5 or more).

Splits: monitor passes by sequence (MOT17-09 and -11 validation), real-scene passes by scenario (every third
scenario validation). The samples are written as a JSON-lines index (picture path, region, count, boxes in region
coordinates); count_train.py cuts and scales the regions on the fly, so no pictures are duplicated on disk.

Usage: count_data.py OUT.jsonl PASS_DIR [PASS_DIR ...] [--crops 6] [--seed 0]
"""
import json
import sys
from pathlib import Path

import numpy as np

MIN_H = 0.12
INSIDE, CUT = 0.8, 0.03  # a person overlapping a region by more than CUT of its box must be INSIDE it
VISIBLE, INVISIBLE = 0.12, 0.06  # share of the box's pixels with an event
MAX_CLASS = 5


PERSON, LOOKALIKE = {1, 2, 7}, {8, 12, 13}


def boxes_of(seg, stem):
    """boxes (x0, y0, x1, y1, visibility): visibility 1 visible, 0 invisible, 0.5 ambiguous.
    With labels_count/ (count_labels.py: every person-like ground-truth entry): a pedestrian hidden behind others
    (visibility < 0.25) is ignored, its box's events are the occluder's; a reflection, distractor or crowd that
    gives events makes the region ambiguous (it looks like a person)."""
    out = []
    cnt = seg / "labels_count" / (stem + ".txt")
    if cnt.exists():
        for line in cnt.read_text().splitlines():
            v = line.split()
            cls, cx, cy, w, h, vis, n_ev = int(v[0]), *(float(x) for x in v[1:6]), int(v[6])
            if vis < 0.25:
                continue
            dens = n_ev / max(w * h * 320 * 320, 1.0)
            if cls in LOOKALIKE:
                flag = 0.5 if dens >= INVISIBLE else 0.0
            else:
                flag = 1.0 if dens >= VISIBLE else (0.0 if dens < INVISIBLE else 0.5)
            out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, flag))
        return out
    full = seg / "labels_all" / (stem + ".txt")
    txt = full if full.exists() else seg / "labels" / (stem + ".txt")
    if txt.exists():
        for line in txt.read_text().splitlines():
            v = line.split()
            cx, cy, w, h = (float(x) for x in v[1:5])
            vis = 1.0
            if len(v) >= 6:
                dens = int(v[5]) / max(w * h * 320 * 320, 1.0)
                vis = 1.0 if dens >= VISIBLE else (0.0 if dens < INVISIBLE else 0.5)
            out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, vis))
    return out


def region_count(boxes, r):
    """(count, boxes in region coordinates) or None when ambiguous"""
    x0, y0, x1, y1 = r
    side = x1 - x0
    inside = []
    for b in boxes:
        if b[4] == 0.0:
            continue  # invisible: no events to count it by
        ix = max(0.0, min(b[2], x1) - max(b[0], x0))
        iy = max(0.0, min(b[3], y1) - max(b[1], y0))
        frac = ix * iy / max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
        if frac < CUT:
            continue
        if frac < INSIDE or (b[3] - b[1]) / side < MIN_H or b[4] < 1.0:
            return None
        inside.append([(max(b[0], x0) - x0) / side, (max(b[1], y0) - y0) / side,
                       (min(b[2], x1) - x0) / side, (min(b[3], y1) - y0) / side])
    return len(inside), inside


def is_val(seg):
    if "MOT17-" in seg.name:
        return any(f"MOT17-{s}-" in seg.name for s in ("09", "11"))
    scen = json.loads((seg / "segment.json").read_text()).get("index", 0) if (seg / "segment.json").exists() else 0
    return scen % 3 == 0


def main(out, dirs, crops=6, seed=0):
    rng = np.random.default_rng(seed)
    n = {"train": np.zeros(MAX_CLASS + 1, int), "val": np.zeros(MAX_CLASS + 1, int)}
    with open(out, "w") as f:
        for d in dirs:
            for seg in sorted(p for p in Path(d).iterdir() if (p / "pictures").is_dir()):
                split = "val" if is_val(seg) else "train"
                for png in sorted((seg / "pictures").glob("pic_*.png")):
                    boxes = boxes_of(seg, png.stem)
                    regions = [(0.0, 0.0, 1.0, 1.0)]
                    for _ in range(crops):
                        s = rng.uniform(0.35, 0.8)
                        x, y = rng.uniform(0, 1 - s, 2)
                        regions.append((x, y, x + s, y + s))
                    for r in regions:
                        rc = region_count(boxes, r)
                        if rc is None:
                            continue
                        c, inside = rc
                        cls = min(c, MAX_CLASS)
                        n[split][cls] += 1
                        f.write(json.dumps({"png": str(png), "region": [round(v, 4) for v in r], "count": c,
                                            "class": cls, "boxes": [[round(v, 4) for v in b] for b in inside],
                                            "pass": Path(d).name, "segment": seg.name, "split": split}) + "\n")
    for split, h in n.items():
        print(f"{split}: {h.sum()} samples, per class 0..{MAX_CLASS - 1}, {MAX_CLASS}+: {h.tolist()}")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    dirs = [x for i, x in enumerate(a[1:], 1) if not x.startswith("--") and not a[i - 1].startswith("--")]
    main(a[0], dirs, opt("--crops", 6), opt("--seed", 0))
