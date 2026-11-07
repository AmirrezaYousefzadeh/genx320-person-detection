"""Several-people room scenes from one-person recordings (real_label.py labels): event pictures are sparse, so the
region around a person in one picture can be pasted into another picture where the person stands elsewhere. Only
that region is copied, with its own background events, so the noise stays at one picture's level. People may not
overlap (box IoU < 0.05, neither box more than 20 % inside the other). A synthetic pass: pictures/ and labels/ like
a recorded one, labels exact by construction, the scenario of every source picture recorded.
Usage: real_composite.py SRC_PASS OUT_PASS [--n 3000] [--people 2,3] [--seed 0]
Split: a composite is validation when all its sources come from validation scenarios (index % 3 == 0), training
when none do; mixed ones are not written.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PIC = 320


def boxes_of(txt):
    out = []
    for line in txt.read_text().splitlines():
        _, cx, cy, w, h = (float(v) for v in line.split()[:5])
        out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def overlap_ok(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter) < 0.05 and inter / min(aa, ab) < 0.2


def main(src, out, n=3000, people=(2, 3), seed=0):
    rng = np.random.default_rng(seed)
    singles = []  # (png, box, scenario index)
    for seg in sorted(Path(src).iterdir()):
        if not (seg / "labels").is_dir():
            continue
        idx = json.loads((seg / "segment.json").read_text())["index"]
        for txt in sorted((seg / "labels").glob("pic_*.txt")):
            b = boxes_of(txt)
            if len(b) == 1:
                singles.append((str(seg / "pictures" / (txt.stem + ".png")), b[0], idx))
    print(f"{len(singles)} one-person pictures")
    dirs = {False: Path(out) / "composites_train", True: Path(out) / "composites_val"}  # split as count_data.py
    for v, d in dirs.items():
        (d / "pictures").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(exist_ok=True)
        (d / "segment.json").write_text(json.dumps({"scenario": d.name, "index": 3 if v else 1, "synthetic": True}))
    meta, k, tries = [], 0, 0
    while k < n and tries < n * 50:
        tries += 1
        m = int(rng.choice(people))
        pick = [singles[i] for i in rng.choice(len(singles), m, replace=False)]
        vals = [p[2] % 3 == 0 for p in pick]
        if any(vals) and not all(vals):
            continue
        if not all(overlap_ok(pick[i][1], pick[j][1]) for i in range(m) for j in range(i + 1, m)):
            continue
        base = cv2.imread(pick[0][0], cv2.IMREAD_GRAYSCALE)
        for png, b, _ in pick[1:]:
            other = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
            w, h = b[2] - b[0], b[3] - b[1]
            x0, y0 = int(max(0, b[0] - 0.1 * w) * PIC), int(max(0, b[1] - 0.05 * h) * PIC)
            x1, y1 = int(min(1, b[2] + 0.1 * w) * PIC), int(min(1, b[3] + 0.05 * h) * PIC)
            base[y0:y1, x0:x1] = other[y0:y1, x0:x1]
        seg_dir = dirs[all(vals)]
        cv2.imwrite(str(seg_dir / "pictures" / f"pic_{k:05d}.png"), base)
        with open(seg_dir / "labels" / f"pic_{k:05d}.txt", "w") as f:
            for _, b, _ in pick:
                f.write(f"0 {(b[0] + b[2]) / 2:.4f} {(b[1] + b[3]) / 2:.4f} {b[2] - b[0]:.4f} {b[3] - b[1]:.4f}\n")
        meta.append({"pic": k, "people": m, "sources": [p[0] for p in pick], "val": all(vals)})
        k += 1
    (Path(out) / "composites.json").write_text(json.dumps(meta))
    print(f"wrote {k} composites ({sum(m['val'] for m in meta)} validation) to {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda key, d: a[a.index(key) + 1] if key in a else d  # noqa: E731
    main(a[0], a[1], int(opt("--n", 3000)), tuple(int(v) for v in opt("--people", "2,3").split(",")),
         int(opt("--seed", 0)))
