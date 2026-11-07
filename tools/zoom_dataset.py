"""Close-up variants of the dataset: random zoomed crops of the pictures with the boxes clipped, for people who
come so close to the camera that only part of them is in the frame (the demo lens is narrow).

Usage: zoom_dataset.py OUT IN_DIR [IN_DIR ...] [--per-picture 0.5] [--zoom 1.4 2.5] [--seed 0]
For a fraction of the pictures (default every second one) a crop of 1/zoom of the picture, placed at random but
preferring a place with a person, is resized back to 320x320 (nearest neighbour: the picture stays three-valued).
A box is kept if at least 35 % of its area or 60 % of its height is inside the crop, clipped to the crop; a crop
that cuts a person down to less than 8 % of the picture height is skipped. Output has the dataset layout
(<segment>_zoom/pictures, labels, labels_all, segment.json), so finetune.py --data takes it like any other pass;
the segment name keeps the MOT17 sequence number, so the validation split stays by sequence.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PIC = 320


def main(argv):
    opts = {argv[i]: argv[i + 1:i + 3 if argv[i] == "--zoom" else i + 2] for i in range(len(argv)) if argv[i].startswith("--")}
    args = [a for i, a in enumerate(argv) if not a.startswith("--") and not (i > 0 and argv[i - 1].startswith("--") and argv[i - 1] != "--zoom") and not (i > 1 and argv[i - 2] == "--zoom")]
    out_root, in_dirs = Path(args[0]), [Path(a) for a in args[1:]]
    frac = float(opts.get("--per-picture", ["0.5"])[0])
    zmin, zmax = (float(v) for v in opts.get("--zoom", ["1.4", "2.5"]))
    rng = np.random.default_rng(int(opts.get("--seed", ["0"])[0]))
    n_pics = n_boxes = 0
    for root in in_dirs:
        for seg in sorted(p for p in root.iterdir() if (p / "pictures").is_dir()):
            dst = out_root / f"{seg.name}_zoom"
            (dst / "pictures").mkdir(parents=True, exist_ok=True)
            (dst / "labels").mkdir(exist_ok=True)
            (dst / "labels_all").mkdir(exist_ok=True)
            k = 0
            for png in sorted((seg / "pictures").glob("pic_*.png")):
                if rng.random() > frac:
                    continue
                txt = seg / "labels" / (png.stem + ".txt")
                boxes = []
                if txt.exists():
                    for line in txt.read_text().splitlines():
                        _, cx, cy, w, h = (float(v) for v in line.split()[:5])
                        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                z = rng.uniform(zmin, zmax)
                side = 1.0 / z
                if boxes and rng.random() < 0.8:  # centre the crop near a person, with jitter
                    b = boxes[rng.integers(len(boxes))]
                    cx, cy = (b[0] + b[2]) / 2 + rng.uniform(-0.3, 0.3) * side, (b[1] + b[3]) / 2 + rng.uniform(-0.3, 0.3) * side
                else:
                    cx, cy = rng.uniform(side / 2, 1 - side / 2, 2)
                x0, y0 = float(np.clip(cx - side / 2, 0, 1 - side)), float(np.clip(cy - side / 2, 0, 1 - side))
                kept, skip = [], False
                for bx0, by0, bx1, by1 in boxes:
                    ix0, iy0, ix1, iy1 = max(bx0, x0), max(by0, y0), min(bx1, x0 + side), min(by1, y0 + side)
                    if ix1 <= ix0 or iy1 <= iy0:
                        continue
                    area_frac = (ix1 - ix0) * (iy1 - iy0) / ((bx1 - bx0) * (by1 - by0) + 1e-9)
                    h_frac = (iy1 - iy0) / (by1 - by0 + 1e-9)
                    nb = ((ix0 - x0) / side, (iy0 - y0) / side, (ix1 - x0) / side, (iy1 - y0) / side)
                    if nb[3] - nb[1] < 0.08:
                        skip = True  # a sliver of a person: ambiguous, drop the crop
                        break
                    if area_frac >= 0.35 or h_frac >= 0.6:
                        kept.append(nb)
                if skip:
                    continue
                img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                px0, py0, ps = int(x0 * PIC), int(y0 * PIC), max(1, int(side * PIC))
                crop = cv2.resize(img[py0:py0 + ps, px0:px0 + ps], (PIC, PIC), interpolation=cv2.INTER_NEAREST)
                name = f"pic_{k:04d}"
                cv2.imwrite(str(dst / "pictures" / f"{name}.png"), crop)
                lines = [f"0 {(b[0] + b[2]) / 2:.4f} {(b[1] + b[3]) / 2:.4f} {b[2] - b[0]:.4f} {b[3] - b[1]:.4f}" for b in kept]
                (dst / "labels" / f"{name}.txt").write_text("".join(l + "\n" for l in lines))
                (dst / "labels_all" / f"{name}.txt").write_text("".join(l + " -1\n" for l in lines))
                k += 1
                n_boxes += len(kept)
            (dst / "segment.json").write_text(json.dumps({"source": str(seg), "zoom": [zmin, zmax], "pictures": k}, indent=1))
            n_pics += k
            print(f"{dst.name}: {k} pictures", flush=True)
    print(f"total {n_pics} pictures, {n_boxes} boxes")


if __name__ == "__main__":
    main(sys.argv[1:])
