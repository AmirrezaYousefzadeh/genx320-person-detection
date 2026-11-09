"""The statistics used to choose the simulator's parameter ranges (tools/evsim.py, RANGES): for 50 ms event pictures,
the share of active pixels inside the video area, the share of active pixels without an active 8-neighbour
("isolated", noise-like) and the share of ON among the active pixels.

The ranges in evsim.py were chosen by hand so that simulated pictures of MOT17-02 at 60 fps (tools/evsim.py with
EVSIM_FIXED to pin parameters) match the monitor recordings of the same video; the paper reports the match
(13.3 vs 13.9 % active, 16 vs 17.5 % isolated, 51 vs 53 % ON). This tool computes the three numbers for any folders of
pictures, so the comparison can be repeated for other sequences or sensors.
Usage: evsim_calib.py PICTURES_DIR [PICTURES_DIR ...] [--roi x0,y0,x1,y1]   (roi in picture pixels; default: all)
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def stats(folder, roi=None):
    act, iso, on, n = 0.0, 0.0, 0.0, 0
    k = np.ones((3, 3), np.uint8)
    for p in sorted(Path(folder).glob("pic_*.png")):
        pic = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if roi:
            x0, y0, x1, y1 = roi
            pic = pic[y0:y1, x0:x1]
        a = (pic != 128).astype(np.uint8)
        if not a.any():
            continue
        neigh = cv2.filter2D(a, -1, k, borderType=cv2.BORDER_CONSTANT) - a
        act += a.mean()
        iso += ((a == 1) & (neigh == 0)).sum() / a.sum()
        on += (pic == 255).sum() / a.sum()
        n += 1
    return {"pictures": n, "active %": round(100 * act / max(n, 1), 1), "isolated %": round(100 * iso / max(n, 1), 1),
            "ON %": round(100 * on / max(n, 1), 1)}


if __name__ == "__main__":
    a = sys.argv[1:]
    roi = None
    if "--roi" in a:
        i = a.index("--roi")
        roi = [int(v) for v in a[i + 1].split(",")]
        del a[i:i + 2]
    if not a:
        sys.exit(__doc__)
    for d in a:
        print(d, stats(d, roi))
