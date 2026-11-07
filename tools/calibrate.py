"""Where is the monitor in the sensor's picture? Fits a homography from a blinking square shown by monitor.py.

Run `tools/show.sh square 0.6` first (a centred square of side 0.6 x monitor height blinking at 2 Hz), then this:
it dumps the detector picture (nn_pic, 320x320) over SWD a few times, keeps the dump with the most events, finds
the square's four corners and computes the homography picture -> monitor pixels. Writes work/calib/calib.json
with the homography, the monitor's resolution and the monitor rectangle that is visible in the picture.
Usage: calibrate.py [square_fraction=0.6] [monitor_w=1920] [monitor_h=1200]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PIC = 320


def nn_pic_addr(elf=Path(os.environ.get("GENX_ELF", Path(__file__).resolve().parent.parent
                                        / "firmware/build/person_detector.elf"))):
    """nn_pic's address in the flashed build: it moves with the network size (0xC041F380 at 192 px,
    0xC0415AE0 at 160 px), and a fixed number read the wrong memory after the change (2026-11-05).
    GENX_ELF names the flashed build when firmware/build holds another one (tools/campaign.py keeps a copy)."""
    out = subprocess.run(["arm-none-eabi-nm", str(elf)], capture_output=True, text=True, check=True).stdout
    return next(int(l.split()[0], 16) for l in out.splitlines() if l.endswith(" nn_pic"))


NN_PIC_ADDR = nn_pic_addr()


def dump_pic(path):
    subprocess.run(["openocd", "-f", "board/stm32f7discovery.cfg", "-c",
                    f"init; halt; dump_image {path} {NN_PIC_ADDR:#x} {PIC * PIC:#x}; resume; shutdown"],
                   capture_output=True, check=True)
    return np.fromfile(path, dtype=np.uint8).reshape(PIC, PIC)


def quad(contour):
    """A contour approximated by four points, ordered top-left, top-right, bottom-right, bottom-left."""
    poly = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True).reshape(-1, 2).astype(np.float32)
    if len(poly) != 4:
        sys.exit(f"found a {len(poly)}-gon, not a square; is `show.sh square` running and fully in view?")
    s, d = poly.sum(1), poly[:, 0] - poly[:, 1]
    return np.array([poly[s.argmin()], poly[d.argmax()], poly[s.argmax()], poly[d.argmin()]], np.float32)


def square_corners(ev):
    """Corners of the square's centre line in a 0/1 event map: the border is dense with events, the rest is
    sparse noise, so a local density threshold isolates it; outer and inner contour are averaged."""
    dens = cv2.blur(ev.astype(np.float32), (9, 9))
    thr = (float(np.median(dens)) + float(dens.max())) / 2  # between the noise floor and the border
    mask = (dens > thr).astype(np.uint8) * 255
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outer_i = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    holes = [i for i in range(len(contours)) if hier[0][i][3] == outer_i]
    if not holes:
        sys.exit("the square has no inside; is `show.sh square` running and fully in view?")
    inner_i = max(holes, key=lambda i: cv2.contourArea(contours[i]))
    return (quad(contours[outer_i]) + quad(contours[inner_i])) / 2


def main(fraction=0.6, mon_w=1920, mon_h=1200):
    out = Path("work/calib")
    out.mkdir(parents=True, exist_ok=True)
    pics = [dump_pic(out / f"dump_{i}.raw") for i in range(8)]
    ev = max(((p != 128).astype(np.uint8) for p in pics), key=lambda e: e.sum())
    cv2.imwrite(str(out / "square_events.png"), ev * 255)
    pic_corners = square_corners(ev)
    a = fraction * mon_h
    x0, y0 = (mon_w - a) / 2, (mon_h - a) / 2
    mon_corners = np.array([[x0, y0], [x0 + a, y0], [x0 + a, y0 + a], [x0, y0 + a]], np.float32)
    H = cv2.getPerspectiveTransform(pic_corners, mon_corners)  # picture px -> monitor px
    pic_rect = np.array([[[0, 0]], [[PIC - 1, 0]], [[PIC - 1, PIC - 1]], [[0, PIC - 1]]], np.float32)
    seen = cv2.perspectiveTransform(pic_rect, H).reshape(-1, 2)
    visible = [float(seen[:, 0].min()), float(seen[:, 1].min()), float(seen[:, 0].max()), float(seen[:, 1].max())]
    # the largest axis-aligned monitor rectangle inside the seen quadrilateral (conservative: inner corners)
    inner = [float(max(seen[0, 0], seen[3, 0])), float(max(seen[0, 1], seen[1, 1])),
             float(min(seen[1, 0], seen[2, 0])), float(min(seen[2, 1], seen[3, 1]))]
    # only the monitor can show anything: clip the stage to it (a view reaching past the monitor's edge, as from
    # the tripod of 2026-11-05, gave a stage partly above the screen, where replayed people were cut off)
    inner = [max(inner[0], 0.0), max(inner[1], 0.0), min(inner[2], float(mon_w)), min(inner[3], float(mon_h))]
    calib = {"monitor": [mon_w, mon_h], "H_pic_to_mon": H.tolist(), "H_mon_to_pic": np.linalg.inv(H).tolist(),
             "pic_corners": pic_corners.tolist(), "seen_quad_mon": seen.tolist(), "visible_mon_bbox": visible,
             "stage_mon": [int(v) for v in inner]}
    (out / "calib.json").write_text(json.dumps(calib, indent=1))
    vis = cv2.cvtColor(ev * 255, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, [pic_corners.astype(np.int32)], True, (0, 140, 255), 1)
    cv2.imwrite(str(out / "square_fit.png"), vis)
    print("square corners in the picture:", pic_corners.round(1).tolist())
    print("monitor pixels seen by the picture (quad):", seen.round(0).tolist())
    print("stage (monitor rectangle fully inside the view): x0 y0 x1 y1 =", calib["stage_mon"])
    print(f"wrote {out / 'calib.json'}")


if __name__ == "__main__":
    main(*(float(v) if i == 0 else int(v) for i, v in enumerate(sys.argv[1:4])))
