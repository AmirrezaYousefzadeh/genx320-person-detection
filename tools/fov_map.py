"""Which part of the monitor does the sensor see? Blinks a square at nine known monitor positions (tools/show.sh
blink, 10 Hz so that every 50 ms picture catches a toggle), dumps the detector picture (nn_pic) over SWD for
each, finds the square's edges in the picture and fits monitor = a * picture + b per axis. Prints the monitor
rectangle in view, the picture position of the monitor's centre, the scale, and what to move. Writes
work/calib/fov_map.png (the nine pictures). Run it after moving the board, before tools/calibrate.py.
Usage: fov_map.py [monitor_w=1920] [monitor_h=1200] [square=200]
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
PIC = 320


def nn_pic_addr(elf=Path(os.environ.get("GENX_ELF", Path(__file__).resolve().parent.parent
                                        / "firmware/build/person_detector.elf"))):
    """nn_pic's address in the flashed build: it moves with the network size (0xC041F380 at 192 px,
    0xC0415AE0 at 160 px), and a fixed number read the wrong memory after the change (2026-11-05).
    GENX_ELF names the flashed build when firmware/build holds another one (tools/campaign.py keeps a copy)."""
    out = subprocess.run(["arm-none-eabi-nm", str(elf)], capture_output=True, text=True, check=True).stdout
    return next(int(l.split()[0], 16) for l in out.splitlines() if l.endswith(" nn_pic"))


NN_PIC_ADDR = nn_pic_addr()
PITCH_MM = 0.27  # 24" 1920x1200 monitor


def dump_pic(path):
    subprocess.run(["openocd", "-f", "board/stm32f7discovery.cfg", "-c",
                    f"init; halt; dump_image {path} {NN_PIC_ADDR:#x} {PIC * PIC:#x}; resume; shutdown"],
                   capture_output=True, check=True)
    return np.fromfile(path, dtype=np.uint8).reshape(PIC, PIC)


def dense_span(profile):
    """(first, last) index of the dense part of an event-count profile, or None (a square gives many times
    the noise floor)."""
    idx = np.nonzero(profile > 3 * np.median(profile) + 5)[0]
    return (int(idx.min()), int(idx.max())) if len(idx) >= 4 else None


def main(mon_w=1920, mon_h=1200, s=200):
    out = ROOT / "work" / "calib"
    out.mkdir(parents=True, exist_ok=True)
    xs = [100, (mon_w - s) // 2, mon_w - s - 100]
    ys = [100, (mon_h - s) // 2, mon_h - s - 100]
    positions = [(x, y) for y in ys for x in xs]
    env = dict(os.environ, BLINK_HZ="10")
    tile = Image.new("L", (3 * (PIC + 10), 3 * (PIC + 10)), 64)
    pairs_x, pairs_y = [], []  # (picture coordinate, monitor coordinate) of square edges fully in view
    for i, (x, y) in enumerate(positions):
        subprocess.run([str(ROOT / "tools" / "show.sh"), "blink", str(x), str(y), str(s)], env=env,
                       capture_output=True, check=True)
        time.sleep(2.5)
        pic = dump_pic(out / "fov_dump.raw")
        ev = pic != 128
        cs, rs = dense_span(ev.sum(0)), dense_span(ev.sum(1))
        note = "not in view"
        if cs and rs:
            note = f"picture cols {cs[0]}-{cs[1]} rows {rs[0]}-{rs[1]}"
            if cs[0] > 0:
                pairs_x.append((cs[0], x))
            if cs[1] < PIC - 1:
                pairs_x.append((cs[1], x + s))
            if rs[0] > 0:
                pairs_y.append((rs[0], y))
            if rs[1] < PIC - 1:
                pairs_y.append((rs[1], y + s))
        print(f"square at monitor ({x}, {y}) size {s}: {note}")
        tile.paste(Image.fromarray(pic), ((i % 3) * (PIC + 10), (i // 3) * (PIC + 10)))
        ImageDraw.Draw(tile).text(((i % 3) * (PIC + 10) + 4, (i // 3) * (PIC + 10) + 2), f"{x},{y}", fill=255)
    subprocess.run([str(ROOT / "tools" / "show.sh"), "black"], capture_output=True)
    tile.save(out / "fov_map.png")
    for axis, pairs, size, name in (("x", pairs_x, mon_w, "columns"), ("y", pairs_y, mon_h, "rows")):
        if len(pairs) < 2:
            print(f"{axis}: fewer than two square edges in view, cannot fit")
            continue
        p = np.array(pairs, float)
        a, b = np.polyfit(p[:, 0], p[:, 1], 1)
        lo, hi = b, a * (PIC - 1) + b
        centre_pic = (size / 2 - b) / a
        print(f"{axis}: {a:.2f} monitor px per picture px, picture {name} 0..{PIC - 1} see monitor {axis} "
              f"{lo:.0f}..{hi:.0f} ({(hi - lo) * PITCH_MM:.0f} mm); monitor centre at picture {axis} = "
              f"{centre_pic:.0f} (ideal {PIC / 2:.0f}, off by {(centre_pic - PIC / 2) * a * PITCH_MM:+.0f} mm "
              f"at the screen)")
    print(f"wrote {out / 'fov_map.png'}")


if __name__ == "__main__":
    main(*(int(v) for v in sys.argv[1:4]))
