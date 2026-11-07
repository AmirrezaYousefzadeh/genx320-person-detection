"""Grab what the LCD shows from the running board over SWD (openocd) and save it as one 480x272 PNG.

Reads the LTDC registers to find the enabled layers (address, window, pixel format, constant alpha, default colour,
blending factors), dumps their frame buffers and blends them like the LTDC does: over the whole screen, with the
layer's default colour outside its window, BF1 = CA or PA x CA, BF2 = 1 - CA or 1 - PA x CA. (Until 2026-11-05 it
blended each layer inside its window only, and showed a panel that the opaque default colour of the layer above
hid on the glass.) The L8 colour table cannot be read back, so L8 layers use the event colours of the firmware.
Usage: grab_lcd.py <out.png> [--portrait]   (--portrait turns the image by 90 degrees, for a board in portrait)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

LTDC = 0x40016800
LAYER_OFFSET = (0x84, 0x104)
EVENT_CLUT = np.array([[30, 37, 52], [216, 223, 236], [64, 126, 201], [255, 140, 0], [225, 35, 35],
                       [255, 255, 255]] + [[255, 0, 255]] * 250, dtype=np.uint8)
BYTES_PER_PIXEL = {0: 4, 1: 3, 2: 2, 5: 1}  # ARGB8888, RGB888, RGB565, L8


def openocd(cmds, resume=True):
    """Halt, run cmds, and resume unless told otherwise (the core then stays halted for the next call, so that
    the registers read in one call and the buffers dumped in the next belong to the same moment)."""
    subprocess.run(["openocd", "-f", "board/stm32f7discovery.cfg", "-c", "; ".join(["init", "halt"] + cmds +
                    (["resume"] if resume else []) + ["shutdown"])], capture_output=True, check=True)


def words(raw):
    return np.frombuffer(raw, dtype="<u4")


def decode(raw, fmt, w, h, pitch):
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(h, pitch)[:, :w * BYTES_PER_PIXEL[fmt]]
    if fmt == 2:
        p = rows.copy().view("<u2").astype(np.uint32)
        rgb = np.stack([(p >> 11) * 255 // 31, ((p >> 5) & 63) * 255 // 63, (p & 31) * 255 // 31], axis=-1)
        return rgb.astype(np.float32), np.ones((h, w), np.float32)
    if fmt == 5:
        return EVENT_CLUT[rows].astype(np.float32), np.ones((h, w), np.float32)
    if fmt == 1:
        return rows.reshape(h, w, 3)[..., ::-1].astype(np.float32), np.ones((h, w), np.float32)
    p = rows.reshape(h, w, 4)
    return p[..., 2::-1].astype(np.float32), p[..., 3] / 255.0


def grab(out_png, portrait=False):
    with tempfile.TemporaryDirectory() as tmp:
        regs_file = Path(tmp) / "ltdc.raw"
        openocd([f"dump_image {regs_file} {LTDC:#x} 0x140"], resume=False)
        r = words(regs_file.read_bytes())
        reg = lambda off: int(r[off // 4])
        ahbp, avbp = reg(0x0C) >> 16 & 0xFFF, reg(0x0C) & 0x7FF
        aaw, aah = reg(0x10) >> 16 & 0xFFF, reg(0x10) & 0x7FF
        width, height = aaw - ahbp, aah - avbp
        bg = reg(0x2C)
        screen = np.tile(np.array([bg >> 16 & 255, bg >> 8 & 255, bg & 255], np.float32), (height, width, 1))
        layers = []
        for n, base in enumerate(LAYER_OFFSET, 1):
            cr, whpcr, wvpcr = reg(base), reg(base + 0x04), reg(base + 0x08)
            if not cr & 1:
                continue
            x0, x1 = (whpcr & 0xFFF) - ahbp - 1, (whpcr >> 16 & 0xFFF) - ahbp
            y0, y1 = (wvpcr & 0x7FF) - avbp - 1, (wvpcr >> 16 & 0x7FF) - avbp
            fmt, alpha, addr = reg(base + 0x10) & 7, (reg(base + 0x14) & 255) / 255.0, reg(base + 0x28)
            dccr, bfcr = reg(base + 0x18), reg(base + 0x1C)
            pitch = reg(base + 0x2C) >> 16 & 0x1FFF
            lines = reg(base + 0x30) & 0x7FF
            layers.append((n, x0, x1, y0, y1, fmt, alpha, addr, pitch, lines, dccr, bfcr))
        cmds = [f"dump_image {tmp}/layer{n}.raw {addr:#x} {pitch * lines}"
                for n, _, _, _, _, _, _, addr, pitch, lines, _, _ in layers]
        openocd(cmds)
        for n, x0, x1, y0, y1, fmt, alpha, addr, pitch, lines, dccr, bfcr in layers:
            # the layer over the whole screen: its default colour outside the window, its pixels inside
            rgb = np.tile(np.array([dccr >> 16 & 255, dccr >> 8 & 255, dccr & 255], np.float32), (height, width, 1))
            pa = np.full((height, width), (dccr >> 24 & 255) / 255.0, np.float32)
            px, a = decode((Path(tmp) / f"layer{n}.raw").read_bytes(), fmt, x1 - x0, lines, pitch)
            h = min(lines, y1 - y0)
            rgb[y0:y0 + h, x0:x1], pa[y0:y0 + h, x0:x1] = px[:h], a[:h]
            pixel_alpha = (bfcr >> 8 & 7) == 6  # BF1 = PA x CA (else CA, and the default colour is opaque)
            f = ((pa if pixel_alpha else np.ones_like(pa)) * alpha)[..., None]
            screen = f * rgb + (1 - f) * screen
            print(f"layer {n}: {x1 - x0}x{lines} at ({x0},{y0}) format {fmt} alpha {alpha:.2f} address {addr:#x} "
                  f"blending {'PAxCA' if pixel_alpha else 'CA'}")
    img = screen.clip(0, 255).astype(np.uint8)
    if portrait:
        img = np.rot90(img, k=-1)
    Image.fromarray(img).save(out_png)
    print(f"saved {out_png}")


if __name__ == "__main__":
    grab(sys.argv[1], "--portrait" in sys.argv[2:])
