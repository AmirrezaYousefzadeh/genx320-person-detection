"""Read a histogram-mode recording of the board (make HISTO=1): a 512-byte text header, then frames of a 16-byte
header {magic "HFRM", board tick ms, sequence, 0} and 320x320 bytes (OFF count in the low nibble, ON count in
the high nibble; raw sensor layout, line = sensor y, byte = sensor x).

  read_frames(path) -> (header text, ticks ms [N], frames uint8 [N, 320, 320])
  picture(frame)    -> the detector's 320x320 picture (grey 128 nothing, white ON wins, black OFF wins, a tie by
                       pixel parity), in picture coordinates X = 319 - y, Y = x, like the firmware (frames.c)
  on_counts(frame)  -> ON counts in picture coordinates (for the blink-marker sync)
Usage: histo_raw.py FILE.raw   prints a summary
"""
import sys

import numpy as np

PIC = 320
MAGIC = 0x4D524648
FRAME_BYTES = 16 + PIC * PIC


def read_frames(path):
    data = np.fromfile(path, dtype=np.uint8)
    header = bytes(data[:512]).decode(errors="replace")
    body = data[512:]
    n = body.size // FRAME_BYTES
    body = body[:n * FRAME_BYTES].reshape(n, FRAME_BYTES)
    hdr = body[:, :16].copy().view("<u4")  # magic, tick, seq, 0
    ok = hdr[:, 0] == MAGIC
    if not ok.all():
        print(f"warning: {(~ok).sum()} frame headers without the magic", file=sys.stderr)
    frames = body[ok, 16:].reshape(-1, PIC, PIC)
    return header, hdr[ok, 1].astype(np.int64), frames


def to_picture(frame_xy):
    """sensor layout [y, x] -> picture [Y, X] with X = 319 - y, Y = x"""
    return frame_xy.T[:, ::-1]


def picture(frame):
    f = to_picture(frame)
    on, off = f >> 4, f & 0xF
    pic = np.full((PIC, PIC), 128, np.uint8)
    yy, xx = np.mgrid[0:PIC, 0:PIC]
    white = np.where(on != off, on > off, ((xx ^ yy) & 1) == 1)
    active = (on | off) != 0
    pic[active & white] = 255
    pic[active & ~white] = 0
    return pic


def on_counts(frame):
    return to_picture(frame) >> 4


if __name__ == "__main__":
    h, t, f = read_frames(sys.argv[1])
    print(h.split("\n")[1])
    if len(t):
        dt = np.diff(t)
        print(f"{len(t)} frames, ticks {t[0]}..{t[-1]} ms, interval mean {dt.mean():.1f} ms (min {dt.min()}, max {dt.max()}),"
              f" active pixels per frame mean {np.mean([(x != 0).sum() for x in f[:50]]):.0f}")
