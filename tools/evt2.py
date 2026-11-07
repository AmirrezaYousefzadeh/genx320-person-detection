"""EVT2.0 helpers shared by the tools: read a Prophesee RAW file, decode the words, build pictures.

EVT2.0 words (32 bit): type in bits 31..28. CD_OFF (0x0) / CD_ON (0x1): timestamp bits 5..0 in 27..22, x in
21..11, y in 10..0. EVT_TIME_HIGH (0x8): timestamp bits 33..6 in 27..0. The firmware's picture is X = 319 - y,
Y = x (X to the right, Y down).
"""
import numpy as np

PIC = 320
HEADER_END = b"% end\n"


def read_raw(path):
    """(header text, words) of a Prophesee EVT2 RAW file (our firmware writes a 512-byte header)."""
    data = open(path, "rb").read()
    end = data.index(HEADER_END) + len(HEADER_END)
    words = np.frombuffer(data[end:len(data) - (len(data) - end) % 4], dtype="<u4")
    return data[:end].decode("ascii", "replace"), words


def decode(words):
    """dict with x, y (sensor coordinates, uint16), p (1 ON / 0 OFF, uint8), t (us, int64) of the CD events."""
    types = words >> 28
    th = np.where(types == 0x8, (words & 0x0FFFFFFF).astype(np.int64) << 6, -1)
    idx = np.where(th >= 0, np.arange(len(words)), 0)
    np.maximum.accumulate(idx, out=idx)
    time_high = th[idx]
    cd = (types <= 1) & (time_high >= 0)
    w = words[cd]
    return {"x": ((w >> 11) & 0x7FF).astype(np.uint16), "y": (w & 0x7FF).astype(np.uint16),
            "p": (types[cd] == 1).astype(np.uint8), "t": time_high[cd] + ((w >> 22) & 0x3F).astype(np.int64)}


def to_picture_coords(ev):
    """X, Y (int32) of the events in the firmware's picture."""
    return (PIC - 1) - ev["y"].astype(np.int32), ev["x"].astype(np.int32)


def picture(ev, sel):
    """The detector's picture of the selected events: 128 nothing, 255 ON, 0 OFF (last event wins)."""
    X, Y = to_picture_coords(ev)
    pic = np.full((PIC, PIC), 128, np.uint8)
    pic[Y[sel], X[sel]] = np.where(ev["p"][sel] == 1, 255, 0).astype(np.uint8)
    return pic


def check(ev):
    """Sanity numbers: event count, duration, negative time steps, gaps over 1 ms."""
    t = ev["t"]
    dt = np.diff(t)
    return {"events": int(len(t)), "duration_s": float((t[-1] - t[0]) / 1e6) if len(t) > 1 else 0.0,
            "negative_steps": int((dt < 0).sum()), "gaps_over_1ms": int((dt > 1000).sum()),
            "max_gap_us": int(dt.max()) if len(dt) else 0}
