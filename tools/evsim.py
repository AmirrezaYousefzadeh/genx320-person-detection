"""Simulated GenX320 event pictures from ordinary video: a big training set from an artificial event camera, to be
fine-tuned on a small recorded set afterwards (Amir, 2026-11-06).

Video -> a fixed square crop per 10 s window, scaled to the 320x320 picture (people a set fraction of the view high,
chosen as in tools/zoom_plan.py) -> grey, slight blur (lens), linear light (gamma 2.2) -> temporal upsampling to
>= 120 Hz with tools/interp_flow.py (at picture size, cheap) -> events per pixel as in ESIM: an event each time the
log intensity moves one contrast threshold away from the pixel's reference level (thresholds differ per pixel) ->
50 ms pictures as the board makes them (128 nothing, 255 ON, 0 OFF, last event wins), with the sensor's event-rate
limit (ERC: 500 kev/s, 25 k events per picture; above it events are dropped at random) and background noise.
Contrast threshold, blur, light level, noise are drawn per crop from ranges (tools/evsim_calib.py fits them to the
monitor recordings of the same videos).

Labels (YOLO, picture coordinates): class 0 = a person (MOT17 classes 1, 2, 7, conf 1, visibility >= 0.25) giving at
least 20 events in its box; class 1 = a region to ignore (reflections, crowds, people in vehicles, people hardly
visible or giving fewer events). labels_all/ adds the event count of every box.
Sources: --mot DIR [DIR ...] (MOT-style sequence folders), --pp22 train|test|all (PersonPath22 videos).
Usage: evsim.py OUT (--mot DIR ... | --pp22 SPLIT) [--zooms 0.3,0.45,0.6] [--ranks 0] [--seed 0] [--max-windows N]
"""
import argparse
import json
import zlib
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import interp_flow  # noqa: E402
import zoom_plan  # noqa: E402

PIC, WIN_S, SEG_S, MIN_EVENTS = 320, 0.05, 10.0, 20
PERSON = (1, 2, 7)
# per-crop parameter ranges (uniform); tools/evsim_calib.py prints the values that match the monitor recordings
# fitted on the monitor recordings (MOT17-02 at 60 fps, 2026-11-06): inside the video 13.9 % of the pixels active, 17.5 %
# of those isolated (noise-like), 53 % ON; the still black monitor around it 2.4-4.5 % active (noise in the dark);
# the lit room ~1 %
RANGES = {"C": (0.22, 0.38), "C_sigma": (0.02, 0.05), "blur": (0.6, 1.2), "gain": (0.3, 1.0),
          "ambient": (0.0, 0.02), "noise": (0.005, 0.04), "on_bias": (0.45, 0.6)}
if os.environ.get("EVSIM_FIXED"):  # calibration runs: some parameters fixed, e.g. '{"C": 0.3, "noise": 0.02}'
    RANGES.update({k: (v, v) for k, v in json.loads(os.environ["EVSIM_FIXED"]).items()})
EPS = 0.004
CAP = 25000  # events per 50 ms picture (ERC at 500 kev/s)
WARMUP_S = 0.5


class Mot:
    """a MOT-style sequence folder (img1/, gt/gt.txt, seqinfo.ini)"""

    def __init__(self, d):
        self.d = Path(d)
        self.fps, self.W, self.H, self.n = _seqinfo(self.d)
        self.name = self.d.name
        self.files = sorted((self.d / "img1").glob("*.jpg"))[:self.n]
        self.gt = {}
        for line in (self.d / "gt" / "gt.txt").read_text().split():
            r = line.split(",")
            f = int(r[0]) - 1
            self.gt.setdefault(f, []).append((int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]),
                                              float(r[8]), int(float(r[7])), int(float(r[6]))))
        self.first, self.last = 0, self.n - 1

    def rows(self, f):  # (id, x, y, w, h, vis, cls, conf)
        return self.gt.get(int(round(f)), [])

    def frames(self, start, count):
        for f in self.files[start:start + count]:
            yield cv2.imread(str(f))


class PP22:
    """a PersonPath22 video with its key-frame labels (interpolated, tools/pp22_prep.py)"""

    def __init__(self, uid):
        import pp22_prep
        self.p = pp22_prep
        meta, self.tracks = pp22_prep.load(uid)
        self.uid, self.name = uid, "PP22-" + uid.split(".")[0]
        cap = cv2.VideoCapture(str(pp22_prep.RAW / uid))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or meta["fps"]
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or meta["number_of_frames"])
        self.W, self.H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        kf = sorted({r[0] for t in self.tracks.values() for r in t})
        self.step = float(np.median(np.diff(kf))) if len(kf) > 1 else 5.0
        self.first, self.last = (kf[0], min(kf[-1], self.n - 1)) if kf else (0, -1)

    def rows(self, f):
        return [(tid, x, y, w, h, vis, cls, conf) for tid, x, y, w, h, vis, cls, conf
                in self.p.rows_at(self.tracks, int(round(f)), self.step)]

    def frames(self, start, count):
        cap = cv2.VideoCapture(str(self.p.RAW / self.uid))
        for _ in range(start):
            cap.grab()
        for _ in range(count):
            ok, fr = cap.read()
            if not ok:
                return
            yield fr
        cap.release()


def _seqinfo(d):
    import configparser
    c = configparser.ConfigParser()
    c.read(Path(d) / "seqinfo.ini")
    s = c["Sequence"]
    return float(s["frameRate"]), int(s["imWidth"]), int(s["imHeight"]), int(s["seqLength"])


class Sim:
    """one crop's sensor: per-pixel reference levels and thresholds"""

    def __init__(self, rng, prm):
        self.prm, self.rng = prm, rng
        self.c_on = np.clip(rng.normal(prm["C"], prm["C_sigma"], (PIC, PIC)), 0.05, None).astype(np.float32)
        self.c_off = np.clip(rng.normal(prm["C"], prm["C_sigma"], (PIC, PIC)), 0.05, None).astype(np.float32)
        self.ref = None
        self.reset_window()

    def reset_window(self):
        self.last = np.zeros((PIC, PIC), np.int8)  # +1 ON, -1 OFF, 0 none (last event wins)
        self.count = np.zeros((PIC, PIC), np.int32)

    def log_image(self, grey):
        g = grey.astype(np.float32) / 255.0
        if self.prm["blur"] > 0:
            g = cv2.GaussianBlur(g, (0, 0), self.prm["blur"])
        return np.log(self.prm["gain"] * g ** 2.2 + self.prm["ambient"] + EPS)

    def step(self, L):
        if self.ref is None:
            self.ref = L.copy()
            return
        d = L - self.ref
        n_on = np.floor(np.maximum(d, 0) / self.c_on)
        n_off = np.floor(np.maximum(-d, 0) / self.c_off)
        self.ref += n_on * self.c_on - n_off * self.c_off
        self.last[n_on > 0] = 1
        self.last[n_off > 0] = -1
        self.count += (n_on + n_off).astype(np.int32)

    def picture(self):
        total = int(self.count.sum())
        active = self.count > 0
        if total > CAP:  # ERC: events dropped evenly; a pixel shows if one of its events survives
            keep = 1.0 - (1.0 - CAP / total) ** self.count
            active &= self.rng.random((PIC, PIC)) < keep
        pic = np.full((PIC, PIC), 128, np.uint8)
        pic[active & (self.last > 0)] = 255
        pic[active & (self.last < 0)] = 0
        noise = self.rng.random((PIC, PIC)) < self.prm["noise"]
        pic[noise] = np.where(self.rng.random(int(noise.sum())) < self.prm["on_bias"], 255, 0).astype(np.uint8)
        return pic, total


def draw_params(rng):
    return {k: float(rng.uniform(*v)) for k, v in RANGES.items()}


def crop_grey(frame, crop, flip):
    cx0, cy0, side = crop
    x0, y0 = int(round(cx0)), int(round(cy0))
    g = cv2.cvtColor(frame[y0:y0 + int(round(side)), x0:x0 + int(round(side))], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (PIC, PIC), interpolation=cv2.INTER_AREA if g.shape[0] > PIC else cv2.INTER_LINEAR)
    return g[:, ::-1].copy() if flip else g


def labels_for(src, pos, crop, flip, pic):
    """(label lines, labels_all lines) at source frame position pos"""
    cx0, cy0, side = crop
    k = PIC / side
    lines, all_lines = [], []
    active = pic != 128
    for tid, x, y, w, h, vis, cls, conf in src.rows(pos):
        x0, y0, x1, y1 = (x - cx0) * k, (y - cy0) * k, (x + w - cx0) * k, (y + h - cy0) * k
        if flip:
            x0, x1 = PIC - x1, PIC - x0
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, PIC), min(y1, PIC)
        if x1 - x0 < 0.02 * PIC or y1 - y0 < 0.02 * PIC:
            continue
        n_ev = int(active[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))].sum())
        person = cls in PERSON and conf > 0 and vis >= 0.25 and n_ev >= MIN_EVENTS
        line = f"{0 if person else 1} {(x0 + x1) / 2 / PIC:.4f} {(y0 + y1) / 2 / PIC:.4f} {(x1 - x0) / PIC:.4f} {(y1 - y0) / PIC:.4f}"
        lines.append(line)
        all_lines.append(f"{line} {n_ev} {cls} {vis:.2f}")
    return lines, all_lines


def people_array(src, start, n):
    p = []
    for f in range(start, start + n):
        for tid, x, y, w, h, vis, cls, conf in src.rows(f):
            if cls in PERSON and conf > 0 and vis >= 0.25:
                p.append((f - start, x, y, x + w, y + h))
    return np.array(p, np.float64).reshape(-1, 5)


def simulate_source(args):
    kind, ident, out, zooms, ranks, seed, max_windows = args
    cv2.setNumThreads(1)
    src = Mot(ident) if kind == "mot" else PP22(ident)
    rng = np.random.default_rng(zlib.crc32(f"{src.name} {seed}".encode()))
    fps = src.fps
    W = int(round(SEG_S * fps))
    starts = list(range(src.first, src.last - W + 2, W))[:max_windows or None]
    K = max(1, int(math.ceil(120.0 / fps)))
    done = 0
    for s in starts:
        crops = []
        p = people_array(src, s, W)
        if len(p) == 0:
            continue
        for z in zooms:
            for r in ranks:
                crop, (per_frame, rel) = zoom_plan.plan_boxes(p, src.W, src.H, W, z, r)
                flip = bool(rng.random() < 0.5)
                tag = f"{src.name}_f{s:05d}_z{round(z * 100)}_r{r}" + ("_flip" if flip else "")
                d = Path(out) / tag
                if (d / "meta.json").exists():
                    continue
                prm = draw_params(rng)
                crops.append({"crop": crop, "flip": flip, "dir": d, "sim": Sim(rng, prm), "prm": prm, "prev": None,
                              "fl": None, "j": 0, "tag": tag})
        if not crops:
            continue
        warm = min(int(round(WARMUP_S * fps)), s - src.first)
        f0 = s - warm
        n_pics = int(SEG_S / WIN_S)
        for c in crops:
            (c["dir"] / "pictures").mkdir(parents=True, exist_ok=True)
            (c["dir"] / "labels").mkdir(exist_ok=True)
            (c["dir"] / "labels_all").mkdir(exist_ok=True)
            c["events"] = []
        for i, frame in enumerate(src.frames(f0, warm + W + 1)):
            for c in crops:
                g = crop_grey(frame, c["crop"], c["flip"])
                if c["prev"] is None:
                    c["sim"].step(c["sim"].log_image(g))
                    c["prev"] = g
                    continue
                ts = [k / K for k in range(1, K)]
                mids = interp_flow.interpolate(c["prev"], g, ts, interp_flow.flows(c["prev"], g, 1.0)) if ts else []
                for k, img in enumerate(mids + [g], 1):
                    t = (i - 1 + k / K - warm) / fps  # s from the window start (the frame time of i - 1 + k/K)
                    if t > 0:
                        j = int(t / WIN_S - 1e-9)
                        while c["j"] < j and c["j"] < n_pics:  # a picture is complete
                            close_picture(src, c, s, fps)
                    c["sim"].step(c["sim"].log_image(img))
                    if t <= 0:  # warm-up: the levels settle, its events are not shown
                        c["sim"].reset_window()
                c["prev"] = g
        for c in crops:
            while c["j"] < n_pics:
                close_picture(src, c, s, fps)
            (c["dir"] / "meta.json").write_text(json.dumps({
                "source": src.name, "kind": kind, "start_frame": s, "fps": fps, "crop": [float(v) for v in c["crop"]],
                "flip": c["flip"], "params": c["prm"], "upsampling": K, "cap_events": CAP,
                "events_per_picture_mean": float(np.mean(c["events"])) if c["events"] else 0.0}, indent=1))
            done += 1
    return f"{src.name}: {done} windows x crops"


def close_picture(src, c, s, fps):
    j = c["j"]
    pic, total = c["sim"].picture()
    c["sim"].reset_window()
    c["events"].append(total)
    cv2.imwrite(str(c["dir"] / "pictures" / f"pic_{j:04d}.png"), pic)
    pos = s + (j + 0.5) * WIN_S * fps
    lines, all_lines = labels_for(src, pos, c["crop"], c["flip"], pic)
    (c["dir"] / "labels" / f"pic_{j:04d}.txt").write_text("".join(l + "\n" for l in lines))
    (c["dir"] / "labels_all" / f"pic_{j:04d}.txt").write_text("".join(l + "\n" for l in all_lines))
    c["j"] = j + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mot", nargs="*", default=[])
    ap.add_argument("--pp22", default="")
    ap.add_argument("--zooms", default="0.3,0.45,0.6")
    ap.add_argument("--ranks", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0)
    a = ap.parse_args()
    zooms = [float(v) for v in a.zooms.split(",")]
    ranks = [int(v) for v in a.ranks.split(",")]
    tasks = [("mot", d, a.out, zooms, ranks, a.seed, a.max_windows) for d in a.mot]
    if a.pp22:
        import pp22_prep
        split_of = {v: k for k, vs in json.loads(pp22_prep.SPLITS.read_text()).items() for v in vs}
        uids = sorted(p.name for p in pp22_prep.RAW.glob("*.mp4")
                      if (pp22_prep.RAW / "anno_amodal_2022" / f"{p.name}.json").exists()
                      and (a.pp22 == "all" or split_of.get(p.name) == a.pp22))
        tasks += [("pp22", u, a.out, zooms, ranks, a.seed, a.max_windows) for u in uids]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with Pool(int(os.environ.get("NUM_THREADS", "2"))) as pool:
        for line in pool.imap_unordered(simulate_source, tasks):
            print(line, flush=True)


if __name__ == "__main__":
    main()
