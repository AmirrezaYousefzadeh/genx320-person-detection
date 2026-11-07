"""PersonPath22 (Amazon, ECCV 2022, CC BY-NC 4.0) as 60 fps MOT-style sequences for the monitor campaign.

One 10 s window per video (the most scenes for the copy time we have): the window inside the annotated part with the
most people that are at least 5 % of the frame high and at least a quarter visible. PersonPath22 labels key frames
(about 5 per second); boxes in between are interpolated per track id (not across gaps of more than three key-frame
steps). Visibility = visible box / amodal box. Classes in MOT17's numbering: 1 person (also occluded or in the
background), 2 person on an open vehicle, 7 sitting or standing (static) person; 8 person inside a vehicle, 12
reflection and 13 crowd are kept with conf 0, so the readers leave them out. 4K videos are scaled to 1920 wide.

Writes work/datasets/pp22/<part>/<uid>_w<start>/ (source frame rate, img1/ gt/ seqinfo.ini meta.json), then the 60 fps
copy work/datasets/pp22_60/<part>/... (tools/make60.py) with READY in each finished part, and inserts the part into the
campaign plan (work/campaign/plan.json) before its first "lit" entry, as pass camp60pp_<part> (zoom 0.4, any light).
Usage: pp22_prep.py [WINDOWS_PER_PART 30] [MAX_WINDOWS]
"""
import configparser
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make60  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "work/datasets/personpath22/raw"
SPLITS = ROOT / "work/datasets/personpath22/splits.json"
OUT, OUT60 = ROOT / "work/datasets/pp22", ROOT / "work/datasets/pp22_60"
PLAN = ROOT / "work/campaign/plan.json"
WINDOW_S, MIN_H, MAX_W = 10.0, 0.05, 1920


def mot_class(labels):
    """(MOT17 class, conf) of a PersonPath22 entity"""
    if labels.get("reflection"):
        return 12, 0
    if labels.get("crowd"):
        return 13, 0
    if labels.get("person_in_vehicle"):
        return 8, 0
    if labels.get("person_on_open_vehicle"):
        return 2, 1
    if labels.get("sitting_person") or labels.get("standing_person"):
        return 7, 1
    return 1, 1


def load(uid):
    """metadata and {id: [(frame, x, y, w, h, vis, cls, conf), ...]} sorted by frame"""
    am = json.loads((RAW / "anno_amodal_2022" / f"{uid}.json").read_text())
    vi = json.loads((RAW / "anno_visible_2022" / f"{uid}.json").read_text())
    vis_box = {(e["id"], e["blob"]["frame_idx"]): e["bb"] for e in vi["entities"]}
    tracks = {}
    for e in am["entities"]:
        f, (x, y, w, h) = e["blob"]["frame_idx"], e["bb"]
        v = vis_box.get((e["id"], f))
        vis = 0.0
        if v is not None and w > 0 and h > 0 and not e["labels"].get("fully_occluded"):
            ix = max(0.0, min(x + w, v[0] + v[2]) - max(x, v[0]))
            iy = max(0.0, min(y + h, v[1] + v[3]) - max(y, v[1]))
            vis = min(1.0, ix * iy / (w * h))
        cls, conf = mot_class(e["labels"])
        tracks.setdefault(e["id"], []).append((f, x, y, w, h, vis, cls, conf))
    for t in tracks.values():
        t.sort()
    return am["metadata"], tracks


def window(meta, tracks, fps, n_frames):
    """start frame of the best 10 s window, or None"""
    H = meta["resolution"]["height"]
    kf = sorted({r[0] for t in tracks.values() for r in t})
    if len(kf) < 2:
        return None
    W = int(round(WINDOW_S * fps))
    lo, hi = kf[0], min(kf[-1], n_frames - 1) - W + 1
    if hi < lo:
        return None
    good = {}
    for t in tracks.values():
        for f, x, y, w, h, vis, cls, conf in t:
            if conf and vis >= 0.25 and h >= MIN_H * H:
                good[f] = good.get(f, 0) + 1
    best = None
    for s in range(lo, hi + 1, max(1, int(round(fps)))):
        score = sum(c for f, c in good.items() if s <= f < s + W)
        if best is None or score > best[0]:
            best = (score, s)
    return best[1] if best and best[0] > 0 else None


def rows_at(tracks, f, step):
    """interpolated (id, x, y, w, h, vis, cls, conf) of every track at frame f"""
    out = []
    for tid, t in tracks.items():
        fr = [r[0] for r in t]
        i = int(np.searchsorted(fr, f, side="right")) - 1
        if i < 0:
            continue
        a = t[i]
        if a[0] == f:
            out.append((tid, *a[1:]))
            continue
        if i + 1 >= len(t):
            continue
        b = t[i + 1]
        if b[0] - a[0] > 3 * step:  # the person left and came back: no box in the gap
            continue
        u = (f - a[0]) / (b[0] - a[0])
        box = [(1 - u) * p + u * q for p, q in zip(a[1:6], b[1:6])]
        near = a if u < 0.5 else b
        out.append((tid, *box, near[6], near[7]))
    return out


def prep(args):
    uid, part, split = args
    cv2.setNumThreads(1)
    meta, tracks = load(uid)
    cap = cv2.VideoCapture(str(RAW / uid))
    fps = cap.get(cv2.CAP_PROP_FPS) or meta["fps"]
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or meta["number_of_frames"])
    s = window(meta, tracks, fps, n)
    if s is None:
        return f"{uid}: no window with people"
    W = int(round(WINDOW_S * fps))
    name = f"{uid.split('.')[0]}_w{s:05d}"
    d = OUT / part / name
    if not (d / "seqinfo.ini").exists():
        (d / "img1").mkdir(parents=True, exist_ok=True)
        (d / "gt").mkdir(exist_ok=True)
        kf = sorted({r[0] for t in tracks.values() for r in t})
        step = float(np.median(np.diff(kf)))
        for _ in range(s):
            cap.grab()
        scale, size = 1.0, None
        for k in range(W):
            ok, fr = cap.read()
            if not ok:
                return f"{uid}: video ended at frame {s + k}"
            if fr.shape[1] > MAX_W:
                scale = MAX_W / fr.shape[1]
                fr = cv2.resize(fr, (MAX_W, int(round(fr.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            size = fr.shape[1], fr.shape[0]
            cv2.imwrite(str(d / "img1" / f"{k + 1:06d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(d / "gt" / "gt.txt", "w") as g:
            for k in range(W):
                for tid, x, y, w, h, vis, cls, conf in rows_at(tracks, s + k, step):
                    g.write(f"{k + 1},{tid},{x * scale:.2f},{y * scale:.2f},{w * scale:.2f},{h * scale:.2f},"
                            f"{conf},{cls},{vis:.4f}\n")
        c = configparser.ConfigParser()
        c.optionxform = str
        c["Sequence"] = {"name": name, "imDir": "img1", "frameRate": f"{fps:.6g}", "seqLength": str(W),
                         "imWidth": str(size[0]), "imHeight": str(size[1]), "imExt": ".jpg"}
        with open(d / "seqinfo.ini", "w") as f:
            c.write(f, space_around_delimiters=False)
        (d / "meta.json").write_text(json.dumps({
            "source": f"PersonPath22 {uid} (Amazon, ECCV 2022), CC BY-NC 4.0; do not use to identify people",
            "personpath22_split": split, "window_frames": [s, s + W - 1], "window_s": [s / fps, (s + W) / fps],
            "fps": fps, "scale": scale, "keyframe_step": step,
            "labels": "key-frame boxes interpolated per track id; classes in MOT17 numbering (tools/pp22_prep.py)"},
            indent=1))
    return make60.make((OUT60 / part, str(d)))


def add_to_plan(part):
    plan = json.loads(PLAN.read_text())
    name = f"camp60pp_{part}"
    if any(e["name"] == name for e in plan["entries"]):
        return
    entry = {"name": name, "light": "any",
             "opts": ["--src", str((OUT60 / part).relative_to(ROOT)), "--zoom", "0.4", "--zoom-rank", "0"]}
    i = next((k for k, e in enumerate(plan["entries"]) if e["light"] == "lit"), len(plan["entries"]))
    plan["entries"].insert(i, entry)
    tmp = PLAN.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, indent=1))
    tmp.replace(PLAN)


if __name__ == "__main__":
    per_part = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    split_of = {v: k for k, vs in json.loads(SPLITS.read_text()).items() for v in vs}
    uids = sorted(p.name for p in RAW.glob("*.mp4") if (RAW / "anno_amodal_2022" / f"{p.name}.json").exists())
    # train and test videos interleaved, so every part holds both
    tr, te = [u for u in uids if split_of.get(u) == "train"], [u for u in uids if split_of.get(u) != "train"]
    order = [u for pair in zip(tr, te) for u in pair] + tr[len(te):] + te[len(tr):]
    if len(sys.argv) > 2:
        order = order[:int(sys.argv[2])]
    parts = [order[i:i + per_part] for i in range(0, len(order), per_part)]
    with Pool(int(os.environ.get("NUM_THREADS", "2"))) as pool:
        for k, uids_k in enumerate(parts, 1):
            part = f"part{k}"
            for line in pool.imap_unordered(prep, [(u, part, split_of.get(u, "?")) for u in uids_k]):
                print(part, line, flush=True)
            if (OUT60 / part).is_dir():
                (OUT60 / part / "READY").touch()
                add_to_plan(part)
                print(f"{part} ready and in the plan", flush=True)
