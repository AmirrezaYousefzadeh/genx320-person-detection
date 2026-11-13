"""Frame-camera baseline: the same detector architecture on RGB frames of the same scenes and the same views.

For every 50 ms event picture of a test segment, this takes the video frame that the monitor showed at the middle of
the picture's window (frames.log and the time fit in events.json), mirrored like the replay and cropped to the square
that the sensor saw. Its boxes come from the segment's own labels_all/, so that the RGB frames and the event pictures
are scored on the same people:
  moving  the event labels (people with >= 20 active pixels); the other annotated people are regions to ignore
  all     every annotated person (a frame camera also sees people who stand still)
Regions to ignore and the minimum person height (12 % of the event picture) are the same for both inputs.
Networks: an RGB model (default: ST's COCO YOLO-LC as shipped, int8, RGB 192 px) on the frames, and an event model
(default: the network on the board) on the event pictures of the same moments.
Usage: rgb_baseline.py OUT.json [--sets mon60z_dark,pp22test] [--rgb MODEL] [--rgb-size 192] [--rgb-scale 1]
                       [--event MODEL] [--event-size 160] [--stride 2]
ST's keras models (work/models_zoo/*.keras) take RGB in [0, 1]: --rgb-scale 0.00392156862745098; the int8 file takes
0..255 (its input quantization does the rest).
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_ap import decode, evaluate, load, predictor, samples, timing_uncertain  # noqa: E402
from eval_suite import sets  # noqa: E402
from record_dataset import local_seq_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PIC = 320


def pc_time_fn(seg_json, ev_json):
    """sensor time (us from the first event) -> PC time, as the labeller did"""
    s = ev_json.get("sync", {})
    start = seg_json["pc_start"]
    if s.get("method") == "blink marker":
        slope = 1 + s.get("drift_ppm", 0.0) / 1e6
        return lambda us: start + s["offset_s"] + slope * us / 1e6
    if str(s.get("method", "")).startswith("motion fit"):
        return lambda us: start + s["offset_s"] + us / 1e6
    return lambda us: start + us / 1e6


def boxes(txt):
    out = []
    if txt.exists():
        for line in txt.read_text().splitlines():
            v = line.split()
            out.append((int(float(v[0])), *(float(x) for x in v[1:5])))
    return out


def to_crop(c, cx, cy, w, h, tx, ty, ts):
    """picture coordinates (warped replay: fractions of 320 px) -> fractions of the crop"""
    x0, y0 = ((cx - w / 2) * PIC - tx) / ts, ((cy - h / 2) * PIC - ty) / ts
    x1, y1 = ((cx + w / 2) * PIC - tx) / ts, ((cy + h / 2) * PIC - ty) / ts
    return c, np.clip([x0, y0, x1, y1], 0, 1)


def segment_items(seg, stride):
    """[(event picture path, RGB crop (uint8 RGB), {target set: (gt, ign)} in crop and in picture coordinates)]"""
    sj = json.loads((seg / "segment.json").read_text())
    ej = json.loads((seg / "events.json").read_text())
    log = [l for l in (seg / "frames.log").read_text().splitlines() if l]
    warp = next(l for l in log if l.startswith("warp ")).split()
    cx0, cy0, side = float(warp[1]), float(warp[2]), float(warp[3])
    tx, ty, ts = (float(v) for v in warp[warp.index("target") + 1:warp.index("target") + 4])
    flipped = any(l == "flip 1" for l in log)
    shown = np.array([[float(a), int(b)] for a, b in (l.split() for l in log if l[0].isdigit())])
    pc_of = pc_time_fn(sj, ej)
    img_dir = local_seq_dir(sj["seq_dir"]) / "img1"
    min_crop = 0.12 * PIC / ts  # the evaluation's 12 % of the event picture, in crop units
    out = []
    pics = sorted((seg / "pictures").glob("pic_*.png"))[::stride]
    for p in pics:
        k = int(p.stem.split("_")[1])
        t_mid = pc_of((k + 0.5) * 50_000)
        i = np.searchsorted(shown[:, 0], t_mid) - 1
        if not (0 <= i < len(shown)) or t_mid - shown[i, 0] > 0.5:
            continue
        frame = cv2.imread(str(img_dir / f"{int(shown[i, 1]) + 1:06d}.jpg"))
        if frame is None:
            continue
        if flipped:
            frame = frame[:, ::-1]
        x0, y0 = int(round(cx0)), int(round(cy0))
        crop = frame[max(y0, 0):y0 + int(round(side)), max(x0, 0):x0 + int(round(side))]
        if crop.size == 0:
            continue
        crop = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        moving = {tuple(np.round(b[1:], 4)) for b in boxes(seg / "labels" / p.with_suffix(".txt").name) if b[0] == 0}
        allb = boxes(seg / "labels_all" / p.with_suffix(".txt").name)
        sets_c = {"moving": ([], []), "all": ([], [])}
        sets_p = {"moving": ([], []), "all": ([], [])}
        for c, cx, cy, w, h in allb:
            _, bc = to_crop(c, cx, cy, w, h, tx, ty, ts)
            bp = np.clip([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0, 1)
            small = (bc[3] - bc[1]) < min_crop
            is_moving = c == 0 and tuple(np.round((cx, cy, w, h), 4)) in moving
            for name, target in (("moving", c == 0 and is_moving and not small), ("all", c == 0 and not small)):
                (sets_c[name][0] if target else sets_c[name][1]).append(bc)
                (sets_p[name][0] if target else sets_p[name][1]).append(bp)
        arr = lambda v: np.array(v, np.float32).reshape(-1, 4)  # noqa: E731
        out.append((str(p), crop, {n: (arr(g), arr(ig)) for n, (g, ig) in sets_c.items()},
                    {n: (arr(g), arr(ig)) for n, (g, ig) in sets_p.items()}, seg.name))
    return out


def main(argv):
    out = Path(argv[0])
    opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
    chosen = opt("--sets", "mon60z_dark,pp22test").split(",")
    rgb_model = opt("--rgb", str(ROOT / "work/models_zoo/st_yololcv1_192_int8.tflite"))
    rgb_size = int(opt("--rgb-size", "192"))
    rgb_scale = float(opt("--rgb-scale", "1"))
    ev_model = opt("--event", str(ROOT / "models/paper/row5_all_160/hf_body_int8.tflite") + "+"
                   + str(ROOT / "models/paper/row5_all_160/hf_head.npz"))
    ev_size = int(opt("--event-size", "160"))
    stride = int(opt("--stride", "2"))
    run_rgb, run_ev = predictor(rgb_model, rgb_size), predictor(ev_model, ev_size)
    res = json.loads(out.read_text()) if out.exists() else {}
    S = sets()
    for name in chosen:
        roots, select, _ = S[name]
        keys = [f"MOT17-{s}-" if s.isdigit() else s for s in select.split(",") if s]
        segs = sorted({Path(p).parent.parent for p, _, _ in samples(roots.split(","), select.split(","), 1, 0.12)})
        segs = [s for s in segs if not timing_uncertain(s) and (not keys or any(k in s.name for k in keys))]
        items = []
        for seg in segs:
            items += segment_items(seg, stride)
        print(f"{name}: {len(segs)} segments, {len(items)} pictures", flush=True)
        results = {(m, t): [] for m in ("rgb", "event") for t in ("moving", "all")}
        groups = [it[4] for it in items]
        for i in range(0, len(items), 128):
            chunk = items[i:i + 128]
            rs = [cv2.resize(c, (rgb_size, rgb_size), interpolation=cv2.INTER_AREA) for _, c, _, _, _ in chunk]
            outs_rgb = run_rgb([r.astype(np.float32) * rgb_scale for r in rs] if rgb_scale != 1 else rs)
            outs_ev = run_ev([load(p, ev_size) for p, _, _, _, _ in chunk])
            for (p, _, crop_sets, pic_sets, _), o_rgb, o_ev in zip(chunk, outs_rgb, outs_ev):
                s_r, b_r = decode(np.asarray(o_rgb))
                s_e, b_e = decode(np.asarray(o_ev))
                for t in ("moving", "all"):
                    results[("rgb", t)].append((s_r, b_r, *crop_sets[t]))
                    results[("event", t)].append((s_e, b_e, *pic_sets[t]))
        for (m, t), r in results.items():
            v = evaluate(r, 0.5, groups=groups)
            res.setdefault(name, {})[f"{m}/{t}"] = {k: v[k] for k in ("AP50", "AP50_std", "AP50_95", "P@0.5", "R@0.5",
                                                                    "people", "pictures", "segments")}
            print(f"{name:12s} {m:5s} {t:6s} AP50 {v['AP50']:.3f} +- {v['AP50_std'] or 0:.3f}  AP50:95 {v['AP50_95']:.3f}  "
                  f"P {v['P@0.5']:.2f} R {v['R@0.5']:.2f}  ({v['people']} people)", flush=True)
        out.write_text(json.dumps({**res, "_models": {"rgb": rgb_model, "rgb_size": rgb_size, "rgb_scale": rgb_scale, "event": ev_model,
                                                     "event_size": ev_size, "stride": stride}}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
