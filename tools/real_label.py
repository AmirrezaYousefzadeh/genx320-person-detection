"""Labels for real-scene recordings (tools/self_record.py: one person moving in an otherwise still room).

Cuts events.raw into 50 ms pictures exactly like record_dataset.py (pictures/pic_NNNN.png: 128 nothing, 255 ON,
0 OFF) and boxes the person from the motion itself, since nothing else moves: event density (a box blur of the
active pixels) above the noise floor, nearby dense parts merged (head, arms and legs can move separately), the
largest merged region, the bounding box of its active pixels, smoothed over neighbouring pictures. The noise floor
is measured on the empty-room segments (scenario names starting with "empty"). A picture whose region holds fewer
than --min-events active pixels gets no box: a person who stands still gives no events and cannot be seen, as in
the monitor passes. Shadows and reflections can stretch a box: check the previews by eye.

Usage: real_label.py label OUT_DIR [--window 50] [--min-events 300]
       real_label.py score OUT_DIR MODEL.tflite [MODEL.tflite ...]   precision/recall at IoU 0.5 on these labels
Writes per segment labels/pic_NNNN.txt (YOLO "0 cx cy w h"), preview_NNNN.png (every 20th picture, box drawn) and
labels.json; OUT_DIR/labels_summary.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evt2 import PIC, check, decode, picture, read_raw  # noqa: E402

BLUR = 15        # density kernel, picture pixels
MERGE = 31       # dense parts closer than this are one person
SMOOTH = 2       # median of the box over +-SMOOTH pictures


def segments(out):
    return sorted(p for p in Path(out).iterdir() if (p / "segment.json").exists() and (p / "events.raw").exists())


def pictures_of(seg, window_ms):
    _, words = read_raw(seg / "events.raw")
    ev = decode(words)
    t = ev["t"] - ev["t"][0]
    n = int(t[-1] // (window_ms * 1000)) + 1
    edges = np.searchsorted(t, np.arange(n + 1) * window_ms * 1000)
    for k in range(n):
        sel = np.zeros(len(t), bool)
        sel[edges[k]:edges[k + 1]] = True
        yield k, picture(ev, sel), check(ev) if k == 0 else None


def density(pic):
    return cv2.blur((pic != 128).astype(np.float32), (BLUR, BLUR))


def noise_floor(out, window_ms):
    """median over the empty-room pictures of their 99.9th percentile of event density (the median: a few noise
    bursts in the empty room reach 10x the usual level and made the threshold miss sparse silhouettes)"""
    vals = [np.percentile(density(pic), 99.9) for seg in segments(out)
            if json.loads((seg / "segment.json").read_text())["scenario"].startswith("empty")
            for _, pic, _ in pictures_of(seg, window_ms)]
    if not vals:
        sys.exit("no empty-room segment to measure the noise floor on")
    return float(np.median(vals))


def person_box(pic, thr, min_events):
    dense = (density(pic) > thr).astype(np.uint8)
    if dense.sum() == 0:
        return None
    merged = cv2.dilate(dense, np.ones((MERGE, MERGE), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(merged)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (lab == k) & (dense > 0) & (pic != 128)
    ys, xs = np.nonzero(region)
    if len(xs) < min_events:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32) / PIC


def smooth(boxes):
    """Median of each coordinate over +-SMOOTH neighbouring pictures that have a box."""
    out = []
    for i, b in enumerate(boxes):
        if b is None:
            out.append(None)
            continue
        near = [boxes[j] for j in range(max(0, i - SMOOTH), min(len(boxes), i + SMOOTH + 1)) if boxes[j] is not None]
        out.append(np.median(np.array(near), axis=0))
    return out


def label(out, window_ms=50, min_events=300):
    out = Path(out)
    thr = noise_floor(out, window_ms) * 3.0
    summary = {"noise_threshold": thr, "window_ms": window_ms, "min_events": min_events, "segments": {}}
    for seg in segments(out):
        (seg / "pictures").mkdir(exist_ok=True)
        (seg / "labels").mkdir(exist_ok=True)
        raw, pics, stats = [], [], None
        for k, pic, st in pictures_of(seg, window_ms):
            stats = st or stats
            cv2.imwrite(str(seg / "pictures" / f"pic_{k:04d}.png"), pic)
            raw.append(person_box(pic, thr, min_events))
            pics.append(pic)
        boxes = smooth(raw)
        for k, b in enumerate(boxes):
            with open(seg / "labels" / f"pic_{k:04d}.txt", "w") as f:
                if b is not None:
                    f.write(f"0 {(b[0] + b[2]) / 2:.4f} {(b[1] + b[3]) / 2:.4f} {b[2] - b[0]:.4f} {b[3] - b[1]:.4f}\n")
            if k % 20 == 0:
                vis = cv2.cvtColor(pics[k], cv2.COLOR_GRAY2BGR)
                if b is not None:
                    cv2.rectangle(vis, (int(b[0] * PIC), int(b[1] * PIC)), (int(b[2] * PIC) - 1, int(b[3] * PIC) - 1),
                                  (0, 140, 255), 1)
                cv2.imwrite(str(seg / f"preview_{k:04d}.png"), vis)
        n_box = sum(b is not None for b in boxes)
        info = {**stats, "pictures": len(boxes), "pictures_with_a_person": n_box, "noise_threshold": thr,
                "min_events": min_events, "method": "motion region (one person in a still room)"}
        (seg / "labels.json").write_text(json.dumps(info, indent=1))
        summary["segments"][seg.name] = {"pictures": len(boxes), "with_a_person": n_box}
        print(f"{seg.name}: {len(boxes)} pictures, a person in {n_box}", flush=True)
    (out / "labels_summary.json").write_text(json.dumps(summary, indent=1))


def score(out, models):
    from finetune import evaluate, tflite_predictor  # TensorFlow only here
    samples = []
    for seg in segments(out):
        for png in sorted((seg / "pictures").glob("pic_*.png")):
            txt = seg / "labels" / (png.stem + ".txt")
            boxes = [tuple(float(v) for v in l.split()[1:5]) for l in txt.read_text().splitlines()] if txt.exists() else []
            samples.append((str(png), boxes))
    for m in models:
        r = evaluate(tflite_predictor(m), samples)
        print(f"{m}: {len(samples)} pictures  P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f}  {r}")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    if len(a) >= 2 and a[0] == "label":
        label(a[1], opt("--window", 50), opt("--min-events", 300))
    elif len(a) >= 3 and a[0] == "score":
        score(a[1], [m for m in a[2:] if not m.startswith("--")])
    else:
        sys.exit(__doc__)
