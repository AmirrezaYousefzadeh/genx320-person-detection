"""Curation with an RGB teacher (Amir's suggestion, 2026-11-06): a strong RGB person detector (Ultralytics YOLO11x,
COCO class person) runs on the source video frames that the monitor passes replay, and its detections are compared
with the labels the dataset inherits from those videos (MOT17 train gt, PersonPath22 windows). Every recording of a
frame shares that frame's labels, so a label fault in the source is a fault in every pass.

  run OUT [--every N] [--imgsz 1280] [--model PATH]   teacher detections per sequence or window -> OUT/<name>.json
  report OUT [--conf 0.6]                               OUT/teacher_report.csv: per sequence, the labelled people the
        teacher sees (IoU >= 0.4 with any label), the confident detections that overlap no label of any class
        ("unlabelled": people missing from the labels, or teacher errors) and the labelled people the teacher does
        not see at all; OUT/unlabelled_<name>.txt (MOT gt lines, class 13 = ignore) for the labeller; montages of
        the frames with the most unlabelled detections (labels orange, teacher cyan) for inspection.
Sources: work/datasets/mot17/unzipped/MOT17-*-FRCNN (MOT17 train, 30/25/14 fps) and work/datasets/pp22/part*/<window>
(PersonPath22 at the source frame rate). Classes: MOT17 numbering (1 pedestrian, 2 person on vehicle, 7 static
person, 8 distractor, 12 reflection, 13 crowd); conf 0 = ignore.
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / "work"


def sources():
    out = []
    for d in sorted((W / "datasets/mot17/unzipped/MOT17/train").glob("MOT17-*-FRCNN")):
        out.append((d.name, d))
    for d in sorted((W / "datasets/pp22").glob("part*/uid_vid_*")):
        out.append((d.name, d))
    return out


def read_gt(d):
    """{frame: [(x0, y0, x1, y1, cls, conf, vis)]}"""
    gt = {}
    for line in (d / "gt/gt.txt").read_text().splitlines():
        f = line.split(",")
        if len(f) < 9:
            continue
        fr, _, x, y, w, h, conf, cls, vis = (float(v) for v in f[:9])
        gt.setdefault(int(fr), []).append((x, y, x + w, y + h, int(cls), conf, vis))
    return gt


def iou_matrix(a, b):
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-6)


def inside_fraction(a, b):
    """fraction of each box of a inside each box of b"""
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + 1e-6)[:, None]


def run(out, every=2, imgsz=1280, model_path=None):
    from ultralytics import YOLO
    out.mkdir(parents=True, exist_ok=True)
    model_path = model_path or W / "models_ext/yolo11x.pt"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    for name, d in sources():
        dst = out / f"{name}.json"
        if dst.exists():
            continue
        frames = sorted((d / "img1").glob("*.jpg"))[::every]
        det = {}
        for i in range(0, len(frames), 16):
            chunk = frames[i:i + 16]
            res = model.predict([str(p) for p in chunk], imgsz=imgsz, conf=0.15, classes=[0], verbose=False, half=True)
            for p, r in zip(chunk, res):
                b = r.boxes
                det[int(p.stem)] = [[round(float(v), 1) for v in xyxy] + [round(float(c), 3)]
                                    for xyxy, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy())]
        dst.write_text(json.dumps({"source": str(d.relative_to(ROOT)), "every": every, "imgsz": imgsz, "frames": det}))
        print(name, len(frames), "frames", sum(len(v) for v in det.values()), "detections", flush=True)


def report(out, conf=0.6):
    rows, worst = [], []
    for name, d in sources():
        src = out / f"{name}.json"
        if not src.exists():
            continue
        det = {int(k): v for k, v in json.loads(src.read_text())["frames"].items()}
        gt = read_gt(d)
        H = cv2.imread(str(next((d / "img1").glob("*.jpg")))).shape[0]
        n_lab = n_seen = n_unl = n_unseen = n_frames = 0
        unl_lines, per_frame = [], []
        for fr, dets in det.items():
            n_frames += 1
            g = gt.get(fr, [])
            gb = [x[:4] for x in g]
            people = [x for x in g if x[4] == 1 and x[5] > 0 and x[6] >= 0.5 and (x[3] - x[1]) >= 0.05 * H]
            db = [x[:4] for x in dets]
            dc = np.array([x[4] for x in dets]) if dets else np.zeros(0)
            m = iou_matrix(db, gb)
            crowd = [x[:4] for x in g if x[4] == 13]
            inc = inside_fraction(db, crowd).max(axis=1) if crowd and dets else np.zeros(len(dets))
            unl = [k for k in range(len(dets)) if dc[k] >= conf and (not len(gb) or m[k].max() < 0.3) and inc[k] < 0.7]
            mp = iou_matrix([x[:4] for x in people], [db[k] for k in range(len(dets)) if dc[k] >= 0.15])
            seen = int(sum(1 for k in range(len(people)) if len(mp[k]) and mp[k].max() >= 0.4))
            unseen = int(sum(1 for k in range(len(people)) if not len(mp[k]) or mp[k].max() < 0.3))
            n_lab += len(people)
            n_seen += seen
            n_unl += len(unl)
            n_unseen += unseen
            for k in unl:
                x0, y0, x1, y1 = db[k]
                unl_lines.append(f"{fr},-1,{x0:.1f},{y0:.1f},{x1 - x0:.1f},{y1 - y0:.1f},0,13,{dc[k]:.2f}")
            per_frame.append((len(unl) + unseen, fr, len(unl), unseen))
        rows.append({"sequence": name, "frames": n_frames, "labelled_people": n_lab,
                     "seen_by_teacher": round(n_seen / max(n_lab, 1), 3), "labels_teacher_misses": n_unseen,
                     "unlabelled_detections": n_unl, "unlabelled_per_frame": round(n_unl / max(n_frames, 1), 3)})
        if unl_lines:
            (out / f"unlabelled_{name}.txt").write_text("\n".join(unl_lines) + "\n")
        per_frame.sort(reverse=True)
        worst.extend((s, name, d, fr, a, b) for s, fr, a, b in per_frame[:3])
    with open(out / "teacher_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    worst.sort(key=lambda r: -r[0])
    tiles = []
    for s, name, d, fr, a, b in worst[:24]:
        img = cv2.imread(str(d / "img1" / f"{fr:06d}.jpg"))
        det = {int(k): v for k, v in json.loads((out / f"{name}.json").read_text())["frames"].items()}
        for x0, y0, x1, y1, c, cf, vis in read_gt(d).get(fr, []):
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 140, 255) if c == 1 else (0, 60, 120), 2)
        for x0, y0, x1, y1, cf in det.get(fr, []):
            if cf >= conf:
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 2)
        cv2.putText(img, f"{name} f{fr} unlabelled {a} unseen {b}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        tiles.append(cv2.resize(img, (640, 360)))
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / "montage_worst.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    tot = {k: sum(r[k] for r in rows) for k in ("frames", "labelled_people", "labels_teacher_misses", "unlabelled_detections")}
    print(f"{len(rows)} sequences, {tot['frames']} frames, {tot['labelled_people']} labelled people, "
          f"{tot['labels_teacher_misses']} of them unseen by the teacher, {tot['unlabelled_detections']} confident "
          f"detections without a label -> {out / 'teacher_report.csv'}")


if __name__ == "__main__":
    cmd, out = sys.argv[1], Path(sys.argv[2])
    a = sys.argv
    if cmd == "run":
        run(out, int(a[a.index("--every") + 1]) if "--every" in a else 2, int(a[a.index("--imgsz") + 1]) if "--imgsz" in a else 1280,
            a[a.index("--model") + 1] if "--model" in a else None)
    elif cmd == "report":
        report(out, float(a[a.index("--conf") + 1]) if "--conf" in a else 0.6)
    else:
        sys.exit(__doc__)
