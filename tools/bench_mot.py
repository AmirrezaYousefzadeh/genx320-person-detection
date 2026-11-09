"""Test bench: play a MOT17 sequence on the monitor, log the board's detections, score them against MOT17's boxes.

Needs: X on :1 (tools/xserver.sh), work/calib/calib.json (tools/calibrate.py), the board on /dev/ttyACM0 (GENX_SERIAL=).
  run   SEQ_DIR SECONDS OUT_DIR   shows SEQ_DIR/img1 at its frame rate (filling the stage, sides cropped),
        logs frame times and detections
  score OUT_DIR [--iou 0.5] [--latency MS] [--extra-latency MS] [--min-h 0.12] [--score S]   matches detections to the frame shown LATENCY ms before the
        network finished (default: network time + window/2) and prints precision, recall and mean IoU
Both: bench_mot.py both SEQ_DIR SECONDS OUT_DIR
MOT17 boxes used as truth: classes 1 (pedestrian), 2 (person on vehicle), 7 (static person), visibility >= 0.25.
"""
import configparser
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PIC = 320
WINDOW_MS = 50
PERSON_CLASSES = {1, 2, 7}


def seq_info(seq_dir):
    cfg = configparser.ConfigParser()
    cfg.read(Path(seq_dir) / "seqinfo.ini")
    s = cfg["Sequence"]
    return float(s["frameRate"]), int(s["imWidth"]), int(s["imHeight"]), int(s["seqLength"])


def run(seq_dir, seconds, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fps, w, h, n = seq_info(seq_dir)
    (out / "sequence.json").write_text(json.dumps({"seq_dir": str(seq_dir), "fps": fps, "w": w, "h": h, "n": n}))
    subprocess.run([ROOT / "tools/show.sh", "frames", str(Path(seq_dir) / "img1"), str(fps), str(out / "frames.log"),
                    "--cover"], check=True)
    time.sleep(2)
    subprocess.run(["sg", "dialout", "-c",
                    f"{ROOT}/work/venv/bin/python -I {ROOT}/tools/det_log.py {out / 'detections.csv'} {seconds}"],
                   check=True)
    t_end = time.time()
    subprocess.run([ROOT / "tools/show.sh", "off"])
    # The replay stalls when the display is not active (someone switched to the login screen, Ctrl+Alt+F1):
    # then the sensor saw a still picture and the run is worthless.
    shown = [l for l in (out / "frames.log").read_text().split("\n") if l and l[0].isdigit()]
    last = float(shown[-1].split()[0]) if shown else 0.0
    if t_end - last > 3.0:
        print(f"WARNING: the replay stopped {t_end - last:.0f} s before the end ({len(shown)} frames shown); "
              f"active terminal: {Path('/sys/class/tty/tty0/active').read_text().strip()} (ours is tty2)")
    with open(out / "detections.csv") as f:
        rates = [int(r["kevps"]) for r in csv.DictReader(f) if r.get("kevps")]
    if rates:
        print(f"event rate during the run: median {int(np.median(rates))} kev/s, max {max(rates)}")


def load_gt(seq_dir):
    gt = {}
    with open(Path(seq_dir) / "gt" / "gt.txt") as f:
        for row in csv.reader(f):
            frame, _, x, y, bw, bh, conf, cls, vis = (float(v) for v in row[:9])
            if int(cls) in PERSON_CLASSES and conf > 0 and vis >= 0.25:
                gt.setdefault(int(frame) - 1, []).append((x, y, x + bw, y + bh))
    return gt


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


MIN_PERSON_H = 0.12  # of the picture height, as in finetune.py


def score(out_dir, iou_thr=0.5, latency_ms=None, extra_ms=0.0, min_h=MIN_PERSON_H, min_score=0.0):
    out = Path(out_dir)
    seq = json.loads((out / "sequence.json").read_text())
    calib = json.loads((ROOT / "work/calib/calib.json").read_text())
    H = np.array(calib["H_mon_to_pic"], np.float64)
    lines = (out / "frames.log").read_text().split("\n")
    rx, ry, rw, rh = (int(v) for v in lines[0].split()[1:5])  # frame rectangle on the monitor
    shown = np.array([[float(a), int(b)] for a, b in (l.split() for l in lines[1:] if l and l[0].isdigit())])  # time, frame
    sx, sy = rw / seq["w"], rh / seq["h"]
    gt = load_gt(seq["seq_dir"])

    def gt_in_pic(frame):
        boxes = []
        for x0, y0, x1, y1 in gt.get(frame, []):
            mon = np.array([[[rx + x0 * sx, ry + y0 * sy]], [[rx + x1 * sx, ry + y0 * sy]],
                            [[rx + x1 * sx, ry + y1 * sy]], [[rx + x0 * sx, ry + y1 * sy]]], np.float64)
            pic = cv2.perspectiveTransform(mon, H).reshape(-1, 2) / (PIC - 1)
            b = (pic[:, 0].min(), pic[:, 1].min(), pic[:, 0].max(), pic[:, 1].max())
            if b[2] > 0 and b[3] > 0 and b[0] < 1 and b[1] < 1:  # at least partly in the picture
                boxes.append(tuple(np.clip(b, 0, 1)))
        return boxes

    tp = fp = fn = 0
    ious, rows = [], []
    with open(out / "detections.csv") as f:
        for r in csv.DictReader(f):
            lat = (latency_ms if latency_ms is not None else int(r["nn_ms"]) + WINDOW_MS / 2 + extra_ms) / 1000
            t_pic = float(r["pc_time"]) - lat
            i = np.searchsorted(shown[:, 0], t_pic) - 1
            if i < 0 or t_pic - shown[i, 0] > 1.0:
                continue  # no frame shown then (before the start)
            frame = int(shown[i, 1])
            all_truth = gt_in_pic(frame)
            truth = [g for g in all_truth if g[3] - g[1] >= min_h]  # people the sensor can resolve
            small = [g for g in all_truth if g[3] - g[1] < min_h]   # ignored: neither counted nor penalised
            dets = [tuple(b[:4]) for b in json.loads(r["boxes"]) if b[4] >= min_score]
            used = set()
            for d in dets:
                best, best_iou = -1, 0.0
                for j, g in enumerate(truth):
                    if j in used:
                        continue
                    v = iou(d, g)
                    if v > best_iou:
                        best, best_iou = j, v
                if best >= 0 and best_iou >= iou_thr:
                    tp += 1
                    used.add(best)
                    ious.append(best_iou)
                elif not any(iou(d, g) >= 0.3 for g in small):
                    fp += 1
            fn += len(truth) - len(used)
            rows.append((r["seq"], frame, len(truth), len(dets), len(used)))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    print(f"{len(rows)} detector runs matched to frames; truth boxes {tp + fn}, detections {tp + fp}")
    print(f"precision {prec:.2f}  recall {rec:.2f}  mean IoU of matches {np.mean(ious) if ious else float('nan'):.2f}"
          f"  (IoU >= {iou_thr})")
    with open(out / "per_run.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["seq", "frame", "truth", "detections", "matched"])
        wr.writerows(rows)
    return prec, rec


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("run", "both"):
        run(sys.argv[2], float(sys.argv[3]), sys.argv[4])
    if cmd in ("score", "both"):
        out = sys.argv[4] if cmd == "both" else sys.argv[2]
        args = sys.argv[5:] if cmd == "both" else sys.argv[3:]
        kw = {}
        if "--iou" in args:
            kw["iou_thr"] = float(args[args.index("--iou") + 1])
        if "--latency" in args:
            kw["latency_ms"] = float(args[args.index("--latency") + 1])
        if "--min-h" in args:
            kw["min_h"] = float(args[args.index("--min-h") + 1])
        if "--extra-latency" in args:  # histogram mode: picture age at network start + the sensor's drain time
            kw["extra_ms"] = float(args[args.index("--extra-latency") + 1])
        if "--score" in args:  # keep only detections with at least this score (a run with make THRESHOLD=0.3
            kw["min_score"] = float(args[args.index("--score") + 1])  # logs everything, then score at several)
        score(out, **kw)
