"""Record labelled event data from MOT17 sequences shown on the monitor, with the board's microSD recorder.

  record OUT SEQ_DIR START_FRAME [START_FRAME ...] [--seconds S] [--flip] [--pan F]
        for each start frame: play the sequence from there (filling the calibrated stage), record the raw
        events to the card (tools/sdrec.py start/stop), save frames.log (wall-clock time of every shown frame)
        and segment.json (recording number, PC times, sequence) in OUT/<sequence>_f<start>/
  fetch OUT_SEGMENT_DIR
        copy the segment's recording from the card over SWD into events.raw (slow, ~100 kB/s; or copy the
        card's /evt/recNNNN.raw there by hand)
  label OUT_SEGMENT_DIR [--window MS] [--min-events N]
        cut events.raw into pictures of WINDOW ms (grey nothing, white ON, black OFF, like the detector's
        input), give each picture the MOT17 person boxes of the frame shown in the middle of its window,
        mapped through the monitor homography: pictures/pic_NNNN.png, labels/pic_NNNN.txt (YOLO "0 cx cy w h",
        normalised), preview_NNNN.png with the boxes drawn. A box with fewer than N events inside (default 20:
        a person who does not move gives no events and cannot be seen) is written to labels_all/ only
Time base: a white square blinks at the stage's top-left corner for 100 ms once a second (monitor.py --marker)
and its switch-on times are logged; the label step finds the bursts of ON events in that square in the recording
and fits sensor time to PC time (reports the residual), then masks the square in the pictures. Without a usable
marker it falls back to the START command time (pc_start, +-0.3 s because of the debug link).
MOT17 boxes used: classes 1 (pedestrian), 2 (person on vehicle), 7 (static person), visibility >= 0.25.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_mot import PERSON_CLASSES, load_gt, seq_info  # noqa: E402
from evt2 import PIC, check, decode, picture, read_raw, to_picture_coords  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "work/venv/bin/python"


def sdrec(*args):
    out = subprocess.run([PY, "-I", ROOT / "tools/sdrec.py", *args], capture_output=True, text=True, check=True)
    return out.stdout


def record_segment(seq_dir, start_frame, out, seconds, extra=()):
    out.mkdir(parents=True, exist_ok=True)
    (out / "calib.json").write_text((ROOT / "work/calib/calib.json").read_text())  # labels need this one later
    fps, _, _, _ = seq_info(seq_dir)
    subprocess.run([ROOT / "tools/show.sh", "frames", str(Path(seq_dir) / "img1"), str(fps), str(out / "frames.log"),
                    "--cover", "--start", str(start_frame), "--marker", *extra], check=True)
    time.sleep(1.5)
    pc_start = time.time()
    sdrec("start")
    time.sleep(seconds)
    pc_stop = time.time()
    status = sdrec("stop")
    subprocess.run([ROOT / "tools/show.sh", "off"])
    file_no = int(status.split("recording ")[1].split(":")[0])
    (out / "segment.json").write_text(json.dumps({"seq_dir": str(seq_dir), "start_frame": start_frame,
                                                  "file_no": file_no, "pc_start": pc_start, "pc_stop": pc_stop,
                                                  "display": list(extra),  # --flip / --pan F / --gain G
                                                  "sdrec_status": status.strip()}, indent=1))
    print(f"{out.name}: recording {file_no:04d}, {status.strip().splitlines()[-1]}")


def fetch_segment(out):
    seg = json.loads((out / "segment.json").read_text())
    print(sdrec("read", str(seg["file_no"]), str(out / "events.raw")))


def load_gt_ignore(seq_dir):
    """Regions to ignore, per 0-based frame (2026-11-07): the source's reflections (12), crowds (13) and distractors
    (8), people marked not to be evaluated (conf 0) or hardly visible (vis < 0.25), and the RGB teacher's confident
    detections that the source never labelled (gt/gt_teacher.txt, tools/teacher_to_gt.py). On an event camera a
    reflection or a half-hidden person moves like a person, so a detection there is neither right nor wrong."""
    import csv
    ign = {}
    for name in ("gt.txt", "gt_teacher.txt"):
        path = Path(seq_dir) / "gt" / name
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.reader(f):
                if len(row) < 9:
                    continue
                frame, _, x, y, bw, bh, conf, cls, vis = (float(v) for v in row[:9])
                if int(cls) in (8, 12, 13) or (int(cls) in PERSON_CLASSES and (conf <= 0 or vis < 0.25)):
                    ign.setdefault(int(frame) - 1, []).append((x, y, x + bw, y + bh))
    return ign


def local_seq_dir(path):
    """The source sequence of a segment on this machine: segment.json stores the path on the recording PC
    (/.../work/datasets/<...>); elsewhere the same sequence is looked up under this repository's work/datasets/."""
    p = Path(path)
    if p.exists():
        return p
    parts = p.parts
    if "datasets" in parts:
        local = ROOT / "work" / Path(*parts[parts.index("datasets"):])
        if local.exists():
            return local
    sys.exit(f"source sequence {path} not found: put MOT17 / PersonPath22 (and their 60 fps renderings, tools/make60.py "
             f"and tools/pp22_prep.py) under {ROOT / 'work/datasets'} (see the README)")


def label_segment(out, window_ms=50, n_previews=4, min_events=20):
    seg = json.loads((out / "segment.json").read_text())
    seq_dir = local_seq_dir(seg["seq_dir"])
    with open(out / "events.raw", "rb") as f:
        if f.read(8).startswith(b"% histo"):
            return label_histo_segment(out, seg, seq_dir, n_previews, min_events)
    header, words = read_raw(out / "events.raw")
    ev = decode(words)
    stats = check(ev)
    # the calibration valid when the segment was recorded (record_segment copies it); the board can move later.
    # 2026-11-07: never fall back to the current calibration (a relabel of the old passes with it put every box in
    # the wrong place); the old passes carry a recovered copy now
    if not (out / "calib.json").exists():
        sys.exit(f"{out}: no calib.json in the segment; refusing to label with the current calibration")
    calib = json.loads((out / "calib.json").read_text())
    H = np.array(calib["H_mon_to_pic"], np.float64)
    lines = [l for l in (out / "frames.log").read_text().split("\n") if l]
    rx, ry, rw, rh = (int(v) for v in [l for l in lines if l.startswith("rect")][-1].split()[1:5])
    flipped = any(l == "flip 1" for l in lines)
    warp = next(([float(v) for v in l.split()[1:4]] for l in lines if l.startswith("warp ")), None)  # cx0 cy0 side
    wl = next((l.split() for l in lines if l.startswith("warp ")), [])
    target = [float(v) for v in wl[wl.index("target") + 1:wl.index("target") + 4]] if "target" in wl else [0, 0, 320]
    shown = np.array([[float(a), int(b)] for a, b in (l.split() for l in lines if l[0].isdigit())])
    marker_rect = next(([int(v) for v in l.split()[1:5]] for l in lines if l.startswith("marker_rect")), None)
    marker_on = np.array([float(l.split()[2]) for l in lines if l.startswith("marker 1")])
    _, img_w, img_h, _ = seq_info(seq_dir)
    sx, sy = rw / img_w, rh / img_h
    gt = load_gt(seq_dir)
    gt_ign = load_gt_ignore(seq_dir)
    (out / "pictures").mkdir(exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    (out / "labels_all").mkdir(exist_ok=True)

    t = ev["t"] - ev["t"][0]  # us from the first event = the START command
    # sensor time -> PC time: from the marker if it is there, else from the START command
    pc_of_t = lambda us: seg["pc_start"] + us / 1e6  # noqa: E731
    mask = None
    sync = {"method": "start command"}
    if marker_rect is not None and len(marker_on) >= 2:
        mon = np.array([[[marker_rect[0], marker_rect[1]]], [[marker_rect[0] + marker_rect[2], marker_rect[1]]],
                        [[marker_rect[0] + marker_rect[2], marker_rect[1] + marker_rect[3]]],
                        [[marker_rect[0], marker_rect[1] + marker_rect[3]]]], np.float64)
        p = cv2.perspectiveTransform(mon, H).reshape(-1, 2)
        mx0, my0 = int(max(p[:, 0].min() - 2, 0)), int(max(p[:, 1].min() - 2, 0))
        mx1, my1 = int(min(p[:, 0].max() + 3, PIC)), int(min(p[:, 1].max() + 3, PIC))
        mask = (mx0, my0, mx1, my1)
        X, Y = to_picture_coords(ev)
        inside = (X >= mx0) & (X < mx1) & (Y >= my0) & (Y < my1) & (ev["p"] == 1)
        bins = np.bincount((t[inside] // 1000).astype(np.int64), minlength=int(t[-1] // 1000) + 1)
        thr = max(5, bins.max() / 4)
        onsets_ms = [i for i in np.where(bins > thr)[0] if i == 0 or bins[i - 1] <= thr]
        # the marker switches on once a second: match the recorded bursts to the logged times by their spacing
        onsets = np.array(onsets_ms, float) / 1000  # s, sensor
        if len(onsets) >= 2:
            best = None
            for i in range(len(marker_on)):
                for j in range(len(onsets)):
                    offset = marker_on[i] - onsets[j]
                    # the marker is periodic: only offsets consistent with the START command (+-0.5 s) are real
                    if abs(offset - seg["pc_start"]) > 0.5:
                        continue
                    d = np.abs((onsets[:, None] + offset) - marker_on[None, :]).min(1)
                    n_match = int((d < 0.03).sum())
                    if best is None or n_match > best[0]:
                        best = (n_match, offset, d)
            if best is not None:
                n_match, offset, d = best
            if best is not None and n_match >= 2:
                matched = d < 0.03
                A = np.c_[onsets[matched], np.ones(matched.sum())]
                b = np.array([marker_on[np.abs(marker_on - (o + offset)).argmin()] for o in onsets[matched]])
                slope, icpt = np.linalg.lstsq(A, b, rcond=None)[0]
                resid = b - (A @ [slope, icpt])
                rms_ms = float(np.sqrt((resid ** 2).mean()) * 1e3)
                fit = {"method": "blink marker", "matched": n_match, "offset_s": float(icpt - seg["pc_start"]),
                       "drift_ppm": float((slope - 1) * 1e6), "residual_rms_ms": rms_ms}
                # a fit from two onsets, or one bent by events of the video near the marker, is worse than none:
                # keep it only when the clocks agree within 2 % and the onsets fit within 50 ms
                if abs(slope - 1) < 0.02 and rms_ms < 50 and n_match >= 3:
                    pc_of_t = lambda us, s_=slope, c_=icpt: c_ + s_ * (us / 1e6)  # noqa: E731
                    sync = fit
                else:
                    sync = {"method": "start command (marker fit rejected)", "rejected": fit}
    n_pics = int(t[-1] // (window_ms * 1000)) + 1
    previews = set(np.linspace(0, n_pics - 1, n_previews).astype(int).tolist())

    def boxes_at(t_mid, src=None):
        """the video's boxes (src: the people, or another per-frame dict) on the monitor at PC time t_mid, in
        picture coordinates (0..1)"""
        i = np.searchsorted(shown[:, 0], t_mid) - 1
        boxes = []
        if 0 <= i < len(shown) and t_mid - shown[i, 0] < 0.5:
            for x0, y0, x1, y1 in (gt if src is None else src).get(int(shown[i, 1]), []):
                if flipped:
                    x0, x1 = img_w - x1, img_w - x0
                if warp:  # monitor.py --warp: the square crop fills the picture, boxes scale with it
                    cx0, cy0, side = warp
                    tx0, ty0, tL = target  # the picture square the crop was shown in (picture pixels)
                    sc = tL / side / PIC
                    # a person cut by the crop's edge ends at the square's edge, not at the picture's (2026-11-05)
                    lo, hi = np.array([tx0, ty0] * 2) / PIC, np.array([tx0 + tL, ty0 + tL] * 2) / PIC
                    b = np.clip([tx0 / PIC + (x0 - cx0) * sc, ty0 / PIC + (y0 - cy0) * sc,
                                 tx0 / PIC + (x1 - cx0) * sc, ty0 / PIC + (y1 - cy0) * sc], lo, hi)
                else:
                    mon = np.array([[[rx + x0 * sx, ry + y0 * sy]], [[rx + x1 * sx, ry + y0 * sy]],
                                    [[rx + x1 * sx, ry + y1 * sy]], [[rx + x0 * sx, ry + y1 * sy]]], np.float64)
                    p = cv2.perspectiveTransform(mon, H).reshape(-1, 2) / (PIC - 1)
                    b = np.clip([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], 0, 1)
                if b[2] - b[0] > 0.02 and b[3] - b[1] > 0.02:
                    boxes.append(b)
        return boxes

    def pic_at(k):
        sel = (t >= k * window_ms * 1000) & (t < (k + 1) * window_ms * 1000)
        pic = picture(ev, sel)
        if mask:
            pic[mask[1]:mask[3], mask[0]:mask[2]] = 128
        return pic

    # The marker blinks once a second, so its fit is only known modulo 1 s; the START command (+-0.5 s above) does not
    # settle it when the first event comes ~0.5 s after the command (2026-11-06: 133 segments, mostly of that day, had
    # their labels one second off). Among the fit and its two one-second aliases keep the one whose boxes hold the most
    # events (sampled pictures).
    if sync.get("method") == "blink marker":
        samples = [(k, pic_at(k) != 128) for k in range(0, n_pics, 4)]
        dens = []
        for a in (-1, 0, 1):
            f = (lambda us, s_=slope, c_=icpt + a: c_ + s_ * (us / 1e6))
            tot = area = 0
            for k, act in samples:
                for x0, y0, x1, y1 in boxes_at(f((k + 0.5) * window_ms * 1000)):
                    X0, Y0, X1, Y1 = int(x0 * 319), int(y0 * 319), int(x1 * 319) + 1, int(y1 * 319) + 1
                    tot += int(act[Y0:Y1, X0:X1].sum())
                    area += (X1 - X0) * (Y1 - Y0)
            dens.append(tot / area if area else 0.0)
        # correct fits peak at 0 by 3-8 %, aliased ones (a negative offset: the first event before the START command,
        # impossible) at +1 by 8-15 % (tested on 9 segments, 2026-11-06)
        a = (-1, 0, 1)[int(np.argmax(dens))]
        if not (a != 0 and dens[a + 1] > 1.03 * dens[1]):
            a = 1 if sync["offset_s"] < 0 and dens[2] >= dens[1] else 0
        if a != 0:
            pc_of_t = lambda us, s_=slope, c_=icpt + a: c_ + s_ * (us / 1e6)  # noqa: E731
            sync = {**sync, "offset_s": sync["offset_s"] + a, "alias_fixed_s": a}
        sync["alias_check_density"] = [round(v, 4) for v in dens]
    n_boxes = n_boxes_all = n_ignore = 0
    for k in range(n_pics):
        pic = pic_at(k)
        cv2.imwrite(str(out / "pictures" / f"pic_{k:04d}.png"), pic)
        t_mid = pc_of_t((k + 0.5) * window_ms * 1000)
        boxes, ignore = boxes_at(t_mid), boxes_at(t_mid, gt_ign)
        active = (pic != 128)
        kept = []
        with open(out / "labels_all" / f"pic_{k:04d}.txt", "w") as fa, open(out / "labels" / f"pic_{k:04d}.txt", "w") as f:
            for x0, y0, x1, y1 in boxes:
                n_ev = int(active[int(y0 * 319):int(y1 * 319) + 1, int(x0 * 319):int(x1 * 319) + 1].sum())
                line = f"0 {(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {x1 - x0:.4f} {y1 - y0:.4f}"
                fa.write(f"{line} {n_ev}\n")
                if n_ev >= min_events:
                    f.write(line + "\n")
                    kept.append((x0, y0, x1, y1))
            for x0, y0, x1, y1 in ignore:  # regions to ignore: kept whatever their events
                n_ev = int(active[int(y0 * 319):int(y1 * 319) + 1, int(x0 * 319):int(x1 * 319) + 1].sum())
                line = f"1 {(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {x1 - x0:.4f} {y1 - y0:.4f}"
                fa.write(f"{line} {n_ev}\n")
                f.write(line + "\n")
        n_ignore += len(ignore)
        n_boxes_all += len(boxes)
        boxes = kept
        n_boxes += len(boxes)
        if k in previews:
            vis = cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR)
            for x0, y0, x1, y1 in boxes:
                cv2.rectangle(vis, (int(x0 * 319), int(y0 * 319)), (int(x1 * 319), int(y1 * 319)), (0, 140, 255), 1)
            cv2.imwrite(str(out / f"preview_{k:04d}.png"), vis)
    (out / "events.json").write_text(json.dumps({**stats, "pictures": n_pics, "boxes": n_boxes, "boxes_all": n_boxes_all,
                                                 "ignore_regions": n_ignore, "min_events": min_events, "sync": sync,
                                                 "marker_mask_pic": mask, "window_ms": window_ms, "header": header},
                                                indent=1))
    print(f"{out.name}: {n_pics} pictures of {window_ms} ms, {n_boxes} boxes ({n_boxes_all} before the event filter), "
          f"sync {sync}, {stats}")


def label_histo_segment(out, seg, seq_dir, n_previews, min_events):
    """Histogram-mode recording (make HISTO=1): one picture per sensor frame. Frame k is given the board tick of
    its arrival; the blink marker (ON counts in the marker square) ties the ticks to PC time, which also absorbs
    the sensor's fixed drain delay. Pictures follow the firmware's rule (tools/histo_raw.py)."""
    from histo_raw import on_counts, picture as histo_picture, read_frames
    header, ticks, frames = read_frames(out / "events.raw")
    if len(ticks) < 2:
        print(f"{out.name}: no frames")
        return
    # the calibration valid when the segment was recorded (record_segment copies it); the board can move later.
    # 2026-11-07: never fall back to the current calibration (a relabel of the old passes with it put every box in
    # the wrong place); the old passes carry a recovered copy now
    if not (out / "calib.json").exists():
        sys.exit(f"{out}: no calib.json in the segment; refusing to label with the current calibration")
    calib = json.loads((out / "calib.json").read_text())
    H = np.array(calib["H_mon_to_pic"], np.float64)
    lines = [l for l in (out / "frames.log").read_text().split("\n") if l]
    rx, ry, rw, rh = (int(v) for v in [l for l in lines if l.startswith("rect")][-1].split()[1:5])
    flipped = any(l == "flip 1" for l in lines)
    shown = np.array([[float(a), int(b)] for a, b in (l.split() for l in lines if l[0].isdigit())])
    marker_rect = next(([int(v) for v in l.split()[1:5]] for l in lines if l.startswith("marker_rect")), None)
    marker_on = np.array([float(l.split()[2]) for l in lines if l.startswith("marker 1")])
    _, img_w, img_h, _ = seq_info(seq_dir)
    sx, sy = rw / img_w, rh / img_h
    gt = load_gt(seq_dir)
    gt_ign = load_gt_ignore(seq_dir)
    for d in ("pictures", "labels", "labels_all"):
        (out / d).mkdir(exist_ok=True)
    t_s = (ticks - ticks[0]) / 1000.0  # s since the first frame
    pc_of_t = lambda t: seg["pc_start"] + t  # noqa: E731
    sync = {"method": "start command"}
    mask = None
    if marker_rect is not None and len(marker_on) >= 2:
        mon = np.array([[[marker_rect[0], marker_rect[1]]], [[marker_rect[0] + marker_rect[2], marker_rect[1]]],
                        [[marker_rect[0] + marker_rect[2], marker_rect[1] + marker_rect[3]]],
                        [[marker_rect[0], marker_rect[1] + marker_rect[3]]]], np.float64)
        p = cv2.perspectiveTransform(mon, H).reshape(-1, 2)
        mx0, my0 = int(max(p[:, 0].min() - 2, 0)), int(max(p[:, 1].min() - 2, 0))
        mx1, my1 = int(min(p[:, 0].max() + 3, PIC)), int(min(p[:, 1].max() + 3, PIC))
        mask = (mx0, my0, mx1, my1)
        bursts = np.array([int(on_counts(f)[my0:my1, mx0:mx1].sum()) for f in frames], float)
        thr = max(5, bursts.max() / 4)
        onsets = np.array([t_s[i] for i in range(len(bursts)) if bursts[i] > thr and (i == 0 or bursts[i - 1] <= thr)])
        best = None
        for i in range(len(marker_on)):
            for j in range(len(onsets)):
                offset = marker_on[i] - onsets[j]
                if abs(offset - seg["pc_start"]) > 0.6:
                    continue
                d = np.abs((onsets[:, None] + offset) - marker_on[None, :]).min(1)
                n_match = int((d < 0.08).sum())  # a frame is 50 ms of accumulation: coarser than events
                if best is None or n_match > best[0]:
                    best = (n_match, offset, d)
        if best is not None and best[0] >= 2:
            n_match, offset, d = best
            matched = d < 0.08
            A = np.c_[onsets[matched], np.ones(matched.sum())]
            b = np.array([marker_on[np.abs(marker_on - (o + offset)).argmin()] for o in onsets[matched]])
            slope, icpt = np.linalg.lstsq(A, b, rcond=None)[0]
            resid = b - (A @ [slope, icpt])
            pc_of_t = lambda t, s_=slope, c_=icpt: c_ + s_ * t  # noqa: E731
            sync = {"method": "blink marker (frames)", "matched": n_match, "offset_s": float(icpt - seg["pc_start"]),
                    "drift_ppm": float((slope - 1) * 1e6), "residual_rms_ms": float(np.sqrt((resid ** 2).mean()) * 1e3)}
    n_boxes = n_boxes_all = 0
    previews = set(np.linspace(0, len(frames) - 1, n_previews).astype(int).tolist())
    for k, f in enumerate(frames):
        pic = histo_picture(f)
        if mask:
            pic[mask[1]:mask[3], mask[0]:mask[2]] = 128
        cv2.imwrite(str(out / "pictures" / f"pic_{k:04d}.png"), pic)
        t_mid = pc_of_t(t_s[k]) - 0.025  # the marker onset is seen at the frame, the picture spans the 50 ms before
        i = np.searchsorted(shown[:, 0], t_mid) - 1

        def mapped(src):
            boxes = []
            if 0 <= i < len(shown) and t_mid - shown[i, 0] < 0.5:
                for x0, y0, x1, y1 in src.get(int(shown[i, 1]), []):
                    if flipped:
                        x0, x1 = img_w - x1, img_w - x0
                    mon = np.array([[[rx + x0 * sx, ry + y0 * sy]], [[rx + x1 * sx, ry + y0 * sy]],
                                    [[rx + x1 * sx, ry + y1 * sy]], [[rx + x0 * sx, ry + y1 * sy]]], np.float64)
                    p = cv2.perspectiveTransform(mon, H).reshape(-1, 2) / (PIC - 1)
                    b = np.clip([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], 0, 1)
                    if b[2] - b[0] > 0.02 and b[3] - b[1] > 0.02:
                        boxes.append(b)
            return boxes

        boxes, ignore = mapped(gt), mapped(gt_ign)
        active = (pic != 128)
        kept = []
        with open(out / "labels_all" / f"pic_{k:04d}.txt", "w") as fa, open(out / "labels" / f"pic_{k:04d}.txt", "w") as fl:
            for x0, y0, x1, y1 in boxes:
                n_ev = int(active[int(y0 * 319):int(y1 * 319) + 1, int(x0 * 319):int(x1 * 319) + 1].sum())
                line = f"0 {(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {x1 - x0:.4f} {y1 - y0:.4f}"
                fa.write(f"{line} {n_ev}\n")
                if n_ev >= min_events:
                    fl.write(line + "\n")
                    kept.append((x0, y0, x1, y1))
            for x0, y0, x1, y1 in ignore:
                n_ev = int(active[int(y0 * 319):int(y1 * 319) + 1, int(x0 * 319):int(x1 * 319) + 1].sum())
                line = f"1 {(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {x1 - x0:.4f} {y1 - y0:.4f}"
                fa.write(f"{line} {n_ev}\n")
                fl.write(line + "\n")
        n_boxes_all += len(boxes)
        n_boxes += len(kept)
        if k in previews:
            vis = cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR)
            for x0, y0, x1, y1 in kept:
                cv2.rectangle(vis, (int(x0 * 319), int(y0 * 319)), (int(x1 * 319), int(y1 * 319)), (0, 140, 255), 1)
            cv2.imwrite(str(out / f"preview_{k:04d}.png"), vis)
    dt = np.diff(ticks)
    stats = {"frames": len(frames), "duration_s": float(t_s[-1]), "frame_interval_ms": float(dt.mean())}
    (out / "events.json").write_text(json.dumps({**stats, "pictures": len(frames), "boxes": n_boxes, "boxes_all": n_boxes_all,
                                                 "min_events": min_events, "sync": sync, "marker_mask_pic": mask,
                                                 "window_ms": 50, "header": header, "mode": "histogram"}, indent=1))
    print(f"{out.name}: {len(frames)} histogram frames, {n_boxes} boxes ({n_boxes_all} before the event filter), "
          f"sync {sync}, {stats}")


def main(argv):
    cmd = argv[0]
    opts = {argv[i]: argv[i + 1] for i in range(1, len(argv) - 1) if argv[i].startswith("--")}
    flags = {"--flip", "--warp"}
    args = [a for i, a in enumerate(argv[1:], 1)
            if not a.startswith("--") and not (argv[i - 1].startswith("--") and argv[i - 1] not in flags)]
    if cmd == "record":
        out_root, seq_dir = Path(args[0]), Path(args[1])
        extra = (["--flip"] if "--flip" in argv else []) + (["--pan", opts["--pan"]] if "--pan" in opts else []) + \
            (["--gain", opts["--gain"]] if "--gain" in opts else []) + \
            (["--warp"] if "--warp" in argv else []) + \
            (["--crop", opts["--crop"]] if "--crop" in opts else [])  # --gain: dimmer frames; --warp: frames fill the sensor's view
        for start in (int(v) for v in args[2:]):
            record_segment(seq_dir, start, out_root / f"{seq_dir.name}_f{start:04d}", float(opts.get("--seconds", 5)),
                           extra)
    elif cmd == "fetch":
        fetch_segment(Path(args[0]))
    elif cmd == "label":
        label_segment(Path(args[0]), int(opts.get("--window", 50)), min_events=int(opts.get("--min-events", 20)))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
