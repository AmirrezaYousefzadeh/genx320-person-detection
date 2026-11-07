"""Record the whole MOT17 training set as event data: every sequence in back-to-back segments, then fetch and label.

  record OUT [--seconds S] [--flip] [--pan F] [--gain G] [--warp] [--zoom TARGET [--zoom-rank R]] [--src ROOT]   plays each sequence from frame 0 in segments of S seconds (default 10) until its end
                             and records each to the card (tools/record_dataset.py record); fast
  fetch OUT                  copies every recorded segment that has no events.raw yet from the card over SWD
                             (slow: ~100 kB/s, hours for the whole set; run in the background) and labels it
Usage: dataset_run.py record|fetch OUT   (a summary goes to OUT/run.log)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_mot import seq_info  # noqa: E402
import zoom_plan  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "work/venv/bin/python"
TRAIN = ROOT / "work/datasets/mot17/unzipped/MOT17/train"
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN",
        "MOT17-13-FRCNN"]


def sequences(src=TRAIN):
    """the sequence folders a pass plays: MOT17 train's seven, or every folder with a seqinfo.ini in src"""
    src = Path(src)
    if src == TRAIN:
        return [TRAIN / n for n in SEQS]
    return sorted(d for d in src.iterdir() if (d / "seqinfo.ini").exists())


def log(out, msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(out / "run.log", "a") as f:
        f.write(line + "\n")


def record(out, seconds, extra=(), zoom=0.0, zoom_rank=0, src=TRAIN):
    """src: the folder of the sequences (TRAIN, or work/datasets/mot17_60/train: the same at 60 fps)"""
    out.mkdir(parents=True, exist_ok=True)
    for seq in sequences(src):
        name = seq.name
        fps, _, _, n = seq_info(seq)
        starts = list(range(0, int(n - seconds * fps) + 1, int(seconds * fps))) or [0]
        for start in starts:
            d = out / f"{name}_f{start:04d}"
            if (d / "segment.json").exists():
                continue
            seg_extra = list(extra)
            if zoom:  # a fixed zoomed crop for this segment (tools/zoom_plan.py), warped onto the sensor's view
                (cx0, cy0, side), (per_frame, rel) = zoom_plan.plan(seq, start, int(seconds * fps), zoom, zoom_rank)
                if "--flip" in extra:  # monitor.py mirrors the frame before cropping: the same people, mirrored
                    cx0 = zoom_plan.seq_meta(seq)[0] - cx0 - side
                seg_extra += ["--warp", "--crop", f"{cx0:.1f},{cy0:.1f},{side:.1f}"]
            r = subprocess.run([PY, "-I", ROOT / "tools/record_dataset.py", "record", str(out), str(seq), str(start),
                                "--seconds", str(seconds), *seg_extra], capture_output=True, text=True)
            log(out, (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[-1])
            time.sleep(1)


def fetch(out):
    for d in sorted(p for p in out.iterdir() if (p / "segment.json").exists()):
        if (d / "events.raw").exists() and (d / "events.json").exists():
            continue
        for step in ("fetch", "label"):
            if step == "fetch" and (d / "events.raw").exists():
                continue
            r = subprocess.run([PY, "-I", ROOT / "tools/record_dataset.py", step, str(d)], capture_output=True, text=True)
            log(out, f"{step} {d.name}: " + (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[-1][:300])
    total = sum(json.loads((d / "events.json").read_text())["boxes"] for d in out.iterdir() if (d / "events.json").exists())
    pics = sum(json.loads((d / "events.json").read_text())["pictures"] for d in out.iterdir() if (d / "events.json").exists())
    log(out, f"DONE: {pics} pictures, {total} boxes")


if __name__ == "__main__":
    cmd, out = sys.argv[1], Path(sys.argv[2])
    secs = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 10.0
    extra = [a for a in sys.argv[3:] if a in ("--flip", "--warp")] + \
        (["--pan", sys.argv[sys.argv.index("--pan") + 1]] if "--pan" in sys.argv else []) + \
        (["--gain", sys.argv[sys.argv.index("--gain") + 1]] if "--gain" in sys.argv else [])
    zoom = float(sys.argv[sys.argv.index("--zoom") + 1]) if "--zoom" in sys.argv else 0.0
    rank = int(sys.argv[sys.argv.index("--zoom-rank") + 1]) if "--zoom-rank" in sys.argv else 0
    src = ROOT / sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else TRAIN
    record(out, secs, extra, zoom, rank, src) if cmd == "record" else fetch(out)
