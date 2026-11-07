"""Data lists for the fine-tuning runs of 2026-11-06 (comma-joined roots for tools/finetune.py --data, and --val keys).
  real   every recorded monitor pass with labels, the room recordings and their composites, the campaign passes
  val    MOT17-09 and -11, the held-out room scenarios (R1), the PersonPath22 test-split videos
  synth  N random simulated segments (tools/evsim.py) linked into work/synth_mix/ (seeded)
Usage: train_lists.py real|val|synth N
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / "work"


def real():
    # sam3* (2026-11-05 17:16-17:40) had been labelled with a calibration from before the tripod head sank ~55 rows
    # (events inside boxes / outside 1.36, other passes 2.3-2.9); relabelled with the 17:52 one on 2026-11-06 (2.37 after relabelling)
    names = ["v1", "v2", "v3", "zoom", "a1", "sam1", "sam_flip", "sam_dim50", "sam_dim25", "sam3", "sam3_flip",
             "sam3_pan_l", "sam3_pan_r", "room_train"]
    roots = [W / f"dataset_{n}" for n in names] + sorted(W.glob("dataset_camp_dark_*")) + sorted(W.glob("dataset_camp60_*")) \
        + sorted(W.glob("dataset_camp60pp_*"))
    # only roots with at least one labelled segment
    return [str(r) for r in roots if r.is_dir() and any((s / "labels").is_dir() for s in r.glob("*_f*"))]


def val():
    sp = json.loads((W / "datasets/personpath22/splits.json").read_text())
    return ["09", "11", "R1"] + [u.split(".")[0] + "_" for u in sp["test"]]


def synth(n):
    segs = sorted(p for p in (W / "synth/v1").glob("*_f*") if (p / "labels").is_dir())
    random.Random(320).shuffle(segs)
    out = W / "synth_mix"
    if out.exists():
        for l in out.iterdir():
            l.unlink()
    out.mkdir(exist_ok=True)
    for s in segs[:n]:
        (out / s.name).symlink_to(s)
    return str(out)


if __name__ == "__main__":
    k = sys.argv[1]
    print(",".join(real()) if k == "real" else ",".join(val()) if k == "val" else synth(int(sys.argv[2])))
