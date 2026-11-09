"""The room part of the training and test data: work/dataset_room_train/ with links to the room recordings
(work/dataset_real1/, labelled by tools/real_label.py) and their two/three-person composites
(work/dataset_real1_comp/, tools/real_composite.py).

Held out for testing (the paper's "room" column): scenarios 03 walk 5 m, 06 sit, 09 occlusion and the validation
composites. Their links get the prefix MOT17-R1- so that the key "R1" selects them in tools/finetune.py --val and
tools/eval_ap.py --select (the readers take every folder whose name ends in _f<number>). The other scenarios and the
training composites are training data.
Usage: room_split.py [OUT work/dataset_room_train]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / "work"
TEST = ("03_walk_5m", "06_sit", "09_occlusion")


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    links = {"composites_f0000": W / "dataset_real1_comp/composites_train",
             "MOT17-R1-composites_f0000": W / "dataset_real1_comp/composites_val"}
    for seg in sorted((W / "dataset_real1").iterdir()):
        if not (seg / "labels").is_dir():
            continue
        if seg.name in TEST:
            links[f"MOT17-R1-{seg.name}_f0000"] = seg
        else:
            links[f"real1_{seg.name}_f0000"] = seg
    for name, target in links.items():
        link = out / name
        if not target.exists():
            print(f"missing {target}, skipped")
            continue
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(target.resolve())
    print(f"{len(links)} links in {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else W / "dataset_room_train")
