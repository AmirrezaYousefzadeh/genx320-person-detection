"""Package N-People for the 4TU.ResearchData upload: one tar file per pass (raw events, logs, calibration, labels;
pictures left out, they are regenerated), the README, licence text, splits.json, the curation files and a manifest with
sizes and SHA-256 sums. Output: work/release/N-People-v<version>/.
Usage: package_dataset.py [--version 1.0] [--with-pictures]
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / "work"

SUBSETS = {
    "MOT17": ["dataset_v1", "dataset_v2", "dataset_v3", "dataset_a1", "dataset_h1", "dataset_sam1", "dataset_sam_flip",
              "dataset_sam_dim50", "dataset_sam_dim25", "dataset_sam3", "dataset_sam3_flip", "dataset_sam3_pan_l",
              "dataset_sam3_pan_r", "dataset_camp_dark_*", "dataset_camp60_dark_*", "dataset_camp60_lit_*"],
    "PersonPath22": ["dataset_camp60pp_*"],
    "Room": ["dataset_real1", "dataset_real1_comp"],
}
KEEP = ("events.raw", "frames.log", "calib.json", "calib_1712.json", "segment.json", "events.json", "labels",
        "labels_all", "labels.json", "detections.csv", "CONDITIONS.json", "composites.json", "run.log")


def passes(patterns):
    out = []
    for pat in patterns:
        out += sorted(p for p in W.glob(pat) if p.is_dir())
    return out


def is_test(seg):
    n = seg.name
    if "MOT17-09-" in n or "MOT17-11-" in n:
        return True
    if n.startswith("uid_vid_"):
        sp = json.loads((W / "datasets/personpath22/splits.json").read_text())
        return any(n.startswith(u.split(".")[0] + "_") for u in sp["test"])
    return n.startswith(("03_", "06_", "09_")) or "composites_val" in n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    a = sys.argv
    version = a[a.index("--version") + 1] if "--version" in a else "1.0"
    with_pictures = "--with-pictures" in a
    out = W / f"release/N-People-v{version}"
    out.mkdir(parents=True, exist_ok=True)
    manifest, splits = [], {"test": [], "train": []}
    for subset, pats in SUBSETS.items():
        (out / subset).mkdir(exist_ok=True)
        for p in passes(pats):
            name = p.name.replace("dataset_", "")
            tar = out / subset / f"{name}.tar"
            if not tar.exists():
                items = []
                for seg in sorted(d for d in p.iterdir()):
                    if seg.is_dir():
                        for k in KEEP + (("pictures",) if with_pictures else ()):
                            if (seg / k).exists():
                                items.append(str((seg / k).relative_to(W)))
                    elif seg.name in KEEP:
                        items.append(str(seg.relative_to(W)))
                subprocess.run(["tar", "-cf", str(tar), "-C", str(W)] + items, check=True)
            for seg in sorted(d for d in p.iterdir() if d.is_dir()):
                splits["test" if is_test(seg) else "train"].append(f"{subset}/{name}/{seg.name}")
            manifest.append({"file": f"{subset}/{tar.name}", "bytes": tar.stat().st_size, "sha256": sha256(tar),
                             "segments": sum(1 for d in p.iterdir() if d.is_dir())})
            print(manifest[-1], flush=True)
    (out / "splits.json").write_text(json.dumps(splits, indent=1))
    shutil.copy2(ROOT / "paper/dataset_README.md", out / "README.md")
    for f in ("flags.json", "align.csv"):
        if (W / "curation" / f).exists():
            (out / "curation").mkdir(exist_ok=True)
            shutil.copy2(W / "curation" / f, out / "curation" / f)
    if (W / "curation/teacher/teacher_report.csv").exists():
        (out / "curation/teacher").mkdir(parents=True, exist_ok=True)
        for f in (W / "curation/teacher").glob("*.csv"):
            shutil.copy2(f, out / "curation/teacher" / f.name)
        for f in (W / "curation/teacher").glob("unlabelled_*.txt"):
            shutil.copy2(f, out / "curation/teacher" / f.name)
    if (W / "results/dataset_stats.md").exists():
        shutil.copy2(W / "results/dataset_stats.md", out / "dataset_stats.md")
    (out / "LICENSE.txt").write_text(
        "N-People is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence\n"
        "(https://creativecommons.org/licenses/by-nc-sa/4.0/). It contains derivatives of MOT17 (CC BY-NC-SA 3.0) and of\n"
        "PersonPath22 (CC BY-NC 4.0; do not use the data to identify people or to derive biometric information).\n")
    (out / "manifest.json").write_text(json.dumps({"version": version, "files": manifest, "total_bytes": sum(m["bytes"] for m in manifest)}, indent=1))
    print(f"{len(manifest)} tar files, {sum(m['bytes'] for m in manifest) / 1e9:.1f} GB, {len(splits['test'])} test and "
          f"{len(splits['train'])} train segments -> {out}")


if __name__ == "__main__":
    main()
