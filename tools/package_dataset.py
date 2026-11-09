"""Package N-People for 4TU.ResearchData in the layout of the published record: one tar file per pass, named
<subset>_<pass>.tar (4TU has no folders), each unpacking to dataset_<pass>/<segment>/ with the raw events, the monitor
log, the calibration, the time fit and the labels (the 50 ms pictures are left out: tools/record_dataset.py label
makes them); curation.tar; README.md (docs/N-People.md); LICENSE.txt; splits.json (test / train segment folders as
unpacked); dataset_stats.md; manifest.json with the size and SHA-256 of every file.
Usage: package_dataset.py OUT_DIR [--version 1.0]
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
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
DOI = "10.4121/e9e04333-cc28-4bfe-9079-7bb252203b02"


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


def main(out, version):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files, splits = [], {"test": [], "train": []}
    for subset, pats in SUBSETS.items():
        for p in passes(pats):
            tar = out / f"{subset}_{p.name.replace('dataset_', '')}.tar"
            segs = sorted(d for d in p.iterdir() if d.is_dir())
            if not tar.exists():
                items = [str((seg / k).relative_to(W)) for seg in segs for k in KEEP if (seg / k).exists()]
                items += [str((p / k).relative_to(W)) for k in KEEP if (p / k).is_file()]
                subprocess.run(["tar", "-cf", str(tar), "-C", str(W)] + items, check=True)
            for seg in segs:
                splits["test" if is_test(seg) else "train"].append(f"{p.name}/{seg.name}")
            files.append({"file": tar.name, "bytes": tar.stat().st_size, "sha256": sha256(tar), "segments": len(segs),
                          "unpacks_to": p.name + "/"})
            print(files[-1]["file"], files[-1]["bytes"], flush=True)
    with tarfile.open(out / "curation.tar", "w") as t:
        t.add(W / "curation", arcname="curation", filter=lambda ti: None if ti.name.endswith(".jpg") else ti)
    (out / "splits.json").write_text(json.dumps({"note": "segment folders as unpacked from the tar files", **splits},
                                                indent=1))
    readme = ROOT / "docs/N-People.md"
    shutil.copy2(readme if readme.exists() else ROOT / "paper/dataset_README.md", out / "README.md")
    (out / "LICENSE.txt").write_text(
        "N-People is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence\n"
        "(https://creativecommons.org/licenses/by-nc-sa/4.0/). It contains derivatives of MOT17 (CC BY-NC-SA 3.0) and of\n"
        "PersonPath22 (CC BY-NC 4.0; do not use the data to identify people or to derive biometric information).\n")
    stats = W / "results/dataset_stats.md"
    if stats.exists():
        shutil.copy2(stats, out / "dataset_stats.md")
    for n in ("curation.tar", "splits.json", "README.md", "LICENSE.txt", "dataset_stats.md"):
        if (out / n).exists():
            files.append({"file": n, "bytes": (out / n).stat().st_size, "sha256": sha256(out / n)})
    (out / "manifest.json").write_text(json.dumps({"dataset": "N-People", "version": version, "doi": DOI,
                                                   "files": files, "total_bytes": sum(f["bytes"] for f in files)},
                                                  indent=1))
    print(f"{len(files) + 1} files, {sum(f['bytes'] for f in files) / 1e9:.1f} GB, {len(splits['test'])} test and "
          f"{len(splits['train'])} train segments -> {out}")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[a.index("--version") + 1] if "--version" in a else "1.0")
