"""Package N-People for 4TU.ResearchData as the record holds it: one zip with the data in logical folders, and the
README, the licence and the manifest next to it (4TU shows a flat list of files; the README can be read without
downloading the data).

  N-People-v<version>.zip   N-People/README.md, LICENSE.txt, splits.json, dataset_stats.md
                            N-People/MOT17/dataset_<pass>/<segment>/...   (and N-People/PersonPath22/..., N-People/Room/...)
                            N-People/curation/...
  README.md, LICENSE.txt    the same files as in the zip
  manifest.json             size and SHA-256 of the record's files; segments and bytes per folder of the zip

Per segment: the raw events, the monitor log, the calibration, the time fit and the labels (the 50 ms pictures are
left out: tools/record_dataset.py label makes them). splits.json names the test and training segments by their path
below N-People/. The README is docs/N-People.md (paper/dataset_README.md in the development repository).
Usage: package_dataset.py OUT_DIR [--version 1.0]
"""
import hashlib
import json
import shutil
import sys
import zipfile
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
LICENSE = ("N-People is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International "
           "licence\n(https://creativecommons.org/licenses/by-nc-sa/4.0/). It contains derivatives of MOT17 (CC BY-NC-SA "
           "3.0) and of\nPersonPath22 (CC BY-NC 4.0; do not use the data to identify people or to derive biometric "
           "information).\n")


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
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def files_of(item):
    """the file itself, or every file below a kept directory (labels/, labels_all/)"""
    return [item] if item.is_file() else sorted(p for p in item.rglob("*") if p.is_file())


def main(out, version):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    readme = ROOT / "docs/N-People.md"
    readme = readme if readme.exists() else ROOT / "paper/dataset_README.md"
    stats = W / "results/dataset_stats.md"
    zpath = out / f"N-People-v{version}.zip"
    splits, contents = {"test": [], "train": []}, []
    # deflate level 1: raw events shrink by about a fifth, at little cost
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for subset, pats in SUBSETS.items():
            for p in passes(pats):
                base, nbytes = f"N-People/{subset}/{p.name}", 0
                segs = sorted(d for d in p.iterdir() if d.is_dir())
                for k in KEEP:  # files of the pass itself (CONDITIONS.json, composites.json, ...)
                    if (p / k).is_file():
                        z.write(p / k, f"{base}/{k}")
                        nbytes += (p / k).stat().st_size
                for seg in segs:
                    for k in KEEP:
                        if (seg / k).exists():
                            for f in files_of(seg / k):
                                z.write(f, f"{base}/{seg.name}/{f.relative_to(seg).as_posix()}")
                                nbytes += f.stat().st_size
                    splits["test" if is_test(seg) else "train"].append(f"{subset}/{p.name}/{seg.name}")
                contents.append({"folder": f"{subset}/{p.name}", "segments": len(segs), "bytes": nbytes})
                print(contents[-1]["folder"], len(segs), nbytes, flush=True)
        for f in sorted((W / "curation").rglob("*")):
            if f.is_file() and f.suffix != ".jpg":
                z.write(f, "N-People/curation/" + f.relative_to(W / "curation").as_posix())
        z.writestr("N-People/splits.json", json.dumps(
            {"note": "segment folders, as paths below N-People/", **splits}, indent=1))
        z.write(readme, "N-People/README.md")
        z.writestr("N-People/LICENSE.txt", LICENSE)
        if stats.exists():
            z.write(stats, "N-People/dataset_stats.md")
    shutil.copy2(readme, out / "README.md")
    (out / "LICENSE.txt").write_text(LICENSE)
    files = [{"file": n, "bytes": (out / n).stat().st_size, "sha256": sha256(out / n)}
             for n in (zpath.name, "README.md", "LICENSE.txt")]
    (out / "manifest.json").write_text(json.dumps(
        {"dataset": "N-People", "version": version, "doi": DOI, "files": files,
         "zip_contents": contents, "segments": {k: len(v) for k, v in splits.items()},
         "unpacked_bytes": sum(c["bytes"] for c in contents)}, indent=1))
    print(f"{zpath.name}: {zpath.stat().st_size / 1e9:.2f} GB zipped, {sum(c['bytes'] for c in contents) / 1e9:.2f} GB "
          f"of segment files; {len(splits['test'])} test and {len(splits['train'])} train segments -> {out}")


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 2:
        sys.exit(__doc__)
    main(a[1], a[a.index("--version") + 1] if "--version" in a else "1.0")
