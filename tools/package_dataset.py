"""Package N-People for 4TU.ResearchData as the record holds it: one zip with the data in logical folders, and the
README, the licence and the manifest next to it (4TU shows a flat list of files; the README can be read without
downloading the data).

  N-People-v<version>.zip   N-People/README.md, LICENSE.txt, splits.json, passes.csv, dataset_stats.md
                            N-People/MOT17/dataset_<pass>/<segment>/...   (and N-People/PersonPath22/..., N-People/Room/...)
                            N-People/curation/...
  README.md, LICENSE.txt    the same files as in the zip
  manifest.json             size and SHA-256 of the record's files; segments and bytes per folder of the zip

Per segment: the raw events, the monitor log, the calibration, the time fit and the labels (the 50 ms pictures are
left out: tools/record_dataset.py pictures makes them; the room composites, which cannot be remade from events alone,
keep theirs). segment.json's source path is written relative to the code repository's work/ folder. splits.json
names the test and training segments by their path below N-People/; passes.csv describes every pass. The README is
docs/N-People.md (paper/dataset_README.md in the development repository).
Usage: package_dataset.py OUT_DIR [--version 1.0]
"""
import csv
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_ap import timing_uncertain  # noqa: E402

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
LICENSE = """N-People v{version}, by Amirreza Yousefzadeh (University of Twente), DOI {doi}

Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0),
https://creativecommons.org/licenses/by-nc-sa/4.0/

N-People contains adaptations of two datasets; their creators are credited here as their licences require:
- MOT17 (MOTChallenge): A. Milan, L. Leal-Taixe, I. Reid, S. Roth, K. Schindler, "MOT16: A Benchmark for
  Multi-Object Tracking", arXiv:1603.00831, 2016; https://motchallenge.net/data/MOT17/ ; licensed CC BY-NC-SA 3.0,
  https://creativecommons.org/licenses/by-nc-sa/3.0/ . Used: the seven training videos, replayed on a monitor and
  recorded with an event camera, and their person annotations, mapped into the event camera's picture.
- PersonPath22 (Amazon): B. Shuai, A. Bergamo, U. Buechler, A. Berneshawi, A. Boden, J. Tighe, "Large Scale
  Real-world Multi-Person Tracking", ECCV 2022; https://amazon-science.github.io/tracking-dataset/personpath22.html ;
  (c) Amazon.com 2022, licensed CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/ ; annotations by AWS.
  Used: one 10 s window of 180 videos, replayed and recorded the same way, and their annotations.
Changes made: the videos were shown on a monitor (cropped, scaled, mirrored, dimmed or interpolated to 60 fps as
described in README.md), recorded with a Prophesee GenX320 event camera, and the annotations were mapped into the
event camera's picture and filtered by the recorded events.

A request, not a licence condition: the PersonPath22 creators ask that their data not be used to identify the people
in the videos or to generate biometric information about them; we ask the same for all of N-People.

The code that recorded and reads N-People is not part of this licence: see
https://github.com/cognimachina/genx320-person-detection (its own licence files).
"""


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


# the passes before the campaign logs (CONDITIONS.json came with the 2026-11-05 night passes)
EARLY = {
    "dataset_v1": ("Eizo", "not logged", "events, ERC 500 kev/s (no AFK)", "recovered from the pass's own labels"),
    "dataset_v2": ("Eizo", "not logged", "events, ERC 500 kev/s (no AFK)", "recovered from the pass's own labels"),
    "dataset_v3": ("Eizo", "not logged", "events, ERC 500 kev/s (no AFK)", "recovered from the pass's own labels"),
    "dataset_a1": ("Eizo", "not logged", "events, ERC 500 kev/s, AFK", "recovered from the pass's own labels"),
    "dataset_h1": ("Eizo", "not logged", "histogram frames (one per ~120 ms)", "recovered from the pass's own labels"),
    "dataset_sam1": ("Samsung", "lit (not logged)", "", "measured before the pass"),
    "dataset_sam_flip": ("Samsung", "lit (not logged)", "", "measured before the pass"),
    "dataset_sam_dim50": ("Samsung", "lit (not logged)", "", "measured before the pass"),
    "dataset_sam_dim25": ("Samsung", "lit (not logged)", "", "measured before the pass"),
}
PAPER_SETS = (("dataset_sam1", "mon30"), ("dataset_sam_flip", "mon30"), ("dataset_sam_dim50", "mon30"),
              ("dataset_sam_dim25", "mon30"), ("dataset_camp60pp_", "pp22test"), ("dataset_camp60_dark_", "mon60z"),
              ("dataset_camp60_lit_", "mon60z"), ("dataset_camp_dark_", "mon30z"))


def pass_row(subset, p, n_segs):
    """one line of passes.csv: how the pass was recorded"""
    n = p.name
    cond = json.loads((p / "CONDITIONS.json").read_text()) if (p / "CONDITIONS.json").exists() else {}
    opts = cond.get("options", "").split()
    opt = lambda k, d="": opts[opts.index(k) + 1] if k in opts else d  # noqa: E731
    monitor, light, sensor, calib = EARLY.get(n, ("", "", "", ""))
    first = next((d for d in sorted(p.iterdir()) if (d / "events.raw").exists()), None)
    if not sensor and first is not None:
        head = open(first / "events.raw", "rb").read(512).decode("ascii", "replace")
        sensor = "events, ERC 500 kev/s, AFK" if "afk=1" in head else "events, ERC 500 kev/s"
    if subset == "Room":
        monitor, light, view, fps = "-", "office light", "real scene", "-"
    else:
        monitor = monitor or ("Samsung" if "Samsung" in cond.get("monitor", "") else cond.get("monitor", ""))
        light = light or str(cond.get("room_light", ""))
        z = opt("--zoom")
        view = f"zoomed crop, people ~{float(z):.0%} of the view high, crop rank {opt('--zoom-rank', '0')}" if z \
            else ("panned " + ("left" if opt("--pan", "0").startswith("-") else "right") if "--pan" in opts else "full view")
        fps = "60 (interpolated)" if ("60" in n.split("_")[1] or "mot17_60" in cond.get("options", "")
                                      or "pp22_60" in cond.get("options", "")) else "source (30/25/14)"
    gain = opt("--gain") or {"dataset_sam_dim50": "0.5", "dataset_sam_dim25": "0.25"}.get(n, "1")
    mirrored = "yes" if ("--flip" in opts or n.endswith("_flip") or "_flip_" in n) else "no"
    calib = calib or ("measured before and after the pass" if subset != "Room" else "-")
    if n.startswith("dataset_sam3"):
        calib = "measured; relabelled after the tripod head sank (calib_1712.json: the calibration before)"
    if n == "dataset_real1_comp":
        view, sensor, calib = "composites of real1 (synthetic)", "-", "-"
    if subset == "Room":
        gain = mirrored = "-"
    paper = next((t for k, t in PAPER_SETS if n == k or (k.endswith("_") and n.startswith(k))), "")
    return {"subset": subset, "pass": n, "segments": n_segs, "monitor": monitor, "room_light": light, "view": view,
            "frame_rate": fps, "gain": gain, "mirrored": mirrored, "sensor": sensor,
            "calibration_drift_px": cond.get("drift_px", ""), "calibration": calib, "paper_test_set": paper}


def main(out, version):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    readme = ROOT / "docs/N-People.md"
    readme = readme if readme.exists() else ROOT / "paper/dataset_README.md"
    stats = W / "results/dataset_stats.md"
    lic = LICENSE.format(version=version, doi=DOI)
    zpath = out / f"N-People-v{version}.zip"
    splits, contents, rows, total = {"test": [], "train": [], "timing_uncertain": []}, [], [], 0
    # deflate level 1: raw events shrink by about a fifth, at little cost
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:

        def put(src, arc):
            nonlocal total
            if src.name == "segment.json":  # the recording PC's absolute path -> relative to the repository's work/
                seg = json.loads(src.read_text())
                if "seq_dir" in seg and "/datasets/" in seg["seq_dir"]:
                    seg["seq_dir"] = "datasets/" + seg["seq_dir"].split("/datasets/", 1)[1]
                data = json.dumps(seg, indent=1).encode()
                z.writestr(arc, data)
                total += len(data)
                return len(data)
            if src.name == "composites.json":  # source pictures as paths below N-People/
                data = src.read_text().replace('"work/dataset_real1/', '"Room/dataset_real1/').encode()
                z.writestr(arc, data)
                total += len(data)
                return len(data)
            z.write(src, arc)
            total += src.stat().st_size
            return src.stat().st_size

        for subset, pats in SUBSETS.items():
            for p in passes(pats):
                base, nbytes = f"N-People/{subset}/{p.name}", 0
                segs = sorted(d for d in p.iterdir() if d.is_dir())
                keep = KEEP + (("pictures",) if p.name == "dataset_real1_comp" else ())
                for k in keep:  # files of the pass itself (CONDITIONS.json, composites.json, ...)
                    if (p / k).is_file():
                        nbytes += put(p / k, f"{base}/{k}")
                for seg in segs:
                    for k in keep:
                        if (seg / k).exists():
                            for f in files_of(seg / k):
                                nbytes += put(f, f"{base}/{seg.name}/{f.relative_to(seg).as_posix()}")
                    splits["test" if is_test(seg) else "train"].append(f"{subset}/{p.name}/{seg.name}")
                    if timing_uncertain(seg):
                        splits["timing_uncertain"].append(f"{subset}/{p.name}/{seg.name}")
                contents.append({"folder": f"{subset}/{p.name}", "segments": len(segs), "bytes": nbytes})
                rows.append(pass_row(subset, p, len(segs)))
                print(contents[-1]["folder"], len(segs), nbytes, flush=True)
        for name in ("align.csv", "flags.json"):  # the curation results (the relabel lists were working files)
            if (W / "curation" / name).exists():
                put(W / "curation" / name, "N-People/curation/" + name)
        for f in sorted((W / "curation/teacher").rglob("*")):
            if f.is_file() and f.suffix != ".jpg":
                put(f, "N-People/curation/teacher/" + f.relative_to(W / "curation/teacher").as_posix())
        buf = io.StringIO()
        wr = csv.DictWriter(buf, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
        extra = {"N-People/passes.csv": buf.getvalue(),
                 "N-People/splits.json": json.dumps({"note": "segment folders, as paths below N-People/; the room "
                                                     "composites are two folders (composites_train, composites_val). "
                                                     "timing_uncertain: test or training segments whose labels are "
                                                     "timed only by the start command (see curation/flags.json); "
                                                     "leave them out of both", **splits}, indent=1),
                 "N-People/LICENSE.txt": lic}
        for arc, text in extra.items():
            z.writestr(arc, text)
            total += len(text.encode())
        put(readme, "N-People/README.md")
        if stats.exists():
            put(stats, "N-People/dataset_stats.md")
    shutil.copy2(readme, out / "README.md")
    (out / "LICENSE.txt").write_text(lic)
    files = [{"file": n, "bytes": (out / n).stat().st_size, "sha256": sha256(out / n)}
             for n in (zpath.name, "README.md", "LICENSE.txt")]
    (out / "manifest.json").write_text(json.dumps(
        {"dataset": "N-People", "version": version, "doi": DOI, "files": files,
         "zip_contents": contents, "segment_folders": {k: len(v) for k, v in splits.items()},
         "note": "segment folders = 837 recorded segments + the 2 folders of room composites",
         "pass_folder_bytes": sum(c["bytes"] for c in contents), "unpacked_bytes": total}, indent=1))
    print(f"{zpath.name}: {zpath.stat().st_size / 1e9:.2f} GB zipped, {total / 1e9:.2f} GB unpacked; "
          f"{len(splits['test'])} test and {len(splits['train'])} train segment folders -> {out}")


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 2:
        sys.exit(__doc__)
    main(a[1], a[a.index("--version") + 1] if "--version" in a else "1.0")
