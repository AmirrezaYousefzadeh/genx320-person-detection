"""The dataset in numbers, for the README and the paper: per group of passes, the segments, 50 ms pictures, labelled
people (boxes with >= 20 events; and all mapped boxes), events, raw bytes and recorded seconds; the videos and the
people (track ids) they come from. Writes work/results/dataset_stats.json and .md.
Usage: dataset_stats.py [--include-card]   (--include-card also counts recorded segments not copied yet)
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / "work"

GROUPS = OrderedDict([
    ("MOT17, source frame rate, Eizo (v1-v3)", ["dataset_v1", "dataset_v2", "dataset_v3"]),
    ("MOT17, source frame rate, Eizo, AFK on (a1)", ["dataset_a1"]),
    ("MOT17, source frame rate, Eizo, histogram frames (h1)", ["dataset_h1"]),
    ("MOT17, source frame rate, Samsung, lit (sam*)", ["dataset_sam1", "dataset_sam_flip", "dataset_sam_dim50",
                                                       "dataset_sam_dim25", "dataset_sam3", "dataset_sam3_flip",
                                                       "dataset_sam3_pan_l", "dataset_sam3_pan_r"]),
    ("MOT17, source frame rate, zoomed, dark (camp_dark_*)", "dataset_camp_dark_*"),
    ("MOT17, 60 fps, zoomed, dark (camp60_dark_*)", "dataset_camp60_dark_*"),
    ("MOT17, 60 fps, zoomed, lit (camp60_lit_*)", "dataset_camp60_lit_*"),
    ("PersonPath22, 60 fps, zoomed, dark (camp60pp_*)", "dataset_camp60pp_*"),
    ("Room, one person, 11 scenarios (real1)", ["dataset_real1"]),
])
DERIVED = OrderedDict([
    ("Room composites (2-3 people, derived from real1)", ["dataset_real1_comp"]),
    ("Close-up crops of v1-v3 (derived)", ["dataset_zoom"]),
])


def passes(spec):
    if isinstance(spec, str):
        return sorted(p for p in W.glob(spec) if p.is_dir())
    return [W / n for n in spec if (W / n).is_dir()]


def seg_stats(seg, include_card):
    ej = seg / "events.json"
    pics = sorted((seg / "pictures").glob("pic_*.png"))
    if not ej.exists():
        if pics and (seg / "labels").is_dir():  # the room recordings (tools/self_record.py: labelled from the motion)
            people = sum(1 for p in pics if (seg / "labels" / (p.stem + ".txt")).exists()
                         for line in (seg / "labels" / (p.stem + ".txt")).read_text().splitlines() if line.startswith("0 "))
            raw = seg / "events.raw"
            return {"segments": 1, "pictures": len(pics), "people": people, "boxes_all": people, "events": 0,
                    "bytes": raw.stat().st_size if raw.exists() else 0, "seconds": len(pics) * 0.05, "video": seg.name}
        if include_card and (seg / "segment.json").exists():
            return {"segments": 1, "on_card": 1}
        return None
    ev = json.loads(ej.read_text())
    raw = seg / "events.raw"
    labels = seg / "labels"
    people = 0
    if labels.is_dir():
        for p in pics:
            t = labels / (p.stem + ".txt")
            if t.exists():
                people += sum(1 for line in t.read_text().splitlines() if line.startswith("0 "))
    src = json.loads((seg / "segment.json").read_text()) if (seg / "segment.json").exists() else {}
    video = Path(str(src.get("seq_dir", src.get("scenario", seg.name)))).name
    return {"segments": 1, "pictures": len(pics) or ev.get("pictures", 0), "people": people,
            "boxes_all": ev.get("boxes_all", 0), "events": ev.get("events", 0),
            "bytes": raw.stat().st_size if raw.exists() else 0, "seconds": ev.get("duration_s", 0.0), "video": video}


def add(tot, s):
    for k, v in s.items():
        if k == "video":
            tot.setdefault("videos", set()).add(v)
        else:
            tot[k] = tot.get(k, 0) + v


def main():
    include_card = "--include-card" in sys.argv
    out = OrderedDict()
    grand = {}
    for name, spec in list(GROUPS.items()) + list(DERIVED.items()):
        tot = {}
        ps = passes(spec)
        for p in ps:
            for seg in sorted(d for d in p.iterdir() if d.is_dir() and ((d / "segment.json").exists() or (d / "pictures").is_dir())):
                s = seg_stats(seg, include_card)
                if s:
                    add(tot, s)
        if not tot:
            continue
        tot["passes"] = len(ps)
        tot["videos"] = len(tot.get("videos", ()))
        out[name] = tot
        if name in GROUPS:
            add(grand, {k: v for k, v in tot.items() if k not in ("videos", "passes")})
    grand.pop("video", None)
    out["TOTAL (real recordings)"] = grand
    (ROOT / "work/results").mkdir(exist_ok=True)
    (ROOT / "work/results/dataset_stats.json").write_text(json.dumps(out, indent=1))
    cols = ["passes", "segments", "videos", "pictures", "people", "boxes_all", "events", "bytes", "seconds"]
    md = ["| group | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, t in out.items():
        md.append(f"| {name} | " + " | ".join(
            f"{t.get(c, 0) / 1e9:.2f} GB" if c == "bytes" else f"{t.get(c, 0) / 60:.1f} min" if c == "seconds"
            else f"{t.get(c, 0):,}" if c != "events" else f"{t.get(c, 0) / 1e6:.0f} M" for c in cols) + " |")
    (ROOT / "work/results/dataset_stats.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
