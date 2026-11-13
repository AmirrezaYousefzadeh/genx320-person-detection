"""The paper's accuracy table: AP of one or more models on every test set (tools/eval_ap.py), as JSON and Markdown.

Test sets (real sensor unless named otherwise; people below 12 % of the picture height are ignored except on PEDRo,
whose protocol counts every box):
  mon30      monitor, MOT17-09/11 at the source frame rate, unzoomed (Samsung passes sam1, sam_flip, sam_dim50/25)
  mon30z     monitor, MOT17-09/11 at the source frame rate, zoomed (campaign passes camp_dark_*)
  mon60z     monitor, MOT17-09/11 at 60 fps, zoomed (campaign passes camp60_*; mon60z_dark / mon60z_lit: by room light)
  pp22test   monitor, PersonPath22 test-split windows at 60 fps, zoomed (campaign passes camp60pp_*)
  room       the room recordings held out (scenarios 03 walk 5 m, 06 sit, 09 occlusion) and their composites
  pedro      PEDRo test split (DAVIS346 on a robot; public benchmark)
Usage: eval_suite.py OUT.json MODEL[=NAME] [MODEL[=NAME] ...] [--sets mon30,room,...] [--size 160]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / "work/venv/bin/python")


def pp22_test_uids():
    sp = json.loads((ROOT / "work/datasets/personpath22/splits.json").read_text())
    return [u.split(".")[0] + "_" for u in sp["test"]]


def sets():
    w = ROOT / "work"
    glob = lambda pat: ",".join(str(p) for p in sorted(w.glob(pat)) if p.is_dir())  # noqa: E731
    return {
        "mon30": (",".join(str(w / f"dataset_{n}") for n in ("sam1", "sam_flip", "sam_dim50", "sam_dim25")), "09,11", 0.12),
        "mon30z": (glob("dataset_camp_dark_*"), "09,11", 0.12),
        "mon60z": (glob("dataset_camp60_*"), "09,11", 0.12),
        "mon60z_dark": (glob("dataset_camp60_dark_*"), "09,11", 0.12),
        "mon60z_lit": (glob("dataset_camp60_lit_*"), "09,11", 0.12),
        "pp22test": (glob("dataset_camp60pp_*"), ",".join(pp22_test_uids()), 0.12),
        "room": (str(w / "dataset_room_train"), "R1", 0.12),
        "pedro": (str(w / "pedro"), "pedro_test", 0.0),
    }


def main():
    args = sys.argv[1:]
    out = Path(args.pop(0))
    size = int(args[args.index("--size") + 1]) if "--size" in args else 160
    chosen = args[args.index("--sets") + 1].split(",") if "--sets" in args else list(sets())
    models = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--sets", "--size"))]
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    S = sets()
    for m in models:
        path, _, name = m.partition("=")
        name = name or Path(path).parent.name + "/" + Path(path).name
        for s in chosen:
            data, select, min_h = S[s]
            if not data:
                continue
            env = dict(os.environ)
            if os.environ.get("EVAL_DUMP_DIR"):  # per-segment detections for tools/paired_bootstrap.py
                env["EVAL_DUMP"] = str(Path(os.environ["EVAL_DUMP_DIR"]) / f"{name.replace('/', '__')}__{s}.json")
            r = subprocess.run([PY, "-I", str(ROOT / "tools/eval_ap.py"), "--model", path, "--data", data, "--select", select,
                                "--size", str(size), "--min-h", str(min_h)], capture_output=True, text=True, env=env)
            try:
                v = json.loads(r.stdout.strip().splitlines()[-1])
            except (IndexError, ValueError):
                print(f"{name} {s}: failed {r.stderr[-300:]}", flush=True)
                continue
            res.setdefault(name, {})[s] = {k: v.get(k) for k in ("AP50", "AP50_std", "AP50_95", "P@0.5", "R@0.5", "F1@0.5",
                                                                  "people", "pictures", "segments")}
            print(f"{name:40s} {s:9s} AP50 {v['AP50']:.3f} +- {v.get('AP50_std') or 0:.3f}  AP50:95 {v['AP50_95']:.3f}  "
                  f"F1@0.5 {v['F1@0.5']:.3f}  ({v['people']} people, {v['pictures']} pictures, {v.get('segments')} segments)", flush=True)
            out.write_text(json.dumps(res, indent=1))
    md = ["| model | " + " | ".join(chosen) + " |", "|---|" + "---|" * len(chosen)]
    for name, r in res.items():
        md.append(f"| {name} | " + " | ".join(f"{r[s]['AP50']:.3f} / {r[s]['AP50_95']:.3f}" if s in r else "-" for s in chosen) + " |")
    out.with_suffix(".md").write_text("AP50 / AP50:95\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
