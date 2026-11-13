"""The paper's plots in a plain IEEE style (serif Times-like text, boxed axes with inward ticks, black-and-white friendly
grey fills and line styles, as in the 'science' + 'ieee' styles of SciencePlots), one column wide.

  data     AP50 per test set for the main training mixes (mean of three trainings, bars = their spread)
           from work/results/ap_seeds_{1,2,3}.json
  acclat   AP50 on PEDRo against the network time on the board, int8 / float head / float head with the
           event-driven first layer (times from work/results/bench_ops.json, else board_timings.json)
  time     where the time of a network run goes on the board (bench, per operator group), dense and event-driven
           first layer, from work/results/bench_ops.json
  pulse    events per millisecond of a replay at the source frame rate and at 60 fps
Usage: paper_figs_ieee.py data|acclat|time|pulse|all
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper/figures"
RES = ROOT / "work/results"
COL = 3.5  # IEEE column width, inches
GREYS = ["white", "0.78", "0.5", "0.15"]
HATCH = ["", "////", "", ""]


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9, "lines.markersize": 3.5, "patch.linewidth": 0.5,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "legend.frameon": False, "legend.handlelength": 1.6, "axes.grid": False, "hatch.linewidth": 0.4,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.015, "pdf.fonttype": 42,
        "axes.prop_cycle": cycler(color=["k", "r", "b", "g"]) + cycler(linestyle=["-", "--", ":", "-."]),
    })


def load_json(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------- data: which training data make the detector work
SETS = [("mon30", "full"), ("mon30z", "zoom"), ("mon60z", "60 fps"), ("pp22test", "PP22"), ("pedro", "PEDRo")]
ROWS = [("row1", "base"), ("row3", "+ zoomed"), ("row2", "+ simulated"), ("row5", "all (board)")]


def fig_data():
    seeds = [load_json(f"ap_seeds_{s}.json") for s in (1, 2, 3)]
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    w = 0.19
    x = np.arange(len(SETS))
    for j, (row, label) in enumerate(ROWS):
        vals = np.array([[sd.get(f"s{s}/{row}", {}).get(k, {}).get("AP50", np.nan) for k, _ in SETS]
                         for s, sd in zip((1, 2, 3), seeds)], float)
        n = np.sum(~np.isnan(vals), 0)
        mean = np.nanmean(vals, 0)
        err = np.where(n > 1, np.nanstd(vals, 0), np.nan)
        ax.bar(x + (j - 1.5) * w, mean, w, color=GREYS[j], edgecolor="k", hatch=HATCH[j], label=label,
               yerr=err, error_kw={"elinewidth": 0.5, "capsize": 1.2, "capthick": 0.5})
    ax.set_xticks(x, [n for _, n in SETS])
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.set_ylabel("AP50")
    ax.set_ylim(0, 0.45)
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), columnspacing=0.9, handlelength=1.2)
    fig.savefig(FIG / "data.pdf")
    print("wrote", FIG / "data.pdf")


# ---------------------------------------------------------------- accuracy against latency on PEDRo
PEDRO = {128: "pedro128_a", 160: "pedro160_d", 192: "pedro192_b"}


def bench_time(ops, size, kind):
    """network time (ms) of a bench build: kind int8 / hf / hf_sparse"""
    names = {(192, "int8"): "build_b192_dense", (192, "hf"): "build_b192hf_dense", (192, "hf_sparse"): "build_b192hf_sparse",
             (160, "hf"): "build_v9_bench", (160, "hf_sparse"): "build_v9s_bench",
             (128, "hf"): "build_b128_dense", (128, "hf_sparse"): "build_b128_sparse"}
    r = ops.get(names.get((size, kind), ""), {})
    return r.get("run_ms")


def fig_acclat():
    ap = load_json("ap_table.json")
    ops = load_json("bench_ops.json")
    old = load_json("board_timings.json")
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    series = {"int8": ([], []), "int8 body + float head": ([], []), "float head + event-driven layer 1": ([], [])}
    for size, run in PEDRO.items():
        a_i8 = ap[f"{run} int8"]["pedro"]["AP50"]
        a_hf = ap[f"{run} int8+float head"]["pedro"]["AP50"]
        t_i8 = old.get(str(size), {}).get("int8_ms")
        t_hf = bench_time(ops, size, "hf") or old.get(str(size), {}).get("hf_ms")
        t_sp = bench_time(ops, size, "hf_sparse")
        series["int8"][0].append(t_i8), series["int8"][1].append(a_i8)
        series["int8 body + float head"][0].append(t_hf), series["int8 body + float head"][1].append(a_hf)
        if t_sp:
            series["float head + event-driven layer 1"][0].append(t_sp)
            series["float head + event-driven layer 1"][1].append(a_hf)  # bit-exact: the same accuracy
        ax.annotate(f"{size} px", (t_hf, a_hf), textcoords="offset points", xytext=(3, -9), fontsize=7)
    marks = {"int8": ("o", "white"), "int8 body + float head": ("s", "k"), "float head + event-driven layer 1": ("D", "0.6")}
    for (name, (xs, ys)), ls in zip(series.items(), ["--", "-", ":"]):
        if xs:
            m, fc = marks[name]
            ax.plot(xs, ys, ls=ls, color="k", marker=m, mfc=fc, mec="k", mew=0.6, label=name)
    ax.set_xlabel("network time on the STM32F746 (ms)")
    ax.set_ylabel("AP50 on PEDRo")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "acclat.pdf")
    print("wrote", FIG / "acclat.pdf")


# ---------------------------------------------------------------- where the time of a network run goes
CONFIGS = [("build_b192hf_dense", "192 px"), ("build_b192hf_sparse", "192 px, event-driven L1"),
           ("build_v9_bench", "160 px"), ("build_v9s_bench", "160 px, event-driven L1"),
           ("build_b128_dense", "128 px"), ("build_b128_sparse", "128 px, event-driven L1")]
GROUPS = [("layer 1", "white", ""), ("layers 2-4", "0.78", ""), ("grid layers", "0.45", ""),
          ("head, copies", "0.15", "")]


def group_times(r):
    """layer 1 (with the input quantization), the three other convolutions on the large maps, the rest, float head"""
    ops = r["ops"]
    convs = [i for i, o in enumerate(ops) if "CONV" in o["name"].upper()]
    l1 = sum(o["ms"] for o in ops[:convs[0] + 1])
    big = sum(ops[i]["ms"] for i in convs[1:4]) + sum(o["ms"] for i, o in enumerate(ops[convs[0] + 1:convs[3]], convs[0] + 1)
                                                      if i not in convs[1:4])
    rest = sum(o["ms"] for o in ops[convs[3] + 1:])
    return [l1, big, rest, max(r["rest_ms"], 0.0)]


def fig_time():
    ops = load_json("bench_ops.json")
    rows = [(label, group_times(ops[b])) for b, label in CONFIGS if b in ops]
    if not rows:
        print("no bench_ops.json yet")
        return
    fig, ax = plt.subplots(figsize=(COL, 0.32 * len(rows) + 0.55))
    y = np.arange(len(rows))[::-1]
    for (label, parts), yy in zip(rows, y):
        left = 0
        for v, (gname, c, h) in zip(parts, GROUPS):
            ax.barh(yy, v, 0.62, left=left, color=c, edgecolor="k", hatch=h)
            left += v
        ax.text(left + 6, yy, f"{left:.0f}", va="center", fontsize=7)
    for gname, c, h in GROUPS:
        ax.barh([0], [0], color=c, edgecolor="k", hatch=h, label=gname)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_xlabel("time of one network run on the bench (ms, 216 MHz)")
    ax.set_xlim(0, max(sum(p) for _, p in rows) * 1.12)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.42, 1.0), columnspacing=0.8, handlelength=1.2,
              borderaxespad=0.2)
    fig.savefig(FIG / "time.pdf")
    print("wrote", FIG / "time.pdf")


# ---------------------------------------------------------------- frame steps of the monitor
def fig_pulse():
    import evt2

    def per_ms(seg, t0_s=3.0, dur_s=0.5):
        _, w = evt2.read_raw(ROOT / seg / "events.raw")
        ev = evt2.decode(w)
        t = (ev["t"] - ev["t"][0]) / 1e3
        sel = (t >= t0_s * 1e3) & (t < (t0_s + dur_s) * 1e3)
        return np.bincount((t[sel] - t0_s * 1e3).astype(int), minlength=int(dur_s * 1e3))[:int(dur_s * 1e3)]

    pairs = [("MOT17-05, 14 fps source", "work/dataset_camp_dark_z40_r0_g100/MOT17-05-FRCNN_f0000",
              "work/dataset_camp60_dark_z40_r0_g100/MOT17-05-FRCNN_f0000"),
             ("MOT17-02, 30 fps source", "work/dataset_camp_dark_z40_r0_g100/MOT17-02-FRCNN_f0000",
              "work/dataset_camp60_dark_z40_r0_g100/MOT17-02-FRCNN_f0000")]
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.1), sharex=True)
    for ax, (title, a, b) in zip(axes, pairs):
        ca, cb = per_ms(a), per_ms(b)
        x = np.arange(len(ca))
        ax.plot(x, ca, "k-", lw=0.7, label="source frame rate")
        ax.plot(x, cb, "r--", lw=0.7, label="60 fps, interpolated")
        ax.axhline(500, color="k", lw=0.5, ls=":")
        ax.set_ylim(0, 640)
        ax.set_ylabel("events/ms")
        ax.text(0.01, 0.93, title, transform=ax.transAxes, fontsize=7, va="top")
    axes[0].text(497, 510, "rate limit", fontsize=6.5, ha="right", va="bottom")
    axes[-1].set_xlabel("time (ms)")
    axes[0].legend(ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.32))
    fig.subplots_adjust(hspace=0.08)
    fig.savefig(FIG / "pulse.pdf")
    print("wrote", FIG / "pulse.pdf")


if __name__ == "__main__":
    style()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, f in (("data", fig_data), ("acclat", fig_acclat), ("time", fig_time), ("pulse", fig_pulse)):
        if cmd in (name, "all"):
            f()
