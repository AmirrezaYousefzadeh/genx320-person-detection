"""The paper's figures (paper/figures/*.pdf), from the recorded data and the result tables.

  pulse    events per millisecond of the same MOT17 video replayed at its source frame rate and at 60 fps
           (needs two segments' events.raw; the frame steps of the monitor are what the sensor sees)
  pareto   PEDRo AP50 against the network time on the board for the input sizes (timings from work/results)
  pair     the same moment three ways: the source frame with its crop, the sensor's picture with the mapped
           boxes, the simulated picture (tools/evsim.py) of the same crop
Usage: paper_figs.py pulse|pareto|pair|all [--sim DIR]
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import evt2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper/figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categorical slots 1-4 (validated palette)
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": "#9a9a96", "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def per_ms(seg, t0_s=3.0, dur_s=0.5):
    _, w = evt2.read_raw(ROOT / seg / "events.raw")
    ev = evt2.decode(w)
    t = (ev["t"] - ev["t"][0]) / 1e3
    sel = (t >= t0_s * 1e3) & (t < (t0_s + dur_s) * 1e3)
    return np.bincount((t[sel] - t0_s * 1e3).astype(int), minlength=int(dur_s * 1e3))[:int(dur_s * 1e3)]


def fig_pulse():
    pairs = [("MOT17-05 (14 fps source)", "work/dataset_camp_dark_z40_r0_g100/MOT17-05-FRCNN_f0000",
              "work/dataset_camp60_dark_z40_r0_g100/MOT17-05-FRCNN_f0000"),
             ("MOT17-02 (30 fps source)", "work/dataset_camp_dark_z40_r0_g100/MOT17-02-FRCNN_f0000",
              "work/dataset_camp60_dark_z40_r0_g100/MOT17-02-FRCNN_f0000")]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.3), sharex=True)
    for ax, (title, a, b) in zip(axes, pairs):
        ca, cb = per_ms(a), per_ms(b)
        x = np.arange(len(ca))
        ax.plot(x, ca, color=ORANGE, lw=0.9, label="source frame rate")
        ax.plot(x, cb, color=BLUE, lw=0.9, label="60 fps, interpolated")
        ax.set_ylabel("events / ms")
        ax.set_title(title, fontsize=8, loc="left", color=INK2)
        ax.set_ylim(0, None)
        ax.grid(axis="y", color="#e4e4e0", lw=0.5)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("time (ms)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.55, -0.01))
    fig.tight_layout(pad=0.3, rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "pulse.pdf")
    print("wrote", FIG / "pulse.pdf")


def fig_pareto(timings=None):
    ap = json.loads((ROOT / "work/results/ap_table.json").read_text())
    rows = {128: "pedro128_a", 160: "pedro160_d", 192: "pedro192_b", 224: "pedro224_a"}
    timings = timings or json.loads((ROOT / "work/results/board_timings.json").read_text())
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    xs, ys, xi, yi = [], [], [], []
    for size, run in rows.items():
        hf = ap[f"{run} int8+float head"]["pedro"]["AP50"]
        i8 = ap[f"{run} int8"]["pedro"]["AP50"]
        t = timings.get(str(size))
        if t is None:
            continue
        xs.append(t["hf_ms"]), ys.append(hf), xi.append(t["int8_ms"]), yi.append(i8)
        ax.annotate(f"{size} px", (t["hf_ms"], hf), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK)
    ax.plot(xi, yi, "o-", color=ORANGE, lw=1.2, ms=4, label="int8", mfc="white")
    ax.plot(xs, ys, "o-", color=BLUE, lw=1.2, ms=4, label="int8 body + float head")
    ax.plot([422], [0.75], "s", color=AQUA, ms=5, label="BSc thesis 2025, int8, same board")
    ax.set_xlabel("network time on the STM32F746 (ms)")
    ax.set_ylabel("AP50 on PEDRo")
    ax.grid(color="#e4e4e0", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "pareto.pdf")
    print("wrote", FIG / "pareto.pdf")


def read_labels(txt):
    out = []
    if txt.exists():
        for line in txt.read_text().splitlines():
            c, cx, cy, w, h = (float(v) for v in line.split()[:5])
            out.append((int(c), cx, cy, w, h))
    return out


def fig_pair(sim=None, seg="work/dataset_camp60_dark_z40_r0_g100/MOT17-02-FRCNN_f0000", k=100, frame=280):
    seg = ROOT / seg
    log = (seg / "frames.log").read_text().splitlines()
    warp = next(l for l in log if l.startswith("warp ")).split()
    cx0, cy0, side = float(warp[1]), float(warp[2]), float(warp[3])
    src = ROOT / "work/datasets/mot17_60/train/MOT17-02-FRCNN/img1" / f"{frame + 1:06d}.jpg"
    img = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
    panels = [(img, "source frame and the crop (blue)", None)]
    crop = img[int(cy0):int(cy0 + side), int(cx0):int(cx0 + side)]
    panels.append((crop, "the crop on the screen", "crop"))
    pic = cv2.imread(str(seg / "pictures" / f"pic_{k:04d}.png"), cv2.IMREAD_GRAYSCALE)
    panels.append((pic, "sensor picture, labels", read_labels(seg / "labels" / f"pic_{k:04d}.txt")))
    if sim:
        sd = sorted(Path(sim).glob("MOT17-02-FRCNN_f00000_z40*"))
        if sd:
            ks = int(round((frame) / 60 / 0.05))
            sp = cv2.imread(str(sd[0] / "pictures" / f"pic_{ks:04d}.png"), cv2.IMREAD_GRAYSCALE)
            lab = read_labels(sd[0] / "labels" / f"pic_{ks:04d}.txt")
            if sd[0].name.endswith("_flip"):  # the simulator mirrored this window: mirror it back
                sp, lab = np.ascontiguousarray(sp[:, ::-1]), [(c, 1 - cx, cy, w, h) for c, cx, cy, w, h in lab]
            panels.append((sp, "simulated picture", lab))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.1, 2.0), gridspec_kw={"width_ratios": [1.5] + [1] * (len(panels) - 1)})
    for ax, (im, title, labels) in zip(axes, panels):
        if labels == "crop":
            ax.imshow(im)
        elif im.ndim == 2:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            S = im.shape[0]
            for c, cx, cy, w, h in labels or []:
                if c == 0 or title.startswith("sensor"):
                    ax.add_patch(plt.Rectangle(((cx - w / 2) * S, (cy - h / 2) * S), w * S, h * S, fill=False,
                                               ec=ORANGE if c == 0 else YELLOW, lw=0.9, ls="-" if c == 0 else ":"))
        else:
            ax.imshow(im)
            ax.add_patch(plt.Rectangle((cx0, cy0), side, side, fill=False, ec=BLUE, lw=1.2))
        ax.set_title(title, fontsize=7, color=INK2)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / "pair.pdf", dpi=200)
    print("wrote", FIG / "pair.pdf")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    sim = sys.argv[sys.argv.index("--sim") + 1] if "--sim" in sys.argv else None
    if cmd in ("pulse", "all"):
        fig_pulse()
    if cmd in ("pareto", "all"):
        fig_pareto()
    if cmd in ("pair", "all"):
        fig_pair(sim)
