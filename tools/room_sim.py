"""The board's counting in a room with several people, replayed: one-person room recordings (real_label.py) are
overlaid picture by picture, so that two or three people move at the same time (each extra person's region is
copied from its own recording at the same moment, with its own background events; moments where two people
overlap are not scored). The detector runs every RUN pictures, as on the board (11 pictures = 0.55 s, one pass;
22 = 1.1 s, mirrored pass), and each counting logic is applied, smoothing over runs included. Scored: the share of
runs whose shown count equals the true count at that moment, per true count; and how often 2 people show as 1.
Usage: room_sim.py PASS_DIR [--people 1,2,3] [--pairs 30] [--detector M.tflite|BODY.tflite+HEAD.npz] [--pass-ms 600]
       [--val-only] [--sweep 0.7,0.6,0.5]
Delay: median seconds from a change of the true count (someone enters or leaves) to the shown count being right.
"""
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from count_eval import GRID, NA, suppress  # noqa: E402
from yolo_ref import decode  # noqa: E402

PIC, SIZE = 320, 192


def boxes_of(txt):
    out = []
    if txt.exists():
        for line in txt.read_text().splitlines():
            _, cx, cy, w, h = (float(v) for v in line.split()[:5])
            out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def load(seg):
    pics = sorted((seg / "pictures").glob("pic_*.png"))
    return [(p, boxes_of(seg / "labels" / (p.stem + ".txt"))) for p in pics]


def compose(seqs, k):
    """picture k of the first sequence with the people of the others pasted in; None when people overlap"""
    base = cv2.imread(str(seqs[0][k][0]), cv2.IMREAD_GRAYSCALE)
    boxes = list(seqs[0][k][1])
    for s in seqs[1:]:
        for b in s[k][1]:
            if any(overlap(b, o) > 0.1 for o in boxes):
                return None, None
            other = cv2.imread(str(s[k][0]), cv2.IMREAD_GRAYSCALE)
            w, h = b[2] - b[0], b[3] - b[1]
            x0, y0 = int(max(0, b[0] - 0.1 * w) * PIC), int(max(0, b[1] - 0.05 * h) * PIC)
            x1, y1 = int(min(1, b[2] + 0.1 * w) * PIC), int(min(1, b[3] + 0.05 * h) * PIC)
            base[y0:y1, x0:x1] = other[y0:y1, x0:x1]
            boxes.append(b)
    return base, boxes




def logics(pass_ms, sweep=""):
    """the counting logics replayed, with their run time on the board: one pass takes pass_ms (network + the 50 ms
    picture + decoding), the mirrored pass about twice the network time. sweep "0.7,0.5,...": one pass with part
    removal at each of these thresholds (a model whose scores are calibrated differently needs another threshold)"""
    out = {}
    if sweep:
        for thr in (float(v) for v in sweep.split(",")):
            for med in (1, -2):
                name = f"1 pass, thr {thr}, parts, {'no smoothing' if med == 1 else 'max of last 2'}"
                out[name] = (False, thr, (0.3, 0.6), med, max(1, round(pass_ms / 50)), pass_ms)
        return out
    for mirror in (True, False):
        run_ms = 2 * pass_ms - 60 if mirror else pass_ms
        for thr, parts in ((0.7, (0.3, 0.6)), (0.6, None)):
            for med in (3, 1, -2):  # -2: max of the last two runs (rises at once, falls one run later)
                smooth = {3: "median 3", 1: "no smoothing", -2: "max of last 2"}[med]
                name = f"{'mirror' if mirror else '1 pass'}, thr {thr}, {'parts' if parts else 'NMS only'}, {smooth}"
                out[name] = (mirror, thr, parts, med, max(1, round(run_ms / 50)), run_ms)
    return out


def main():
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d  # noqa: E731
    from ai_edge_litert.interpreter import Interpreter
    det, head = opt("--detector", "work/finetune/finetune_v4/int8.tflite"), None
    if "+" in det:  # BODY_int8.tflite+HEAD.npz: the board's HEAD_FLOAT build (tools/headfloat.py)
        det, hp = det.split("+")
        hw = np.load(hp)
        head = (hw["W"], hw["b"])
    it = Interpreter(det)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    size = int(inp["shape"][1])
    grid = size // 16
    LOG = logics(float(opt("--pass-ms", 600)), opt("--sweep", ""))

    def run(x):
        it.set_tensor(inp["index"], x[None])
        it.invoke()
        o = it.get_tensor(outp["index"])[0]
        if head is not None:
            s, z = outp["quantization"]
            o = ((o.astype(np.float32) - z) * s) @ head[0] + head[1]
        return o.reshape(grid, grid, NA, 6).copy()

    def keep(p):  # --val-only: the validation scenarios (index % 3 == 0) and the empty room, for a room-trained model
        if "--val-only" not in a:
            return True
        meta = json.loads((p / "segment.json").read_text())
        return meta["index"] % 3 == 0 or meta["scenario"].startswith("empty")

    segs = {p.name: load(p) for p in sorted(Path(a[0]).iterdir()) if (p / "labels").is_dir()
            and "composite" not in p.name and keep(p)}
    movers = [n for n in segs if not n.split("_", 1)[1].startswith("empty")]
    empties = [n for n in segs if n.split("_", 1)[1].startswith("empty")]
    rng = np.random.default_rng(0)
    combos = [(e,) for e in empties] + [(m,) for m in movers]
    for k in (2, 3):
        if str(k) in opt("--people", "1,2,3").split(","):
            allc = list(itertools.permutations(movers, k))
            combos += [allc[i] for i in rng.choice(len(allc), min(int(opt("--pairs", 30)), len(allc)), replace=False)]
    res = {name: {} for name in LOG}
    two_as_one = {name: [0, 0] for name in LOG}
    delays = {name: [] for name in LOG}
    for combo in combos:
        seqs = [segs[n] for n in combo]
        n_pics = min(len(s) for s in seqs)
        truth_at = []  # true count per picture (None where people overlap)
        for k in range(n_pics):
            boxes = [b for s in seqs for b in s[k][1]]
            truth_at.append(None if any(overlap(boxes[i], boxes[j]) > 0.1 for i in range(len(boxes))
                                        for j in range(i + 1, len(boxes))) else len(boxes))
        grids = {}
        for name, (mirror, thr, parts, med, every, run_ms) in LOG.items():
            hist, shown_at = [], {}
            for k in range(0, n_pics, every):
                if truth_at[k] is None:
                    continue
                if k not in grids:
                    pic, _ = compose(seqs, k)
                    x = np.repeat(cv2.resize(pic, (size, size), interpolation=cv2.INTER_NEAREST)[..., None], 3, -1)
                    g1 = run(x)
                    g2 = run(x[:, ::-1].copy())[:, ::-1].copy()
                    g2[..., 0] = -g2[..., 0]
                    grids[k] = (g1, (g1 + g2) / 2)
                boxes = decode(grids[k][1 if mirror else 0].reshape(grid, grid, NA * 6), threshold=thr)
                if parts:
                    boxes = suppress(boxes, *parts)
                hist = (hist + [len(boxes)])[-abs(med):]
                if med == -2:
                    shown = max(hist)
                else:
                    shown = int(np.median(hist)) if len(hist) == med else hist[-1]
                shown_at[k] = shown
                t = truth_at[k]
                ok = res[name].setdefault(t, [0, 0])
                ok[0] += shown == t
                ok[1] += 1
                if t == 2:
                    two_as_one[name][0] += shown == 1
                    two_as_one[name][1] += 1
            # delay: from each change of the true count until a run shows it (its result appears run_ms later)
            for k in range(1, n_pics):
                if truth_at[k] is None or truth_at[k - 1] is None or truth_at[k] == truth_at[k - 1]:
                    continue
                later = [j for j in sorted(shown_at) if j >= k]
                hit = next((j for j in later if shown_at[j] == truth_at[k] and
                            all(truth_at[i] == truth_at[k] for i in range(k, j + 1) if truth_at[i] is not None)), None)
                if hit is not None:
                    delays[name].append((hit - k) * 0.05 + run_ms / 1000)
    counts = sorted({t for r in res.values() for t in r})
    print(f"{len(combos)} room scenes (empty, 1 person, overlaid {opt('--people', '1,2,3')} people); "
          f"{size} px model, one pass {opt('--pass-ms', 600)} ms")
    print(f"{'counting logic':40s} {'run s':>6s} " + " ".join(f"{'ok@' + str(t):>6s}" for t in counts)
          + f" {'mean':>6s} {'2->1':>5s} {'delay s':>8s}")
    for name, spec in LOG.items():
        acc = [res[name][t][0] / res[name][t][1] for t in counts]
        two = two_as_one[name][0] / max(two_as_one[name][1], 1)
        n = " ".join(f"{v:6.2f}" for v in acc)
        d = np.median(delays[name]) if delays[name] else float("nan")
        print(f"{name:40s} {spec[5] / 1000:6.2f} {n} {np.mean(acc):6.3f} {two:5.2f} {d:8.2f}", flush=True)
    print("runs per true count (first logic):", {t: res[next(iter(LOG))][t][1] for t in counts})


if __name__ == "__main__":
    main()
