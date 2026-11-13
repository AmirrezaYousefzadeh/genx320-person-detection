"""Paired comparison of two networks on the same test set: the AP50 difference B - A with a bootstrap over the test
segments, resampled jointly for both networks (the same segments in every resample), so that the scene-to-scene spread
that both networks share cancels. Input: the per-segment dumps that tools/eval_ap.py writes with EVAL_DUMP=FILE
(tools/eval_suite.py with EVAL_DUMP_DIR=DIR).
Usage: paired_bootstrap.py A.json B.json [--n 2000]   prints AP50 of A and B, the difference, its 95 % interval and the
       share of resamples in which B is better.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_ap import ap_from  # noqa: E402,F401  (ap_fast reproduces it on arrays)


def ap_fast(scores, tps, n_gt):
    """eval_ap.ap_from (COCO 101-point AP) on arrays"""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores, kind="stable")
    tp = np.cumsum(tps[o])
    fp = np.cumsum(1 - tps[o])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    return float(np.mean(np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)))


def arrays(segs):
    return {k: (np.array([d[0] for d in v["dets"]], np.float64), np.array([d[1] for d in v["dets"]], np.float64), v["n"])
            for k, v in segs.items()}


def ap_of(arr, keys):
    sc = np.concatenate([arr[k][0] for k in keys]) if keys else np.zeros(0)
    tp = np.concatenate([arr[k][1] for k in keys]) if keys else np.zeros(0)
    return ap_fast(sc, tp, sum(arr[k][2] for k in keys))


def compare(fa, fb, n=2000, seed=0):
    a, b = arrays(json.loads(Path(fa).read_text())), arrays(json.loads(Path(fb).read_text()))
    keys = sorted(set(a) & set(b))
    if set(a) != set(b):
        print(f"warning: segments differ ({len(a)} vs {len(b)}); using the {len(keys)} common ones", file=sys.stderr)
    base_a, base_b = ap_of(a, keys), ap_of(b, keys)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        diffs.append(ap_of(b, pick) - ap_of(a, pick))
    d = np.array(diffs)
    return {"A": round(base_a, 4), "B": round(base_b, 4), "diff": round(base_b - base_a, 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "p_B_better": round(float((d > 0).mean()), 3), "segments": len(keys)}


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[args.index("--n") + 1]) if "--n" in args else 2000
    files = [x for x in args if x.endswith(".json")]
    print(json.dumps(compare(files[0], files[1], n)))
