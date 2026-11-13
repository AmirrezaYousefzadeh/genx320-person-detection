"""Per-operator time of a benchmark build on the board (make NN_BENCH=1 ...): reads nn_bench over SWD and names each
profiled operator from the model files the build embeds. With SPARSE_L1 the first entry is the event-driven first
layer (App/sparse_conv1.c), then the second convolution alone (part A2) and part B; the bench's exactness check runs
part A and part B again afterwards, those entries are dropped. With HEAD_FLOAT the float head is not a TFLM operator:
it is the rest of the run time (with the copies between the parts).
Usage: bench_ops.py ELF MODEL SIZE [--sparse] [--hf] [--json OUT.json --name NAME]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nn_bench import read_symbol  # noqa: E402

RUNS, MAX_OPS = 3, 32
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "firmware/PersonDetection/models"


def ops_of(path):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(str(path))
    it.allocate_tensors()
    T = {t["index"]: t for t in it.get_tensor_details()}
    out = []
    for op in it._get_ops_details():
        if op["op_name"] == "DELEGATE":
            continue
        shp = T[op["outputs"][0]]["shape"].tolist() if len(op["outputs"]) else []
        macs = 0
        if op["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D") and len(op["inputs"]) > 1:
            w = T[op["inputs"][1]]["shape"].tolist()  # conv [out_c, kh, kw, in_c]; depthwise [1, kh, kw, c]
            oh, ow, oc = shp[1], shp[2], shp[3]
            macs = oh * ow * oc * w[1] * w[2] * (w[3] if op["op_name"] == "CONV_2D" else 1)
        out.append((f"{op['op_name']} {'x'.join(str(v) for v in shp[1:])}", int(macs)))
    return out


def main(elf, model, size, sparse=False, hf=False):
    raw = read_symbol(elf, "nn_bench")
    h = np.frombuffer(raw, dtype="<u4", count=5 + RUNS + 3 + MAX_OPS)
    magic, status, cpu_hz = int(h[0]), int(np.int32(h[1])), int(h[2])
    if magic != 0x32424E4E:
        sys.exit(f"bench not finished (magic {magic:#x})")
    runs = (h[5:5 + RUNS] / cpu_hz * 1e3).tolist()
    n_ops = int(h[5 + RUNS])
    cyc = h[8 + RUNS:8 + RUNS + MAX_OPS] / cpu_hz * 1e3
    if sparse:  # the event-driven layer computes layer 1 of part A (its multiply-adds are those of the dense layer)
        l1 = ops_of(MODELS / f"{model}_part_a.tflite")[1][1]
        names = [("SPARSE_CONV1 (event-driven layer 1)", l1)] + ops_of(MODELS / f"{model}_part_a2.tflite")
    else:
        names = ops_of(MODELS / f"{model}_part_a.tflite")
    names += ops_of(MODELS / f"{model}_part_b.tflite")
    ops = [{"name": n, "macs": m, "ms": round(float(cyc[i]), 3)} for i, (n, m) in enumerate(names[:n_ops])]
    run = float(np.median(runs))
    res = {"elf": str(elf), "model": model, "size": int(size), "sparse": sparse, "hf": hf, "status": status,
           "run_ms": round(run, 2), "runs_ms": [round(r, 2) for r in runs], "ops": ops,
           "rest_ms": round(run - sum(o["ms"] for o in ops), 2),
           "l1_mismatch": int(h[6 + RUNS]) if sparse else None, "out_diff": int(h[7 + RUNS]) / 1000 if sparse else None}
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    r = main(a[0], a[1], a[2], "--sparse" in a, "--hf" in a)
    print(f"{r['model']} {r['size']} px {'sparse' if r['sparse'] else 'dense'}: run {r['run_ms']} ms, ops "
          f"{sum(o['ms'] for o in r['ops']):.1f} ms, rest {r['rest_ms']} ms; first ops: "
          + ", ".join(f"{o['name']} {o['ms']:.1f}" for o in r["ops"][:5])
          + (f"; check: {r['l1_mismatch']} mismatches, output diff {r['out_diff']}" if r["sparse"] else ""))
    if "--json" in a:
        p = Path(a[a.index("--json") + 1])
        allr = json.loads(p.read_text()) if p.exists() else {}
        allr[a[a.index("--name") + 1]] = r
        p.write_text(json.dumps(allr, indent=1))
