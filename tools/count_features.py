"""The detector's output grid (12x12 cells x 5 anchors x 6 values, int8 model, mirrored pass fused as on the board)
for every sample of a counting index, saved for training counting heads on top of the detector (count_head.py).
Usage: count_features.py INDEX.jsonl MODEL.tflite OUT.npz [--max-train 12000]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from count_eval import GRID, NA, region_input  # noqa: E402


def main(index, model, out, max_train=12000):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]

    def run(x):
        it.set_tensor(inp["index"], x[None].astype(np.uint8))
        it.invoke()
        return it.get_tensor(outp["index"])[0].reshape(GRID, GRID, NA, 6).copy()

    rows = [json.loads(l) for l in open(index)]
    rng = np.random.default_rng(0)
    train = [r for r in rows if r["split"] == "train"]
    if len(train) > max_train:
        train = [train[i] for i in sorted(rng.choice(len(train), max_train, replace=False))]
    sel = train + [r for r in rows if r["split"] == "val"]
    single, fused, counts, split = [], [], [], []
    for k, r in enumerate(sel):
        x = region_input(r["png"], r["region"])
        a = run(x)
        b = run(x[:, ::-1].copy())[:, ::-1].copy()
        b[..., 0] = -b[..., 0]
        single.append(a.astype(np.float16))
        fused.append(((a + b) / 2).astype(np.float16))
        counts.append(r["count"])
        split.append(r["split"] == "val")
        if k % 2000 == 0:
            print(f"{k} / {len(sel)}", flush=True)
    np.savez_compressed(out, single=np.array(single), fused=np.array(fused), count=np.array(counts),
                        val=np.array(split))
    print(f"wrote {out}: {len(sel)} samples")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], int(a[a.index("--max-train") + 1]) if "--max-train" in a else 12000)
