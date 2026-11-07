"""How sparse are the activations of the int8 person network on event pictures? Per operator: the fraction of
output values equal to the tensor's zero point (true zeros), and the fraction of 8x8 output tiles that are
constant (every value equal: a tile that saw no event gives the bias through the activation, the same everywhere).

Usage: act_sparsity.py MODEL.tflite [--data a,b] [--n 100] [--tile 8]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--data", default=",".join(f"work/dataset_v{i}" for i in (1, 2, 3)))
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--tile", type=int, default=8)
args = ap.parse_args()
import finetune as ft  # noqa: E402
from ai_edge_litert.interpreter import Interpreter  # noqa: E402

train, val = ft.list_samples(args.data, ["09", "11"])
rng = np.random.default_rng(0)
pics = [val[i][0] for i in rng.choice(len(val), min(args.n, len(val)), replace=False)]
it = Interpreter(args.model, experimental_preserve_all_tensors=True)
it.allocate_tensors()
inp = it.get_input_details()[0]
ops = [op for op in it._get_ops_details() if op["op_name"] != "DELEGATE"]
tensors = {t["index"]: t for t in it.get_tensor_details()}
acc = {}
for p in pics:
    img = ft.load_picture(p).numpy().astype(np.uint8)[None]
    it.set_tensor(inp["index"], img)
    it.invoke()
    for op in ops:
        for ti in op["outputs"]:
            t = tensors[ti]
            if t["dtype"] not in (np.int8, np.uint8):
                continue
            a = it.get_tensor(ti)
            zp = t["quantization_parameters"]["zero_points"]
            zp = int(zp[0]) if len(zp) else 0
            zeros = float((a == zp).mean())
            words = pixels = np.nan
            if a.ndim == 4 and a.shape[-1] % 4 == 0:  # int8 NHWC: 4 channels = one 32-bit word = one SMLAD pair
                z = (a[0] == zp)
                words = float(z.reshape(z.shape[0], z.shape[1], -1, 4).all(-1).mean())
                pixels = float(z.all(-1).mean())
            const = np.nan
            if a.ndim == 4 and a.shape[1] >= args.tile:
                h, w, c = a.shape[1] // args.tile * args.tile, a.shape[2] // args.tile * args.tile, a.shape[3]
                b = a[0, :h, :w].reshape(h // args.tile, args.tile, w // args.tile, args.tile, c).transpose(0, 2, 1, 3, 4)
                b = b.reshape(h // args.tile * (w // args.tile), args.tile * args.tile, c)
                const = float((b == b[:, :1]).all(axis=(1, 2)).mean())
            k = (op["index"], op["op_name"], tuple(a.shape))
            s = acc.setdefault(k, [0.0, 0.0, 0, 0.0, 0.0])
            s[0] += zeros; s[1] += 0 if np.isnan(const) else const; s[2] += 1
            s[3] += 0 if np.isnan(words) else words; s[4] += 0 if np.isnan(pixels) else pixels
print(f"{len(pics)} validation pictures, tile {args.tile}x{args.tile}")
print("op  name                shape              zero values   zero 4-ch words   zero pixels   constant tiles")
for (i, name, shape), (z, c, n, w, px) in sorted(acc.items()):
    print(f"{i:2d}  {name:18s}  {str(shape):18s}  {100 * z / n:6.1f} %     {100 * w / n:6.1f} %          {100 * px / n:6.1f} %      {100 * c / n:6.1f} %")
