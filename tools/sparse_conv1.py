"""PC reference of the first convolution of st_yololcv1 (3x3, stride 2, SAME, ReLU, int8) written the sparse way:
the "nothing" pixel is exactly 0 after quantisation (grey 128, zero point -128), so the output is the bias
everywhere except around the active pixels, whose 3x3x16 contributions are scattered. Bit-exact with TFLite's
reference kernel (same per-channel requantisation: doubling high multiply, rounding shift).

Usage: sparse_conv1.py MODEL.tflite [--n 5]   checks exactness on validation pictures (at the model's input size N)
and prints the active-pixel statistics that set the kernel's cost. Also exports the kernel's constants as a C header
with --header FILE (firmware/PersonDetection/models/<MODEL>_conv1.h, used by make SPARSE_L1=1). MODEL is the whole
network or its part A (the first layers are the same).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def quantize_multiplier(scale):
    """TFLite QuantizeMultiplier: scale = m * 2^shift, m in [0.5, 1) as int32 (<< 31)"""
    if scale == 0:
        return 0, 0
    m, shift = np.frexp(scale)
    q = int(round(m * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        shift += 1
    return q, int(shift)


def requant(acc, mult, shift):
    """MultiplyByQuantizedMultiplier for int64 accumulators (vectorised): SaturatingRoundingDoublingHighMul then
    RoundingDivideByPOT, as in gemmlowp / TFLite reference."""
    acc = acc.astype(np.int64)
    left = max(shift, 0)
    right = max(-shift, 0)
    x = acc * (1 << left)
    # doubling high mul with rounding
    ab = x * np.int64(mult)
    nudge = np.where(ab >= 0, np.int64(1 << 30), np.int64(1 - (1 << 30)))
    hi = (ab + nudge) >> 31
    if right:
        mask = (np.int64(1) << right) - 1
        rem = hi & mask
        thr = (mask >> 1) + np.where(hi < 0, 1, 0)
        hi = (hi >> right) + np.where(rem > thr, 1, 0)
    return hi


class Conv1:
    def __init__(self, model_path):
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        # TFLite's reference kernels (as CMSIS-NN on the board), not the XNNPACK delegate, which rounds differently
        self.it = Interpreter(model_path, experimental_preserve_all_tensors=True,
                              experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        self.it.allocate_tensors()
        ops = [op for op in self.it._get_ops_details() if op["op_name"] != "DELEGATE"]
        conv = ops[1]
        T = {t["index"]: t for t in self.it.get_tensor_details()}
        w_t, b_t, in_t, out_t = T[conv["inputs"][1]], T[conv["inputs"][2]], T[conv["inputs"][0]], T[conv["outputs"][0]]
        self.w = self.it.get_tensor(w_t["index"]).astype(np.int32)  # [16, 3, 3, 3]
        self.wsum = self.w.sum(-1)  # the three input channels are the same picture: [16, 3, 3]
        self.bias = self.it.get_tensor(b_t["index"]).astype(np.int64)
        in_scale, out_scale = float(in_t["quantization_parameters"]["scales"][0]), float(out_t["quantization_parameters"]["scales"][0])
        self.out_zp = int(out_t["quantization_parameters"]["zero_points"][0])
        self.in_zp = int(in_t["quantization_parameters"]["zero_points"][0])
        self.mult, self.shift = zip(*(quantize_multiplier(in_scale * float(s) / out_scale) for s in w_t["quantization_parameters"]["scales"]))
        self.out_index = out_t["index"]
        self.in_index = self.it.get_input_details()[0]["index"]
        self.n = int(self.it.get_input_details()[0]["shape"][1])  # the input side N (192, 160, 128)
        self.m = self.n // 2
        # The conv works on (int8 - input zero point) = the uint8 pixel itself: nothing 128, ON 255, OFF 0, and the
        # padding counts as 0. An all-"nothing" interior window therefore gives bias + 128 * sum of all taps; at the
        # last row/column (SAME padding puts the one pad row/column after) the taps outside are missing.
        self.wtap = self.wsum.sum(axis=(1, 2))  # [16]
        self.acc0 = self.bias + 128 * self.wtap
        self.const = self.finish(self.acc0[None, None, :])

    def finish(self, acc):
        out = np.empty(acc.shape, np.int64)
        for c in range(acc.shape[-1]):
            out[..., c] = requant(acc[..., c], self.mult[c], self.shift[c]) + self.out_zp
        return np.clip(out, self.out_zp, 127).astype(np.int8)  # ReLU: max(out, zp) then int8 saturation

    def dense(self, grey):
        """straightforward reference: full conv on the int8 picture (pad after only: SAME with stride 2, N -> N/2)"""
        n, m = self.n, self.m
        xp = np.zeros((n + 1, n + 1), np.int32)  # (int8 - zp) = the uint8 value; the pad row/column after is 0
        xp[:n, :n] = grey.astype(np.int32)
        acc = np.zeros((m, m, 16), np.int64) + self.bias
        for ky in range(3):
            for kx in range(3):
                patch = xp[ky:ky + n:2, kx:kx + n:2]  # [m, m]
                acc += patch[..., None] * self.wsum[:, ky, kx][None, None, :]
        return self.finish(acc)

    def sparse(self, grey):
        """the event-driven way: start from the constant, scatter the active pixels"""
        m = self.m
        acc = np.broadcast_to(self.acc0, (m, m, 16)).copy()
        acc[m - 1, :, :] -= 128 * self.wsum[:, 2, :].sum(-1)   # last output row: tap ky = 2 is padding
        acc[:, m - 1, :] -= 128 * self.wsum[:, :, 2].sum(-1)   # last output column: tap kx = 2 is padding
        acc[m - 1, m - 1, :] += 128 * self.wsum[:, 2, 2]          # the corner lost (2,2) twice
        ys, xs = np.nonzero(grey != 128)
        vals = grey[ys, xs].astype(np.int64) - 128
        for y, x, v in zip(ys, xs, vals):
            # input (y, x) is tap (ky, kx) of output (oy, ox) when y = 2*oy + ky, x = 2*ox + kx
            for ky in range(3):
                oy2 = y - ky
                if oy2 < 0 or oy2 % 2:
                    continue
                oy = oy2 // 2
                if oy >= m:
                    continue
                for kx in range(3):
                    ox2 = x - kx
                    if ox2 < 0 or ox2 % 2:
                        continue
                    ox = ox2 // 2
                    if ox >= m:
                        continue
                    acc[oy, ox] += v * self.wsum[:, ky, kx]
        return self.finish(acc)

    def tflite(self, grey):
        rgb = np.repeat(grey[..., None], 3, -1)[None].astype(np.uint8)
        self.it.set_tensor(self.in_index, rgb)
        self.it.invoke()
        return self.it.get_tensor(self.out_index)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--header")
    args = ap.parse_args()
    import cv2
    import finetune as ft
    k = Conv1(args.model)
    root = Path(__file__).resolve().parent.parent / "work"
    _, val = ft.list_samples(",".join(str(root / f"dataset_v{i}") for i in (1, 2, 3)), ["09", "11"])
    rng = np.random.default_rng(1)
    src = (np.arange(k.n) * 320 // k.n)  # the board's nearest-neighbour stretch (detector.c make_input)
    ok = True
    for i in rng.choice(len(val), args.n, replace=False):
        g = cv2.imread(val[i][0], cv2.IMREAD_GRAYSCALE)[src][:, src]
        ref, dense, sparse = k.tflite(g), k.dense(g), k.sparse(g)
        ok &= np.array_equal(sparse, ref)
        active = int((g != 128).sum())
        print(f"{Path(val[i][0]).parent.parent.name}/{Path(val[i][0]).name}: active {active} of {k.n * k.n} ({100 * active / k.n ** 2:.1f} %), "
              f"dense == tflite: {np.array_equal(dense, ref)}, sparse == tflite: {np.array_equal(sparse, ref)}, "
              f"max diff {np.abs(sparse.astype(int) - ref.astype(int)).max()}")
    if not ok:
        sys.exit("the sparse kernel is NOT bit-exact with TFLite on this model: no header written")
    if args.header:
        with open(args.header, "w") as f:
            f.write(f"/* Generated by tools/sparse_conv1.py from {Path(args.model).name} ({k.n} px): constants of the first\n"
                    "   convolution for firmware/App/sparse_conv1.c. Do not edit. */\n#pragma once\n#include <stdint.h>\n")
            f.write(f"#define CONV1_OUT_ZP ({k.out_zp})\n")
            f.write("/* bias + 128 * (sum of all taps): the accumulator of an interior window with no event */\n")
            f.write("static const int32_t conv1_acc0[16] = {" + ", ".join(str(int(b)) for b in k.acc0) + "};\n")
            f.write("static const int32_t conv1_mult[16] = {" + ", ".join(str(m) for m in k.mult) + "};\n")
            f.write("static const int32_t conv1_shift[16] = {" + ", ".join(str(s) for s in k.shift) + "};\n")
            f.write("static const int8_t conv1_const[16] = {" + ", ".join(str(int(c)) for c in k.const[0, 0]) + "};\n")
            f.write("/* summed over the three identical input channels: [ky][kx][c] */\nstatic const int16_t conv1_w[3][3][16] = {\n")
            for ky in range(3):
                f.write("  {" + ", ".join("{" + ", ".join(str(int(k.wsum[c, ky, kx])) for c in range(16)) + "}" for kx in range(3)) + "},\n")
            f.write("};\n")
        print("wrote", args.header)
