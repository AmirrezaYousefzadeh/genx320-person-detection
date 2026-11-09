"""On-board timing and output check of a float-head build (make NN_BENCH=1 HEAD_FLOAT=1): reads nn_bench over SWD
(time per network run, arena use, the float head's output on the built-in test image) and compares the output with
the PC (the int8 body with TFLite's reference kernels + the float head in numpy) on the same input bytes.
Usage: bench_hf.py ELF BODY_int8.tflite HEAD.npz SIZE [TEST_IMAGE.rgb]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nn_bench import read_symbol  # noqa: E402

RUNS, MAX_OPS = 3, 32
ROOT = Path(__file__).resolve().parent.parent


def main(elf, body, head, size, test=None):
    size = int(size)
    raw = read_symbol(elf, "nn_bench")
    h = np.frombuffer(raw, dtype="<u4", count=5 + RUNS + 3 + MAX_OPS)
    magic, status, cpu_hz, arena = int(h[0]), int(np.int32(h[1])), int(h[2]), h[3:5]
    if magic != 0x32424E4E:
        sys.exit(f"bench not finished (magic {magic:#x})")
    runs = h[5:5 + RUNS] / cpu_hz * 1e3
    out_len = (size // 16) ** 2 * 30
    out = np.frombuffer(raw, dtype="<f4", offset=4 * (8 + RUNS + MAX_OPS), count=out_len)
    print(f"status {status}; per network run (int8 body + float head): " + ", ".join(f"{r:.1f} ms" for r in runs)
          + f"; arena A {arena[0] / 1024:.0f} KiB, B {arena[1] / 1024:.0f} KiB")
    # the same input bytes on the PC: the first size*size*3 bytes of the built-in test image
    img = np.fromfile(test or ROOT / "firmware/PersonDetection/test/room_192x192.rgb", np.uint8)[:size * size * 3]
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    it = Interpreter(body, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], img.reshape(1, size, size, 3))
    it.invoke()
    s, z = o["quantization"]
    hw = np.load(head)
    ref = (((it.get_tensor(o["index"])[0].astype(np.float32) - z) * s) @ hw["W"] + hw["b"]).reshape(-1)
    d = np.abs(out - ref)
    print(f"float head output vs PC: max |difference| {d.max():.5f}, mean {d.mean():.6f} over {out_len} values")


if __name__ == "__main__":
    main(*sys.argv[1:])
