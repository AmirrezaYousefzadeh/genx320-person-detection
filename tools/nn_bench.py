"""Read the person-detector benchmark (nn_bench in person_nn.cc) from the board over SWD and check it against the PC.

Prints the time per run and per operator, compares the board output with the PC reference output of
tools/yolo_ref.py (same model, same test image, TFLite reference kernels) and decodes the boxes.
Usage: nn_bench.py <elf> <model.tflite> <reference_output.npy>
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from yolo_ref import decode  # noqa: E402

RUNS, MAX_OPS, OUT_LEN = 3, 32, 12 * 12 * 30


def read_symbol(elf, name):
    out = subprocess.run(["arm-none-eabi-nm", "-S", elf], capture_output=True, text=True, check=True).stdout
    addr, size = next((int(p[0], 16), int(p[1], 16)) for p in (l.split() for l in out.splitlines())
                      if len(p) == 4 and p[3] == name)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "sym.raw"
        subprocess.run(["openocd", "-f", "board/stm32f7discovery.cfg", "-c",
                        f"init; halt; dump_image {raw} {addr:#x} {size}; resume; shutdown"],
                       capture_output=True, check=True)
        return raw.read_bytes()


def op_names(model):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model)
    it.allocate_tensors()
    return [op["op_name"] for op in it._get_ops_details() if op["op_name"] != "DELEGATE"]


def main(elf, model, ref_npy):
    raw = read_symbol(elf, "nn_bench")
    head = np.frombuffer(raw, dtype="<u4", count=5 + RUNS + 3 + MAX_OPS)
    magic, status, cpu_hz, arena = int(head[0]), int(np.int32(head[1])), int(head[2]), head[3:5]
    runs = head[5:5 + RUNS]
    n_ops = int(head[5 + RUNS])
    l1_mismatch = int(head[6 + RUNS])
    out_diff = int(head[7 + RUNS]) / 1000
    op_cycles = head[8 + RUNS:8 + RUNS + MAX_OPS]
    out_len = (len(raw) - 4 * (8 + RUNS + MAX_OPS)) // 4  # the model's output (12x12x30 at 192, less at 160/128)
    out = np.frombuffer(raw, dtype="<f4", offset=4 * (8 + RUNS + MAX_OPS), count=out_len)
    if magic != 0x32424E4E:
        sys.exit(f"bench not finished (magic {magic:#x})")
    print(f"status {status}, CPU {cpu_hz / 1e6:.0f} MHz, arena used: part A {arena[0] / 1024:.1f} KiB (SDRAM), "
          f"part B {arena[1] / 1024:.1f} KiB (SRAM)")
    print("per run: " + ", ".join(f"{c / cpu_hz * 1e3:.1f} ms" for c in runs))
    names = op_names(model)
    print(f"ops profiled {n_ops} (model has {len(names)}); conv2 output mismatches {l1_mismatch}; final output max diff between the paths {out_diff}")
    if n_ops != len(names):  # SPARSE_L1 build: our kernel replaces QUANTIZE + the first CONV_2D (and the check adds part A's ops)
        names = ["SPARSE_CONV1"] + names[2:] + ["(check) " + n for n in names[:3]]
    for i in range(n_ops):
        name = names[i] if i < len(names) else "?"
        print(f"  op {i:2d} {name:18s} {op_cycles[i] / cpu_hz * 1e3:7.2f} ms")
    ref = np.load(ref_npy).reshape(-1)
    diff = np.abs(out - ref)
    print(f"output vs PC: {np.sum(diff == 0)}/{out_len} values identical, max abs difference {diff.max():.6f}")
    for b in decode(out.reshape(12, 12, 30)):
        print("board: person %.2f  box x0=%.2f y0=%.2f x1=%.2f y1=%.2f" % tuple(b))


if __name__ == "__main__":
    main(*sys.argv[1:4])
