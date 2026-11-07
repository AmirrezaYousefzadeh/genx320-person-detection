"""Split a single-subgraph .tflite model in two after its first N operators, and check the parts on the PC.

Why: the first layers of st_yololcv1_192 need a 258 KiB arena (too big for internal SRAM, so it lives in slow
SDRAM), the rest needs about 110 KiB. With two models the firmware can give the second part a fast SRAM arena.
Each part keeps only its own tensors, buffers and operator codes; signatures and metadata are dropped (TFLM
does not use them). The check runs both parts with the TFLite reference kernels on a test input and requires the
same output as the full model, bit for bit.
Usage: split_model.py <model.tflite> <n_ops_first_part> <out_a.tflite> <out_b.tflite> <test_input.bin>
"""
import copy
import sys

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def make_part(model, ops, inputs, outputs):
    m = copy.deepcopy(model)
    sg = m.subgraphs[0]
    sg.operators = copy.deepcopy(ops)
    used = sorted(set(inputs) | set(outputs) | {t for op in ops for t in list(op.inputs) + list(op.outputs) if t >= 0})
    remap = {old: new for new, old in enumerate(used)}
    sg.tensors = [sg.tensors[i] for i in used]
    for op in sg.operators:
        op.inputs = [remap[t] if t >= 0 else -1 for t in op.inputs]
        op.outputs = [remap[t] for t in op.outputs]
    sg.inputs = [remap[t] for t in inputs]
    sg.outputs = [remap[t] for t in outputs]

    used_buffers = sorted({0} | {t.buffer for t in sg.tensors})
    bremap = {old: new for new, old in enumerate(used_buffers)}
    m.buffers = [m.buffers[b] for b in used_buffers]
    for t in sg.tensors:
        t.buffer = bremap[t.buffer]

    used_codes = sorted({op.opcodeIndex for op in sg.operators})
    cremap = {old: new for new, old in enumerate(used_codes)}
    m.operatorCodes = [m.operatorCodes[c] for c in used_codes]
    for op in sg.operators:
        op.opcodeIndex = cremap[op.opcodeIndex]

    m.signatureDefs, m.metadata, m.metadataBuffer = [], [], []
    builder = flatbuffers.Builder(1024)
    builder.Finish(m.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def run(model_bytes, x):
    it = Interpreter(model_content=model_bytes, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])


def main(path, n_first, out_a, out_b, test_input):
    full = open(path, "rb").read()
    model = schema.ModelT.InitFromPackedBuf(full, 0)
    sg = model.subgraphs[0]
    cut = sg.operators[n_first - 1].outputs[0]
    a = make_part(model, sg.operators[:n_first], list(sg.inputs), [cut])
    b = make_part(model, sg.operators[n_first:], [cut], list(sg.outputs))
    open(out_a, "wb").write(a)
    open(out_b, "wb").write(b)

    side = int(sg.tensors[sg.inputs[0]].shape[1])  # 192, 160, 128, ...
    x = np.fromfile(test_input, dtype=np.uint8).reshape(1, side, side, 3)
    mid = run(a, x)
    same = np.array_equal(run(b, mid), run(full, x))
    print(f"part A: {len(a)} bytes, output {mid.shape} {mid.dtype}; part B: {len(b)} bytes; "
          f"A then B equals the full model: {same}")
    if not same:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *sys.argv[3:6])
