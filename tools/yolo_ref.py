"""PC reference for st_yololcv1_192: run the int8 model with the TFLite reference kernels, decode the boxes, draw them.

Preprocessing as in ST's config: RGB, resize to 192x192 ignoring the aspect ratio ("fit"), nearest neighbour, uint8.
Decoding as in ST's postprocess.py (YOLOv2 head, 5 anchors, 1 class) followed by NMS.
Usage: yolo_ref.py <model.tflite> <image> <out_prefix>
  writes <out_prefix>_input.bin (192x192x3 uint8), <out_prefix>_output.npy (12x12x30 float32), <out_prefix>_boxes.png
"""
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from PIL import Image, ImageDraw

SIZE, GRID, N_ANCHORS = 192, 12, 5
ANCHORS = np.array([0.076023, 0.258508, 0.163031, 0.413531, 0.234769, 0.702585, 0.427054, 0.715892, 0.748154,
                    0.857092], np.float32).reshape(N_ANCHORS, 2)


def preprocess(path):
    return np.asarray(Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.NEAREST), dtype=np.uint8)


def run(model, image):
    it = Interpreter(model, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], image[None])
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decode(out, threshold=0.5, nms_iou=0.5):
    """Return [(score, x0, y0, x1, y1)] in 0..1 image coordinates, best first."""
    g = int(round((out.size / (N_ANCHORS * 6)) ** 0.5))  # 12 for 192 px, 10 for 160, 8 for 128
    f = out.reshape(g, g, N_ANCHORS, 6)
    col, row = np.meshgrid(np.arange(g), np.arange(g))
    x = (sigmoid(f[..., 0]) + col[..., None]) / g
    y = (sigmoid(f[..., 1]) + row[..., None]) / g
    w = np.exp(f[..., 2]) * ANCHORS[:, 0] / g
    h = np.exp(f[..., 3]) * ANCHORS[:, 1] / g
    score = sigmoid(f[..., 4])  # softmax over a single class is 1
    boxes = np.stack([score, x - w / 2, y - h / 2, x + w / 2, y + h / 2], -1).reshape(-1, 5)
    boxes[:, 1:] = boxes[:, 1:].clip(0, 1)
    boxes = boxes[boxes[:, 0] >= threshold]
    boxes = boxes[np.argsort(-boxes[:, 0])]
    keep = []
    for b in boxes:
        if all(iou(b[1:], k[1:]) < nms_iou for k in keep):
            keep.append(b)
    return keep


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def draw(path, boxes, out_png):
    img = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(img)
    for s, x0, y0, x1, y1 in boxes:
        d.rectangle([x0 * img.width, y0 * img.height, x1 * img.width, y1 * img.height], outline=(255, 0, 0), width=3)
        d.text((x0 * img.width + 4, y0 * img.height + 2), f"{s:.2f}", fill=(255, 0, 0))
    img.save(out_png)


if __name__ == "__main__":
    model, path, prefix = sys.argv[1:4]
    image = preprocess(path)
    out = run(model, image)
    image.tofile(prefix + "_input.bin")
    np.save(prefix + "_output.npy", out)
    boxes = decode(out)
    draw(path, boxes, prefix + "_boxes.png")
    for b in boxes:
        print("person %.2f  box x0=%.2f y0=%.2f x1=%.2f y1=%.2f" % tuple(b))
