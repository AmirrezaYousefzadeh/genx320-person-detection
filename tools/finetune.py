"""Fine-tune st_yololcv1_192 (ST model zoo, COCO person) on the event pictures of the recorded dataset.

Data: work/dataset_v1/<segment>/pictures/pic_NNNN.png (320x320 grey: 128 nothing, 255 ON, 0 OFF) with
labels/pic_NNNN.txt (YOLO "0 cx cy w h"). The pictures are stretched to 192x192 and the grey copied to R, G, B,
exactly as the firmware does. Validation = the segments of the sequences named in --val (default 09 and 11),
training = the rest.
Loss: YOLOv2 for one class. Each truth box goes to the cell of its centre and the anchor whose shape fits
best; the head predicts (tx, ty, tw, th, to, class): sigmoid(tx)+col over 12 = x, exp(tw)*anchor/12 = w,
sigmoid(to) = objectness. Weights: coordinates 5, object 1, no-object 0.5 (predictions with IoU > 0.6 to any
truth are not punished), the class output is ignored (single class, softmax of one = 1).
Output in --out: best.keras (lowest validation loss), float.tflite, int8.tflite (quantised with training pictures,
uint8 input / float output like ST's), metrics.json (precision/recall/F1 at IoU 0.5, score > 0.6, on the
validation pictures, before and after fine-tuning, float and int8), log.txt.

Usage: finetune.py --data work/dataset_v1[,work/dataset_v2,...] --model work/models_zoo/st_yololcv1_192.keras
                   --out work/finetune/v1 [--epochs 60] [--batch 32] [--lr 1e-4] [--val 09,11] [--patience 15]
                   [--size 160] [--exclude KEYS] [--aug-scale 0.3] [--aug-shift 0.1] [--small drop|ignore]
                   [--cpu] [--limit N]          (all options and their meaning: finetune.py --help)
--size is read before anything else (it sets the grid of the loss), so it must be given on the command line.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from yolo_ref import ANCHORS, decode, iou as box_iou  # noqa: E402
import evpic  # noqa: E402

SIZE, GRID, NA = 192, 12, 5  # --size changes SIZE and GRID (= SIZE / 16) before anything else runs
if "--size" in sys.argv:
    SIZE = int(sys.argv[sys.argv.index("--size") + 1])
    GRID = SIZE // 16
    import yolo_ref  # its decode() reads these at call time
    yolo_ref.SIZE, yolo_ref.GRID = SIZE, GRID
A = np.array(ANCHORS, np.float32).reshape(NA, 2)  # (w, h) in grid cells


# ---- data --------------------------------------------------------------------------------------------

MIN_PERSON_H = 0.12  # of the picture height (38 px of 320): smaller people are below what the sensor resolves
AUG = (0.0, 0.0)  # --aug-scale, --aug-shift (set in main)
MIN_H_AUG = 0.0  # a person shrunk below this by the augmentation becomes a region to ignore (= --min-h with --small ignore)


def timing_uncertain(seg):
    """a monitor segment whose labels are timed only by the START command (no marker fit, no clear motion fit:
    +-0.5 s or worse); N-People keeps it out of training and testing (curation/flags.json)"""
    ej = Path(seg) / "events.json"
    if not ej.exists():
        return False
    return json.loads(ej.read_text()).get("sync", {}).get("method", "").startswith("start command")


def list_samples(roots, val_seqs, min_h=MIN_PERSON_H, small="drop", exclude_seqs=()):
    """roots: one directory or several separated by commas (e.g. work/dataset_v1,work/dataset_v2).
    val_seqs: MOT17 numbers ("09") or parts of segment names ("pedro_val"); those segments are validation.
    Label lines are "class cx cy w h": class 0 a person, class 1 a region to ignore (a reflection, a crowd, a person
    hardly visible). A person smaller than min_h: small="drop" leaves the picture out (it would teach "no person"
    on a person), small="ignore" keeps the picture and makes the person a region to ignore (tiny
    people only add noise). Pictures without people stay as negatives. Boxes are (class, cx, cy, w, h)."""
    train, val = [], []
    n_small = n_ign = 0
    segs = sorted(seg for root in str(roots).split(",") for seg in Path(root).glob("*_f*"))
    keys = [f"MOT17-{s}-" if s.isdigit() else s for s in val_seqs if s]
    ex_keys = [f"MOT17-{s}-" if s.isdigit() else s for s in exclude_seqs if s]
    for seg in segs:
        if any(k in seg.name for k in ex_keys):  # neither training nor validation (the test segments)
            continue
        if timing_uncertain(seg):  # labels not timed by a marker or the motion (since 2026-11-09)
            continue
        is_val = any(k in seg.name for k in keys)
        for png in sorted((seg / "pictures").glob("pic_*.png")):
            txt = seg / "labels" / (png.stem + ".txt")
            boxes = []
            if txt.exists():
                for line in txt.read_text().splitlines():
                    c, cx, cy, w, h = (float(v) for v in line.split()[:5])
                    boxes.append((int(c), cx, cy, w, h))
            if any(c == 0 and h < min_h for c, _, _, _, h in boxes):
                if small == "drop":
                    n_small += 1
                    continue
                n_ign += sum(c == 0 and h < min_h for c, _, _, _, h in boxes)
                boxes = [(1 if c == 0 and h < min_h else c, cx, cy, w, h) for c, cx, cy, w, h in boxes]
            (val if is_val else train).append((str(png), boxes))
    if n_small:
        print(f"{n_small} pictures with a person smaller than {min_h:.2f} of the height left out")
    if n_ign:
        print(f"{n_ign} people smaller than {min_h:.2f} of the height made regions to ignore")
    return train, val


def load_picture(path):
    if evpic.TEMPORAL > 1:  # current picture and the two before it (tools/evpic.py)
        img = tf.numpy_function(lambda q: evpic.network_input(q.decode() if isinstance(q, bytes) else str(q), SIZE)
                                .astype(np.float32), [path], tf.float32)
        img.set_shape((SIZE, SIZE, 3))
        return img
    img = tf.io.decode_png(tf.io.read_file(path), channels=1)
    img = tf.image.resize(img, (SIZE, SIZE), method="nearest")
    return tf.cast(tf.repeat(img, 3, axis=-1), tf.float32)  # 0..255 like the firmware's input


def encode_targets(boxes):
    """(12, 12, 5, 7): [tx_target(sigmoid space), ty, tw, th, obj, has_box, ignore] and the truth boxes (N, 4)
    corners. A class-1 box marks the cells whose centre it covers as ignore (no no-object loss there)."""
    y = np.zeros((GRID, GRID, NA, 7), np.float32)
    corners = []
    for c, cx, cy, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        if c != 0:
            c0, c1 = int(np.floor((cx - w / 2) * GRID - 0.5)) + 1, int(np.ceil((cx + w / 2) * GRID - 0.5))
            r0, r1 = int(np.floor((cy - h / 2) * GRID - 0.5)) + 1, int(np.ceil((cy + h / 2) * GRID - 0.5))
            y[max(r0, 0):min(r1, GRID), max(c0, 0):min(c1, GRID), :, 6] = 1.0
            y[min(max(int(cy * GRID), 0), GRID - 1), min(max(int(cx * GRID), 0), GRID - 1), :, 6] = 1.0  # small: its own cell
            corners.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            continue
        col, row = min(int(cx * GRID), GRID - 1), min(int(cy * GRID), GRID - 1)
        wh = np.array([w * GRID, h * GRID])
        inter = np.minimum(wh, A).prod(1)
        best = int(np.argmax(inter / (wh.prod() + A.prod(1) - inter)))
        y[row, col, best, :6] = [cx * GRID - col, cy * GRID - row, np.log(w * GRID / A[best, 0]),
                             np.log(h * GRID / A[best, 1]), 1.0, 1.0]
        corners.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return y, corners


MAX_BOXES = 64
DATA_SEED = 1  # the order of the training pictures (--seed changes it, and the random augmentations)


def make_dataset(samples, batch, training):
    """pictures and their targets; targets are encoded on the fly from the boxes (big sets do not fit as constants)"""
    if training:  # a global shuffle once; the buffer below only mixes neighbours
        order = np.random.default_rng(DATA_SEED).permutation(len(samples))
        samples = [samples[i] for i in order]
    paths = np.array([p for p, _ in samples])
    boxes = np.zeros((len(samples), MAX_BOXES, 5), np.float32)
    boxes[..., 0] = -1  # padding
    for i, (_, bs) in enumerate(samples):
        for j, b in enumerate(bs[:MAX_BOXES]):
            boxes[i, j] = b
    ds = tf.data.Dataset.from_tensor_slices((paths, boxes))
    if training:
        ds = ds.shuffle(min(len(samples), 20000), seed=DATA_SEED, reshuffle_each_iteration=True)

    def enc(b):
        y, corners = encode_targets([tuple(r) for r in b if r[0] >= 0])
        t = np.zeros((MAX_BOXES, 4), np.float32)
        for j, c in enumerate(corners[:MAX_BOXES]):
            t[j] = c
        return y, t

    def load(path, b):
        y, t = tf.numpy_function(enc, [b], [tf.float32, tf.float32])
        y.set_shape((GRID, GRID, NA, 7))
        t.set_shape((MAX_BOXES, 4))
        return load_picture(path), (y, t)

    def load_aug(path, b):  # random scale and shift (AUG), boxes follow; then the targets
        img, y, t = tf.numpy_function(scale_shift, [path, b], [tf.float32, tf.float32, tf.float32])
        img.set_shape((SIZE, SIZE, 3))
        y.set_shape((GRID, GRID, NA, 7))
        t.set_shape((MAX_BOXES, 4))
        return img, (y, t)

    def scale_shift(path, b):
        import cv2
        pic = evpic.picture_stack(path.decode())  # 3 channels (the same picture, or with --temporal the ones before)
        h, w = pic.shape[:2]
        s = np.exp(np.random.uniform(-np.log(1 + AUG[0]), np.log(1 + AUG[0])))
        tx, ty = np.random.uniform(-AUG[1], AUG[1], 2)
        # picture point (u, v) in 0..1 -> s * (u - 0.5) + 0.5 + t
        M = np.array([[s, 0, (0.5 - 0.5 * s + tx) * w], [0, s, (0.5 - 0.5 * s + ty) * h]], np.float32)
        out = cv2.warpAffine(pic, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(128, 128, 128))
        out = cv2.resize(out, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        boxes = []
        for c, cx, cy, bw, bh in (r for r in b if r[0] >= 0):
            x0, y0 = s * (cx - bw / 2 - 0.5) + 0.5 + tx, s * (cy - bh / 2 - 0.5) + 0.5 + ty
            x1, y1 = s * (cx + bw / 2 - 0.5) + 0.5 + tx, s * (cy + bh / 2 - 0.5) + 0.5 + ty
            cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, 1), min(y1, 1)
            if cx1 - cx0 < 0.01 or cy1 - cy0 < 0.01:
                continue
            kept = (cx1 - cx0) * (cy1 - cy0) / max((x1 - x0) * (y1 - y0), 1e-9)
            cls = c if kept >= 0.5 and not (c == 0 and (cy1 - cy0) < MIN_H_AUG) else 1  # cut or now small: ignore
            boxes.append((cls, (cx0 + cx1) / 2, (cy0 + cy1) / 2, cx1 - cx0, cy1 - cy0))
        y, corners = encode_targets(boxes)
        t = np.zeros((MAX_BOXES, 4), np.float32)
        for j, cc in enumerate(corners[:MAX_BOXES]):
            t[j] = cc
        return out.astype(np.float32), y, t

    ds = ds.map(load_aug if (training and AUG[0] + AUG[1] > 0) else load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def augment(img, yt):
    """Horizontal flip (labels follow) and a little extra event noise."""
    y, t = yt
    flip = tf.random.uniform([]) < 0.5

    def flipped():
        img_f = tf.image.flip_left_right(img)
        y_f = tf.reverse(y, axis=[1])  # columns
        tx = 1.0 - y_f[..., 0]  # centre within the cell mirrors
        y_f = tf.concat([tx[..., None], y_f[..., 1:]], -1)
        t_f = tf.stack([1.0 - t[:, 2], t[:, 1], 1.0 - t[:, 0], t[:, 3]], -1)
        t_f = tf.where(tf.reduce_any(t != 0, -1, keepdims=True), t_f, t)
        return img_f, y_f, t_f

    img, y, t = tf.cond(flip, flipped, lambda: (img, y, t))
    noise = tf.random.uniform(tf.shape(img)[:2]) < tf.random.uniform([], 0.0, 0.01)
    dots = tf.where(tf.random.uniform(tf.shape(img)[:2]) < 0.5, 255.0, 0.0)
    img = tf.where(noise[..., None], tf.repeat(dots[..., None], 3, -1), img)
    return img, (y, t)


# ---- loss --------------------------------------------------------------------------------------------

COL = tf.constant(np.tile(np.arange(GRID, dtype=np.float32)[None, :, None], (GRID, 1, NA)))
ROW = tf.constant(np.tile(np.arange(GRID, dtype=np.float32)[:, None, None], (1, GRID, NA)))
ANC = tf.constant(A)


def pred_boxes(out):
    """Corner boxes (B, 12, 12, 5, 4) in 0..1 from the raw head."""
    x = (tf.sigmoid(out[..., 0]) + COL) / GRID
    y = (tf.sigmoid(out[..., 1]) + ROW) / GRID
    w = tf.exp(tf.clip_by_value(out[..., 2], -6, 6)) * ANC[:, 0] / GRID
    h = tf.exp(tf.clip_by_value(out[..., 3], -6, 6)) * ANC[:, 1] / GRID
    return tf.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], -1)


def iou_many(pred, truth):
    """IoU of every prediction (B, 12, 12, 5, 4) with every truth box (B, M, 4) -> (B, 12, 12, 5, M)."""
    p = pred[:, :, :, :, None, :]
    t = truth[:, None, None, None, :, :]
    ix = tf.maximum(0.0, tf.minimum(p[..., 2], t[..., 2]) - tf.maximum(p[..., 0], t[..., 0]))
    iy = tf.maximum(0.0, tf.minimum(p[..., 3], t[..., 3]) - tf.maximum(p[..., 1], t[..., 1]))
    inter = ix * iy
    area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    valid = tf.cast(tf.reduce_any(truth != 0, -1), tf.float32)[:, None, None, None, :]
    return inter / (area_p + area_t - inter + 1e-9) * valid


def yolo_loss(y_true, truth, out):
    out = tf.reshape(out, (-1, GRID, GRID, NA, 6))
    has, ign = y_true[..., 5], y_true[..., 6]
    xy_loss = tf.reduce_sum(has * (tf.square(tf.sigmoid(out[..., 0]) - y_true[..., 0]) +
                                   tf.square(tf.sigmoid(out[..., 1]) - y_true[..., 1])))
    wh_loss = tf.reduce_sum(has * (tf.square(out[..., 2] - y_true[..., 2]) + tf.square(out[..., 3] - y_true[..., 3])))
    best_iou = tf.reduce_max(iou_many(pred_boxes(out), truth), -1)
    obj = tf.sigmoid(out[..., 4])
    obj_loss = tf.reduce_sum(has * tf.square(obj - 1.0))
    ignore = tf.cast(best_iou > 0.6, tf.float32)
    noobj_loss = tf.reduce_sum((1 - has) * (1 - ignore) * (1 - ign) * tf.square(obj))
    n = tf.cast(tf.shape(out)[0], tf.float32)
    return (5.0 * (xy_loss + wh_loss) + obj_loss + 0.5 * noobj_loss) / n


# ---- evaluation --------------------------------------------------------------------------------------

def evaluate(predict, samples, threshold=0.6, iou_thr=0.5):
    tp = fp = fn = 0
    for path, boxes in samples:
        out = predict(path)
        dets = decode(out, threshold=threshold)
        truth = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for c, cx, cy, w, h in boxes if c == 0]
        ign = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for c, cx, cy, w, h in boxes if c != 0]
        used = set()
        for d in dets:
            best, best_iou = -1, 0.0
            for j, t in enumerate(truth):
                if j in used:
                    continue
                v = box_iou(d[1:], t)
                if v > best_iou:
                    best, best_iou = j, v
            if best >= 0 and best_iou >= iou_thr:
                tp += 1
                used.add(best)
            elif not any(box_iou(d[1:], g) >= iou_thr for g in ign):
                fp += 1
        fn += len(truth) - len(used)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def keras_predictor(model):
    def predict(path):
        img = load_picture(path)[None]
        return model(img, training=False).numpy()[0]
    return predict


def tflite_predictor(path):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(path)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]

    def predict(png):
        img = load_picture(png).numpy()
        x = img.astype(np.uint8) if inp["dtype"] == np.uint8 else img[None].astype(np.float32)
        it.set_tensor(inp["index"], x[None] if x.ndim == 3 else x)
        it.invoke()
        return it.get_tensor(outp["index"])[0]
    return predict


# ---- main --------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="work/dataset_v1")
    ap.add_argument("--model", default="work/models_zoo/st_yololcv1_192.keras")
    ap.add_argument("--out", default="work/finetune/v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val", default="09,11")
    ap.add_argument("--exclude", default="", help="segments (keys as --val) left out of training and validation: the test segments when --val is another set")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--eval-n", type=int, default=600, help="validation pictures used for precision/recall")
    ap.add_argument("--limit", type=int, default=0, help="use only this many training pictures (smoke test)")
    ap.add_argument("--sparse-lambda", type=float, default=0.0,
                    help="structured activation sparsity: weight of the group-lasso penalty on the ReLU outputs of the early "
                         "layers, groups of --sparse-group consecutive channels (one 32-bit word of int8 NHWC, the unit the "
                         "SMLAD kernel skips)")
    ap.add_argument("--sparse-pixel-lambda", type=float, default=0.0,
                    help="weight of the group-lasso penalty over all channels of a pixel (whole-pixel skipping)")
    ap.add_argument("--sparse-group", type=int, default=4)
    ap.add_argument("--sparse-layers", default="activation,activation_1,activation_2,activation_3",
                    help="Keras layer names whose outputs are regularised (st_yololcv1: the ReLUs after conv 1-4)")
    ap.add_argument("--size", type=int, default=SIZE, help="network input side (192, 160, 128): the model must match, see tools/resize_model.py")
    ap.add_argument("--min-h", type=float, default=MIN_PERSON_H, help="smallest person (fraction of the picture height) kept; pictures with a smaller one are left out")
    ap.add_argument("--temporal", type=int, default=1, help="3: the input channels hold the current picture and the two before it (tools/evpic.py)")
    ap.add_argument("--aug-scale", type=float, default=0.0, help="random zoom in or out by up to this fraction (0.3: x0.77..x1.3), boxes follow")
    ap.add_argument("--aug-shift", type=float, default=0.0, help="random shift by up to this fraction of the picture")
    ap.add_argument("--epoch-size", type=int, default=0, help="train on this many randomly drawn pictures per epoch (a new draw each epoch; 0: all)")
    ap.add_argument("--val-n", type=int, default=0, help="validation loss on at most this many pictures (a fixed random subset; 0: all)")
    ap.add_argument("--small", choices=("drop", "ignore"), default="drop", help="a person below --min-h: leave the picture out, or keep it with that person as a region to ignore")
    ap.add_argument("--patience", type=int, default=15, help="stop when the validation loss has not improved for this many epochs")
    ap.add_argument("--seed", type=int, default=None, help="another training of the same recipe: picture order, augmentations and "
                    "TensorFlow's random state from this seed (default: the fixed order of the paper's first runs)")
    args = ap.parse_args()
    if args.seed is not None:
        global DATA_SEED
        DATA_SEED = args.seed
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
    if args.cpu:  # the queue's thread count, not every core of the shared machine
        tf.config.set_visible_devices([], "GPU")
        n = int(os.environ.get("NUM_THREADS", "4"))
        try:
            tf.config.threading.set_intra_op_parallelism_threads(n)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            pass
    try:
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError:
        pass  # the module-level constants already initialised the devices
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def say(*a):
        line = " ".join(str(v) for v in a)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    global AUG, MIN_H_AUG
    evpic.TEMPORAL = args.temporal
    AUG = (args.aug_scale, args.aug_shift)
    MIN_H_AUG = args.min_h if args.small == "ignore" else 0.0
    train, val = list_samples(args.data, args.val.split(","), args.min_h, args.small, args.exclude.split(","))
    if args.limit:
        train, val = train[:args.limit], val[:max(args.limit // 4, 8)]
    say(f"train {len(train)} pictures ({sum(b[0] == 0 for _, bs in train for b in bs)} people), val {len(val)} "
        f"pictures ({sum(b[0] == 0 for _, bs in val for b in bs)} people); GPUs {tf.config.list_physical_devices('GPU')}")
    rng = np.random.default_rng(0)
    val_eval = [val[i] for i in sorted(rng.choice(len(val), min(args.eval_n, len(val)), replace=False))]

    model = tf.keras.models.load_model(args.model, compile=False)
    # Structured sparsity: a model with the regularised activations as extra outputs. The penalty is the mean L2 norm
    # of groups of G consecutive channels at each pixel (group lasso): a group goes to zero as a whole, which is
    # exactly what an SMLAD-based int8 kernel can skip (one 32-bit load, test, branch). The pixel term does the
    # same over all channels of a pixel. ReLU outputs are >= 0, so the norm is smooth almost everywhere.
    sparse_layers = [l for l in args.sparse_layers.split(",") if l] if (args.sparse_lambda or args.sparse_pixel_lambda) else []
    taps = tf.keras.Model(model.inputs, [model.output] + [model.get_layer(n).output for n in sparse_layers]) if sparse_layers else None
    G = args.sparse_group

    def sparsity_penalty(acts):
        group_terms, pixel_terms = [], []
        for a in acts:
            c = a.shape[-1]
            g = tf.reshape(a, tf.concat([tf.shape(a)[:-1], [c // G, G]], 0))
            group_terms.append(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(g), -1) + 1e-8)))
            pixel_terms.append(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(a), -1) + 1e-8)))
        return args.sparse_lambda * tf.add_n(group_terms) + args.sparse_pixel_lambda * tf.add_n(pixel_terms)

    def structured_sparsity(samples):
        """fraction of zero channel groups and zero pixels per regularised layer, on the float model"""
        if taps is None:
            return {}
        zero_groups = [0.0] * len(sparse_layers); zero_pixels = [0.0] * len(sparse_layers); n = 0
        for path, _ in samples:
            outs = taps(load_picture(path)[None], training=False)[1:]
            for i, a in enumerate(outs):
                a = a.numpy()[0]
                g = a.reshape(a.shape[0], a.shape[1], -1, G)
                zero_groups[i] += float((np.abs(g).max(-1) <= 1e-6).mean())
                zero_pixels[i] += float((np.abs(a).max(-1) <= 1e-6).mean())
            n += 1
        return {name: {"zero_groups": round(zg / n, 4), "zero_pixels": round(zp / n, 4)}
                for name, zg, zp in zip(sparse_layers, zero_groups, zero_pixels)}

    metrics = {"before": evaluate(keras_predictor(model), val_eval)}
    if taps is not None:
        metrics["sparsity_before"] = structured_sparsity(val_eval[:100])
        say("structured sparsity before:", metrics["sparsity_before"])
    say("before fine-tuning (float):", metrics["before"])

    ds_train = make_dataset(train, args.batch, True)
    steps = (args.epoch_size // args.batch) if args.epoch_size else None
    train_iter = iter(ds_train.repeat()) if steps else None  # each epoch continues where the last one stopped
    _unpack = lambda e: (e[0], e[1][0], e[1][1])  # noqa: E731
    if args.val_n and len(val) > args.val_n:
        val = [val[i] for i in sorted(rng.choice(len(val), args.val_n, replace=False))]
    ds_val = make_dataset(val, args.batch, False)
    opt = tf.keras.optimizers.Adam(args.lr)
    best = float("inf")

    @tf.function
    def train_step(img, y, t):
        with tf.GradientTape() as tape:
            if taps is not None:
                outs = taps(img, training=True)
                loss = yolo_loss(y, t, outs[0]) + sparsity_penalty(outs[1:])
            else:
                loss = yolo_loss(y, t, model(img, training=True))
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    @tf.function
    def val_step(img, y, t):
        return yolo_loss(y, t, model(img, training=False))

    since_best = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        tl = ([float(train_step(*_unpack(next(train_iter)))) for _ in range(steps)] if steps
              else [float(train_step(img, y, t)) for img, (y, t) in ds_train])
        vl = [float(val_step(img, y, t)) for img, (y, t) in ds_val]
        say(f"epoch {epoch + 1:3d}  train {np.mean(tl):.3f}  val {np.mean(vl):.3f}  {time.time() - t0:.0f} s")
        if np.mean(vl) < best:
            best, since_best = float(np.mean(vl)), 0
            model.save(out / "best.keras")
        elif (since_best := since_best + 1) >= args.patience:
            say(f"no improvement for {args.patience} epochs, stopping")
            break

    model = tf.keras.models.load_model(out / "best.keras", compile=False)
    metrics["after_float"] = evaluate(keras_predictor(model), val_eval)
    say("after fine-tuning (float):", metrics["after_float"])
    if taps is not None:
        taps = tf.keras.Model(model.inputs, [model.output] + [model.get_layer(n).output for n in sparse_layers])
        metrics["sparsity_after"] = structured_sparsity(val_eval[:100])
        say("structured sparsity after:", metrics["sparsity_after"])

    # float and int8 TFLite, int8 like ST: uint8 input, float output
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    (out / "float.tflite").write_bytes(conv.convert())
    rep = [train[i][0] for i in rng.choice(len(train), min(200, len(train)), replace=False)]

    def representative():
        for p in rep:
            yield [load_picture(p)[None]]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.float32
    (out / "int8.tflite").write_bytes(conv.convert())
    metrics["after_int8"] = evaluate(tflite_predictor(str(out / "int8.tflite")), val_eval)
    say("after fine-tuning (int8 tflite):", metrics["after_int8"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
