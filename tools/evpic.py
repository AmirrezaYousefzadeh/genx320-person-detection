"""The network's input from event pictures (shared by the training, evaluation and conversion tools).

The network takes three channels. Until 2026-11-06 the board copied the grey event picture (128 nothing, 255 ON,
0 OFF) into all three. With TEMPORAL = 3 the channels hold the current 50 ms picture and the two before it (R, G, B):
150 ms of motion history for the same network and the same cost on the board. GENX_TEMPORAL=3 in the environment
switches the tools; tools/finetune.py --temporal 3 sets it for training.
"""
import os
from pathlib import Path

import cv2
import numpy as np

TEMPORAL = int(os.environ.get("GENX_TEMPORAL", "1"))


def picture_stack(path, temporal=None):
    """(H, W, 3) uint8 at the picture's own resolution"""
    t = TEMPORAL if temporal is None else temporal
    p = Path(path)
    cur = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if t == 1:
        return np.repeat(cur[..., None], 3, -1)
    k = int(p.stem.split("_")[-1])
    chans = [cur]
    for d in (1, 2):  # the pictures before it in the same segment (the first ones repeat what exists)
        q = p.with_name(f"pic_{max(k - d, 0):04d}.png")
        img = cv2.imread(str(q), cv2.IMREAD_GRAYSCALE) if q.exists() else None
        chans.append(img if img is not None and img.shape == cur.shape else chans[-1])
    return np.stack(chans, -1)


def network_input(path, size, temporal=None):
    """(size, size, 3) uint8, squeezed with nearest neighbour as the board does"""
    return cv2.resize(picture_stack(path, temporal), (size, size), interpolation=cv2.INTER_NEAREST)
