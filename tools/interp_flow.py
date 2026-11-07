"""Motion-compensated frame interpolation with OpenCV's DIS optical flow (no learned model, nothing downloaded).

Between frames I0 and I1 the flows F01 and F10 are computed on grey copies (at FLOW_SCALE of the resolution, scaled
back up); the flows from time t to both frames follow the linear-motion approximation of Super SloMo (Jiang et al.,
CVPR 2018): Ft0 = -(1-t) t F01 + t^2 F10, Ft1 = (1-t)^2 F01 - t (1-t) F10. Both frames are warped to t and blended
with weights (1-t) and t, each scaled down where its flow fails the forward-backward check (occluded or unreliable
pixels), so a pixel visible in only one frame comes from that frame.
"""
import cv2
import numpy as np

FLOW_SCALE = 0.5
SIGMA_PX = 2.0
_dis = None


def _flow(a, b):
    global _dis
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _dis.calc(a, b, None)


def flows(i0, i1, scale=FLOW_SCALE):
    """(F01, F10) at full resolution, float32 HxWx2"""
    g0, g1 = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i for i in (i0, i1))
    h, w = g0.shape
    if scale != 1.0:
        small = (int(round(w * scale)), int(round(h * scale)))
        g0, g1 = cv2.resize(g0, small, interpolation=cv2.INTER_AREA), cv2.resize(g1, small, interpolation=cv2.INTER_AREA)
    f01, f10 = _flow(g0, g1), _flow(g1, g0)
    if scale != 1.0:
        f01 = cv2.resize(f01, (w, h), interpolation=cv2.INTER_LINEAR) / scale
        f10 = cv2.resize(f10, (w, h), interpolation=cv2.INTER_LINEAR) / scale
    return f01, f10


def _grid(h, w):
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


def _warp(img, gx, gy, f):
    return cv2.remap(img, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def interpolate(i0, i1, ts, fl=None):
    """frames at the times ts (each in (0, 1)) between i0 and i1"""
    f01, f10 = fl if fl is not None else flows(i0, i1)
    h, w = f01.shape[:2]
    gx, gy = _grid(h, w)
    # forward-backward error of each frame's flow, in that frame's coordinates
    e0 = np.linalg.norm(f01 + _warp(f10, gx, gy, f01), axis=2)
    e1 = np.linalg.norm(f10 + _warp(f01, gx, gy, f10), axis=2)
    out = []
    for t in ts:
        ft0 = -(1 - t) * t * f01 + t * t * f10
        ft1 = (1 - t) * (1 - t) * f01 - t * (1 - t) * f10
        w0 = (1 - t) * np.exp(-_warp(e0, gx, gy, ft0) / SIGMA_PX) + 1e-4 * (1 - t)
        w1 = t * np.exp(-_warp(e1, gx, gy, ft1) / SIGMA_PX) + 1e-4 * t
        a, b = _warp(i0, gx, gy, ft0).astype(np.float32), _warp(i1, gx, gy, ft1).astype(np.float32)
        if a.ndim == 3:
            w0, w1 = w0[..., None], w1[..., None]
        out.append(np.clip((w0 * a + w1 * b) / (w0 + w1), 0, 255).astype(np.uint8))
    return out
