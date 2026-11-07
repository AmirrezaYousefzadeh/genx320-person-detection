"""Grab one picture from the webcam that looks at the board (colour camera, 1920x1080) and save it as PNG.

The PC user must be in group `video`; from an older session run it as `sg video -c "python grab_cam.py ..."`.
Usage: grab_cam.py <out.png> [x0 y0 x1 y1]   (optional crop box as fractions of the picture, e.g. the board)
"""
import sys

import cv2


def grab(out_png, crop=None, device=0):
    cam = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(20):  # let exposure settle
        ok, frame = cam.read()
    cam.release()
    if not ok:
        sys.exit("no picture from the webcam")
    if crop:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = crop
        frame = frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    cv2.imwrite(out_png, frame)
    print(f"saved {out_png} {frame.shape[1]}x{frame.shape[0]}")


if __name__ == "__main__":
    grab(sys.argv[1], [float(v) for v in sys.argv[2:6]] if len(sys.argv) >= 6 else None)
