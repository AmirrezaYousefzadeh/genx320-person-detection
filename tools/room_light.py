"""Room light meter: the mean grey level of a webcam frame at a fixed exposure (10 ms, auto exposure off, restored
afterwards). Daylight with the office lights on gave 185 (2026-11-05 noon); a dark room gives a few tens at most.
Usage: room_light.py            prints the mean (0..255)
"""
import time

import cv2

EXPOSURE = 100  # V4L2 exposure_time_absolute, units of 100 us


def measure(device=0):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit("webcam not available")
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual
        cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
        time.sleep(1.0)
        frames = [cap.read()[1] for _ in range(12)][-4:]
        return sum(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()) for f in frames) / len(frames)
    finally:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # back to automatic
        cap.release()


if __name__ == "__main__":
    print(f"{measure():.1f}")
