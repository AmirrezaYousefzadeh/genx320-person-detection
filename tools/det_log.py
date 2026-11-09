"""Log the board's detections from the serial port (ST-LINK virtual COM, 115200 baud).

The detector task prints one line per run:  D <board ms> <seq> <network ms> <n> [x0 y0 x1 y1 score]...
Each line is stored with the PC's receive time, as CSV: pc_time, board_ms, seq, nn_ms, n, boxes (json).
Usage: det_log.py <out.csv> [seconds]   (default: until Ctrl-C). Run as `sg dialout -c "..."` if needed.
"""
import os
import csv
import json
import sys
import time

import serial


def main(out_csv, seconds=None):
    port = serial.Serial(os.environ.get("GENX_SERIAL", "/dev/ttyACM0"), 115200, timeout=1)
    time.sleep(0.3)  # lines queued before we opened the port are stale: drop them and the first partial line
    port.reset_input_buffer()
    port.readline()
    t_end = time.time() + float(seconds) if seconds else None
    n_lines = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pc_time", "board_ms", "seq", "nn_ms", "n", "boxes", "kevps"])
        kevps = ""  # the board's event rate (kev/s) from its last 'E' line: the light/monitor reference of a bench
        try:
            while t_end is None or time.time() < t_end:
                line = port.readline().decode("ascii", "replace").strip()
                if line.startswith("E "):
                    kevps = line.split()[1]
                if not line.startswith("D "):
                    continue
                parts = line.split()
                board_ms, seq, nn_ms, n = (int(p) for p in parts[1:5])
                vals = [float(p) for p in parts[5:5 + 5 * n]]
                boxes = [vals[i:i + 5] for i in range(0, len(vals), 5)]
                w.writerow([f"{time.time():.3f}", board_ms, seq, nn_ms, n, json.dumps(boxes), kevps])
                f.flush()
                n_lines += 1
        except KeyboardInterrupt:
            pass
    print(f"{n_lines} detections written to {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
