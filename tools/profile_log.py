"""Read the detector's profile lines (P ...) from the board for a few seconds and print the mean breakdown.

Usage: profile_log.py SECONDS [label]     (serial /dev/ttyACM0, run with sg dialout -c)
P <make_input us> <decode+smoothing us> <network ms> <total ms> <cpu %> <events %> <net %> <display %> <sd %>
"""
import sys
import time

import serial

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
label = sys.argv[2] if len(sys.argv) > 2 else ""
with serial.Serial("/dev/ttyACM0", 115200, timeout=0.5) as port:
    port.reset_input_buffer()
    t0 = time.time()
    rows = []
    while time.time() - t0 < secs:
        line = port.readline().decode(errors="replace").strip()
        if line.startswith("P "):
            try:
                rows.append([float(v) for v in line.split()[1:10]])
            except ValueError:
                pass
rows = rows[1:]  # the first may be stale
if not rows:
    sys.exit("no profile lines")
m = [sum(r[i] for r in rows) / len(rows) for i in range(9)]
print(f"{label} {len(rows)} runs: make_input {m[0]:.0f} us, decode+smooth {m[1]:.0f} us, network {m[2]:.0f} ms, "
      f"cycle {m[3]:.0f} ms ({1000 / m[3]:.2f}/s); cpu {m[4]:.0f} % = events {m[5]:.0f} + network {m[6]:.0f} + "
      f"display {m[7]:.0f} + sd {m[8]:.0f}")
