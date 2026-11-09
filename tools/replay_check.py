"""Counting check on the board: replay room recordings from the card through the live pipeline (tools/sdrec.py play,
one item at a time) and read the board's own counts from its serial 'D' lines ("... N <count> <score>").
Prints, per recording, how often the board showed the expected count. Run with the dialout group
(`sg dialout -c "..."`) unless the shell has it.
Usage: replay_check.py OUT.txt NO:FROM-TO=EXPECTED [NO:FROM-TO=EXPECTED ...]
"""
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import serial

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / "work/venv/bin/python")


def main(out, *items):
    port = serial.Serial(os.environ.get("GENX_SERIAL", "/dev/ttyACM0"), 115200, timeout=1)
    log = open(out, "w")
    summary = []
    for item in items:
        spec, _, expected = item.partition("=")
        no, _, win = spec.partition(":")
        f, _, t = win.partition("-")
        seconds = float(t) - float(f)
        subprocess.run([PY, "-I", str(ROOT / "tools/sdrec.py"), "play", "--once", spec], capture_output=True)
        port.reset_input_buffer()
        t_end = time.time() + seconds
        counts, nn_ms = [], []
        while time.time() < t_end:
            line = port.readline().decode("ascii", "replace").strip()
            log.write(f"{item} {line}\n")
            m = re.search(r" N (-?\d+)", line)
            if line.startswith("D ") and m:
                counts.append(int(m.group(1)))
                nn_ms.append(int(line.split()[3]))
        subprocess.run([PY, "-I", str(ROOT / "tools/sdrec.py"), "stop"], capture_output=True)
        c = Counter(counts)
        right = c.get(int(expected), 0) / max(len(counts), 1)
        summary.append(f"recording {spec}: expected {expected}, {len(counts)} updates, counts {dict(sorted(c.items()))}, "
                       f"right {right:.2f}; network {sum(nn_ms) / max(len(nn_ms), 1):.0f} ms")
        print(summary[-1], flush=True)
        time.sleep(1.0)
    log.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
