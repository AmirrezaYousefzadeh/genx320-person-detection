"""Read global variables of the running firmware over SWD without stopping it: peek.py SYMBOL [SYMBOL ...]
(32-bit words; ELF from firmware/build unless PEEK_ELF is set). Prints "name value" per line."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ELF = Path(os.environ.get("PEEK_ELF", ROOT / "firmware/build/person_detector.elf"))


def main(symbols):
    nm = subprocess.run(["arm-none-eabi-nm", str(ELF)], capture_output=True, text=True, check=True).stdout
    addr = {}
    for line in nm.splitlines():
        parts = line.split()
        if len(parts) == 3:
            addr[parts[2]] = int(parts[0], 16)
    cmds = ["init"] + [f"mdw {addr[s]:#x}" for s in symbols] + ["shutdown"]
    args = ["openocd", "-f", "board/stm32f7discovery.cfg"]
    for c in cmds:  # one -c per command: a joined script does not echo the reads
        args += ["-c", c]
    r = subprocess.run(args, cwd=str(ROOT / "firmware"), capture_output=True, text=True)
    vals = {}
    for line in (r.stdout + "\n" + r.stderr).splitlines():
        line = line.strip()
        if line.startswith("0x") and ":" in line:
            a, rest = line.split(":", 1)
            vals[int(a, 16)] = int(rest.split()[0], 16)
    for s in symbols:
        print(s, vals.get(addr[s], "?"))


if __name__ == "__main__":
    main(sys.argv[1:])
