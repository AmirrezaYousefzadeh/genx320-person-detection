"""Drive the board's microSD recorder over SWD (firmware/App/sdcard.c, control block sd_ctrl).

  status                 card state, free space, last recording
  start / stop           start a recording (/evt/recNNNN.raw on the card), stop it
  record SECONDS         start, wait, stop; prints the recording number and size
  list                   directory listing of the card
  clean                  delete everything on the card (asks nothing: only use on the dedicated card)
  format                 make a new file system on the card (FatFs picks FAT32/exFAT)
  read NO OUT.raw [MAX_BYTES]   copy recording NO from the card to the PC over SWD (slow, ~100 kB/s)
  play [--once] NO[:FROM-TO] ...   replay recordings on the board (seconds into each, e.g. 182:2-18), looped
                         unless --once; the board paints the recorded events instead of the sensor's and
                         detects on them as live; "stop" ends it
Usage: sdrec.py <command> [args]
"""
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import os
ELF = Path(os.environ.get("SDREC_ELF", ROOT / "firmware/build/person_detector.elf"))  # SDREC_ELF=firmware/build_histo/... for the histogram build
CFG = "board/stm32f7discovery.cfg"
FIELDS = ["magic", "cmd", "state", "result", "file_no", "bytes", "dropped", "start_tick", "stop_tick", "arg0", "arg1",
          "buf_addr", "buf_len", "card_mb", "free_mb"]
CMD = {"start": 1, "stop": 2, "list": 3, "clean": 4, "read": 5, "format": 6, "play": 7}
STATE = {0: "no card", 1: "idle", 2: "recording", 3: "busy", 4: "done", 5: "error", 6: "playing"}
PLAY_MAX = 16  # app.h; the playlist follows FIELDS in sd_ctrl: play_n, play_loop, play_file[], play_from_ms[],
               # play_to_ms[], play_item
MAGIC = 0x31444453


def openocd(*commands):
    out = subprocess.run(["openocd", "-f", CFG, "-c", "adapter speed 4000", "-c", "init",  # 4 MHz: 1.5x faster dumps
                          *sum((["-c", c] for c in commands), []),
                          "-c", "shutdown"], capture_output=True, text=True, check=True)
    return out.stdout + out.stderr


def symbol(name):
    nm = subprocess.run(["arm-none-eabi-nm", ELF], capture_output=True, text=True, check=True).stdout
    return int(re.search(rf"^([0-9a-f]+) \w {name}$", nm, re.M).group(1), 16)


CTRL = symbol("sd_ctrl")


def read_ctrl():
    out = openocd(f'echo "CTRL [read_memory {CTRL:#x} 32 {len(FIELDS)}]"')
    vals = [int(v, 0) for v in re.search(r"CTRL (.*)", out).group(1).split()]
    c = dict(zip(FIELDS, vals))
    c["result"] = struct.unpack("<i", struct.pack("<I", c["result"]))[0]
    assert c["magic"] == MAGIC, f"no SD recorder in the firmware (magic {c['magic']:#x})"
    return c


def command(name, arg0=0, arg1=0, wait=True, timeout=120):
    openocd(f"mww {CTRL + 9 * 4:#x} {arg0}", f"mww {CTRL + 10 * 4:#x} {arg1}", f"mww {CTRL + 4:#x} {CMD[name]}")
    if not wait:
        return read_ctrl()
    t0 = time.time()
    while True:
        time.sleep(0.3)
        c = read_ctrl()
        if c["cmd"] == 0 and c["state"] != 3:
            return c
        if time.time() - t0 > timeout:
            sys.exit(f"timeout waiting for {name}: {c}")


def dump_result(c, path):
    openocd(f"dump_image {path} {c['buf_addr']:#x} {c['bytes']}")
    return Path(path).read_bytes()


def show(c):
    print(f"state {STATE.get(c['state'], c['state'])}, result {c['result']}, card {c['card_mb'] / 1024:.1f} GB, "
          f"free {c['free_mb'] / 1024:.1f} GB, last/current recording {c['file_no']:04d}: {c['bytes']} bytes, "
          f"dropped words {c['dropped']}, ticks {c['start_tick']}..{c['stop_tick']}")


def main(argv):
    cmd = argv[0]
    if cmd == "status":
        show(read_ctrl())
    elif cmd == "play":
        loop = 0 if "--once" in argv else 1
        items = []
        for a in argv[1:]:
            if a.startswith("--"):
                continue
            no, _, win = a.partition(":")
            f, _, t = win.partition("-")
            items.append((int(no), int(float(f or 0) * 1000), int(float(t or 0) * 1000)))
        if not items or len(items) > PLAY_MAX:
            sys.exit(f"give 1..{PLAY_MAX} recordings")
        base = len(FIELDS)
        cmds = [f"mww {CTRL + base * 4:#x} {len(items)}", f"mww {CTRL + (base + 1) * 4:#x} {loop}"]
        for i, (no, f, t) in enumerate(items):
            cmds += [f"mww {CTRL + (base + 2 + i) * 4:#x} {no}", f"mww {CTRL + (base + 2 + PLAY_MAX + i) * 4:#x} {f}",
                     f"mww {CTRL + (base + 2 + 2 * PLAY_MAX + i) * 4:#x} {t}"]
        openocd(*cmds, f"mww {CTRL + 4:#x} {CMD['play']}")
        time.sleep(1.0)
        show(read_ctrl())
    elif cmd in ("start", "stop", "clean", "format"):
        show(command(cmd, wait=cmd != "start", timeout=600))
    elif cmd == "record":
        c = command("start", wait=False)
        t0 = time.time()
        time.sleep(float(argv[1]))
        c = command("stop")
        show(c)
        print(f"recording {c['file_no']:04d}: {c['bytes'] / 1e6:.2f} MB in {time.time() - t0:.1f} s"
              f" (board {c['stop_tick'] - c['start_tick']} ms), dropped words {c['dropped']}")
    elif cmd == "list":
        c = command("list")
        tmp = Path(tempfile.gettempdir())
        print(dump_result(c, tmp / "sd_list.bin").decode("ascii", "replace") if c["bytes"] else "(empty)")
    elif cmd == "read":
        no, out = int(argv[1]), argv[2]
        max_bytes = int(argv[3]) if len(argv) > 3 else None
        data, offset = bytearray(), 0
        while True:
            c = command("read", no, offset)
            if c["state"] != 4:
                sys.exit(f"read failed: result {c['result']}")
            if c["bytes"] == 0:
                break
            data += dump_result(c, Path(out).with_suffix(".part"))
            offset += c["bytes"]
            print(f"  {offset / 1e6:.1f} MB", flush=True)
            if c["bytes"] < c["buf_len"] or (max_bytes and offset >= max_bytes):
                break
        Path(out).write_bytes(data[:max_bytes] if max_bytes else data)
        Path(out).with_suffix(".part").unlink(missing_ok=True)
        print(f"wrote {out}: {len(data)} bytes")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
