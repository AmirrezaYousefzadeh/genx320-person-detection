"""Guided recording of real scenes: a person in front of the sensor instead of the monitor, with live feedback.

The board is turned to face the room; the monitor stays behind it and faces the person. The monitor shows what the
sensor sees right now (the detector picture, read over SWD without stopping the board, about once a second,
mirrored like a selfie view), the board's detections (from the serial port) and, in a session, each scenario's
instruction, a countdown and a REC indicator. Each scenario is recorded to the board's card (tools/sdrec.py) and
the board's live detections are logged next to it.

Usage: self_record.py session OUT_DIR [--only 1,3] [--seconds 20] [--prep 12]   guided recording
       self_record.py mirror                                                    live view only (testing the demo)
       self_record.py fetch OUT_DIR                                             copy the recordings from the card
Keys on the PC keyboard while it runs: SPACE starts the next scenario at once, S skips it, ESC ends.
Writes OUT_DIR/<NN>_<name>/segment.json (card file number, PC times, scenario) and detections.csv (the board's
raw boxes per run, with the event rate and the number of active input pixels).
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pygame
import serial

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from record_dataset import fetch_segment, sdrec  # noqa: E402

ELF = ROOT / "firmware/build/person_detector.elf"
PIC = 320

SCENARIOS = [  # (name, what the person does)
    ("empty", "Step out of the camera's view. Nothing moves."),
    ("walk_3m", "Walk slowly from left to right and back, about 3 m from the camera."),
    ("walk_5m", "The same, about 5 m from the camera."),
    ("approach", "Walk towards the camera from far away, stop at about 1.5 m, then walk back."),
    ("still_wave", "Stand still facing the camera for 5 seconds, then wave your arms."),
    ("sit", "Sit down on a chair and stand up again, a few times."),
    ("close", "Close up, about 1.5 m away: move your upper body, turn around."),
    ("diagonal", "Walk diagonally through the view, then in a circle."),
    ("occlusion", "Walk behind a desk or a chair, so that you are partly hidden."),
    ("fast", "Fast movements: jog through the view, jump."),
    ("empty_end", "Step out of the view again. Nothing moves."),
]

BG = (12, 14, 20)
TEXT = (225, 230, 240)
DIM = (130, 140, 160)
ORANGE = (255, 140, 0)
RED = (230, 40, 40)
PIC_COLOURS = np.array([[30, 37, 52], [216, 223, 236], [64, 126, 201]], np.uint8)  # none, ON, OFF (as the LCD)


def symbol_addr(name):
    out = subprocess.run(["arm-none-eabi-nm", str(ELF)], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        p = line.split()
        if len(p) == 3 and p[2] == name:
            return int(p[0], 16)
    sys.exit(f"{name} is not in {ELF}")


class Board(threading.Thread):
    """Reads the board's serial lines: the last boxes (D), event rate (E), active pixels (A); logs D lines."""

    def __init__(self):
        super().__init__(daemon=True)
        self.port = serial.Serial("/dev/ttyACM0", 115200, timeout=0.5)
        self.lock = threading.Lock()
        self.boxes, self.kevps, self.active, self.gated, self.nn_ms, self.t_boxes = [], 0, 0, 0, 0, 0.0
        self.log, self.log_file, self.runs, self.runs_with_person = None, None, 0, 0

    def start_log(self, path):
        with self.lock:
            self.log_file = open(path, "w", newline="")
            self.log = csv.writer(self.log_file)
            self.log.writerow(["pc_time", "board_ms", "seq", "nn_ms", "n", "boxes", "kevps", "active", "gated"])
            self.runs = self.runs_with_person = 0

    def stop_log(self):
        with self.lock:
            if self.log_file:
                self.log_file.close()
            self.log, self.log_file = None, None
            return self.runs, self.runs_with_person

    def run(self):
        time.sleep(0.2)
        self.port.reset_input_buffer()
        while True:
            p = self.port.readline().decode("ascii", "replace").split()
            if not p:
                continue
            now = time.time()
            with self.lock:
                try:
                    if p[0] == "D" and len(p) >= 5:
                        n = int(p[4])
                        vals = [float(v) for v in p[5:5 + 5 * n]]
                        self.boxes = [vals[i:i + 5] for i in range(0, len(vals), 5)]
                        self.nn_ms, self.t_boxes = int(p[3]), now
                        if self.log:
                            self.log.writerow([f"{now:.3f}", p[1], p[2], p[3], n, json.dumps(self.boxes),
                                               self.kevps, self.active, self.gated])
                            self.log_file.flush()
                            self.runs += 1
                            self.runs_with_person += n > 0
                    elif p[0] == "E":
                        self.kevps = int(p[1])
                    elif p[0] == "A":
                        self.active, self.gated = int(p[1]), int(p[2])
                except ValueError:
                    pass  # a line cut by the port opening


class Preview(threading.Thread):
    """Reads the detector picture over SWD about once a second while the board runs (no halt). Stops while
    sdrec needs the debug link (hold `swd`)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.addr = symbol_addr("nn_pic")
        self.swd = threading.Lock()
        self.pic, self.t_pic = None, 0.0
        self.tmp = Path(tempfile.mkdtemp()) / "pic.raw"

    def run(self):
        while True:
            with self.swd:
                subprocess.run(["openocd", "-f", "board/stm32f7discovery.cfg", "-c",
                                f"init; dump_image {self.tmp} {self.addr:#x} {PIC * PIC:#x}; shutdown"],
                               capture_output=True)
            try:
                data = self.tmp.read_bytes()
                if len(data) == PIC * PIC:
                    self.pic, self.t_pic = np.frombuffer(data, np.uint8).reshape(PIC, PIC), time.time()
            except OSError:
                pass
            time.sleep(0.3)


class Screen:
    def __init__(self):
        os.environ["DISPLAY"] = ":1"  # the monitor, also from a shell whose DISPLAY is an SSH-forwarded one
        pygame.display.init()
        pygame.font.init()
        pygame.mouse.set_visible(False)
        self.s = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.w, self.h = self.s.get_size()
        f = lambda size, bold=False: pygame.font.SysFont("dejavusans", size, bold=bold)  # noqa: E731
        self.f_status, self.f_text, self.f_big, self.f_info, self.f_small = f(78, True), f(56), f(150, True), f(42), f(30)
        self.clock = pygame.time.Clock()

    def text(self, s, font, colour, x, y):
        img = font.render(s, True, colour)
        self.s.blit(img, (x, y))
        return img.get_width(), img.get_height()

    def frame(self, board, preview, status, status_colour, instruction="", big="", progress=None, rec=False):
        s = self.s
        s.fill(BG)
        # the sensor's view, mirrored (move right, the picture moves right), with the board's latest boxes
        side = min(self.h - 260, self.w // 2 - 80)
        px, py = self.w - side - 60, 150
        pic = preview.pic
        if pic is not None:
            idx = np.where(pic == 255, 1, np.where(pic == 0, 2, 0))[:, ::-1]
            img = pygame.surfarray.make_surface(PIC_COLOURS[idx].swapaxes(0, 1))
            s.blit(pygame.transform.scale(img, (side, side)), (px, py))
        else:
            pygame.draw.rect(s, (30, 37, 52), (px, py, side, side))
        with board.lock:
            boxes, kevps, nn_ms, fresh = list(board.boxes), board.kevps, board.nn_ms, time.time() - board.t_boxes < 3
        if fresh:
            for x0, y0, x1, y1, score in boxes:
                r = pygame.Rect(px + (1 - x1) * side, py + y0 * side, (x1 - x0) * side, (y1 - y0) * side)
                pygame.draw.rect(s, ORANGE, r, 5)
                self.text(f"person {min(99, round(score * 100))}%", self.f_small, ORANGE, r.x + 6, r.y + 4)
        pygame.draw.rect(s, (70, 80, 100), (px - 2, py - 2, side + 4, side + 4), 2)
        self.text("what the camera sees (mirrored)", self.f_small, DIM, px, py + side + 10)
        # left: status, instruction, countdown, the board's numbers
        x = 70
        if rec:
            pygame.draw.circle(s, RED, (x + 30, 108), 26)
            self.text(status, self.f_status, status_colour, x + 75, 62)
        else:
            self.text(status, self.f_status, status_colour, x, 62)
        y = 200
        for line in textwrap.wrap(instruction, 30):
            self.text(line, self.f_text, TEXT, x, y)
            y += 72
        if big:
            self.text(big, self.f_big, TEXT, x, max(y + 30, 520))
        n = len(boxes) if fresh else 0
        info = f"board sees {n} person{'s' if n != 1 else ''}" if fresh else "board: no data"
        self.text(info, self.f_info, ORANGE if n else DIM, x, self.h - 250)
        self.text(f"events {kevps} k/s    network {nn_ms} ms", self.f_info, DIM, x, self.h - 190)
        if progress is not None:
            pygame.draw.rect(s, (40, 46, 60), (x, self.h - 110, self.w // 2 - 160, 26))
            pygame.draw.rect(s, RED, (x, self.h - 110, int((self.w // 2 - 160) * progress), 26))
        self.text("SPACE start now     S skip     ESC end", self.f_small, (90, 98, 115), x, self.h - 60)
        pygame.display.flip()
        self.clock.tick(15)

    def keys(self):
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return "esc"
                if ev.key == pygame.K_SPACE:
                    return "space"
                if ev.key == pygame.K_s:
                    return "skip"
        return None


def stop_monitor_py():
    """Stop tools/monitor.py and any other self_record.py (not via show.sh, which would stop this one too)."""
    pids = subprocess.run(["pgrep", "-f", r"tools/(monitor|self_record)\.py"], capture_output=True, text=True)
    for pid in pids.stdout.split():
        try:
            if int(pid) != os.getpid() and Path(f"/proc/{pid}/comm").read_text().startswith("python"):
                os.kill(int(pid), 15)
        except (OSError, ValueError):
            pass


def mirror():
    stop_monitor_py()
    board, preview, screen = Board(), Preview(), Screen()
    board.start()
    preview.start()
    while screen.keys() != "esc":
        screen.frame(board, preview, "LIVE VIEW", TEXT, "Move in front of the camera: the orange boxes are what "
                     "the board detects, on its own screen too.")


def session(out_dir, only=None, seconds=20.0, prep=12.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_monitor_py()
    board, preview, screen = Board(), Preview(), Screen()
    board.start()
    preview.start()
    todo = [(i, s) for i, s in enumerate(SCENARIOS, 1) if not only or i in only]
    for k, (i, (name, what)) in enumerate(todo, 1):
        seg = out_dir / f"{i:02d}_{name}"
        if (seg / "segment.json").exists():
            continue
        t_end, key = time.time() + prep, None
        while key not in ("space", "skip", "esc") and time.time() < t_end:  # get ready
            screen.frame(board, preview, f"NEXT  {k} / {len(todo)}", TEXT, what,
                         f"{int(t_end - time.time()) + 1}")
            key = screen.keys()
        if key == "esc":
            break
        if key == "skip":
            continue
        screen.frame(board, preview, f"{k} / {len(todo)}", RED, what, "starting", rec=True)
        seg.mkdir(parents=True, exist_ok=True)
        with preview.swd:
            sdrec("start")
        pc_start = time.time()
        board.start_log(seg / "detections.csv")
        key = None
        while key not in ("esc", "skip") and time.time() < pc_start + seconds:  # recording
            left = pc_start + seconds - time.time()
            screen.frame(board, preview, f"{k} / {len(todo)}", RED, what, f"{int(left) + 1} s",
                         progress=1 - left / seconds, rec=True)
            key = screen.keys()
        pc_stop = time.time()
        runs, with_person = board.stop_log()
        screen.frame(board, preview, "saving", DIM, what, "")
        with preview.swd:
            status = sdrec("stop")
        file_no = int(status.split("recording ")[1].split(":")[0])
        (seg / "segment.json").write_text(json.dumps({
            "scenario": name, "instruction": what, "index": i, "file_no": file_no, "pc_start": pc_start,
            "pc_stop": pc_stop, "stopped_early": key in ("esc", "skip"), "scene": "real (board facing the room)",
            "runs": runs, "runs_with_a_person": with_person, "sdrec_status": status.strip()}, indent=1))
        print(f"{seg.name}: recording {file_no:04d}, {runs} runs, a person in {with_person}", flush=True)
        t_end = time.time() + 3
        while time.time() < t_end:
            screen.frame(board, preview, "saved", TEXT, f"The board saw a person in {with_person} of {runs} runs.")
        if key == "esc":
            break
    t_end = time.time() + 5
    while time.time() < t_end:
        screen.frame(board, preview, "DONE", TEXT, "Thank you. The recordings are on the board's card.")


def fetch(out_dir):
    for seg in sorted(p for p in Path(out_dir).iterdir() if (p / "segment.json").exists()):
        if not (seg / "events.raw").exists():
            fetch_segment(seg)


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    if a and a[0] == "session":
        only = [int(v) for v in opt("--only", "").split(",") if v]
        session(a[1], only, opt("--seconds", 20.0), opt("--prep", 12.0))
    elif a and a[0] == "mirror":
        mirror()
    elif a and a[0] == "fetch":
        fetch(a[1])
    else:
        sys.exit(__doc__)
