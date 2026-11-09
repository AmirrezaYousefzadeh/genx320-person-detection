"""Dataset campaign on the monitor bench: passes, calibrations and copies run unattended for hours.

The plan is work/campaign/plan.json, read again before every step, so it can change while the campaign runs:
  {"end": "2026-11-06T20:00:00", "copy_s": 155, "record_s": 20,
   "entries": [{"name": "camp60_dark_z40_r0_g100", "light": "dark", "opts": ["--src", "...", "--zoom", "0.4", ...]}, ...]}
Each entry is one pass, tools/dataset_run.py record OUT OPTS (--src: the sequences, e.g. the 60 fps copies of
tools/make60.py; --zoom/--zoom-rank: tools/zoom_plan.py crops; --gain; --flip), recorded when the room light matches
("dark", "lit" or "any"; webcam at a fixed exposure, tools/room_light.py: below 25 dark, from 50 lit, nothing in
between). A source is used only once it holds a READY file. The list order is the priority: an entry is recorded only if
the copy of everything waiting, of the entries before it that are still to record (their light may come later) and of
itself ends before "end" (copy_s per 10 s segment over SWD, the bottleneck; record_s to record one). Plan passes are
copied and labelled first, oldest first; other recordings still on the card (the 30 fps passes of 2026-11-05) only with
time to spare. At "end" the board sleeps and the campaign exits; anything left stays on the card.

Around every pass the board is recalibrated (the drift between the two calibrations goes to the pass's
CONDITIONS.json, drift_px). With nothing to do the board sleeps. A board found running while the
campaign left it asleep belongs to someone else for up to 3 h: the campaign leaves it alone. The build on the board
must be work/campaign/flashed.elf (the tools read sd_ctrl and nn_pic from it): with another build flashed it waits.

One queue job:
  after the campaign, the rest of the card: campaign.py fetch
  work/venv/bin/python -I tools/campaign.py run
  touch work/campaign/PAUSE   finish the current step, put the board to sleep, leave it alone until the file goes
  touch work/campaign/STOP    the same, then exit
Usage: campaign.py run [--take-board] | status | check
"""
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_mot import seq_info  # noqa: E402
from dataset_run import TRAIN, sequences  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / "work/venv/bin/python")
CAMP = ROOT / "work/campaign"
STATE = CAMP / "state.json"
PLAN_FILE = CAMP / "plan.json"
LOG = CAMP / "campaign.log"
CALIB = ROOT / "work/calib/calib.json"
ELF = CAMP / "flashed.elf"
OCD = ["openocd", "-f", "board/stm32f7discovery.cfg", "-c", "adapter speed 4000"]
DARK_MAX, LIT_MIN = 25.0, 50.0  # webcam mean at the fixed exposure: 0.1 with the light off (2026-11-05 18:20), 185 in daylight
SECONDS = 10.0
IDLE_S, FOREIGN_S, CALIB_BACKOFF_S, CALIB_FRESH_S = 600, 3 * 3600, 1800, 180
MONITOR = "Samsung SyncMaster 2493HM, brightness 100"
DRIFT_PTS = np.array([[40, 20], [280, 20], [280, 250], [40, 250]], np.float64)[:, None]  # picture px, on the screen
ENV = {"GENX_ELF": str(ELF), "SDREC_ELF": str(ELF), "PEEK_ELF": str(ELF)}


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, timeout=None, env=None):
    """the command's CompletedProcess; never raises (a timeout gives returncode 124)"""
    try:
        return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout, cwd=ROOT,
                              env=dict(os.environ, **ENV, **(env or {})))
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"timeout after {timeout} s")


def tail(r, n=300):
    return (r.stdout.strip().splitlines() or [r.stderr.strip()])[-1][-n:] if r.returncode == 0 else \
        f"rc {r.returncode}: {(r.stderr.strip() or r.stdout.strip())[-n:]}"


def load_state():
    s = json.loads(STATE.read_text()) if STATE.exists() else {}
    for k, v in (("board", None), ("foreign_since", None), ("attempts", {}), ("fail", {}), ("calib_at", 0),
                 ("calib_fail_at", 0)):
        s.setdefault(k, v)
    return s


def save_state(s):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(STATE)


# ---------------------------------------------------------------- plan, passes and segments
def load_plan():
    p = json.loads(PLAN_FILE.read_text())
    p["end_t"] = dt.datetime.fromisoformat(p["end"]).timestamp()
    return p


def entry_src(e):
    o = e["opts"]
    return ROOT / o[o.index("--src") + 1] if "--src" in o else TRAIN


def src_ready(src):
    return src == TRAIN or (src / "READY").exists()


_n_seg = {}


def n_segments(src):
    """10 s segments of a pass over src, as tools/dataset_run.py record cuts them"""
    if src not in _n_seg:
        total = 0
        for d in sequences(src):
            fps, _, _, n = seq_info(d)
            total += len(range(0, int(n - SECONDS * fps) + 1, int(SECONDS * fps))) or 1
        _n_seg[src] = total
    return _n_seg[src]


def pass_dir(name):
    return ROOT / "work" / f"dataset_{name}"


def seg_dirs(d):
    return sorted(p for p in d.iterdir() if (p / "segment.json").exists()) if d.is_dir() else []


def copied(seg):
    return (seg / "events.raw").exists() and (seg / "events.json").exists()


def waiting_segments(state, plan):
    """(plan passes' segments still only on the card, the other monitor recordings'), oldest pass first; a segment
    that failed three times is left out"""
    names = {f"dataset_{e['name']}" for e in plan["entries"]}
    groups = []
    for d in (ROOT / "work").glob("dataset_*"):
        segs = [s for s in seg_dirs(d) if not copied(s) and state["fail"].get(str(s.relative_to(ROOT)), 0) < 3
                and "seq_dir" in (s / "segment.json").read_text()]
        if segs:
            groups.append((d.name in names, min(s.joinpath("segment.json").stat().st_mtime for s in segs), segs))
    groups.sort(key=lambda g: g[1])
    return ([s for p, _, segs in groups if p for s in segs], [s for p, _, segs in groups if not p for s in segs])


def choose(plan, state, cond, n_waiting):
    """(entry to record now or None, why not)"""
    now, ahead, why = time.time(), 0, "plan done for this light"
    for e in plan["entries"]:
        if state["attempts"].get(e["name"], 0) >= 3:
            continue
        src = entry_src(e)
        match = e["light"] in (cond, "any")
        if not src_ready(src):
            if match:
                return None, f"{src.relative_to(ROOT)} not ready yet"
            continue
        left = n_segments(src) - len(seg_dirs(pass_dir(e["name"])))
        if left <= 0:
            continue
        finish = now + (n_waiting + ahead + left) * plan["copy_s"] + (ahead + left) * plan["record_s"]
        if finish > plan["end_t"]:
            why = f"{e['name']} would finish copying at {dt.datetime.fromtimestamp(finish):%H:%M}, after the end"
            continue
        if match:
            return e, None
        ahead += left  # a pass of the other light, recorded when its light comes: its time stays reserved
    return None, why


# ---------------------------------------------------------------- room, board, calibration
def room_light():
    r = run([PY, "-I", "tools/room_light.py"], timeout=60)
    try:
        return float(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def condition(light):
    if light is None:
        return None
    return "dark" if light < DARK_MAX else "lit" if light >= LIT_MIN else None


def board_state():
    """'asleep' (parked by tools/board_sleep.sh: WFI loop in SRAM1, sensor supply PG7 low), 'running' or
    'unreachable'; reads memory without stopping the CPU"""
    r = run(OCD + ["-c", "init", "-c", "mdw 0x20010000", "-c", "mdw 0x40021814", "-c", "shutdown"], timeout=60)
    txt = r.stdout + r.stderr
    m1, m2 = re.search(r"0x20010000:\s+([0-9a-fA-F]{8})", txt), re.search(r"0x40021814:\s+([0-9a-fA-F]{8})", txt)
    if not (m1 and m2):
        return "unreachable"
    return "asleep" if int(m1.group(1), 16) == 0xE7FDBF30 and not int(m2.group(1), 16) & 0x80 else "running"


def build_matches():
    """the flash holds work/campaign/flashed.elf (vector table, code and constants compared byte for byte)"""
    cmds, pairs = ["init"], []
    for sec in (".isr_vector", ".text", ".rodata"):
        want, got = CAMP / f"want{sec}.bin", CAMP / f"got{sec}.bin"
        run(["arm-none-eabi-objcopy", "-O", "binary", "-j", sec, ELF, want], timeout=60)
        hdr = run(["arm-none-eabi-objdump", "-h", "-j", sec, ELF], timeout=60).stdout
        addr = int(re.search(rf"{re.escape(sec)}\s+[0-9a-f]+\s+([0-9a-f]+)", hdr).group(1), 16)
        got.unlink(missing_ok=True)
        cmds.append(f"dump_image {got} {addr:#x} {want.stat().st_size:#x}")
        pairs.append((want, got))
    run(OCD + sum((["-c", c] for c in cmds + ["shutdown"]), []), timeout=300)
    return all(g.exists() and g.read_bytes() == w.read_bytes() for w, g in pairs)


def board_sleep(state):
    r = run(["tools/board_sleep.sh", "sleep"], timeout=120)
    state["board"] = "asleep"
    save_state(state)
    log("board asleep" if r.returncode == 0 else f"board sleep FAILED: {tail(r)}")


def take_board(state):
    """True when the board runs the campaign's build and is the campaign's to use (woken if asleep)"""
    st = board_state()
    if st == "unreachable":
        log("board not reachable over SWD (unplugged, or another openocd is using the ST-LINK)")
        return False
    if st == "running" and state["board"] != "awake":
        now = time.time()
        if not state["foreign_since"]:
            state["foreign_since"] = now
            save_state(state)
            log("the board is running although the campaign left it asleep: someone may be using it, so the campaign "
                "leaves it alone for up to 3 h (touch work/campaign/PAUSE to pause the campaign instead)")
        if now - state["foreign_since"] < FOREIGN_S:
            return False
        log("the board has been running for 3 h without the campaign: taking it back")
    if state["board"] != "awake":  # asleep, or taken back: check the build before using it
        if not build_matches():
            log("the board holds another build than work/campaign/flashed.elf: waiting (flash that build again, or "
                "copy the new one there if the change was meant)")
            state["foreign_since"] = time.time()
            save_state(state)
            return False
        if st == "asleep":
            r = run(["tools/board_sleep.sh", "wake"], timeout=120)
            time.sleep(8)
            log("board woken" if r.returncode == 0 else f"board wake FAILED: {tail(r)}")
        state["board"], state["foreign_since"] = "awake", None
        save_state(state)
    return True


def calibrate():
    """the monitor homography again (as tools/recalibrate.sh): the 0.6 square, else the 0.4 one; the square used or
    None"""
    run(["tools/xserver.sh", "start"], timeout=60)
    run(["xset", "s", "off", "s", "noblank", "-dpms"], env={"DISPLAY": ":1"}, timeout=30)
    used = None
    for frac in ("0.6", "0.4"):
        run(["tools/show.sh", "square", frac], env={"BLINK_HZ": "10"}, timeout=60)
        time.sleep(3)
        r = run([PY, "-I", "tools/calibrate.py", frac, "1920", "1200"], timeout=300)
        (CAMP / "last_calibration.txt").write_text(r.stdout + r.stderr)
        if r.returncode == 0:
            used = frac
            break
    run(["tools/show.sh", "black"], timeout=60)
    return used


def drift_px(before, after):
    """how far a picture point moves between two calibrations (largest over points on the screen), picture px"""
    mon = cv2.perspectiveTransform(DRIFT_PTS, np.array(before["H_pic_to_mon"], np.float64))
    back = cv2.perspectiveTransform(mon, np.array(after["H_mon_to_pic"], np.float64))
    return round(float(np.linalg.norm((back - DRIFT_PTS).reshape(-1, 2), axis=1).max()), 2)


# ---------------------------------------------------------------- the two jobs
def record_pass(state, e, cond, light):
    name, opts = e["name"], e["opts"]
    n_seg = n_segments(entry_src(e))
    d = pass_dir(name)
    state["attempts"][name] = state["attempts"].get(name, 0) + 1
    save_state(state)
    frac = "kept"
    if time.time() - state["calib_at"] > CALIB_FRESH_S:  # right after the previous pass: its closing calibration
        before = json.loads(CALIB.read_text()) if CALIB.exists() else None
        frac = calibrate()
        if not frac:
            state["calib_fail_at"] = time.time()
            save_state(state)
            log(f"calibration failed before {name} (is the board facing the monitor?): no recording for 30 min")
            return
        state["calib_at"] = time.time()
        save_state(state)
        if before:
            log(f"calibrated ({frac} square); the view moved {drift_px(before, json.loads(CALIB.read_text()))} px "
                "since the previous calibration")
    c0 = json.loads(CALIB.read_text())
    t0 = dt.datetime.now()
    log(f"recording {name} ({' '.join(opts)}), {n_seg - len(seg_dirs(d))} segments, room light {light:.1f}")
    r = run([PY, "-I", "tools/dataset_run.py", "record", d, *opts], timeout=4 * 3600)
    light_after = room_light()
    frac_after = calibrate()
    drift = None
    if frac_after:
        state["calib_at"] = time.time()
        drift = drift_px(c0, json.loads(CALIB.read_text()))
    else:
        state["calib_fail_at"] = time.time()
    save_state(state)
    segs = seg_dirs(d)
    path = d / "CONDITIONS.json"
    info = json.loads(path.read_text()) if path.exists() else {"sessions": []}
    info["sessions"].append({"start": f"{t0:%F %T}", "end": f"{dt.datetime.now():%F %T}",
                             "webcam_mean_fixed_exposure": light, "webcam_after": light_after,
                             "light_changed": condition(light_after) != cond, "calibration_square": frac,
                             "drift_px": drift, "segments_after": len(segs)})
    info.update({"room_light": "off" if cond == "dark" else "on", "webcam_mean_fixed_exposure": light,
                 "monitor": MONITOR, "options": " ".join(opts), "recorded": info["sessions"][0]["start"],
                 "playback": "60 fps with vsync" if "60" in str(entry_src(e)) else "source frame rate",
                 "drift_px": max((s["drift_px"] for s in info["sessions"] if s["drift_px"] is not None), default=None),
                 "light_changed": any(s["light_changed"] for s in info["sessions"])})
    path.write_text(json.dumps(info, indent=1))
    log(f"recorded {name}: {len(segs)}/{n_seg} segments in {(dt.datetime.now() - t0).seconds // 60} min, room light "
        f"after {light_after}, view drift during the pass {drift} px"
        + ("" if r.returncode == 0 else f", dataset_run {tail(r)}")
        + ("" if frac_after else ", closing calibration FAILED"))


def fetch_one(state, seg):
    key = str(seg.relative_to(ROOT))
    t0 = time.time()
    if not (seg / "events.raw").exists():
        r = run([PY, "-I", "tools/record_dataset.py", "fetch", seg], timeout=3600)
        if r.returncode != 0 or not (seg / "events.raw").exists():
            state["fail"][key] = state["fail"].get(key, 0) + 1
            save_state(state)
            log(f"copy of {key} FAILED ({state['fail'][key]}/3): {tail(r)}")
            return
    size, t1 = (seg / "events.raw").stat().st_size, time.time()
    r = run([PY, "-I", "tools/record_dataset.py", "label", seg], timeout=3600)
    if r.returncode != 0:
        state["fail"][key] = state["fail"].get(key, 0) + 1
        save_state(state)
        log(f"labelling of {key} FAILED ({state['fail'][key]}/3): {tail(r)}")
        return
    ev = json.loads((seg / "events.json").read_text())
    log(f"copied {key}: {size / 1e6:.1f} MB in {t1 - t0:.0f} s ({size / 1e3 / max(t1 - t0, 1):.0f} kB/s), "
        f"{ev['pictures']} pictures, {ev['boxes']} boxes, sync {ev['sync'].get('method')}")
    if not any(not copied(s) for s in seg_dirs(seg.parent)):
        r = run([PY, "-I", "tools/dataset_run.py", "fetch", seg.parent], timeout=600)
        log(f"pass {seg.parent.name} complete: {tail(r)}")


# ---------------------------------------------------------------- loop and reports
def campaign(take=False):
    CAMP.mkdir(parents=True, exist_ok=True)
    if not ELF.exists():
        shutil.copy2(ROOT / "firmware/build/person_detector.elf", ELF)
    state = load_state()
    if take or state["board"] is None:
        state["board"] = "awake" if board_state() == "running" else "asleep"
    state["foreign_since"] = None
    save_state(state)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    plan = load_plan()
    log(f"campaign running until {plan['end']}: {len(plan['entries'])} plan entries; board {state['board']}")
    paused, last_why = False, None
    try:
        while True:
            if (CAMP / "STOP").exists():
                log("STOP file: ending")
                break
            if (CAMP / "PAUSE").exists():
                if not paused:
                    log("paused (work/campaign/PAUSE)")
                    if state["board"] == "awake":
                        board_sleep(state)
                    paused = True
                time.sleep(60)
                continue
            if paused:
                log("resumed")
                paused = False
            try:
                try:
                    plan = load_plan()
                except (OSError, ValueError, KeyError) as err:
                    log(f"plan.json unreadable ({err}): keeping the previous plan")
                if time.time() >= plan["end_t"]:
                    log("end of the campaign")
                    break
                pri, other = waiting_segments(state, plan)
                did = False
                if time.time() - state["calib_fail_at"] > CALIB_BACKOFF_S:
                    light = room_light()
                    cond = condition(light)
                    e, why = choose(plan, state, cond, len(pri)) if cond else (None, f"room light {light}: neither dark nor lit")
                    if why != last_why and why:
                        log(f"no recording: {why}")
                    last_why = why
                    if e and take_board(state):
                        record_pass(state, e, cond, light)
                        did = True
                if not did:
                    seg = pri[0] if pri else (other[0] if other and time.time() + plan["copy_s"] < plan["end_t"] else None)
                    if seg is not None and take_board(state):
                        fetch_one(state, seg)
                        did = True
                if not did:
                    if state["board"] == "awake":
                        board_sleep(state)
                    for _ in range(IDLE_S // 60):
                        if (CAMP / "STOP").exists() or (CAMP / "PAUSE").exists() or time.time() >= plan["end_t"]:
                            break
                        time.sleep(60)
            except Exception:  # noqa: BLE001  a bad step must not end the campaign
                log("step failed:\n" + traceback.format_exc())
                time.sleep(120)
    finally:
        if state["board"] == "awake":
            board_sleep(state)
        log("campaign stopped")


def fetch_all():
    """after the campaign: copy and label every segment still on the card (plan passes first, then the others), then
    the board to sleep. The board must hold work/campaign/flashed.elf."""
    state, plan = load_state(), load_plan()
    state["board"] = "awake" if board_state() == "running" else "asleep"
    state["foreign_since"] = None
    save_state(state)
    try:
        while not (CAMP / "STOP").exists():
            pri, other = waiting_segments(state, plan)
            segs = pri + other
            if not segs:
                log("fetch: nothing left on the card")
                break
            if not take_board(state):
                time.sleep(300)
                continue
            log(f"fetch: {len(segs)} segments left")
            fetch_one(state, segs[0])
    finally:
        if state["board"] == "awake":
            board_sleep(state)
        log("fetch stopped")


def status():
    state, plan = load_state(), load_plan()
    pri, other = waiting_segments(state, plan)
    print(f"end {plan['end']}; waiting on the card: {len(pri)} plan segments (~{len(pri) * plan['copy_s'] / 3600:.1f} h "
          f"of copying), {len(other)} other")
    for e in plan["entries"]:
        src = entry_src(e)
        d = pass_dir(e["name"])
        segs = seg_dirs(d)
        n = n_segments(src) if src_ready(src) else "?"
        print(f"  {e['name']:34s} {e['light']:4s} recorded {len(segs)}/{n}, copied {sum(copied(s) for s in segs)}")
    pics = boxes = 0
    for e in plan["entries"]:
        for s in seg_dirs(pass_dir(e["name"])):
            if copied(s):
                ev = json.loads((s / "events.json").read_text())
                pics, boxes = pics + ev["pictures"], boxes + ev["boxes"]
    print(f"plan data copied and labelled: {pics} pictures, {boxes} boxes")
    print(f"board {state['board']}" + ("; PAUSE file present" if (CAMP / "PAUSE").exists() else "")
          + ("; STOP file present" if (CAMP / "STOP").exists() else ""))
    if LOG.exists():
        print("".join(LOG.read_text().splitlines(keepends=True)[-10:]), end="")


def check():
    """the pieces the loop needs, without recording: board state, build, room light, a calibration"""
    CAMP.mkdir(parents=True, exist_ok=True)
    if not ELF.exists():
        shutil.copy2(ROOT / "firmware/build/person_detector.elf", ELF)
    print("board", board_state())
    print("build matches flashed.elf:", build_matches())
    light = room_light()
    print("room light", light, "->", condition(light))
    before = json.loads(CALIB.read_text())
    frac = calibrate()
    print("calibration", frac, "drift since the previous one", drift_px(before, json.loads(CALIB.read_text())) if frac else "-")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        campaign(take="--take-board" in sys.argv)
    elif cmd == "status":
        status()
    elif cmd == "check":
        check()
    elif cmd == "fetch":
        fetch_all()
    else:
        sys.exit(__doc__)
