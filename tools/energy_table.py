"""Indicative energy of the detector's implementations (160 px, float head): measured times on the board times the
STM32F746's datasheet currents (tools/power_estimate.py), not a power measurement.

  P_run   core in Run mode at 216 MHz + the peripherals the firmware enables: 3.3 V x (108 + 43) mA = 0.50 W
  P_sleep the same in Sleep mode (core clock stopped, peripherals and DMA running): 3.3 V x (35 + 43) mA = 0.26 W
  sensor  GenX320 ~3 mW (product brief), not added; SDRAM and display not included
Energy per update = P_run x (network time + 4 ms of other work) + P_x x (the wait for the next picture), with P_x =
P_run in today's firmware (the core never sleeps) and P_sleep with an idle sleep. Power when nothing moves (frame gate,
no network): event mode keeps decoding events (the event task's share at the empty room's ~100 kev/s); histogram mode
only reads a frame every ~120 ms.
Inputs: work/results/bench_ops.json (bench times), replay results (work/results/replay_build_v9*.txt, live network
time per run). Usage: energy_table.py [--latex]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "work/results"
V = 3.3
P_RUN = V * (108 + 43) / 1000
P_SLEEP = V * (35 + 43) / 1000
WORK_MS = 4.0          # stretching the picture, decoding the boxes, display, log (< 4 ms measured)
WINDOW_MS = 50.0       # an event picture; the next one completes on average half a window after the network ends
HISTO_FRAME_MS = 119.0  # histogram frames (EHC) every ~119 ms
EVENT_TASK_PER_KEVS = 0.0004  # share of the core per kev/s (Table V: 9-20 % at 190-500 kev/s)
EMPTY_ROOM_KEVS = 102.0       # the empty-room recordings (Room/dataset_real1/01_empty, 11_empty_end)


def bench(name):
    ops = json.loads((RES / "bench_ops.json").read_text()) if (RES / "bench_ops.json").exists() else {}
    return ops.get(name, {}).get("run_ms")


def replay_ms(build):
    p = RES / f"replay_{build}.txt"
    if not p.exists():
        return None
    v = [int(m) for m in re.findall(r"network (\d+) ms", p.read_text())]
    return float(np.median(v)) if v else None


LIVE_OFFICE_NET_MS, LIVE_OFFICE_UPDATE_MS = 378.0, 440.0  # dense layer 1, live in an office (683 updates, 104 kev/s)
LIVE_ED_NET_MS, LIVE_ED_UPDATE_MS = 339.0, 402.0  # event-driven layer 1, live in the office (2026-11-13, 728 updates, 136 kev/s)
FRAME_CAMERA_PERIOD_MS = 1000 / 30  # a frame sensor at 30 fps (e.g. HM01B0, < 2 mW at QVGA)


def row(name, net_ms, wait_ms, idle_share):
    """net_ms: the network as it runs (live); wait_ms: waiting for the next picture; idle_share: the core's busy share
    while nothing moves (frame gate)"""
    upd = net_ms + WORK_MS + wait_ms
    return {"name": name, "net_ms": round(net_ms, 1), "update_ms": round(upd, 1), "per_s": round(1000 / upd, 2),
            "E_mJ": round(P_RUN * upd, 1), "E_sleep_mJ": round(P_RUN * (net_ms + WORK_MS) + P_SLEEP * wait_ms, 1),
            "idle_mW": round(P_RUN * 1000), "idle_sleep_mW": round((P_SLEEP + idle_share * (P_RUN - P_SLEEP)) * 1000)}


def main(latex=False):
    t_d, t_s = bench("build_v9_bench"), bench("build_v9s_bench")
    live = LIVE_OFFICE_NET_MS / t_d  # the event task's preemption, measured live with the dense layer
    ev_wait = LIVE_OFFICE_UPDATE_MS - LIVE_OFFICE_NET_MS - WORK_MS  # the 50 ms window and scheduling
    share_ev = EVENT_TASK_PER_KEVS * EMPTY_ROOM_KEVS
    rows = [row("Events, dense layer 1", LIVE_OFFICE_NET_MS, ev_wait, share_ev),
            row("Events, event-driven layer 1", LIVE_ED_NET_MS, LIVE_ED_UPDATE_MS - LIVE_ED_NET_MS - WORK_MS, share_ev),
            row("Histogram frames, event-driven layer 1", t_s, HISTO_FRAME_MS / 2, 0.01),
            row("RGB frames, same network (projected)", t_d, FRAME_CAMERA_PERIOD_MS / 2, 0.015)]
    for r in rows:
        print(f"{r['name']:42s} net {r['net_ms']:6.1f} ms  update {r['update_ms']:6.1f} ms ({r['per_s']:.2f}/s)  "
              f"E {r['E_mJ']:5.0f} mJ (sleep {r['E_sleep_mJ']:5.0f})  idle {r['idle_mW']:4.0f} mW (sleep {r['idle_sleep_mW']:4.0f})")
    if latex:
        for r in rows:
            print(f"{r['name']} & {r['net_ms']:.0f} & {r['update_ms']:.0f} & {r['E_mJ']:.0f} / {r['E_sleep_mJ']:.0f} & "
                  f"{r['idle_mW']:.0f} / {r['idle_sleep_mW']:.0f} \\\\")
    (RES / "energy_table.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main("--latex" in sys.argv)
