#!/bin/sh
# After the board was moved back to face the monitor: check the aim (tools/fov_map.py, nine blinking squares) and
# refit the picture <-> monitor homography (tools/calibrate.py, a blinking centred square). About one minute;
# nobody should stand between the board and the monitor.
cd "$(dirname "$0")/.." || exit 1
DISPLAY=:1 xset s off s noblank -dpms
work/venv/bin/python -I tools/fov_map.py | tail -3
for frac in 0.6 0.4; do  # the smaller square when the larger one is not fully in view
    BLINK_HZ=10 tools/show.sh square $frac > /dev/null
    sleep 3
    if work/venv/bin/python -I tools/calibrate.py $frac 1920 1200 > work/calib/last_calibration.txt 2>&1; then
        tail -2 work/calib/last_calibration.txt; break
    fi
    tail -1 work/calib/last_calibration.txt
done
tools/show.sh black > /dev/null
