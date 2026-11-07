#!/bin/sh
# Show something on the monitor in front of the sensor: stops the running tools/monitor.py (or self_record.py) and starts a new one
# in the background on display :1 (started by tools/xserver.sh).
# Usage: show.sh <mode> [file]   (modes: see monitor.py; "off" only stops)
cd "$(dirname "$0")/.." || exit 1
for p in $(pgrep -f "tools/(monitor|self_record)\.py"); do  # both own the monitor
    case "$(ps -o comm= -p "$p")" in python*) kill "$p" ;; esac
done
[ "$1" = off ] && exit 0
DISPLAY=:1 nohup work/venv/bin/python -I tools/monitor.py "$@" > work/monitor.log 2>&1 &
sleep 1
pgrep -f "tools/monitor.py" > /dev/null && echo "showing: $*" || { echo "monitor.py did not start:"; cat work/monitor.log; exit 1; }
