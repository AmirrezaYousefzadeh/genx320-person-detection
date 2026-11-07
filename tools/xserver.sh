#!/bin/sh
# The X server for the monitor in front of the sensor (display :1). Since 2026-11-04 the only monitor (Samsung
# SyncMaster 2493HM) hangs on the Intel graphics, where the login screen's X server also runs: ours takes virtual
# terminal 2 (the login screen is on terminal 1, Ctrl+Alt+F1 brings it back; Ctrl+Alt+F2 ours). No config file
# needed: /etc/X11/xorg.conf.d/90-only-boot-gpu.conf keeps both servers on the Intel card. Screen blanking and
# power saving are off: nobody types on this display, and after 10 min the monitor went dark (2026-11-05).
# Usage: xserver.sh start|stop|status
cd "$(dirname "$0")/.." || exit 1
case "$1" in
start)
    if pgrep -f "Xorg :1" > /dev/null; then echo "X :1 already running"; exit 0; fi
    (Xorg :1 vt2 -nolisten tcp -noreset -s 0 -dpms > work/xorg1.log 2>&1 &)  # -s 0 -dpms: never blank
    for i in 1 2 3 4 5 6 7 8; do sleep 1; [ -S /tmp/.X11-unix/X1 ] && break; done
    DISPLAY=:1 xrandr 2>/dev/null | grep " connected" || { echo "X :1 did not start:"; tail -5 work/xorg1.log; exit 1; }
    DISPLAY=:1 xset s off s noblank -dpms ;;
stop)
    "$(dirname "$0")/show.sh" off
    pkill -f "Xorg :1" && echo "X :1 stopped (login screen back on the monitor)" ;;
status)
    pgrep -af "Xorg :1" | grep -v pgrep || echo "X :1 not running"
    echo "active terminal: $(cat /sys/class/tty/tty0/active)" ;;
*) echo "usage: $0 start|stop|status" >&2; exit 1 ;;
esac
