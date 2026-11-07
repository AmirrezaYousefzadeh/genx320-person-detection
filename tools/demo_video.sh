#!/bin/sh
# The demo playlist on the monitor in front of the sensor, looped: MOT17-09 then MOT17-02 (both filmed with a static
# camera, so only the people give events and the event view stays clean). For filming the board.
# MOT17 is CC BY-NC-SA 3.0: credit it in the video description (motchallenge.net) and do not monetise the video.
# Usage: demo_video.sh [start|stop]
cd "$(dirname "$0")/.." || exit 1
case "${1:-start}" in
stop) exec tools/show.sh black ;;
esac
PL=work/demo_playlist
if [ ! -d $PL ]; then
    mkdir -p $PL
    for s in 09 02; do
        for f in work/datasets/mot17/unzipped/MOT17/train/MOT17-$s-FRCNN/img1/*.jpg; do
            ln -s "$(realpath "$f")" "$PL/s${s}_$(basename "$f")"
        done
    done
fi
DISPLAY=:1 xset s off s noblank -dpms
exec tools/show.sh frames $PL 30 --cover
