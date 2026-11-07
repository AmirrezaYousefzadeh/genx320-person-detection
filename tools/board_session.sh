#!/bin/sh
# After the campaign (2026-11-06): on-board timing and float-head output checks at 128/160/192 px, then the demo
# firmware (firmware/build = int8 body + float head, counting threshold 0.5), a counting check on replayed room
# recordings, and the board to sleep. Output: work/results/board_<time>.txt
cd "$(dirname "$0")/.." || exit 1
P=work/venv/bin/python
OUT=work/results/board_$(date +%Y%m%d_%H%M).txt
flash() { openocd -f board/stm32f746g-disco.cfg -c "program $1 verify reset exit" > /dev/null 2>&1 && echo "flashed $1" >> $OUT; }
tools/board_sleep.sh wake >> $OUT
for b in "build_p128_bench work/finetune/pedro128_a 128" "build_v7hf_bench work/finetune/finetune_v7room_160 160" \
         "build_v7hf192_bench work/finetune/finetune_v7room 192"; do
    set -- $b
    flash firmware/$1/person_detector.elf
    sleep 25  # the bench runs the network three times at start-up
    echo "== $1 ($3 px)" >> $OUT
    $P -I tools/bench_hf.py firmware/$1/person_detector.elf $2/hf_body_int8.tflite $2/hf_head.npz $3 >> $OUT 2>&1
done
flash firmware/build/person_detector.elf
sleep 15
echo "== demo firmware: replayed room recordings (180 empty room, 181 walk 3 m, 182 walk 5 m, 185 sit, 188 occlusion)" >> $OUT
SDREC_ELF=firmware/build/person_detector.elf $P -I tools/replay_check.py work/results/replay_demo.txt \
    180:2-12=0 181:4-14=1 182:2-18=1 185:2-18=1 188:2-18=1 >> $OUT 2>&1
tools/board_sleep.sh sleep >> $OUT
echo "ALL_DONE" >> $OUT
cat $OUT
