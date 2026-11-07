#!/bin/sh
# Put the board to sleep (display and GenX320 off) or wake it up, over the ST-LINK. No firmware change needed.
#
# sleep: reset the MCU, keep it from booting, drive LCD backlight (PK3), LCD_DISP (PI12) and the three sensor
#        supply enables PSU_EN_1V1 (PG7), PSU_EN_1V8 (PI0), PSU_EN_2V5 (PH6) low, then park the CPU in a WFI loop
#        in SRAM1 with the debugger released. Only the hardware power LEDs stay on.
# wake:  reset and run the firmware (same as the black RESET button or unplugging the USB cable).
# Usage: board_sleep.sh sleep|wake
set -e
cfg=board/stm32f746g-disco.cfg

case "$1" in
sleep)
    openocd -f $cfg -c init -c "reset halt" \
        -c "mww 0x40023830 0x001005C0" \
        -c "mww 0x40022800 0x00000040; mww 0x40022000 0x01000001" \
        -c "mww 0x40021800 0x00004000; mww 0x40021C00 0x00001000" \
        -c "mww 0x20010000 0xE7FDBF30; resume 0x20010000; mww 0xE000EDF0 0xA05F0000" \
        -c shutdown >/dev/null 2>&1
    echo "board asleep" ;;
wake)
    openocd -f $cfg -c init -c "reset run" -c shutdown >/dev/null 2>&1
    echo "board running" ;;
*)
    echo "usage: $0 sleep|wake" >&2
    exit 1 ;;
esac
