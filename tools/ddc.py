"""Brightness and contrast of the monitor in front of the sensor over DDC/CI (Samsung SyncMaster 2493HM on
/dev/i2c-10, the Intel HDMI output). monitorcontrol fails on this monitor because it reads the reply in two
transactions and the monitor restarts its reply on each one; here one write, a pause, one read.
Usage: ddc.py                    print brightness, contrast, input
       ddc.py brightness N       set brightness 0..100 (and print the readback)
       ddc.py contrast N         set contrast 0..100
The PC user must be in group i2c.
"""
import fcntl
import os
import sys
import time

BUS = os.environ.get("DDC_BUS", "/dev/i2c-10")
I2C_SLAVE = 0x0703
VCP = {"brightness": 0x10, "contrast": 0x12, "input": 0x60}


class Ddc:
    def __init__(self, bus=BUS):
        self.fd = os.open(bus, os.O_RDWR)
        fcntl.ioctl(self.fd, I2C_SLAVE, 0x37)

    def close(self):
        os.close(self.fd)

    def _write(self, payload):
        pkt = [0x51, 0x80 | len(payload)] + payload
        chk = 0x6E
        for b in pkt:
            chk ^= b
        os.write(self.fd, bytes(pkt + [chk]))

    def get(self, code, tries=3):
        """(current, maximum) of a VCP feature."""
        r = b""
        for _ in range(tries):
            self._write([0x01, code])
            time.sleep(0.05)
            r = os.read(self.fd, 12)
            if r[0] == 0x6E and r[1] == 0x88 and r[2] == 0x02 and r[3] == 0x00 and r[4] == code:
                return (r[8] << 8) | r[9], (r[6] << 8) | r[7]
            time.sleep(0.1)
        raise OSError(f"no valid DDC/CI reply for VCP {code:#x}: {r.hex(' ')}")

    def set(self, code, value):
        self._write([0x03, code, (value >> 8) & 0xFF, value & 0xFF])
        time.sleep(0.05)


def main():
    d = Ddc()
    if len(sys.argv) == 3:
        name, value = sys.argv[1], int(sys.argv[2])
        d.set(VCP[name], value)
        time.sleep(0.2)
        cur, mx = d.get(VCP[name])
        print(f"{name} {cur}/{mx}")
        return
    for name, code in VCP.items():
        cur, mx = d.get(code)
        print(f"{name} {cur}/{mx}")


if __name__ == "__main__":
    main()
