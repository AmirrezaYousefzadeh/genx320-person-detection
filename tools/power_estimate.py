"""Indicative power of the detector, from ST's characterisation data (the numbers STM32CubeMX's Power Consumption
Calculator uses), not measured.

STM32F746 datasheet (DS10916, DocID027590 Rev 1), typical values at 25 degC, VDD 3.3 V:
  Table 26: Run mode, code and data from Flash (ART on, L1 cache on), regulator on, 216 MHz (scale 1, over-drive):
            108 mA with all peripherals disabled (186 mA with all enabled)
  Table 30: Sleep mode at 216 MHz: 35 mA with all peripherals disabled
  Table 36: per-peripheral current in uA/MHz (scale 1), times its bus clock
The firmware enables (RCC clocks in firmware/): GPIOA-K, DMA2 (4 streams), DMA2D, DCMI, FMC, the bus matrix
(AHB, 216 MHz); TIM2, TIM6, I2C1, I2C3, PWR (APB1, 54 MHz); TIM1, USART1, SDMMC1, SYSCFG, LTDC (APB2, 108 MHz).
FreeRTOS has no idle hook (no WFI), so the core is in Run mode all the time.
Not included: the external SDRAM, the LCD panel and backlight, the ST-LINK, regulators, the sensor's own supply
(GenX320: a few mW, Prophesee's product brief).
Usage: power_estimate.py [network ms per run ...]
"""
import sys

VDD = 3.3
HCLK, APB1, APB2 = 216, 54, 108
BASE_RUN_MA = 108.0   # Table 26, all peripherals disabled, 216 MHz
BASE_SLEEP_MA = 35.0  # Table 30
UA_PER_MHZ = {        # Table 36, scale 1
    "AHB1": {"GPIOA": 2.2, "GPIOB": 2.1, "GPIOC": 2.3, "GPIOD": 2.2, "GPIOE": 2.2, "GPIOF": 2.2, "GPIOG": 2.1,
             "GPIOH": 2.0, "GPIOI": 2.3, "GPIOJ": 2.2, "GPIOK": 2.0, "DMA2 (4 streams)": 2.2 * 4 + 4.9,
             "DMA2D": 87.1, "bus matrix": 21.06},
    "AHB2": {"DCMI": 5.1},
    "AHB3": {"FMC": 18.9},
    "APB1": {"TIM2": 5.5, "TIM6": 1.2, "I2C1": 2.7, "I2C3": 2.4, "PWR": 0.2},
    "APB2": {"TIM1": 13.0, "USART1": 5.5, "SDMMC1": 4.6, "SYSCFG": 1.5, "LTDC": 28.8},
}
CLOCK = {"AHB1": HCLK, "AHB2": HCLK, "AHB3": HCLK, "APB1": APB1, "APB2": APB2}


def main(times_ms):
    periph = {n: v * CLOCK[bus] / 1000 for bus, d in UA_PER_MHZ.items() for n, v in d.items()}
    run_ma = BASE_RUN_MA + sum(periph.values())
    p_run = run_ma * VDD / 1000
    print(f"MCU in Run mode at 216 MHz: {BASE_RUN_MA:.0f} mA core + {sum(periph.values()):.1f} mA peripherals "
          f"= {run_ma:.0f} mA, {p_run * 1000:.0f} mW at {VDD} V")
    for n, v in sorted(periph.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {n:18s} {v:5.1f} mA")
    sleep_ma = BASE_SLEEP_MA + sum(periph.values())
    print(f"(for reference: Sleep mode with the same peripherals {sleep_ma:.0f} mA, {sleep_ma * VDD:.0f} mW)")
    for t in times_ms:
        print(f"network run of {t:g} ms: {p_run * t:.0f} mJ")


if __name__ == "__main__":
    main([float(a) for a in sys.argv[1:]] or [230, 360, 514, 450])
