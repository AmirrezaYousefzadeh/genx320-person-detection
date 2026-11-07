/*
 * On-sensor event processing of the GenX320: the event rate controller (ERC). The sensor drops events
 * evenly within each 100 us reference period so that at most `target` events per period get out: a hard cap
 * on what the microcontroller and the SD card have to handle, with no holes in time.
 *
 * Register sequence after Prophesee's OpenEB (GenX320Erc::activate_dyn, Apache 2.0); register names and
 * fields from the kit's register map (Drivers/Prophesee/genx320/genx320_all_pub_registers.h).
 *
 * Copyright (c) 2026 Amirreza Yousefzadeh, University of Twente. Licensed under the Apache License 2.0.
 */
#include "app.h"

#include "genx320.h"

/* register access of the kit driver (psee_sensor_wrapper.c) */
void psee_sensor_write(uint16_t reg, uint32_t data);
void psee_sensor_read(uint16_t reg, uint32_t *data);

#define ERC_REF_PERIOD_US 100

static uint32_t erc_kevps = 0;

uint32_t sensor_erc_kevps(void) { return erc_kevps; }

static void wait_bit(uint16_t reg, uint32_t mask, uint32_t want) {
  for (int i = 0; i < 100; i++) {
    uint32_t v = 0;
    psee_sensor_read(reg, &v);
    if ((v & mask) == want) return;
    vTaskDelay(1);
  }
}

void sensor_erc_set(uint32_t kevps) {
  erc_kevps = kevps;
  if (kevps == 0) { /* off: bypass the block */
    psee_sensor_write(ERC_PIPELINE_CONTROL, ERC_PIPELINE_CONTROL_ENABLE | ERC_PIPELINE_CONTROL_BYPASS);
    return;
  }
  uint32_t target = kevps * ERC_REF_PERIOD_US / 1000; /* events per reference period */
  if (target < 1) target = 1;

  /* hold and flush, then configure in bypass */
  psee_sensor_write(ERC_PIPELINE_CONTROL, 0);
  psee_sensor_write(ERC_PIPELINE_CONTROL, ERC_PIPELINE_CONTROL_ENABLE | ERC_PIPELINE_CONTROL_BYPASS);
  /* power up and initialise the delay FIFO SRAM */
  uint32_t v = 0;
  psee_sensor_read(SRAM_INITN, &v);
  psee_sensor_write(SRAM_INITN, v | SRAM_INITN_ERC_DL);
  psee_sensor_read(SRAM_PD1, &v);
  psee_sensor_write(SRAM_PD1, v & ~SRAM_PD1_ERC_DL_PD);
  /* use the delay FIFO (no bypass), delayed average drop rate */
  psee_sensor_write(ERC_DELAY_FIFO_FLUSH_AND_BYPASS, 0);
  wait_bit(ERC_DELAY_FIFO_FLUSH_AND_BYPASS, ERC_DELAY_FIFO_FLUSH_AND_BYPASS_STATUS, 0);
  psee_sensor_write(ERC_REF_PERIOD_FLAVOR, ERC_REF_PERIOD_FLAVOR_AVG_DROP_RATE_DELAYED |
                                               (ERC_REF_PERIOD_US << ERC_REF_PERIOD_FLAVOR_REFERENCE_Pos));
  psee_sensor_write(ERC_TD_TARGET_EVENT_COUNT, target << ERC_TD_TARGET_EVENT_COUNT_VAL_Pos);
  psee_sensor_write(ERC_DELAY_FIFO_NON_TD_RSVD_AREA,
                    ERC_DELAY_FIFO_NON_TD_RSVD_AREA_AUTO_RAISE | (28u << ERC_DELAY_FIFO_NON_TD_RSVD_AREA_VAL_Pos));
  /* monitoring events: period start (average drop rate) and end (TD count) tags */
  psee_sensor_write(ERC_MONITORING_EVENT_CONTROL, ERC_MONITORING_EVENT_CONTROL_AVG_DROP_RATE_EN |
                                                      ERC_MONITORING_EVENT_CONTROL_IN_TD_CNT_EN |
                                                      ERC_MONITORING_EVENT_CONTROL_TD_EVT_CNT_EN);
  /* dropping in time only (not whole rows/columns), all TD dropped when the drop level reaches 512/1024 */
  wait_bit(ERC_AHVT_DROPPING_CONTROL, ERC_AHVT_DROPPING_CONTROL_STATUS, ERC_AHVT_DROPPING_CONTROL_STATUS);
  psee_sensor_write(ERC_AHVT_DROPPING_CONTROL,
                    ERC_AHVT_DROPPING_CONTROL_T_EN | (512u << ERC_AHVT_DROPPING_CONTROL_DROP_ALL_TD_WHEN_GEQ_Pos));
  psee_sensor_write(ERC_RESET_TDROP_COUNTER_ON_MTAG_FIRST, 0);
  /* go */
  psee_sensor_write(ERC_PIPELINE_CONTROL, ERC_PIPELINE_CONTROL_ENABLE);
}

/* ---- Event Histogram Computer (EHC, make HISTO=1). The sensor counts the events of every pixel during an
   accumulation period and sends the 320x320 counts over the CPI. Histo3D: two 4-bit saturating counters per
   pixel, OFF events in the low nibble and ON events in the high one (Diff3D, one signed count, cancels an ON and
   an OFF on the same pixel: 3x sparser pictures than event mode on MOT17, and lower recall). Sequence after
   psee_ehc_activate() of the GenX320 driver Prophesee ships with OpenMV (Apache 2.0); the EHC shares its SRAM
   with the STC filter, so the two exclude each other. */
#define EHC_ALGO_HISTO3D 1u
#define EHC_TRIGGER_PERIOD 1u

void sensor_ehc_start(uint32_t period_us) {
  if (period_us < 0x10) period_us = 0x10;
  if (period_us > 0x1FFF0) period_us = 0x1FFF0;
  psee_sensor_write(EHC_PIPELINE_CONTROL, 0); /* hold the block while configuring */
  uint32_t v = 0;                             /* power up and initialise its SRAM */
  psee_sensor_read(SRAM_INITN, &v);
  psee_sensor_write(SRAM_INITN, v | SRAM_INITN_EHC_STC);
  psee_sensor_read(SRAM_PD0, &v);
  psee_sensor_write(SRAM_PD0, v & ~(SRAM_PD0_EHC_PD | SRAM_PD0_STC0_PD | SRAM_PD0_STC1_PD));
  psee_sensor_write(EHC_INITIALISATION, EHC_INITIALISATION_REQ_INIT);
  psee_sensor_write(EHC_EHC_CONTROL, (EHC_ALGO_HISTO3D << EHC_CONTROL_ALGO_SEL_Pos) |
                                         (EHC_TRIGGER_PERIOD << EHC_CONTROL_TRIG_SEL_Pos)); /* close the period by time */
  psee_sensor_write(EHC_BITS_SPLITTING, (4u << EHC_BITS_SPLITTING_NEGATIVE_BIT_LENGTH_Pos) |
                                            (4u << EHC_BITS_SPLITTING_POSITIVE_BIT_LENGTH_Pos));
  psee_sensor_write(EHC_INTEGRATION_PERIOD, period_us);                                    /* 16 us steps */
  wait_bit(EHC_INITIALISATION, EHC_INITIALISATION_FLAG_INIT_DONE, EHC_INITIALISATION_FLAG_INIT_DONE);
  psee_sensor_write(EHC_PIPELINE_CONTROL, EHC_PIPELINE_CONTROL_ENABLE);
}

/* ---- Spatio-Temporal Contrast filter (STC): the sensor passes an event only if the same pixel had another event
   within `threshold_ms` before it (a second event confirms real motion; isolated noise events never get a
   second one). Register sequence and timestamping parameters after psee_stc_only_activate() of the GenX320
   driver Prophesee ships with OpenMV (Apache 2.0), its table for the 10 MHz event clock the kit uses. The STC shares
   its SRAM with the EHC, so it is for event mode only. */
typedef struct { uint16_t presc, mult, timeout; } stc_params_t;
static const stc_params_t kStcParams10[101] = { /* index = threshold in ms */
    {0, 0, 0}, {11, 9, 31}, {11, 7, 27}, {12, 11, 30}, {12, 9, 31}, {9, 1, 27}, {12, 7, 32}, {11, 3, 38},
    {13, 11, 30}, {12, 5, 34}, {13, 9, 38}, {10, 1, 43}, {10, 1, 27}, {14, 15, 30}, {13, 7, 32}, {14, 13, 35},
    {12, 3, 38}, {12, 3, 38}, {14, 11, 42}, {13, 5, 47}, {13, 5, 47}, {14, 9, 52}, {14, 9, 52}, {11, 1, 91},
    {11, 1, 59}, {11, 1, 59}, {11, 1, 27}, {11, 1, 27}, {15, 15, 30}, {15, 15, 30}, {14, 7, 32}, {14, 7, 32},
    {15, 13, 35}, {15, 13, 35}, {15, 13, 35}, {13, 3, 38}, {13, 3, 38}, {15, 11, 89}, {15, 11, 42}, {15, 11, 42},
    {15, 11, 42}, {14, 5, 47}, {14, 5, 47}, {14, 5, 47}, {14, 5, 47}, {15, 9, 109}, {15, 9, 52}, {15, 9, 52},
    {15, 9, 52}, {15, 9, 52}, {12, 1, 123}, {12, 1, 123}, {12, 1, 59}, {12, 1, 59}, {12, 1, 59}, {12, 1, 59},
    {16, 15, 64}, {16, 15, 64}, {16, 15, 64}, {15, 7, 69}, {15, 7, 69}, {15, 7, 69}, {15, 7, 69}, {15, 7, 69},
    {16, 13, 74}, {16, 13, 74}, {16, 13, 74}, {16, 13, 74}, {16, 13, 74}, {14, 3, 81}, {14, 3, 81}, {14, 3, 81},
    {14, 3, 81}, {14, 3, 81}, {16, 11, 182}, {16, 11, 89}, {16, 11, 89}, {16, 11, 89}, {16, 11, 89}, {16, 11, 89},
    {16, 11, 89}, {15, 5, 200}, {15, 5, 98}, {15, 5, 98}, {15, 5, 98}, {15, 5, 98}, {15, 5, 98}, {15, 5, 98},
    {15, 5, 98}, {16, 9, 223}, {16, 9, 223}, {16, 9, 223}, {16, 9, 109}, {16, 9, 109}, {16, 9, 109}, {16, 9, 109},
    {16, 9, 109}, {16, 9, 109}, {16, 9, 109}, {13, 1, 251}, {13, 1, 251}};

void sensor_stc_start(uint32_t threshold_ms) {
  if (threshold_ms < 1) threshold_ms = 1;
  if (threshold_ms > 100) threshold_ms = 100;
  const stc_params_t *k = &kStcParams10[threshold_ms];
  psee_sensor_write(STC_PIPELINE_CONTROL, STC_PIPELINE_CONTROL_ENABLE | STC_PIPELINE_CONTROL_BYPASS);
  uint32_t v = 0; /* power up and initialise its SRAM */
  psee_sensor_read(SRAM_INITN, &v);
  psee_sensor_write(SRAM_INITN, v | SRAM_INITN_EHC_STC);
  psee_sensor_read(SRAM_PD0, &v);
  psee_sensor_write(SRAM_PD0, v & ~(SRAM_PD0_STC0_PD | SRAM_PD0_STC1_PD));
  psee_sensor_write(STC_INITIALISATION, STC_INITIALISATION_REQ_INIT);
  psee_sensor_write(STC_STC_PARAM, STC_PARAM_ENABLE | ((threshold_ms * 1000) << STC_PARAM_THRESHOLD_Pos));
  psee_sensor_write(STC_TRAIL_PARAM, 0);
  psee_sensor_write(STC_TIMESTAMPING, ((uint32_t)k->presc << STC_TIMESTAMPING_PRESCALER_Pos) |
                                          ((uint32_t)k->mult << STC_TIMESTAMPING_MULTIPLIER_Pos));
  psee_sensor_write(STC_INVALIDATION, (uint32_t)k->timeout << STC_INVALIDATION_DT_FIFO_TIMEOUT_Pos);
  wait_bit(STC_INITIALISATION, STC_INITIALISATION_FLAG_INIT_DONE, STC_INITIALISATION_FLAG_INIT_DONE);
  psee_sensor_write(STC_PIPELINE_CONTROL, STC_PIPELINE_CONTROL_ENABLE);
}
