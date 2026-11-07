/*
 * CPU load from FreeRTOS run-time statistics (DWT cycle counter): how busy the processor is, and the share of
 * each task, over the last second. A cheap power proxy: active cycles per second.
 *
 * Copyright (c) 2026 Amirreza Yousefzadeh, University of Twente. Licensed under the Apache License 2.0.
 */
#include "app.h"

volatile uint32_t cpu_busy_percent = 0;
volatile uint32_t cpu_task_percent[CPU_TASKS] = {0}; /* events, detector, display, sd */

static const char *const task_names[CPU_TASKS] = {"events", "detector", "display", "sd"};

/* Call about once a second (from the display task). */
void cpu_load_update(void) {
  static TaskStatus_t status[12];
  static uint32_t last_total = 0, last_idle = 0, last_task[CPU_TASKS] = {0};
  uint32_t total;
  UBaseType_t n = uxTaskGetSystemState(status, 12, &total);
  uint32_t idle = 0, task[CPU_TASKS] = {0};
  for (UBaseType_t i = 0; i < n; i++) {
    if (strcmp(status[i].pcTaskName, "IDLE") == 0) idle = status[i].ulRunTimeCounter;
    for (int k = 0; k < CPU_TASKS; k++)
      if (strcmp(status[i].pcTaskName, task_names[k]) == 0) task[k] = status[i].ulRunTimeCounter;
  }
  uint32_t d_total = total - last_total;
  if (d_total > 0) {
    cpu_busy_percent = 100 - (uint32_t)((uint64_t)(idle - last_idle) * 100 / d_total);
    for (int k = 0; k < CPU_TASKS; k++) cpu_task_percent[k] = (uint32_t)((uint64_t)(task[k] - last_task[k]) * 100 / d_total);
  }
  last_total = total, last_idle = idle;
  memcpy(last_task, task, sizeof(task));
}
