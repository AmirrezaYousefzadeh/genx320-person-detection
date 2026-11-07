/*
 * microSD recorder: writes the raw EVT2.0 stream to files on the card, for the dataset (see app.h, sd_ctrl_t).
 *
 * The event task pushes every chunk it takes from the DMA ring into the 3.5 MB SDRAM ring (sd_push); this
 * task drains the ring to the open file in 32 KB pieces. Recordings are /evt/recNNNN.raw (Prophesee EVT2 RAW:
 * a 512-byte text header, then the words as they came from the sensor) plus recNNNN.txt with the board
 * ticks and counters. Commands come from the PC over SWD (tools/sdrec.py writes sd_ctrl.cmd); LIST and READ
 * leave their result in the SDRAM ring for the PC to dump. Copyright (c) 2026 Amirreza Yousefzadeh,
 * University of Twente. Licensed under the Apache License 2.0.
 */
#include "app.h"

#include <stdio.h>

#include "ff.h"
#include "stm32746g_discovery_sd.h"

#define CHUNK_WORDS 8192 /* 32 KB per f_write, a multiple of the ring and of a sector */
#define HEADER_BYTES 512
#define DIR_NAME "evt"

/* FatFs objects in the uncached SDRAM: their sector buffers are DMA targets */
static FATFS fs __attribute__((section("Uncached_SDRAM"), aligned(32)));
static FIL file __attribute__((section("Uncached_SDRAM"), aligned(32)));
static uint8_t work[4096] __attribute__((section("Uncached_SDRAM"), aligned(32)));

__attribute__((used)) sd_ctrl_t sd_ctrl = {SD_MAGIC, 0, SD_NOCARD, 0, 0, 0, 0, 0, 0, 0, 0,
                                           (uint32_t)record_buf, REC_WORDS * 4, 0, 0};

static volatile uint32_t head = 0, tail = 0; /* word indices into record_buf */
static volatile uint32_t recording = 0;
static uint32_t words_total = 0;

static inline uint32_t used(void) { return (head + REC_WORDS - tail) % REC_WORDS; }

void sd_push(const uint32_t *words, uint32_t n) {
  if (!recording || n == 0) return;
  if (used() + n >= REC_WORDS) {
    sd_ctrl.dropped += n;
    return;
  }
  uint32_t h = head, first = REC_WORDS - h;
  if (first > n) first = n;
  memcpy(record_buf + h, words, first * sizeof(uint32_t));
  if (n > first) memcpy(record_buf, words + first, (n - first) * sizeof(uint32_t));
  head = (h + n) % REC_WORDS;
}

/* ---- files ----------------------------------------------------------------------------------------- */

static FRESULT write_header(void) {
  char hdr[HEADER_BYTES + 1];
  int n = snprintf(hdr, sizeof(hdr),
#if HISTO_MODE
                   "%% histo 1.0\n%% format HISTO3D;height=320;width=320;frame_header=16;off_bits=0-3;on_bits=4-7\n"
                   "%% geometry 320x320\n"
#else
                   "%% evt 2.0\n%% format EVT2;height=320;width=320\n%% geometry 320x320\n"
#endif
                   "%% camera_integrator_name Prophesee\n%% sensor_generation 320\n"
                   "%% recorder GenX320 person detector, STM32F746G-DISCO, file %lu, board tick %lu ms\n"
                   "%% sensor_config erc_kevps=%lu afk=%d stc=%d stc_ms=%d histo=%d\n"
                   "%% pad ",
                   (unsigned long)sd_ctrl.file_no, (unsigned long)sd_ctrl.start_tick,
                   (unsigned long)sensor_erc_kevps(), SENSOR_AFK, SENSOR_STC, SENSOR_STC ? STC_THRESHOLD_MS : 0,
                   HISTO_MODE);
  memset(hdr + n, ' ', HEADER_BYTES - n);
  memcpy(hdr + HEADER_BYTES - 7, "\n% end\n", 7);
  UINT bw;
  return f_write(&file, hdr, HEADER_BYTES, &bw);
}

static uint32_t highest_file_no(void) {
  DIR dir;
  FILINFO fno;
  uint32_t best = 0;
  if (f_opendir(&dir, "/" DIR_NAME) != FR_OK) return 0;
  while (f_readdir(&dir, &fno) == FR_OK && fno.fname[0]) {
    unsigned no;
    if (sscanf(fno.fname, "rec%u.raw", &no) == 1 && no > best) best = no;
  }
  f_closedir(&dir);
  return best;
}

static FRESULT start_recording(void) {
  char name[32];
  FRESULT r = f_mkdir("/" DIR_NAME);
  if (r != FR_OK && r != FR_EXIST) return r;
  sd_ctrl.file_no = highest_file_no() + 1;
  snprintf(name, sizeof(name), "/" DIR_NAME "/rec%04lu.raw", (unsigned long)sd_ctrl.file_no);
  if ((r = f_open(&file, name, FA_CREATE_ALWAYS | FA_WRITE)) != FR_OK) return r;
  head = tail = 0;
  sd_ctrl.bytes = sd_ctrl.dropped = 0;
  sd_ctrl.start_tick = HAL_GetTick();
  sd_ctrl.stop_tick = 0;
  words_total = 0;
  if ((r = write_header()) != FR_OK) return r;
  recording = 1;
  return FR_OK;
}

static FRESULT write_words(uint32_t n) {
  UINT bw;
  FRESULT r = f_write(&file, record_buf + tail, n * sizeof(uint32_t), &bw);
  tail = (tail + n) % REC_WORDS;
  words_total += n;
  sd_ctrl.bytes += bw;
  return r;
}

static FRESULT stop_recording(void) {
  recording = 0;
  sd_ctrl.stop_tick = HAL_GetTick();
  FRESULT r = FR_OK;
  while (used() >= CHUNK_WORDS && r == FR_OK) r = write_words(CHUNK_WORDS);
  if (used() && r == FR_OK) r = write_words(used());
  FRESULT rc = f_close(&file);
  if (r == FR_OK) r = rc;
  char name[32], text[400];
  snprintf(name, sizeof(name), "/" DIR_NAME "/rec%04lu.txt", (unsigned long)sd_ctrl.file_no);
  if (f_open(&file, name, FA_CREATE_ALWAYS | FA_WRITE) == FR_OK) {
    int n = snprintf(text, sizeof(text),
                     "file=rec%04lu.raw\nstart_tick_ms=%lu\nstop_tick_ms=%lu\nwords=%lu\ndropped_words=%lu\n"
#if HISTO_MODE
                     "ring_overruns=%lu\nformat=HISTO3D frames 320x320, 512-byte header, per frame 16-byte header "
                     "{magic HFRM, tick_ms, seq, 0} + 102400 bytes (OFF count low nibble, ON count high nibble)\n"
                     "picture=X=319-y,Y=x\n",
#else
                     "ring_overruns=%lu\nformat=EVT2 320x320, 512-byte header\npicture=X=319-y,Y=x\n",
#endif
                     (unsigned long)sd_ctrl.file_no, (unsigned long)sd_ctrl.start_tick,
                     (unsigned long)sd_ctrl.stop_tick, (unsigned long)words_total, (unsigned long)sd_ctrl.dropped,
                     (unsigned long)ev_overruns);
    UINT bw;
    f_write(&file, text, n, &bw);
    f_close(&file);
  }
  return r;
}

/* Directory listing as text into the SDRAM buffer */
static FRESULT list_dir(const char *path, char *out, uint32_t cap, uint32_t *len) {
  DIR dir;
  FILINFO fno;
  FRESULT r = f_opendir(&dir, path);
  if (r != FR_OK) return r;
  while (f_readdir(&dir, &fno) == FR_OK && fno.fname[0] && *len + 300 < cap) {
    *len += snprintf(out + *len, cap - *len, "%s/%s %c %lu\n", path, fno.fname, (fno.fattrib & AM_DIR) ? 'd' : 'f',
                     (unsigned long)fno.fsize); /* newlib-nano printf has no long long */
    if (fno.fattrib & AM_DIR) {
      char sub[260];
      snprintf(sub, sizeof(sub), "%s/%s", path, fno.fname);
      list_dir(sub, out, cap, len);
    }
  }
  f_closedir(&dir);
  return FR_OK;
}

/* Delete everything below path (path itself stays) */
static FRESULT clean_dir(const char *path) {
  DIR dir;
  FILINFO fno;
  FRESULT r = f_opendir(&dir, path);
  if (r != FR_OK) return r;
  for (;;) {
    r = f_readdir(&dir, &fno);
    if (r != FR_OK || !fno.fname[0]) break;
    char sub[260];
    snprintf(sub, sizeof(sub), "%s/%s", path, fno.fname);
    if (fno.fattrib & AM_DIR) clean_dir(sub);
    f_unlink(sub);
    f_closedir(&dir); /* the directory changed: start over */
    if ((r = f_opendir(&dir, path)) != FR_OK) return r;
  }
  f_closedir(&dir);
  return r;
}

static FRESULT read_file(uint32_t no, uint32_t offset, uint8_t *out, uint32_t cap, uint32_t *len) {
  char name[32];
  snprintf(name, sizeof(name), "/" DIR_NAME "/rec%04lu.raw", (unsigned long)no);
  FRESULT r = f_open(&file, name, FA_READ);
  if (r != FR_OK) return r;
  UINT br = 0;
  if ((r = f_lseek(&file, offset)) == FR_OK) r = f_read(&file, out, cap, &br);
  *len = br;
  f_close(&file);
  return r;
}

static void update_free(void) {
  DWORD nclst;
  FATFS *p;
  HAL_SD_CardInfoTypeDef info;
  BSP_SD_GetCardInfo(&info);
  sd_ctrl.card_mb = (uint32_t)(((uint64_t)info.LogBlockNbr * info.LogBlockSize) >> 20);
  if (f_getfree("", &nclst, &p) == FR_OK) sd_ctrl.free_mb = (uint32_t)(((uint64_t)nclst * p->csize) / 2048);
}

/* ---- replay --------------------------------------------------------------------------------------------
   The playlist's files are read in 32 KB pieces into the SDRAM ring (idle while not recording); the event task
   takes the words and paints them at their recorded pace. An item starts only when the previous one is drained;
   a file's last piece is padded to a whole piece with words the event task ignores (type 0xE). */
volatile uint32_t sd_playing = 0;
static volatile uint32_t p_head, p_tail, p_seq, p_from, p_to, p_done;
static volatile int64_t p_t0; /* time of the file's first time-high word, us: the item's time zero */

/* Time (us) of the first time-high word at or after byte pos of the open file, -1 if there is none */
static int64_t time_at(uint32_t pos) {
  UINT br = 0;
  if (f_lseek(&file, pos) != FR_OK || f_read(&file, work, sizeof(work), &br) != FR_OK) return -1;
  const uint32_t *w = (const uint32_t *)work;
  for (uint32_t i = 0; i < br / 4; i++)
    if ((w[i] >> 28) == 0x8) return (int64_t)(w[i] & 0x0FFFFFFFu) << 6;
  return -1;
}

/* Byte position about 0.2 s before from_ms, by bisection on the time-high words (a few 4 KB reads): the event
   task then skips little, instead of the card task reading the file from its start (seconds of no picture) */
static uint32_t seek_position(int64_t t0, uint32_t from_ms) {
  uint32_t lo = HEADER_BYTES, hi = (uint32_t)f_size(&file);
  int64_t target = t0 + (int64_t)from_ms * 1000 - 200000;
  if (from_ms == 0 || target <= t0) return HEADER_BYTES;
  while (hi - lo > 2 * sizeof(work)) {
    uint32_t mid = (lo + (hi - lo) / 2) & ~3u;
    int64_t t = time_at(mid);
    if (t < 0 || t > target) hi = mid;
    else lo = mid;
  }
  return lo;
}

uint32_t sd_play_peek(const uint32_t **words) {
  uint32_t h = p_head, t = p_tail;
  if (h == t) return 0;
  *words = record_buf + t;
  return h > t ? h - t : REC_WORDS - t;
}
void sd_play_consume(uint32_t n) { p_tail = (p_tail + n) % REC_WORDS; }
uint32_t sd_play_item_seq(void) { return p_seq; }
int64_t sd_play_t0(void) { return p_t0; }
void sd_play_window(uint32_t *from_ms, uint32_t *to_ms) { *from_ms = p_from, *to_ms = p_to; }
void sd_play_item_done(void) { p_done = 1; }

static int stop_requested(void) {
  if (sd_ctrl.cmd != SD_CMD_STOP) return 0;
  sd_ctrl.cmd = SD_CMD_NONE;
  return 1;
}

static FRESULT play(void) {
  FRESULT r = FR_OK;
  int stop = 0;
  p_head = p_tail = 0;
  sd_ctrl.state = SD_PLAYING;
  sd_playing = 1;
  do {
    for (uint32_t i = 0; i < sd_ctrl.play_n && i < PLAY_MAX && !stop; i++) {
      char name[32];
      snprintf(name, sizeof(name), "/" DIR_NAME "/rec%04lu.raw", (unsigned long)sd_ctrl.play_file[i]);
      if ((r = f_open(&file, name, FA_READ)) != FR_OK) continue; /* a missing file is skipped */
      int64_t t0 = time_at(HEADER_BYTES);
      if (t0 >= 0 && (r = f_lseek(&file, seek_position(t0, sd_ctrl.play_from_ms[i]))) == FR_OK) {
        while (p_head != p_tail && !(stop = stop_requested())) vTaskDelay(1); /* the previous item drains */
        p_from = sd_ctrl.play_from_ms[i], p_to = sd_ctrl.play_to_ms[i], p_done = 0, p_t0 = t0;
        sd_ctrl.play_item = i;
        p_seq++;
        while (!stop && !p_done && !(stop = stop_requested())) {
          if (REC_WORDS - 1 - (p_head + REC_WORDS - p_tail) % REC_WORDS < CHUNK_WORDS) {
            vTaskDelay(1); /* the ring is full: the event task is behind the recording's pace */
            continue;
          }
          UINT br = 0;
          if ((r = f_read(&file, record_buf + p_head, CHUNK_WORDS * 4, &br)) != FR_OK) break;
          uint32_t n = br / 4;
          if (n < CHUNK_WORDS) /* end of file: pad the piece, so that the next one starts on a piece boundary */
            for (uint32_t k = n; k < CHUNK_WORDS; k++) record_buf[p_head + k] = 0xE0000000u;
          if (n) p_head = (p_head + CHUNK_WORDS) % REC_WORDS;
          if (n < CHUNK_WORDS) break;
        }
      }
      f_close(&file);
    }
  } while (sd_ctrl.play_loop && sd_ctrl.play_n && !stop);
  while (!stop && p_head != p_tail && !(stop = stop_requested())) vTaskDelay(1);
  sd_playing = 0;
  return r;
}

/* ---- task ------------------------------------------------------------------------------------------ */

static void mount(void) {
  FRESULT r = f_mount(&fs, "", 1);
  sd_ctrl.result = r;
  if (r == FR_OK) {
    update_free();
    sd_ctrl.file_no = highest_file_no();
    sd_ctrl.state = SD_IDLE;
  } else {
    sd_ctrl.state = (BSP_SD_IsDetected() == SD_PRESENT) ? SD_ERROR : SD_NOCARD;
  }
}

void sd_task(void *argument) {
  (void)argument;
  vTaskDelay(pdMS_TO_TICKS(200)); /* let the card power up */
  mount();
  for (;;) {
    if (recording) {
      if (used() >= CHUNK_WORDS) {
        FRESULT r = write_words(CHUNK_WORDS);
        if (r != FR_OK) {
          sd_ctrl.result = r;
          stop_recording();
          sd_ctrl.state = SD_ERROR;
        }
      } else {
        vTaskDelay(1);
      }
      if (sd_ctrl.cmd == SD_CMD_STOP) {
        sd_ctrl.cmd = SD_CMD_NONE;
        sd_ctrl.result = stop_recording();
        update_free();
        sd_ctrl.state = sd_ctrl.result == FR_OK ? SD_DONE : SD_ERROR;
      }
      continue;
    }
    uint32_t cmd = sd_ctrl.cmd;
    if (cmd == SD_CMD_NONE) {
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }
    sd_ctrl.cmd = SD_CMD_NONE;
    if (sd_ctrl.state == SD_NOCARD || sd_ctrl.state == SD_ERROR) mount();
    if (sd_ctrl.state == SD_NOCARD) continue;
    sd_ctrl.state = SD_BUSY;
    uint32_t len = 0;
    FRESULT r;
    switch (cmd) {
      case SD_CMD_START:
        r = start_recording();
        sd_ctrl.state = r == FR_OK ? SD_RECORDING : SD_ERROR;
        break;
      case SD_CMD_LIST:
        r = list_dir("", (char *)record_buf, REC_WORDS * 4, &len);
        sd_ctrl.bytes = len;
        sd_ctrl.state = r == FR_OK ? SD_DONE : SD_ERROR;
        break;
      case SD_CMD_CLEAN:
        r = clean_dir("");
        update_free();
        sd_ctrl.file_no = 0;
        sd_ctrl.state = r == FR_OK ? SD_DONE : SD_ERROR;
        break;
      case SD_CMD_READ:
        r = read_file(sd_ctrl.arg0, sd_ctrl.arg1, (uint8_t *)record_buf, REC_WORDS * 4, &len);
        sd_ctrl.bytes = len;
        sd_ctrl.state = r == FR_OK ? SD_DONE : SD_ERROR;
        break;
      case SD_CMD_PLAY:
        r = play();
        sd_ctrl.state = r == FR_OK ? SD_DONE : SD_ERROR;
        break;
      case SD_CMD_FORMAT: {
        MKFS_PARM opt = {FM_ANY, 0, 0, 0, 0};
        r = f_mkfs("", &opt, work, sizeof(work));
        if (r == FR_OK) mount();
        else sd_ctrl.state = SD_ERROR;
        break;
      }
      default:
        r = FR_INVALID_PARAMETER;
        sd_ctrl.state = SD_ERROR;
    }
    sd_ctrl.result = r;
  }
}
