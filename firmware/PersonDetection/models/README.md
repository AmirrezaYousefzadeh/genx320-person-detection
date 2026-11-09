# Networks

All are ST's YOLO-LC v1 (`st_yololcv1`, STM32 model zoo, COCO person detector, 283 k parameters) or networks
fine-tuned from it on event pictures; licence SLA0044 (`LICENSE.md`). Each model is split in two after its third
operator (`tools/split_model.py`), so that the second part runs with its tensor arena in the fast internal SRAM:
`<model>_part_a.tflite` (input to N/4 x N/4 x 32, arena in SDRAM) and `<model>_part_b.tflite` (the rest). The split
is checked on the PC: part A followed by part B gives the full model's output bit for bit.

Float-head models (`*_hf_int8`) end before the last 1x1 convolution; that convolution runs in float on the
Cortex-M7's FPU (`firmware/App/head_float.c`) with the weights in `<model>_head.h` (`tools/headfloat.py`; the
Makefile picks the file of the chosen model).

| model | input | what | paper |
|---|---|---|---|
| `person_all_v8_160_hf_int8` | 160 | **the demo** (default build): trained on all of N-People (monitor passes, zoomed passes, simulated pictures, PersonPath22 windows, room recordings), float head; stopping epoch chosen on ten PersonPath22 training windows; the same network as `models/paper/row5_all_160` | the network on the board: last row of the test-set table, speed, power |
| `person_all_v8_160_int8` | 160 | the same network, whole int8 (`HEAD_FLOAT=0`) | mixed precision (float head against whole int8) |
| `person_room_v7_160_hf_int8` | 160 | the earlier demo network: monitor passes + room recordings, float head; stopping epoch picked on the test sequences | the earlier board network (test-set section); the live log of the throughput table |
| `person_room_v7_160_int8` | 160 | the same network, whole int8 (`HEAD_FLOAT=0`) | - |
| `person_e1b_160_hf_int8` | 160 | the earlier demo network fine-tuned further with simulated events | - (an earlier "+ simulated" network) |
| `person_room_v7_192_hf_int8` | 192 | the earlier demo network at 192 px, float head | timing at 192 px |
| `person_pedro128_hf_int8` | 128 | fine-tuned on PEDRo at 128 px, float head | PEDRo 128 px, timing at 128 px |
| `person_evt_v3_int8` | 192 | monitor passes only (first week), int8 | sensor-preprocessing table (rate controller, anti-flicker, contrast filter rows) |
| `person_evt_v4_int8` | 192 | as v3, trained with the minimum person size rule, int8; `_part_a2`: the input layer alone, for `SPARSE_L1=1` | sensor-preprocessing table (histogram row), the sparse first layer and the profile |
| `person_hist_v1_int8` | 192 | v4 fine-tuned on the sensor's histogram frames (`HISTO=1`) | sensor-preprocessing table (histogram row) |
| `st_yololcv1_192_int8` | 192 | ST's COCO model as shipped, unchanged | the starting point |

Build with another model: `make MODEL=<model> NN_SIZE=<input> [HEAD_FLOAT=0]` (float-head models need
`HEAD_FLOAT=1`, the default). A new model: `tools/build_hf.sh <training run> <model name> <count threshold> <build dir> <size>`.
