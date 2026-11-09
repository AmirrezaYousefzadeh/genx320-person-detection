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
| `person_room_v7_160_hf_int8` | 160 | **the demo** (default build): monitor passes + room recordings, float head | the deployed network: test-set table (row 1), speed and power |
| `person_room_v7_160_int8` | 160 | the same network, whole int8 (`HEAD_FLOAT=0`) | int8 rows |
| `person_e1b_160_hf_int8` | 160 | the demo network fine-tuned further with simulated events | "+ simulated pictures" |
| `person_room_v7_192_hf_int8` | 192 | the demo network at 192 px, float head | timing at 192 px |
| `person_pedro128_hf_int8` | 128 | fine-tuned on PEDRo at 128 px, float head | PEDRo 128 px, timing at 128 px |
| `person_evt_v3_int8` | 192 | monitor passes only (first week), int8 | sensor-preprocessing table (rate controller, anti-flicker, contrast filter rows) |
| `person_evt_v4_int8` | 192 | as v3, trained with the minimum person size rule, int8; `_part_a2`: the input layer alone, for `SPARSE_L1=1` | sensor-preprocessing table (histogram row), the sparse first layer and the profile |
| `person_hist_v1_int8` | 192 | v4 fine-tuned on the sensor's histogram frames (`HISTO=1`) | sensor-preprocessing table (histogram row) |
| `st_yololcv1_192_int8` | 192 | ST's COCO model as shipped, unchanged | the starting point |

Build with another model: `make MODEL=<model> NN_SIZE=<input> [HEAD_FLOAT=0]` (float-head models need
`HEAD_FLOAT=1`, the default). A new model: `tools/build_hf.sh <training run> <model name> <count threshold> <build dir> <size>`.
