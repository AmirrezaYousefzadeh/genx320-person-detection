# The networks behind the paper's accuracy tables

All are ST's YOLO-LC v1 (STM32 model zoo, 283 k parameters) fine-tuned on event pictures with `tools/finetune.py`;
licence SLA0044 (`firmware/PersonDetection/models/LICENSE.md`): use only on STMicroelectronics devices. Each folder holds

- `int8.tflite`: the whole network in int8 (uint8 input, float output);
- `hf_body_int8.tflite` + `hf_head.npz`: the int8 body and the float head (W 256 x 30, b 30) that the firmware runs with
  `HEAD_FLOAT=1` (`tools/headfloat.py`; `tools/headfloat.py --header-from hf_head.npz OUT.h` writes the firmware header).

Evaluate with `tools/eval_ap.py --model <folder>/hf_body_int8.tflite+<folder>/hf_head.npz --size <px> ...` or
`tools/eval_suite.py` (see the README).

| folder | input | trained on (the chain starts from the COCO model) | paper |
|---|---|---|---|
| `pedro_128`, `pedro_160`, `pedro_192`, `pedro_224` | 128-224 | PEDRo train split (early stopping on its val split) | PEDRo table |
| `row1_as_deployed_160` | 160 | full-view monitor passes + room recordings (the first demo's recipe) | test-set table, row 1 |
| `row2_sim_160` | 160 | + simulated pictures (`tools/evsim.py`) | row 2 |
| `row3_zoomed_160` | 160 | + the zoomed monitor passes | row 3 |
| `row4_sim_zoomed_160` | 160 | + both | row 4 |
| `row5_all_160` | 160 | + the PersonPath22 training windows | row 5 |
| `run_to_run/a`, `run_to_run/b` | 160 | row 5's recipe trained twice under the earlier selection | "two trainings of one recipe differ by up to 0.014" |

Rows 1-5 stop at the epoch with the lowest validation loss on ten PersonPath22 *training* windows (real recordings;
`tools/finetune.py --val uid_vid_00000_w,... --exclude <the test segments>`), so no choice was made on test data.
Rows 3-5 were trained again on 11 November 2026, after the dataset review fixed the timing of the segments without a
marker fit (34 timed from the motion, 57 left out): their training data had changed; rows 1-2 saw none of those
segments. The network on the board is row 5 (`firmware/PersonDetection/models/person_all_v9_160_hf_int8`; v8 was row 5
before the timing fix). The earlier board network (`person_room_v7_160_hf_int8`, row 1's data) had its stopping epoch
picked on the test sequences; the paper reports it next to row 1.

The firmware's own copies (split in two for the board, with their head headers) are in
`firmware/PersonDetection/models/`.
