# The networks behind the paper's accuracy tables

All are ST's YOLO-LC v1 (STM32 model zoo, 283 k parameters) fine-tuned on event pictures with `tools/finetune.py`;
licence SLA0044 (`firmware/PersonDetection/models/LICENSE.md`): use only on STMicroelectronics devices. Each folder holds

- `int8.tflite`: the whole network in int8 (uint8 input, float output);
- `hf_body_int8.tflite` + `hf_head.npz`: the int8 body and the float head (W 256 x 30, b 30) that the firmware runs with
  `HEAD_FLOAT=1` (`tools/headfloat.py`; `tools/headfloat.py --header-from hf_head.npz OUT.h` writes the firmware header).

Evaluate with `tools/eval_ap.py --model <folder>/hf_body_int8.tflite+<folder>/hf_head.npz --size <px> ...` or
`tools/eval_suite.py` (see the README).

| folder | input | trained on (all starting from the COCO model) | paper |
|---|---|---|---|
| `pedro_128`, `pedro_160`, `pedro_192`, `pedro_224` | 128-224 | PEDRo train split (early stopping on its val split) | PEDRo table |
| `deployed_160` | 160 | full-view monitor passes + room recordings (the demo's network) | test-set table, row 1 |
| `deployed+sim_160` | 160 | + simulated pictures (`tools/evsim.py`) | row 2 |
| `deployed+zoomed_160` | 160 | + the zoomed monitor passes | row 3 |
| `deployed+sim+zoomed_160` | 160 | + both | row 4 |
| `deployed+sim+zoomed+pp22_160` | 160 | + the PersonPath22 training windows | row 5 |
| `..._disjoint_selection` | 160 | row 5 retrained, stopping epoch chosen on ten PersonPath22 training windows | selection check |
| `..._repeat` | 160 | row 5 retrained with the original selection | selection check |

The firmware's own copies (split in two for the board, with their head headers) are in
`firmware/PersonDetection/models/`.
