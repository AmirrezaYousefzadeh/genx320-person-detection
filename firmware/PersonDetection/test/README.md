Built-in test pictures (192x192 RGB, raw uint8, the grey event picture copied to R, G and B as the firmware does), used
only to check the board against the PC (`make NN_BENCH=1`, `tools/bench_hf.py`, `tools/nn_bench.py`) and the box
drawing (`make TEST_IMAGE=1`):

- `room_192x192.rgb` (the default): a 50 ms event picture of a person walking 3 m from the sensor, from the authors'
  own room recordings (N-People, `dataset_real1/02_walk_3m`, picture 122), stretched from 320x320 with nearest
  neighbour; 6 % active pixels.
- `evt_192x192.rgb` (`make TEST_IMG=PersonDetection/test/evt_192x192.rgb`): an event picture of a MOT17 sequence
  replayed on the monitor (N-People, CC BY-NC-SA 4.0, derived from MOT17, CC BY-NC-SA 3.0); 15.6 % active pixels,
  the picture of the sparse first layer's timing in the paper.
