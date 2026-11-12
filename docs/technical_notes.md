# Technical notes

The engineering details behind the paper: how the firmware, the sensor set-up, the dataset recording and the
measurements work, with the numbers we measured. The paper gives the story and the main results; this page keeps the
rest. Board: STM32F746G-DISCO (Cortex-M7 at 216 MHz, 320 KB SRAM, 8 MB SDRAM) with the Prophesee GenX320 starter kit.

## 1. Firmware data path

- **Event input.** The GenX320 streams EVT2.0 words over its parallel interface into the STM32's camera interface
  (DCMI). Its DMA fills a 60 KB ring in the core-coupled DTCM, about 15 ms of events at 1 Mev/s. The event reader
  stays eight words behind the DMA counter: the counter leads the memory by up to one FIFO burst, and reading up to it
  returns stale words of the previous lap.
- **Event picture.** An event task decodes the ring into a 320x320 picture over a 50 ms window: one byte per pixel,
  128 where nothing happened, 255 if the last event was ON, 0 if it was OFF. Each event is a single byte store. The
  display shows the same picture.
- **Network.** Each picture is stretched (nearest neighbour) to the network's input of N x N pixels (160 in the demo).
  The network is ST's YOLO-LC v1 (283 k parameters), fine-tuned on event pictures, split in two parts: the first layers
  (input down to N/4 x N/4 x 32) need a tensor arena of 176 KiB at 160 px (253 KiB at 192 px) and run with their arena
  in the SDRAM (cacheable MPU region); the rest needs 93-126 KiB and runs from the internal SRAM. Weights are in the
  1 MB internal flash; the whole image is about 570 KB.
- **Float head.** The last 1x1 convolution (256 -> 30, batch normalisation folded in) runs in single precision on the
  FPU after the int8 body (0.49, 0.77, 1.1 M multiply-adds and 8, 13, 15 ms at 128, 160, 192 px). Its output equals
  the PC's to 1e-5. Reason: one int8 scale cannot serve 30 outputs of very different ranges (box offsets, sizes and
  objectness of five anchors). The int8 calibration set is not the problem: 200 and 2000 pictures give the same result.
- **Boxes and count.** The firmware decodes the boxes, drops a box that overlaps a higher-scored one (IoU above 0.3,
  or more than 60 % of the smaller box covered) as a part of the same person, and draws and counts the boxes scored
  above 0.5. Results are not smoothed over time. A frame gate skips the network when a picture has fewer than 150
  active pixels.
- **Recorder.** A microSD recorder writes the raw EVT2.0 stream (3-5 MB/s sustained) with a header that states the
  sensor configuration; recordings can be replayed through the live pipeline for regression tests. A button on the
  screen starts and stops recording without a PC.

## 2. Sensor-side preprocessing

- **ERC** (event rate controller): drops events evenly above a programmed rate (500 kev/s in all our work). A 1.3 Mev/s
  test pattern overruns the DMA ring and the card without it; with it there are no overruns, and the network is
  preempted less (1000 -> 707 ms on that pattern, measured at 200 MHz with ST's original network).
- **AFK** (anti-flicker): removes 50-520 Hz (lamps, monitor refresh). No visible effect on the detector (precision /
  recall 0.61 / 0.40 against 0.62 / 0.41 without).
- **STC** (spatio-temporal contrast): keeps an event only if its pixel fired again within a time window. Set up with the
  register values of the vendor's driver, it removed almost everything: on the monitor 450 -> 17 kev/s at 10 ms (recall
  0.41 -> 0.14; 0.29 at 50 ms); in the room, for a person walking, waving and sitting, 150-350 kev/s -> 8, 10 and 11
  kev/s at 10, 50 and 100 ms, the same for all three movements, so few that the frame gate skipped every picture (with
  the filter off, 70 % of the network runs found the person). A software filter with the same rule keeps 11-25 % of
  the room events at 10 ms and 24-40 % at 100 ms, even when it demands the same polarity. So the sensor's filter, as
  configured, does not act as a time window: only a constant trickle from a few hot pixels passes. The timestamp
  settings come from the driver's table for a 10 MHz event clock, which may not be this kit's. The filter is off.
- **EHC** (histogram frames): 320x320 count frames with two 4-bit counters per pixel, one every ~119 ms at a 50 ms
  integration period; the sensor needs ~55 ms to drain a frame and its VSYNC comes once per two frames, so the DMA
  counters serve as the frame clock. The kit's driver lacks the EHC register sequences: the start-up sequence comes
  from the GenX320 driver that Prophesee ships with OpenMV, our configuration is written after it. With histogram
  frames the processor load fell from 91-100 % to 66 % at the same detector rate, independent of the event rate (a
  4 Mev/s test pattern costs the same as an empty room). An event-trained model lost recall (0.45 -> 0.30); fine-tuned
  on 379 recorded frames it matched the event mode again (0.68 / 0.43 against 0.65 / 0.45).

## 3. Training

YOLOv2-style loss with an ignore mask; Adam at 1e-4 (5e-5 and 3e-5 for the later stages), batch 32, horizontal flips,
random scale and shift. The COCO weights are re-instantiated at 128, 160 and 224 px (grids 8, 10, 14) and fine-tuned
per size. Training stops on the validation loss of ten PersonPath22 *training* windows (PEDRo's validation split for
the PEDRo networks), so no choice is made on test data; choosing the stopping point on the test sequences instead
inflated AP50 by up to 0.07 in our first runs. Deployment: post-training int8 with 200 representative pictures, then
the float head. A three-channel input with the current picture and the two before it added only 0.012 AP50 on PEDRo
and was not deployed. The training data of every network in the paper is listed in `models/paper/README.md`.

## 4. Recording the dataset

- A 24" 1920x1200 monitor stands in front of the board on a tripod; the sensor sees a ~250 mm square of the screen.
- A pattern of blinking squares gives the monitor-to-sensor homography (median corner error 1-2 px), calibrated before
  and after every pass from the campaign on; the drift between the two is 1-6 px. One exception: the tripod head sank
  by 55 rows between two calibrations; the four passes recorded in between were relabelled.
- The video is warped so that the chosen square crop fills the largest square of the sensor picture that lies on the
  screen (280 of 320 px). A white square blinks for 100 ms once per second inside the view; the labeller finds its
  bursts and fits sensor time to the PC's clock (4-10 ms RMS; ~34 ms for histogram frames).
- A campaign tool runs the passes, calibrations and card copies unattended and picks dark or lit passes from a
  webcam's reading of the room light.
- The monitor's backlight stays at 100 % (PWM dimming makes every pixel flicker); brightness is varied in software.

## 5. Frame steps of a monitor

At the source frame rate every frame change reaches the sensor as one jump and the event rate pulses: 46 % of its
fluctuation power at 14 Hz for a 14 fps video, 44 % at 30 Hz for 30 fps, with peaks at the 500 kev/s cap in 16 % of
the milliseconds; a real person gives under 1 % at any such line. Rendering at 60 fps with motion-compensated
interpolation (dense optical flow both ways, occlusion-aware blending; half the error of frame repetition on held-out
frames) and playing with vertical sync moves the pulsing to the 60 Hz refresh. The picture-to-picture change of the
event density falls from 19 % to 3 % (14 fps source) and from 9 % to 6 % (30 fps); a real person: 3 %. Time structure:
in the 60 fps replays 28-31 % of the events follow an event of the same pixel within 10 ms, against 36-60 % for a
person in the room (9-10 % and 10-26 % within 1 ms). The monitor's own noise is ~0.7 events per pixel per second on a
still grey screen.

## 6. Labels and their checks

See also `docs/N-People.md`. A box is a label only if at least 20 of its pixels hold an event; every mapped box is kept
with its count in `labels_all/`. Ignore regions: the sources' reflections, crowds, distractors, people in vehicles and
hardly visible people, and confident RGB-detector finds without annotation; people below 12 % of the picture height are
ignored in evaluation. RGB-teacher check (YOLO11x at 1280 px on every second source frame, 27.6 k frames, 394 k people
at least 5 % high): it confirms 94.9 % of the MOT17 labels and 89.7 % of PersonPath22's (IoU >= 0.4, any confidence;
78.0 and 76.3 % with confidence >= 0.5); the missed labels are small or hidden (median height 8.9 % of the frame, 77 %
below 12 %, 41 % with visibility below 0.75); confident detections (>= 0.6) on no label: 0.25 per frame on MOT17 and
0.33 on PersonPath22. Timing: the marker fixes time modulo 1 s; an event-density check moved 178 segments by one
second; 91 segments without a usable marker fit: 34 timed from the motion, 57 left out of training and testing.

## 7. Where the time goes

192 px int8 network on a stored picture: 499 ms at 216 MHz; the four convolutions on the large maps 313 ms (73, 88, 77,
75), the 12x12 tail 186 ms. Live, the event task preempts it in proportion to the event rate: 550 ms in a quiet room
(190 kev/s), 591 on a crowd scene (450 kev/s), 620 at the 500 kev/s cap (9, 16, 20 % of the network's time). The rest
of an update: the wait for the next complete picture and under 4 ms of work (stretch 2-2.5 ms, box decoding and
selection 0.5-0.7 ms, display and log), so an update is the network time plus ~53 ms. 160 px with float head: 362 ms
on the bench picture; live in an office (683 updates in 5 min, median 104 kev/s, measured with an earlier network of
the same size and the dense first layer) 378 ms median (410 at the 90th percentile), an update every 0.44 s; the event
task takes 3-11 % of the core. Replayed room recordings: 448-565 ms per run.

## 8. Sparsity

- **First layer.** 94-98 % of the input pixels are "nothing". After quantisation the first convolution sees
  (int8 - zero point) = the pixel value itself, so an event-free window gives bias + 128 x (sum of taps), the same 16
  values everywhere. The kernel (`firmware/App/sparse_conv1.c`) starts every output row from that constant, scatters
  the active pixels (at most 2x2 outputs, 64 multiplies each) and requantises only the touched columns. Bit-exact with
  TFLite's reference kernels and CMSIS-NN; 21.6 ms instead of 73 ms on a picture with 15.6 % active pixels at 192 px,
  71 ms on a dense photograph; the 192 px network: 443 instead of 499 ms. Any input size (multiple of 4) and the float
  head are supported; the constants per model are made by `tools/sparse_conv1.py` (check against TFLite's *reference*
  kernels: XNNPACK rounds differently).
- **Deeper layers.** Activations are 40-60 % zero, but CMSIS-NN computes int8 in SIMD pairs (SMLAD), so only whole
  four-channel words can be skipped, and 1-11 % of the words are zero. A group-lasso term on the words raised this to
  16-18 % in the first three layers at no cost (F1 0.446 against 0.447) and 20-24 % for -0.015 F1, far below the ~75 %
  at which skipping pays off on this core.
- **Delta inference** across pictures does not fit event pictures: 76 % of the 8x8 input tiles change between two
  pictures, and from the second layer on all activations change.

## 9. Power (indicative)

From ST's characterisation data for the STM32F746 (typical, 25 degC, 3.3 V; the data behind STM32CubeMX's power
calculator): the core in Run mode at 216 MHz 108 mA, the peripherals the firmware enables (camera interface, DMA,
display controller, memory controller, card interface) 43 mA: about 0.5 W at all times. A network run costs about
0.18 J at 160 px (0.11 J at 128, 0.26 J at 192). The GenX320 adds a few milliwatts; SDRAM and display not included.
