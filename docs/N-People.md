# N-People: person detection with a GenX320 event camera, recorded from a monitor and in a room

**Version 1.0 (November 2026).** Event-camera recordings made with a Prophesee GenX320 (320×320 pixels) on the
STM32F746G-DISCO starter kit, labelled with person boxes. The sensor's own stream is recorded (EVT2.0, with the event
rate controller at 500 kev/s), so every representation (event pictures, histograms, time surfaces) can be derived.

Three subsets:

1. **N-People/MOT17** — the seven MOT17 training sequences (Milan et al. 2016, CC BY-NC-SA 3.0) replayed on a monitor
   in front of the sensor, under controlled variations: two monitors; full view or a fixed square crop per 10 s segment
   with people 30–70 % of the view high ("zoom"); mirroring; displayed brightness (gain 1, 0.6, 0.35); room dark or lit;
   the video at its own frame rate (30, 25 or 14 fps) or rendered to 60 fps with motion-compensated interpolation;
   sensor configuration: plain events, events with the anti-flicker filter (AFK), or the sensor's histogram frames.
2. **N-People/PersonPath22** — one 10 s window (the one with the most visible people) of each of 180 PersonPath22
   videos (Shuai et al. 2022, CC BY-NC 4.0), replayed at 60 fps, zoom 0.4, dark room. 107 of the videos belong to
   PersonPath22's training split and 73 to its test split.
3. **N-People/Room** — one person (an author, who consents to publication) in an office in eleven scripted scenarios of
   20 s (empty room, walking at 3 and 5 m, approaching, standing still and waving, sitting, close, diagonal, fast
   walking, occlusion, empty room again), labelled from the motion; plus two- and three-person composites made by
   overlaying scenarios at the same moments (`composites/`, synthetic, with the overlay recipe).

Size (version 1.0): 837 recorded segments (10 s on the monitor, 20 s in the room; 2.4 h, 2.8 G events), 172 k
pictures of 50 ms, 1.21 M labelled people, 15.8 GB of raw events (16.0 GB with the labels and logs, 12.4 GB as one zip). The exact per-pass numbers are in `dataset_stats.md` (made by `tools/dataset_stats.py`).

## How the labels were made

The monitor–sensor homography is calibrated with a pattern of blinking squares before and after every pass (median
corner error 1–2 px; the drift between the two calibrations is in each pass's `CONDITIONS.json`, 1–6 px). A white
square blinks for 100 ms once a second inside the view; the labeller finds its bursts in the recording and fits sensor
time to the PC's clock (residual 4–10 ms RMS for events, ~34 ms for histogram frames). The video's boxes are mapped
through the homography and the time fit into every 50 ms picture.

- A box becomes a **label** (`labels/`) only if at least 20 events fall inside it: a person who stands still gives no
  events. `labels_all/` keeps every mapped box with its event count, so other rules can be applied.
- **Ignore regions** (class 1): reflections, crowds, people in vehicles, hardly visible people (as annotated in the
  source), and people lower than 12 % of the picture height (38 px). In training and scoring a detection on them is
  neither right nor wrong.
- **Curation.** Every segment was checked by its events (`tools/curate.py align`: the offset and time shift that put
  the most events inside the boxes, and the contrast of event density inside/outside). The blink marker fixes the time
  only modulo one second; 121 segments whose labels were one second off were relabelled, and 14 with too little motion
  to decide are listed in `curation/flags.json`, as are 37 segments without a marker fit (timed by the start command,
  ±0.3 s). The four `sam3*` passes were relabelled after the tripod head sank (`calib_1712.json` keeps the old
  calibration). A strong RGB detector (YOLO11x) was run on the source frames; confident detections that overlap no
  source annotation are listed per sequence in `curation/teacher/` and are ignore regions in the labels.

## Files

The record holds the data as one zip file and, next to it, this README, the licence and a manifest:

```
README.md, LICENSE.txt     this description; the licence (both also inside the zip)
manifest.json              size and SHA-256 of the record's files; segments and bytes per folder of the zip
N-People-v1.0.zip          all data; unzip N-People-v1.0.zip gives the folder N-People/:

N-People/splits.json       the test segments (MOT17-09/11 in every pass; the 73 PersonPath22 test-split videos; room
                           scenarios 03, 06, 09 and their composites) and the training segments (everything else),
                           as paths below N-People/
N-People/dataset_stats.md  the numbers per pass
N-People/curation/         flags.json (segments with uncertain timing), align.csv (label offset, time shift and event
                           contrast per segment), teacher/ (RGB-teacher report, unlabelled detections per sequence)
N-People/MOT17/            the MOT17 passes, one folder per pass (dataset_v1, ..., dataset_camp60_dark_z40_r0_g100, ...)
N-People/PersonPath22/     the PersonPath22 passes (dataset_camp60pp_part1 ... part7)
N-People/Room/             the room recordings (dataset_real1) and their composites (dataset_real1_comp)

<subset>/dataset_<pass>/CONDITIONS.json           room light (webcam mean at a fixed exposure), monitor, options,
                                                  calibration drift
<subset>/dataset_<pass>/<segment>/events.raw      Prophesee EVT2.0 RAW: 512-byte text header, then 32-bit words
                       (% format EVT2;height=320;width=320; % sensor_config erc_kevps=500 afk=.. stc=.. histo=..);
                       histogram passes: 512-byte header, then frames of a 16-byte header
                       {"HFRM", board tick ms, sequence, 0} and 320x320 bytes (two 4-bit counters)
                       frames.log    wall-clock time and index of every monitor frame shown; the crop and the
                                     on-screen square ("warp cx cy side ... target x y side"); marker times
                       calib.json    the homography used for the labels
                       segment.json  source sequence, start frame, recording number, PC times
                       events.json   counts, time-sync fit (method, offset, drift, residual, alias check)
                       labels/pic_NNNN.txt   YOLO lines "c cx cy w h" (normalised) per 50 ms picture; c = 0 person,
                                             1 ignore
                       labels_all/           every mapped box with its event count
```

With the code repository, link the pass folders into its `work/` folder:
`for d in /path/to/N-People/*/dataset_*; do ln -s "$d" work/; done`.

The 50 ms pictures are not in the record: `tools/record_dataset.py label <segment>` of the code repository makes
them from `events.raw` (`pictures/pic_NNNN.png`, 320x320 8-bit: 128 nothing, 255 ON, 0 OFF, last event wins) and
writes the labels again from the source annotations, also with other window lengths or rules. The room recordings
(`Room/dataset_real1`) have no monitor log; their labels come from the motion (`tools/real_label.py`).

## Evaluation protocol

`tools/eval_ap.py`: AP50 and AP50:95 (COCO style, 101-point) on the 50 ms pictures; class-1 regions and people below
12 % of the height are ignored. Report the test sets separately: MOT17 full view (source fps), MOT17 zoomed (source fps),
MOT17 zoomed 60 fps (dark and lit), PersonPath22 test windows. The room scenarios are held out too, but one person in
one room is too small a sample for a meaningful accuracy figure. Keep the test segments out of training and of every
choice made during training (stopping epoch, thresholds): choose on training segments instead, e.g. a few PersonPath22
training windows. With four segments per MOT17 set, report a bootstrap over segments (`eval_ap.py` does), not over
pictures. Reference results of a 0.28 M-parameter detector are in the paper and the code repository.

## What a monitor cannot measure (please read before using the data for temporal methods)

A monitor shows frames: even at 60 fps an edge moves in 16.7 ms steps. This changes the time structure of the events
somewhat: in the 60 fps replays 28-31 % of the events follow an event of the same pixel within 10 ms, against 36-60 %
for a person in the room (9-10 % and 10-26 % within 1 ms). Sensor-side temporal filters (STC) and temporal
representations therefore see the replay a little differently from real motion; the dataset evaluates networks and
spatial representations best. At the source frame
rate the event rate pulses at the frame rate (46 % of its fluctuation power at 14 Hz for a 14 fps video, 44 % at 30 Hz
for 30 fps); at 60 fps the pulsing moves to the refresh rate and the picture-to-picture density change falls from 9–19 %
to 3–6 % (a real person: 3 %). The monitor's noise (~0.7 events per pixel per second on a still grey screen) and
contrast differ from a real scene; the room recordings are there for that check.

## Simulated part (optional, regenerate)

`tools/evsim.py` turns the same videos into the board's pictures with an event-camera simulator fitted on these
recordings (active pixels 13.3 vs 13.9 %, isolated 16 vs 17.5 %, ON share 51 vs 53 %); 140 k pictures from MOT17 and
the PersonPath22 training videos. The simulator is in the code repository; its output is not part of this record.

## Licence and attribution

The record is licensed **CC BY-NC-SA 4.0**. It contains derivatives of MOT17 (CC BY-NC-SA 3.0, Milan et al., "MOT16:
A benchmark for multi-object tracking", arXiv:1603.00831) and of PersonPath22 (CC BY-NC 4.0, Shuai et al., "Large
scale real-world multi-person tracking", ECCV 2022; Amazon asks that the data not be used to identify people or to
derive biometric information, which we pass on to every user of this record). The room recordings are published with
the consent of the person recorded. Tools, firmware and training code: Apache 2.0,
https://github.com/AmirrezaYousefzadeh/genx320-person-detection.

## Citation

Dataset: N-People v1.0, 4TU.ResearchData, DOI 10.4121/e9e04333-cc28-4bfe-9079-7bb252203b02.
