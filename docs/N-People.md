# N-People: person detection with a GenX320 event camera, recorded from a monitor and in a room

**Version 1.0 (November 2026).** Event-camera recordings made with a Prophesee GenX320 (320×320 pixels) on the
STM32F746G-DISCO starter kit, labelled with person boxes. The sensor's own stream is recorded (EVT2.0, with the
sensor's event rate controller at 500 kev/s), so every representation (event pictures, histograms, time surfaces) can
be derived. Code to read, label and use the data: https://github.com/cognimachina/genx320-person-detection.

Three subsets:

1. **N-People/MOT17** — the seven MOT17 training videos (Milan et al. 2016, CC BY-NC-SA 3.0) replayed on a monitor
   in front of the sensor, 34 passes under controlled variations: two monitors (Eizo, Samsung); full view, panned, or
   a fixed square crop per 10 s segment in which people are 30–70 % of the view high ("zoomed"); mirroring; displayed
   brightness (frame values × 1, 0.6, 0.5, 0.35, 0.25); room dark or lit; the video at its own frame rate (30, 25 or
   14 fps) or interpolated to 60 fps; sensor configuration: events (rate controller only, v1–v3), events with the
   anti-flicker filter (AFK; all later event passes), or the sensor's histogram frames (h1). `passes.csv` lists every
   pass with its settings.
2. **N-People/PersonPath22** — one 10 s window (the one with the most visible people) of each of 180 PersonPath22
   videos (Shuai et al. 2022, CC BY-NC 4.0), replayed at 60 fps, zoomed (people ~40 % of the view high), dark room,
   AFK. 107 of the videos belong to PersonPath22's training split and 73 to its test split.
3. **N-People/Room** — one person (an author, who consents to publication) in an office in eleven
   scripted scenarios of 20 s (empty room, walking at 3 and 5 m, approaching, standing still and waving, sitting,
   close, diagonal, fast walking, occlusion, empty room again), labelled from the motion; plus 4000 two- and
   three-person pictures composed from them (`dataset_real1_comp`, synthetic, with the recipe).

Size (version 1.0): 837 recorded segments in 42 passes (10 s on the monitor, 20 s in the room; 2.4 h, 2.9 G events),
172 k pictures of 50 ms, 1.21 M labelled people, 15.8 GB of raw events (16.0 GB unpacked, 12.5 GB as one zip).
`dataset_stats.md` has the numbers per group of passes.

## Getting started

```
unzip N-People-v1.0.zip                       # -> N-People/
git clone --branch n-people-v1.0 https://github.com/cognimachina/genx320-person-detection   # the tools as of this release
cd genx320-person-detection && pip install -r requirements.txt
mkdir -p work && for d in /path/to/N-People/*/dataset_*; do ln -s "$d" work/; done
python tools/record_dataset.py pictures work/dataset_sam1/*    # 50 ms pictures from events.raw (labels untouched)
```

`record_dataset.py pictures` writes `pictures/pic_NNNN.png` next to `labels/pic_NNNN.txt` in each segment, the
layout every tool of the repository reads (training: `tools/finetune.py`, evaluation: `tools/eval_ap.py`,
`tools/eval_suite.py`). It needs only `events.raw` and `events.json`; the room recordings work the same way. Do
not use `record_dataset.py label` for this: it rewrites the labels from the source videos (MOT17, PersonPath22 and
their 60 fps renderings must then be present under `work/datasets/`), which is how they were made.

Without the repository: the format below is all you need. A minimal reader in Python (NumPy):

```python
import json, numpy as np
def events(path):                                  # EVT2.0 RAW -> x, y (sensor), p (1 ON), t (us)
    data = open(path, "rb").read(); end = data.index(b"% end\n") + 6
    w = np.frombuffer(data[end:end + (len(data) - end) // 4 * 4], "<u4"); typ = w >> 28
    th = np.where(typ == 8, (w & 0x0FFFFFFF).astype(np.int64) << 6, -1)
    i = np.maximum.accumulate(np.where(th >= 0, np.arange(len(w)), 0)); th = th[i]
    cd = (typ <= 1) & (th >= 0); w = w[cd]
    return (w >> 11) & 0x7FF, w & 0x7FF, (typ[cd] == 1).astype(np.uint8), th[cd] + ((w >> 22) & 0x3F)
def pictures(seg, window_ms=50):                   # the labelled 50 ms pictures of a segment (event passes)
    x, y, p, t = events(f"{seg}/events.raw"); t = t - t[0]
    X, Y = 319 - y.astype(int), x.astype(int)      # picture coordinates (see below)
    mask = json.load(open(f"{seg}/events.json")).get("marker_mask_pic")
    for k in range(int(t[-1] // (window_ms * 1000)) + 1):
        s = (t >= k * window_ms * 1000) & (t < (k + 1) * window_ms * 1000)
        pic = np.full((320, 320), 128, np.uint8); pic[Y[s], X[s]] = np.where(p[s] == 1, 255, 0)
        if mask: pic[mask[1]:mask[3], mask[0]:mask[2]] = 128
        yield pic                                  # labels: f"{seg}/labels/pic_{k:04d}.txt"
```

## Coordinates and time

- **Sensor and picture coordinates.** `events.raw` holds the sensor's coordinates (x, y in 0..319) as the GenX320
  sends them. Pictures and labels are in the board's picture frame, the sensor turned by 90°: **X = 319 − y,
  Y = x** (X to the right, Y down; people stand upright). Boxes drawn on sensor coordinates (for example in
  Prophesee's Metavision tools) must be turned the same way.
- **Label format.** `labels/pic_NNNN.txt`: one line per box, `c cx cy w h`, c = 0 person, 1 region to ignore;
  centre, width and height as fractions of the picture, where 1.0 = 319 pixels (pixel = value × 319).
  `labels_all/pic_NNNN.txt`: every mapped box with a sixth column, its active pixels (pixels with an event in the
  picture) inside the box. Boxes are amodal, as in the sources: a partly hidden person keeps the whole extent.
- **Time base.** t = 0 is the first event of the recording (the board starts recording on a command from the PC).
  Picture k holds the events with t in [50 k, 50 (k + 1)) ms. Its labels are the boxes of the video frame on the
  monitor at the middle of that window, mapped through the time fit in `events.json` (sensor time → PC time).
- **Blink marker.** A white square blinks once a second inside the view for the time fit. Its events are in
  `events.raw`; in the pictures the rectangle `events.json` `marker_mask_pic` = [x0, y0, x1, y1) (picture
  coordinates) is set to "nothing". Mask it the same way in any representation you build, or the network learns the
  marker.
- **Pictures.** 320×320, 8 bit: 128 no event, 255 ON, 0 OFF (the last event of a pixel in the window wins), the
  detector's input on the board.
- **Histogram pass (h1).** `events.raw` is a HISTO3D file: a 512-byte text header, then frames of a 16-byte header
  {"HFRM", board tick in ms, sequence number, 0} and 320×320 bytes in sensor layout (row = sensor y, byte = sensor
  x; ON count in the high 4 bits, OFF count in the low 4 bits). Each frame accumulates 50 ms and the sensor needs
  ~70 ms to drain it, so there is one picture per frame, ~120 ms apart (`events.json` `frame_interval_ms`), not one
  per 50 ms; picture k is frame k (`tools/histo_raw.py` turns it into a picture).
- **Monitor log** (`frames.log`, monitor passes): Unix time (s, PC clock) and video frame index of every frame
  shown; the stage rectangle and, for zoomed passes, the crop (`warp cx cy side`, source pixels) and the square it
  filled in the picture (`target x y side`, picture pixels); the marker's rectangle (monitor pixels) and switch
  times. Room-light and session times in `CONDITIONS.json` are local time (Europe/Amsterdam).

## How the labels were made

The monitor–sensor homography is calibrated with a pattern of blinking squares before the pass and, from the night
passes on, again after it (median corner error 1–2 px; the drift between the two calibrations is in each pass's
`CONDITIONS.json` and in `passes.csv`, 1–6 px). For the first five passes (v1–v3, a1, h1, Eizo monitor) the
calibration was not kept with the pass; their `calib.json` was recovered afterwards from the pass's original labels
(its `note` field says so), and their geometry is less accurate (see Curation). A white
square blinks for 100 ms once a second inside the view; the labeller finds its bursts in the recording and fits
sensor time to the PC's clock (residual 4–10 ms RMS for events, ~34 ms for histogram frames). The video's boxes are
mapped through the homography and the time fit into every 50 ms picture.

- A box becomes a **label** (`labels/`) only if at least 20 active pixels fall inside it: a person who stands still
  gives no events. `labels_all/` keeps every mapped box with its active-pixel count, so other rules can be applied.
- **Regions to ignore** (class 1): reflections, crowds, people in vehicles and hardly visible people as annotated in
  the source (MOT17 classes and visibility < 0.25), and confident detections of a strong RGB detector (YOLO11x on the
  source frames) that overlap no source annotation (`curation/teacher/`). In training and scoring a detection on them
  is neither right nor wrong.
- **Small people.** People lower than 12 % of the picture height (38 px) are ordinary person labels (class 0) in the
  files; the evaluation protocol ignores them (`eval_ap.py --min-h 0.12`), and our training leaves such pictures out.
- **Room recordings** have no monitor: the person is boxed from the motion (`tools/real_label.py`: event density
  above the empty room's noise, nearby parts merged, smoothed over neighbouring pictures; at least 300 active pixels).
  `labels.json` per scenario summarises it. `detections.csv` is the board's own live detector output during the
  recording (PC time, board time ms, sequence, network ms, count, boxes [x0, y0, x1, y1, score] in picture
  fractions, kev/s, active pixels, gated), kept for reference.

## Curation

- **Time.** 735 monitor segments are timed by their blink-marker fit. The marker fixes the time only modulo one
  second; the labeller chooses among the fit and its one-second neighbours by event density, which moved 178
  segments by one second, and 14 segments with too little motion to decide are flagged. 91 segments have no usable
  marker fit (the marker hidden or ambiguous). The start command alone times them badly: on the marker-timed
  segments of the same passes the first event follows it by −0.4 to +1.5 s. 34 of them are therefore timed
  from the motion: the shift of the boxes (−1 to +2 s, 25 ms steps) that puts the most active pixels inside them, kept
  when its peak stands out (at least 1.10 times the median of the curve, not at the end of the range; on 60
  marker-timed segments the 20 fits that passed were 38 ms from the marker in the median and at most 152 ms). The
  other 57 stay timed by the start command and are listed as `timing_uncertain` in `splits.json` and in
  `curation/flags.json`: leave them out of training and testing (the repository's tools do, by their `events.json`).
- **Geometry.** `curation/align.csv` gives, per monitor segment, the label offset (2 px steps) and time shift that put
  the most active pixels inside the boxes, and the contrast of active-pixel density inside and outside the boxes. Read
  it with care: the offset estimates a label error only on full-view scenes from a static camera (MOT17-02, -04,
  -09). With a moving camera (MOT17-05, -10, -11, -13) or a zoomed crop the events fill the view and the maximum
  drifts to the dense parts of the body or to the search limit (±48 px); a large offset there is not a label error.
  On the static full-view scenes the Samsung passes give 2 px (median), the Eizo passes 5–11 px (v1 the most): the
  Eizo labels sit a few pixels above the events. The four `sam3*` passes were relabelled after the tripod head sank
  (`calib_1712.json` keeps the calibration from before; the event check had found boxes 52–56 px off).
- **RGB teacher.** A strong RGB detector (YOLO11x at 1280 px) on every second source frame confirms 94.9 % of MOT17's
  labels and 89.7 % of PersonPath22's (IoU ≥ 0.4); the labels it misses are mostly small or hidden. Its confident
  detections that overlap no annotation (0.25 per frame on MOT17, 0.33 on PersonPath22) are listed in
  `curation/teacher/` and are regions to ignore in the labels.

## Files

The record holds the data as one zip file and, next to it, this README, the licence and a manifest:

```
README.md, LICENSE.txt     this description; the licence and attributions (both also inside the zip)
manifest.json              size and SHA-256 of the record's files; segment folders and bytes per pass in the zip
N-People-v1.0.zip          all data; unzip gives the folder N-People/:

N-People/splits.json       the test and the training segments, and the timing-uncertain ones to leave out of both, as
                           paths below N-People/ (see Evaluation)
N-People/passes.csv        one line per pass: subset, monitor, room light, view, frame rate, gain, mirrored, sensor
                           configuration, calibration (and its drift), and the paper test set its test segments form
N-People/dataset_stats.md  the numbers per group of passes
N-People/curation/         align.csv (per segment: label offset, time shift and event contrast, see Curation),
                           flags.json (segments with a known label or timing limitation), teacher/ (RGB-detector
                           report: unlabelled detections per sequence, made ignore regions)
N-People/MOT17/            34 passes: dataset_v1..v3, a1, h1 (Eizo), dataset_sam* (Samsung, full view),
                           dataset_camp_dark_* and dataset_camp60_{dark,lit}_* (zoomed)
N-People/PersonPath22/     dataset_camp60pp_part1 ... part7
N-People/Room/             dataset_real1 (11 scenarios) and dataset_real1_comp (composites)

Pass names: camp = recorded in the campaign at the source frame rate, camp60 = at 60 fps; dark / lit = room light;
z40 = zoomed, people ~40 % of the view high; r0, r1, r2 = the best, second and third crop position of the segment;
g100, g60, g35 = frame values × 1, 0.6, 0.35; flip = mirrored. sam = Samsung monitor, full view (dim50 / dim25:
× 0.5 / 0.25; pan_l / pan_r: the picture shifted left / right); v1–v3, a1 (AFK), h1 (histogram) = Eizo monitor.

<subset>/dataset_<pass>/CONDITIONS.json   room light (webcam mean at a fixed exposure), monitor, options,
                                          calibration drift (passes from 5 November 2026 on)
<subset>/dataset_<pass>/run.log           the recorder's log of the pass (one line per segment: card status)
<subset>/dataset_<pass>/<segment>/        segment = <video>_f<first frame> (frames counted at the replay rate:
                                          f0600 at 60 fps is the same moment as f0300 at 30 fps); PersonPath22:
                                          uid_vid_<video>_w<window start frame>_f0000
    events.raw     Prophesee EVT2.0 RAW: 512-byte text header (% format EVT2;height=320;width=320; from a1 on
                   % sensor_config erc_kevps=500 afk=.. stc=.. histo=..), then 32-bit words; h1: HISTO3D (above)
    frames.log     the monitor log (above)
    calib.json     the homography used for the labels (H_mon_to_pic: monitor pixels -> picture pixels)
    calib_1712.json  (sam3* passes) the calibration before the tripod head sank
    segment.json   source video (path below the code repository's work/), first frame, recording number, PC times
    events.json    event counts, the time fit (method, offset, drift, residual, alias check), the marker mask,
                   window_ms
    labels/        pic_NNNN.txt, one per picture (above)
    labels_all/    pic_NNNN.txt with the active-pixel count
Room/dataset_real1/<NN_scenario>/   events.raw, labels/, labels.json, detections.csv, segment.json (scenario,
                                    instruction, PC times)
Room/dataset_real1_comp/            composites_train/, composites_val/: pictures/ and labels/ (pic_NNNNN = the
                                    composite's index), composites.json (per composite: the source pictures,
                                    paths below N-People/, and the split)
```

## Evaluation protocol

`splits.json` holds out MOT17-09 and MOT17-11 in every MOT17 pass, the 73 PersonPath22 test-split videos, the room
scenarios 03, 06 and 09, and the composites made only from them (`composites_val`); everything else is training
data. No test segment shares a source video with a training segment of the same subset, and no composite mixes the
two.

`tools/eval_ap.py`: AP50 and AP50:95 (COCO style, 101-point) on the 50 ms pictures; class-1 regions and people below
12 % of the height are ignored. The paper reports four test sets (`passes.csv`, column paper_test_set, and
`tools/eval_suite.py`): **mon30**, MOT17-09/11 in full view at the source frame rate (passes sam1, sam_flip,
sam_dim50, sam_dim25); **mon30z**, zoomed at the source frame rate (camp_dark_*); **mon60z**, zoomed at 60 fps
(camp60_dark_* and camp60_lit_*); **pp22test**, the PersonPath22 test-split windows (camp60pp_*; 54 of the 73, the
other 19 are timing-uncertain). The other held-out
MOT17 segments (v1–v3, a1, h1, sam3*) are not in those sets; keep them out of training all the same. Report the sets
separately. The room scenarios are held out too, but one person in one room is too small a sample for a meaningful
accuracy figure. Keep the test segments out of training and of every choice made during training (stopping epoch,
thresholds): choose on training segments instead, e.g. a few PersonPath22 training windows. With four segments per
MOT17 set, report a bootstrap over segments (`eval_ap.py` does), not over pictures.

## What a monitor cannot measure (please read before using the data for temporal methods)

A monitor shows frames: even at 60 fps an edge moves in 16.7 ms steps. This changes the time structure of the events
somewhat: in the 60 fps replays 28–31 % of the events follow an event of the same pixel within 10 ms, against 36–60 %
for a person in the room (9–10 % and 10–26 % within 1 ms). Sensor-side temporal filters (STC) and temporal
representations therefore see the replay a little differently from real motion; the dataset evaluates networks and
spatial representations best. At the source frame rate the event rate pulses at the frame rate (46 % of its
fluctuation power at 14 Hz for a 14 fps video, 44 % at 30 Hz for 30 fps); at 60 fps the pulsing moves to the refresh
rate and the picture-to-picture density change falls from 9–19 % to 3–6 % (a real person: 3 %). The monitor's noise
(~0.7 events per pixel per second on a still grey screen) and contrast differ from a real scene; the room recordings
are there for that check.

## Simulated part (optional, regenerate)

`tools/evsim.py` turns the same videos into the board's pictures with an event-camera simulator fitted on these
recordings (active pixels 13.3 vs 13.9 %, isolated 16 vs 17.5 %, ON share 51 vs 53 %); 140 k pictures from MOT17 and
the PersonPath22 training videos. The simulator is in the code repository; its output is not part of this record.

## Licence and attribution

The record is licensed **CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/). It contains
adaptations of MOT17 (Milan et al., "MOT16: A Benchmark for Multi-Object Tracking", arXiv:1603.00831;
https://motchallenge.net/data/MOT17/; CC BY-NC-SA 3.0) and of PersonPath22 (Shuai et al., "Large Scale Real-world
Multi-Person Tracking", ECCV 2022; https://amazon-science.github.io/tracking-dataset/personpath22.html; © Amazon.com
2022, CC BY-NC 4.0, annotations by AWS); `LICENSE.txt` gives the full attribution and what was changed. The
PersonPath22 creators ask that their data not be used to identify the people in the videos or to generate biometric
information about them; we ask the same for all of N-People (a request, not a licence condition). The room
recordings are published with the consent of the person recorded. The code repository has its own licences
(open source for the tools and firmware; the networks, which derive from ST's model zoo, and ST's board code under
ST's licence SLA0044).

## Citation

Yousefzadeh, A. (2026). N-People: person detection with a GenX320 event camera, recorded from a monitor and in a
room (Version 1.0) [Data set]. 4TU.ResearchData. https://doi.org/10.4121/e9e04333-cc28-4bfe-9079-7bb252203b02

```bibtex
@misc{npeople2026,
  author    = {Yousefzadeh, Amirreza},
  title     = {{N-People}: person detection with a {GenX320} event camera, recorded from a monitor and in a room},
  year      = {2026},
  version   = {1.0},
  publisher = {4TU.ResearchData},
  doi       = {10.4121/e9e04333-cc28-4bfe-9079-7bb252203b02}
}
```

The DOI above always points to the latest version; the record page lists a DOI for each version. The paper that
describes the recording set-up, the labels and baseline results is listed in the code repository.
