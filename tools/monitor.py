"""Show things on the monitor in front of the GenX320 (X display :1, started by tools/xserver.sh).

Runs full screen until killed. Modes:
  pattern      a big letter F orbiting slowly: its edges give events, its shape shows the orientation
  bars         one bright bar sweeping left to right, then top to bottom (orientation and framing)
  image FILE   a still picture that sways left-right by 6 % of the screen at 0.5 Hz so that its edges give events
  video FILE   a video, looped (OpenCV decodes it, pygame shows it)
  frames DIR FPS [LOG] [--cover|--warp [--crop X,Y,SIDE]] [--start N] [--marker] [--flip] [--pan F]  the numbered JPEG/PNG frames of a directory (e.g. a
               MOT17 sequence's img1/), looped at FPS from frame N; --marker blinks a white square (MARKER px)
               near the stage's top-left corner for 100 ms once a second and logs each change ("marker 1/0 t"),
               which ties the sensor's clock to the PC's; --flip mirrors the frames, --pan F (-1..1) moves the
               crop of --cover to the left or right part of the frame; with LOG, the wall-clock time of every shown frame goes to that file ("time
               index"), its first line being the frame rectangle on the monitor (x y w h); --cover fills the
               stage and crops the sides instead of letterboxing; --warp shows a square crop of the frame warped
               exactly onto the sensor's view (tools/calibrate.py), filling its picture, nothing outside it
  frame        a white border along the monitor's edge, blinking at 2 Hz (shows how much of the monitor is seen)
  blink X Y SIZE  a white square at monitor pixel (X, Y), side SIZE, blinking at 2 Hz
  ramp [SECONDS]  whole screen brightens linearly over SECONDS (4) then drops: a polarity check
  square [FRACTION]  a centred square of side FRACTION (0.6) x height, blinking at 2 Hz: calibrates where the
               monitor is in the sensor's picture (tools/calibrate.py)
  grey [LEVEL] a still uniform grey (128): the backlight's own flicker, nothing else
  black        nothing (dark screen)
Usage: DISPLAY=:1 monitor.py <mode> [args]
"""
import math
import os
import sys
import time

import cv2
import numpy as np
import pygame

CALIB = "work/calib/calib.json"  # written by tools/calibrate.py; without it the whole monitor is the stage
MARKER = 80  # side of the blink marker square, monitor pixels
MARKER_MARGIN = 30  # still black border around it (no video there)
BLINK_HZ = float(os.environ.get("BLINK_HZ", "2"))  # frame, blink and square modes; 10 Hz makes every 50 ms picture catch a toggle
BG = int(os.environ.get("BLINK_BG", "0"))  # background grey level of the pattern modes (does a toggle elsewhere disturb a grey area?)
MARKER_INSET = (96, 72)  # from the stage's top-left corner: the sensor's corners see nothing (lens)
MARKER_ON_S, MARKER_PERIOD_S = 0.1, 1.0


def surface_from_bgr(frame, size):
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), size, "RGB")


def main(mode, path=None, *args):
    os.environ.setdefault("DISPLAY", ":1")
    pygame.display.init()
    pygame.font.init()
    pygame.mouse.set_visible(False)
    # vsync: each flip waits for the monitor's refresh (59.95 Hz), so a 60 fps video shows every frame for exactly
    # one refresh; without it the loop ran at 62 Hz against the screen (skipped and torn frames). MONITOR_VSYNC=0: off
    info = pygame.display.Info()
    vsync = os.environ.get("MONITOR_VSYNC", "1") != "0"
    screen = None
    if vsync:
        try:
            screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN | pygame.SCALED, vsync=1)
        except pygame.error as e:
            print("no vsync:", e, flush=True)
            vsync = False
    if screen is None:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    w, h = screen.get_size()
    # Pictures and videos go into the "stage": the part of the monitor the sensor sees (tools/calibrate.py)
    try:
        import json
        sx0, sy0, sx1, sy1 = json.load(open(CALIB))["stage_mon"]
        stage = (sx0, sy0, min(sx1, w) - sx0, min(sy1, h) - sy0)
    except (OSError, KeyError, ValueError):
        stage = (0, 0, w, h)
    print("stage x y w h:", stage, flush=True)

    cover = "--cover" in args  # fill the stage and crop the sides instead of letterboxing (frames mode)

    def fit(shape):
        """Size and position of a frame of this shape letterboxed into (or, with --cover, filling) the stage."""
        scale = (max if cover else min)(stage[2] / shape[1], stage[3] / shape[0])
        size = (int(shape[1] * scale), int(shape[0] * scale))
        return size, (stage[0] + (stage[2] - size[0]) // 2, stage[1] + (stage[3] - size[1]) // 2)
    clock = pygame.time.Clock()
    t0 = time.time()

    if mode == "pattern":
        font = pygame.font.SysFont("dejavusans", int(h * 0.8), bold=True)
        glyph = font.render("F", True, (255, 255, 255))
    elif mode == "image":
        img = cv2.imread(path)
        if img is None:
            sys.exit(f"cannot read {path}")
        base, base_pos = fit(img.shape)
        base_surf = surface_from_bgr(img, base)
    elif mode == "video":
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sys.exit(f"cannot open {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
    elif mode == "frames":
        files = sorted(f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png")))
        if not files:
            sys.exit(f"no frames in {path}")
        fps = float(args[0])
        log = open(args[1], "w") if len(args) > 1 and not args[1].startswith("--") else None
        size, pos = fit(cv2.imread(os.path.join(path, files[0])).shape)
        print("frames x y w h:", pos + size, flush=True)
        if log:
            log.write("rect %d %d %d %d\n" % (pos + size))
        index = int(args[args.index("--start") + 1]) % len(files) if "--start" in args else 0
        marker = "--marker" in args
        marker_state = None
        flip = "--flip" in args  # mirror the frames left-right (labels follow in record_dataset.py)
        pan = float(args[args.index("--pan") + 1]) if "--pan" in args else 0.0  # -1..1: shift the crop of --cover
        gain = float(args[args.index("--gain") + 1]) if "--gain" in args else 1.0  # pixel values x gain (dimmer scene)
        if pan:
            pos = (int(stage[0] + (stage[2] - size[0]) * (0.5 + 0.5 * pan)), pos[1])
            log and log.write("rect %d %d %d %d\n" % (pos + size))
        # --warp: a square crop of the frame (moved by --pan) is warped onto the monitor area the sensor sees, so
        # that it fills the sensor's picture exactly, upright, and nothing is shown outside the sensor's view
        warp = None
        if "--warp" in args:
            import json
            import numpy as np
            H = np.array(json.load(open(CALIB))["H_pic_to_mon"], np.float64)
            # the largest square of the sensor's picture whose corners all lie on the screen: the video goes there
            def on_screen(x0, y0, L):
                c = np.array([[x0, y0], [x0 + L, y0], [x0 + L, y0 + L], [x0, y0 + L]], np.float64) - 0.5
                q = cv2.perspectiveTransform(c[:, None], H).reshape(-1, 2)
                return (q[:, 0] >= 0).all() and (q[:, 0] <= w - 1).all() and (q[:, 1] >= 0).all() and (q[:, 1] <= h - 1).all()
            tgt = next(((x0, y0, L) for L in range(320, 63, -4) for y0 in range(0, 321 - L, 4)
                        for x0 in range(0, 321 - L, 4) if on_screen(x0, y0, L)), (0, 0, 320))
            tx0, ty0, tL = tgt
            pic = np.array([[tx0, ty0], [tx0 + tL, ty0], [tx0 + tL, ty0 + tL], [tx0, ty0 + tL]], np.float64) - 0.5
            quad = cv2.perspectiveTransform(pic[:, None], H).reshape(-1, 2)
            fh, fw = cv2.imread(os.path.join(path, files[0])).shape[:2]
            if "--crop" in args:  # X,Y,SIDE in frame pixels (tools/zoom_plan.py): a zoomed crop
                cx0, cy0, side = (float(v) for v in args[args.index("--crop") + 1].split(","))
            else:
                side = min(fh, fw)
                cx0, cy0 = (fw - side) * (0.5 + 0.5 * pan), (fh - side) / 2
            crop = np.array([[cx0, cy0], [cx0 + side, cy0], [cx0 + side, cy0 + side], [cx0, cy0 + side]], np.float64)
            bx0, by0 = np.floor(quad.min(0)).astype(int)
            bx1, by1 = np.ceil(quad.max(0)).astype(int)
            # warp the crop alone, so the rest of the quad's bounding box stays black (warping the whole frame showed
            # the frame around the crop there, people's legs included, outside the labelled square; 2026-11-05)
            x0i, y0i = max(int(np.floor(cx0)), 0), max(int(np.floor(cy0)), 0)
            x1i, y1i = min(int(np.ceil(cx0 + side)), fw), min(int(np.ceil(cy0 + side)), fh)
            W = cv2.getPerspectiveTransform((crop - [x0i, y0i]).astype(np.float32), (quad - [bx0, by0]).astype(np.float32))
            warp = (W, (int(bx1 - bx0), int(by1 - by0)), (int(bx0), int(by0)), (x0i, y0i, x1i, y1i))
            log and log.write(f"warp {cx0:.1f} {cy0:.1f} {side:.1f} frame {fw} {fh} target {tx0} {ty0} {tL}\n")
            print(f"warp: frame crop x {cx0:.0f} y {cy0:.0f} side {side:.0f} -> picture square x {tx0} y {ty0} side {tL} "
                  f"-> monitor quad {quad.round(0).tolist()}", flush=True)
        if log:
            log.write("stage %d %d %d %d\n" % stage)
            log.write("flip %d\n" % int(flip))
            if marker:
                log.write("marker_rect %d %d %d %d\n" % (stage[0] + MARKER_INSET[0], stage[1] + MARKER_INSET[1], MARKER, MARKER))
            log.write("vsync %d\n" % int(vsync))

        # frames are read, mirrored, dimmed and warped ahead in a thread (OpenCV releases the GIL), so the loop
        # only blits and flips: 60 fps needs a new picture every 16.7 ms
        import queue
        import threading
        prepared = queue.Queue(maxsize=8)

        def produce(i):
            while True:
                frame = cv2.imread(os.path.join(path, files[i]))
                if flip:
                    frame = frame[:, ::-1]
                if gain != 1.0:
                    frame = cv2.convertScaleAbs(frame, alpha=gain)
                if warp:
                    Wm, isize, ipos, (x0i, y0i, x1i, y1i) = warp
                    img = cv2.warpPerspective(frame[y0i:y1i, x0i:x1i], Wm, isize)
                else:
                    isize, ipos = size, pos
                    img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                prepared.put((i, cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes(), isize, ipos))
                i = (i + 1) % len(files)

        threading.Thread(target=produce, args=(index,), daemon=True).start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        t = time.time() - t0
        screen.fill((BG, BG, BG))
        if mode == "pattern":
            r = h * 0.12  # orbit radius; one turn per 1.5 s, so all edges move and give events
            x = w / 2 - glyph.get_width() / 2 + r * math.cos(t * 2 * math.pi / 1.5)
            y = h / 2 - glyph.get_height() / 2 + r * math.sin(t * 2 * math.pi / 1.5)
            screen.blit(glyph, (x, y))
        elif mode == "bars":
            phase = (t % 8) / 8  # 4 s horizontal sweep, 4 s vertical sweep
            if phase < 0.5:
                x = int(phase * 2 * w)
                pygame.draw.rect(screen, (255, 255, 255), (x, 0, int(w * 0.03), h))
            else:
                y = int((phase - 0.5) * 2 * h)
                pygame.draw.rect(screen, (255, 255, 255), (0, y, w, int(h * 0.05)))
        elif mode == "image":
            dx = int(0.06 * stage[2] * math.sin(2 * math.pi * t * 0.5))
            screen.blit(base_surf, (base_pos[0] + dx, base_pos[1]))
        elif mode == "frames":
            index, rgb, isize, ipos = prepared.get()
            screen.blit(pygame.image.frombuffer(rgb, isize, "RGB"), ipos)
            if marker:
                on = (t % MARKER_PERIOD_S) < MARKER_ON_S
                mx, my = stage[0] + MARKER_INSET[0], stage[1] + MARKER_INSET[1]
                # a still black margin around the marker: zoomed video moving there drowned its flashes (2026-11-05)
                pygame.draw.rect(screen, (0, 0, 0), (mx - MARKER_MARGIN, my - MARKER_MARGIN,
                                                     MARKER + 2 * MARKER_MARGIN, MARKER + 2 * MARKER_MARGIN))
                pygame.draw.rect(screen, (255, 255, 255) if on else (0, 0, 0), (mx, my, MARKER, MARKER))
            pygame.display.flip()
            now = time.time()
            if log:
                log.write(f"{now:.4f} {index}\n")
                if marker and on != marker_state:
                    log.write(f"marker {int(on)} {now:.4f}\n")
                    marker_state = on
                log.flush()
            # the next frame comes from the producer thread (prepared), in order
            clock.tick(fps)
            continue
        elif mode == "frame":
            if int(t * 2 * BLINK_HZ) % 2 == 0:  # appearing and disappearing gives events on the whole border
                pygame.draw.rect(screen, (255, 255, 255), (0, 0, w, h), 24)
        elif mode == "blink":  # blink X Y SIZE: a white square blinking (where does it land in the picture?)
            if int(t * 2 * BLINK_HZ) % 2 == 0:
                bx, by, bs = int(path), int(args[0]), int(args[1])
                pygame.draw.rect(screen, (255, 255, 255), (bx, by, bs, bs))
        elif mode == "square":
            if int(t * 2 * BLINK_HZ) % 2 == 0:  # centred square of side FRACTION x h, blinking (tools/calibrate.py)
                a = int(float(path or 0.6) * h)
                pygame.draw.rect(screen, (255, 255, 255), ((w - a) // 2, (h - a) // 2, a, a), 16)
        elif mode == "grey":  # grey [LEVEL]: a still uniform grey (default 128): what the backlight alone gives (flicker)
            g = int(path or 128)
            screen.fill((g, g, g))
        elif mode == "ramp":  # ramp [SECONDS]: the whole screen brightens linearly over SECONDS (default 4), then drops:
            period = float(path or 4)  # ON events only during the ramp, a burst of OFF at the drop (polarity check)
            g = int(255 * (t % period) / period)
            screen.fill((g, g, g))
        elif mode == "video":
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            size, pos = fit(frame.shape)
            screen.blit(surface_from_bgr(frame, size), pos)
        pygame.display.flip()
        clock.tick(fps if mode == "video" else 60)


if __name__ == "__main__":
    main(*sys.argv[1:])
