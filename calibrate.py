"""Guided calibration — tune the spotter to your body + camera (roadmap #4).

    python calibrate.py                 # webcam, writes calib.json
    python calibrate.py --out me.json   # choose the output file
    python calibrate.py --source clip.mp4   # calibrate from footage

Runs two short timed phases with on-screen prompts:
  1) hold your GUARD still,
  2) throw STRAIGHT punches.
Then it derives spotter thresholds, saves them to a config file, and shows the
result. Load them later with:  python main.py --config calib.json

Keys:  r = redo    q / Esc = quit (the file is saved once calibration completes)
"""

from __future__ import annotations

import argparse

import cv2

from boxing_cv.calibrate import Calibrator
from boxing_cv.capture import open_source
from boxing_cv.constants import Config
from boxing_cv.features import FeatureExtractor
from boxing_cv.overlay import Overlay, _text
from boxing_cv.pose import PoseEstimator


def _phases(guard_secs: float, punch_secs: float):
    # (name, duration, instruction)
    return [
        ("ready", 2.0, "Get in your GUARD..."),
        ("guard", guard_secs, "Hold GUARD - stay still"),
        ("ready", 2.0, "Get ready to PUNCH..."),
        ("punch", punch_secs, "Throw STRAIGHT punches!"),
    ]


def _banner(frame, text, sub=""):
    h, w = frame.shape[:2]
    scale, thick = 1.2, 3
    (tw, _t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    _text(frame, text, ((w - tw) // 2, h // 2), scale=scale, color=(0, 215, 255), thick=thick)
    if sub:
        (sw, _s), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        _text(frame, sub, ((w - sw) // 2, h // 2 + 40), scale=0.7, color=(255, 255, 255), thick=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Guided spotter calibration")
    ap.add_argument("--source", default=None, metavar="SPEC",
                    help="webcam index, video file, or image folder (default: webcam)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--source-fps", type=float, default=30.0)
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--config", default=None, help="base config to start from")
    ap.add_argument("--out", default="calib.json", help="where to write the calibrated config")
    ap.add_argument("--guard-secs", type=float, default=4.0)
    ap.add_argument("--punch-secs", type=float, default=8.0)
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    cfg.camera_index = args.camera
    cfg.mirror = not args.no_mirror
    cfg.model_complexity = args.complexity
    cfg.source_fps = args.source_fps

    overlay = Overlay(cfg)
    phases = _phases(args.guard_secs, args.punch_secs)
    win = "Calibration"

    try:
        src = open_source(args.source, cfg, realtime=None)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    calib = Calibrator(cfg)
    idx, phase_start, done, result = 0, None, False, None

    with src, PoseEstimator(cfg) as pose:
        if not src.opened():
            raise SystemExit(f"Could not open source {args.source or cfg.camera_index!r}.")
        extractor = FeatureExtractor(cfg)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        for frame, t, _dt in src.frames():
            if phase_start is None:
                phase_start = t

            if not done:
                name, dur, instr = phases[idx]
                lm = pose.process(frame)
                ff = extractor.update(lm, t) if lm is not None else None
                if ff is not None:
                    if name == "guard":
                        calib.observe_guard(ff)
                    elif name == "punch":
                        calib.observe_punch(ff)

                remaining = dur - (t - phase_start)
                overlay.draw_skeleton(frame, lm)
                seen = "person detected" if lm is not None else "NO PERSON in frame"
                _banner(frame, instr, f"{max(0.0, remaining):.0f}s   |   {seen}")

                if t - phase_start >= dur:
                    idx += 1
                    phase_start = t
                    if idx >= len(phases):
                        done = True
                        result = calib.compute()
                        result.apply(cfg)
                        out = cfg.save(args.out)
                        print(result.summary())
                        print(f"saved {out}")
            else:
                overlay.draw_skeleton(frame, None)
                status = "SAVED to " + args.out if result and result.ok else "FALLBACK (defaults)"
                _banner(frame, "Calibration done", status)
                _text(frame, "r redo   |   q quit", (10, frame.shape[0] - 12),
                      scale=0.6, color=(180, 180, 180))

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                calib = Calibrator(cfg)
                extractor = FeatureExtractor(cfg)
                idx, phase_start, done, result = 0, None, False, None

        # Source ended before finishing (e.g. a short video): compute anyway.
        if not done:
            result = calib.compute()
            result.apply(cfg)
            out = cfg.save(args.out)
            print(result.summary())
            print(f"saved {out} (source ended before all phases completed)")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
