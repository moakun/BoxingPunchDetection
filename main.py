"""Real-time boxing punch detection — live overlay + counter.

    python main.py                 # default webcam
    python main.py --camera 1      # pick a device
    python main.py --no-mirror     # disable the mirror flip
    python main.py --complexity 0  # faster pose model on weak hardware
    python main.py --nn model.onnx # use the v2 neural classifier if trained

Keys:  q / Esc = quit   r = reset counter

This is the Phase-3 deliverable: it counts punches and labels each as
(type, zone) using the rule classifier. Point a webcam at yourself, ideally at
~30-45 deg rather than dead-on (see plan §2.3).
"""

from __future__ import annotations

import argparse

import cv2

from boxing_cv.capture import WebcamCapture
from boxing_cv.constants import Config
from boxing_cv.overlay import Overlay
from boxing_cv.pipeline import Pipeline


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.camera_index = args.camera
    cfg.mirror = not args.no_mirror
    cfg.model_complexity = args.complexity
    cfg.enable_hook = args.hook
    return cfg


def load_classifier(cfg: Config, onnx_path: str | None):
    if not onnx_path:
        return None
    # Imported lazily so the rule-based app never needs torch/onnxruntime.
    from boxing_cv.classify_nn import NNClassifier
    return NNClassifier(onnx_path, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time boxing punch detection")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--hook", action="store_true", help="enable experimental hook rule")
    ap.add_argument("--nn", type=str, default=None, metavar="MODEL.onnx",
                    help="use the v2 ONNX classifier instead of rules")
    args = ap.parse_args()

    cfg = build_config(args)
    classifier = load_classifier(cfg, args.nn)
    overlay = Overlay(cfg)
    win = "Boxing Punch Detection"

    with WebcamCapture(cfg) as cam:
        if not cam.opened():
            raise SystemExit(
                f"Could not open camera {cfg.camera_index}. "
                "Try a different --camera index."
            )
        with Pipeline(cfg, classifier=classifier) as pipe:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            for frame, t, _dt in cam.frames():
                result = pipe.process(frame, t)
                for ev in result.finalized:
                    overlay.flash(ev)
                    print(f"[{pipe.total:3d}] {ev.display:<14s} "
                          f"(side={ev.side} elbow={ev.peak_elbow:.0f} zone={ev.zone})")

                overlay.render(
                    frame, result.lm, result.ff, cam.fps,
                    pipe.stance, pipe.spotter, pipe.total, pipe.counts,
                )
                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    pipe.reset_counts()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
