"""Data collection tool (Phase 4): auto-clip punches, label them by keypress.

Runs the same detection pipeline as ``main.py``. Every detected punch is
snapshotted (its landmark-sequence window is frozen immediately, before the ring
buffer can discard it) and queued. You then label the queue with one key each.

    python record.py --angle 30 --out data

Keys while running:
    s = straight    u = uppercut    h = hook      (label + save oldest pending)
    z = flip zone (head<->body) of the oldest pending
    x = discard oldest pending (false trigger / not a punch)
    q / Esc = quit

Each clip is saved as an .npz holding the normalized coord sequence (what the
model eats), the per-frame feature vectors, visibility, timestamps, and label
metadata (trajectory, role, zone, side, stance, camera angle). An ``index.csv``
is appended for easy loading in ``train.py``. Store hard negatives too — throw
guard fidgets and feints and hit ``x`` so the spotter learns not to fire (§8).
"""

from __future__ import annotations

import argparse
import csv
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List

import cv2
import numpy as np

from boxing_cv.capture import WebcamCapture
from boxing_cv.constants import Config
from boxing_cv.features import window_to_array
from boxing_cv.overlay import Overlay, _text
from boxing_cv.pipeline import Pipeline
from boxing_cv.spotter import PunchEvent


@dataclass
class Clip:
    event: PunchEvent
    coords: np.ndarray      # (T, 33, 3)
    vis: np.ndarray         # (T, 33)
    ts: np.ndarray          # (T,)
    seq: np.ndarray         # (T, F)
    stance_name: str
    stance_flag: float


def stance_flag(lead: str | None) -> float:
    return {"left": 1.0, "right": -1.0}.get(lead or "", 0.0)


def snapshot(pipe: Pipeline, event: PunchEvent) -> Clip:
    cfg = pipe.cfg
    window = pipe.buffer.window(
        event.peak_t - cfg.window_before_s, event.peak_t + cfg.window_after_s
    )
    coords = np.stack([ff.coords for ff in window]) if window else np.zeros((0, 33, 3), np.float32)
    vis = np.stack([ff.visibility for ff in window]) if window else np.zeros((0, 33), np.float32)
    ts = np.array([ff.t for ff in window], np.float32)
    flag = stance_flag(pipe.stance.lead_side)
    seq = window_to_array(window, flag)
    return Clip(event, coords, vis, ts, seq, pipe.stance.name, flag)


def save_clip(clip: Clip, trajectory: str, out: Path, angle: str) -> Path:
    ev = clip.event
    role = ev.role if ev.role in ("lead", "rear") else "unknown"
    label_dir = out / "clips" / trajectory
    label_dir.mkdir(parents=True, exist_ok=True)
    uid = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    path = label_dir / f"{uid}.npz"

    np.savez_compressed(
        path,
        coords=clip.coords, vis=clip.vis, ts=clip.ts, seq=clip.seq,
        trajectory=trajectory, role=role, zone=ev.zone, side=ev.side,
        stance=clip.stance_name, stance_flag=clip.stance_flag,
        angle=angle, peak_elbow=ev.peak_elbow,
    )

    index = out / "index.csv"
    new = not index.exists()
    with index.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["path", "trajectory", "role", "zone", "side",
                        "stance", "angle", "frames"])
        w.writerow([path.as_posix(), trajectory, role, ev.zone, ev.side,
                    clip.stance_name, angle, len(clip.ts)])
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Punch clip recorder + labeler")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--angle", default="front", help="camera angle tag: front/30/45/other")
    ap.add_argument("--out", default="data", help="output dataset directory")
    ap.add_argument("--hook", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.camera_index = args.camera
    cfg.mirror = not args.no_mirror
    cfg.model_complexity = args.complexity
    cfg.enable_hook = args.hook
    out = Path(args.out)

    overlay = Overlay(cfg)
    pending: Deque[Clip] = deque()
    saved = 0
    win = "Recorder — s/u/h label, z zone, x discard, q quit"

    with WebcamCapture(cfg) as cam:
        if not cam.opened():
            raise SystemExit(f"Could not open camera {cfg.camera_index}.")
        with Pipeline(cfg) as pipe:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            for frame, t, _dt in cam.frames():
                result = pipe.process(frame, t)
                for ev in result.finalized:
                    overlay.flash(ev)
                    pending.append(snapshot(pipe, ev))

                overlay.draw_skeleton(frame, result.lm)
                overlay.draw_readouts(frame, result.lm, result.ff)
                overlay.draw_flash(frame)

                # Labeling panel.
                _text(frame, f"angle={args.angle}  saved={saved}  pending={len(pending)}",
                      (10, 26), scale=0.6)
                if pending:
                    g = pending[0].event
                    _text(frame,
                          f"LABEL: guess {g.type_}/{g.role}/{g.zone} side={g.side}",
                          (10, 54), scale=0.6, color=(0, 200, 255))
                    _text(frame, "s straight  u uppercut  h hook  z zone  x discard",
                          (10, 80), scale=0.55)
                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if not pending:
                    continue
                if key == ord("z"):
                    ev = pending[0].event
                    ev.zone = "head" if ev.zone == "body" else "body"
                elif key == ord("x"):
                    pending.popleft()
                elif key in (ord("s"), ord("u"), ord("h")):
                    traj = {"s": "straight", "u": "uppercut", "h": "hook"}[chr(key)]
                    clip = pending.popleft()
                    p = save_clip(clip, traj, out, args.angle)
                    saved += 1
                    print(f"saved {p}  [{traj}/{clip.event.role}/{clip.event.zone}]")

    cv2.destroyAllWindows()
    print(f"Done. {saved} clips saved under {out}/")


if __name__ == "__main__":
    main()
