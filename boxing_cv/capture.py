"""Webcam capture with frame timing.

Thin wrapper over ``cv2.VideoCapture`` that yields ``(frame_bgr, t, dt)`` where
``t`` is a monotonic timestamp (seconds) and ``dt`` the gap since the previous
frame. Timestamps drive all kinematics, so we use the wall clock rather than a
nominal FPS (webcams rarely hit their advertised rate).
"""

from __future__ import annotations

import time
from typing import Iterator, Tuple

import cv2

from .constants import Config, DEFAULT_CONFIG


class WebcamCapture:
    """Context-managed webcam source with a smoothed FPS estimate."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
        if cfg.frame_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        if cfg.frame_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        self._prev_t: float | None = None
        self._fps = 0.0

    def opened(self) -> bool:
        return self.cap.isOpened()

    @property
    def fps(self) -> float:
        return self._fps

    def frames(self) -> Iterator[Tuple["cv2.Mat", float, float]]:
        """Yield ``(frame_bgr, t, dt)`` until the camera stops or is released."""
        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            t = time.perf_counter()
            dt = 0.0 if self._prev_t is None else t - self._prev_t
            self._prev_t = t
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
            if self.cfg.mirror:
                frame = cv2.flip(frame, 1)
            yield frame, t, dt

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "WebcamCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
