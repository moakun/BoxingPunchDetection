"""OpenCV overlay: skeleton, live readouts, counter, punch flash.

Drawing uses the **raw image-normalized** landmarks (x, y in [0, 1]) to place
pixels, while the numeric readouts (elbow angle, wrist speed) come from the
torso-normalized ``FrameFeatures``.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import cv2
import numpy as np

from .constants import (
    ARM_JOINTS, Config, DEFAULT_CONFIG, L_ELB, R_ELB, L_WRI, R_WRI,
    UPPER_BODY, UPPER_BODY_CONNECTIONS,
)
from .features import FrameFeatures
from .spotter import PunchEvent
from .stance import StanceDetector

# BGR colors.
_BONE = (80, 220, 80)
_JOINT = (255, 255, 255)
_JOINT_LOW = (60, 60, 220)     # low-visibility joint
_TEXT = (255, 255, 255)
_PANEL = (0, 0, 0)
_ACCENT = (0, 200, 255)
_FLASH = (0, 215, 255)


def _px(lm_row: np.ndarray, w: int, h: int):
    return int(lm_row[0] * w), int(lm_row[1] * h)


def _text(img, s, org, scale=0.6, color=_TEXT, thick=1, bg=True):
    if bg:
        (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base), _PANEL, -1)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Overlay:
    """Stateful renderer (holds the transient punch-flash timer)."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self._flash_text = ""
        self._flash_until = 0.0

    def flash(self, event: PunchEvent) -> None:
        label = event.display or event.type_ or "punch"
        self._flash_text = label.upper()
        self._flash_until = time.time() + 0.7

    # -- pieces -----------------------------------------------------------
    def draw_skeleton(self, frame, lm: Optional[np.ndarray]) -> None:
        if lm is None:
            return
        h, w = frame.shape[:2]
        min_vis = self.cfg.min_landmark_visibility
        for a, b in UPPER_BODY_CONNECTIONS:
            if lm[a, 3] < min_vis or lm[b, 3] < min_vis:
                continue
            cv2.line(frame, _px(lm[a], w, h), _px(lm[b], w, h), _BONE, 2, cv2.LINE_AA)
        for j in UPPER_BODY:
            color = _JOINT if lm[j, 3] >= min_vis else _JOINT_LOW
            cv2.circle(frame, _px(lm[j], w, h), 4, color, -1, cv2.LINE_AA)

    def draw_readouts(self, frame, lm: Optional[np.ndarray], ff: Optional[FrameFeatures]) -> None:
        """Live elbow angle at each elbow, wrist speed at each wrist (Phase 1)."""
        if lm is None or ff is None:
            return
        h, w = frame.shape[:2]
        for side in ("left", "right"):
            _, elb_i, wri_i = ARM_JOINTS[side]
            if lm[elb_i, 3] >= self.cfg.min_landmark_visibility:
                _text(frame, f"{ff.elbow[side]:.0f}", _px(lm[elb_i], w, h),
                      scale=0.5, color=_ACCENT)
            if lm[wri_i, 3] >= self.cfg.min_landmark_visibility:
                _text(frame, f"{ff.speed[side]:.1f}", _px(lm[wri_i], w, h),
                      scale=0.5, color=_ACCENT)

    def draw_hud(
        self,
        frame,
        fps: float,
        stance: StanceDetector,
        spotter,
        total: int,
        counts: Dict[str, int],
    ) -> None:
        y = 26
        _text(frame, f"FPS {fps:4.1f}", (10, y), scale=0.6)
        y += 26
        _text(frame, f"stance: {stance.name} (lead {stance.lead_side})", (10, y), scale=0.6)
        y += 26
        _text(frame, f"L:{spotter.state('left'):9s} R:{spotter.state('right')}",
              (10, y), scale=0.55)
        y += 30
        _text(frame, f"PUNCHES: {total}", (10, y), scale=0.8, color=_ACCENT, thick=2)
        y += 26
        if counts:
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
            for name, c in top:
                _text(frame, f"  {name}: {c}", (10, y), scale=0.5)
                y += 20

    def draw_flash(self, frame) -> None:
        if time.time() >= self._flash_until or not self._flash_text:
            return
        h, w = frame.shape[:2]
        scale, thick = 1.6, 3
        (tw, th), _ = cv2.getTextSize(self._flash_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = ((w - tw) // 2, h - 50)
        _text(frame, self._flash_text, org, scale=scale, color=_FLASH, thick=thick)

    def draw_session(self, frame, round_no, ppm, secs_left=None) -> None:
        """Round / round-timer / cadence, top-right (right-aligned)."""
        w = frame.shape[1]
        lines = [f"round {round_no}"]
        if secs_left is not None:
            m, s = divmod(int(max(0.0, secs_left)), 60)
            lines.append(f"{m}:{s:02d}")
        lines.append(f"{ppm:.1f}/min")
        y = 26
        for i, ln in enumerate(lines):
            scale = 0.7 if i == 0 else 0.6
            (tw, _th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            _text(frame, ln, (w - tw - 12, y), scale=scale, color=_ACCENT)
            y += 26

    # -- one-call compose -------------------------------------------------
    def render(self, frame, lm, ff, fps, stance, spotter, total, counts,
               round_no=None, ppm=None, secs_left=None) -> None:
        self.draw_skeleton(frame, lm)
        self.draw_readouts(frame, lm, ff)
        self.draw_hud(frame, fps, stance, spotter, total, counts)
        if round_no is not None:
            self.draw_session(frame, round_no, ppm or 0.0, secs_left)
        self.draw_flash(frame)
        _text(frame, "q quit  |  r reset  |  n next round", (10, frame.shape[0] - 12),
              scale=0.5, color=(180, 180, 180))
