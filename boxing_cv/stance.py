"""Stance detection: which hand leads.

The lead side is whichever shoulder/hip sits **closer to the camera** — smaller
MediaPipe ``z`` (plan §2.2). We vote over a window of frames with a little
hysteresis so the estimate is stable but can still flip if you switch stance.

We report a mirror-invariant ``lead_side`` (``"left"`` / ``"right"``); jab vs
cross is derived from that, so it's correct regardless of the display mirror.
The orthodox/southpaw *word* is cosmetic and can read inverted when the frame is
mirrored (see README §The mirror caveat).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np

from .constants import (
    Config, DEFAULT_CONFIG, L_HIP, L_SHO, R_HIP, R_SHO,
)
from .features import FrameFeatures

_HYSTERESIS = 0.03   # dead-band on the z-difference (torso units)


class StanceDetector:
    """Running lead-side estimate from shoulder/hip depth."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self._diffs: Deque[float] = deque(maxlen=cfg.stance_vote_frames)
        self._lead: Optional[str] = None

    def reset(self) -> None:
        self._diffs.clear()
        self._lead = None

    def update(self, ff: FrameFeatures) -> Optional[str]:
        """Feed one frame; returns the current ``lead_side`` (or ``None``)."""
        z = ff.coords[:, 2]
        vis = ff.visibility

        # Prefer shoulders+hips; fall back to shoulders if hips are occluded.
        if min(vis[L_HIP], vis[R_HIP]) >= self.cfg.min_landmark_visibility:
            left_z = 0.5 * (z[L_SHO] + z[L_HIP])
            right_z = 0.5 * (z[R_SHO] + z[R_HIP])
        elif min(vis[L_SHO], vis[R_SHO]) >= self.cfg.min_landmark_visibility:
            left_z, right_z = z[L_SHO], z[R_SHO]
        else:
            return self._lead   # not enough signal this frame

        # d > 0  => right side farther => LEFT leads.
        self._diffs.append(float(right_z - left_z))
        mean_d = float(np.mean(self._diffs))

        if mean_d > _HYSTERESIS:
            self._lead = "left"
        elif mean_d < -_HYSTERESIS:
            self._lead = "right"
        # else: inside the dead-band -> keep previous decision.
        return self._lead

    @property
    def lead_side(self) -> Optional[str]:
        return self._lead

    @property
    def rear_side(self) -> Optional[str]:
        if self._lead is None:
            return None
        return "right" if self._lead == "left" else "left"

    def role(self, side: str) -> str:
        """Map a physical side to ``"lead"`` / ``"rear"`` (``"?"`` if unknown)."""
        if self._lead is None:
            return "?"
        return "lead" if side == self._lead else "rear"

    @property
    def name(self) -> str:
        """Cosmetic orthodox/southpaw label (see mirror caveat)."""
        if self._lead is None:
            return "unknown"
        return "orthodox" if self._lead == "left" else "southpaw"
