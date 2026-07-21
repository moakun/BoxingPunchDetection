"""Feature engineering: normalization, joint angles, kinematics, ring buffer.

Everything downstream (spotter, classifier, NN) consumes *torso-normalized*
coordinates so it generalizes across where you stand and how far you are from
the camera. See plan §5.

Design notes
------------
* Elbow angle is computed in the **image plane (x, y only)**. MediaPipe's ``z``
  is relative and noisy; a 2-D angle is robust and stays meaningful on an
  angled camera (plan §2.3). Forearm foreshortening is kept as a separate depth
  cue instead.
* Coordinate smoothing is a short **median** (edge-preserving) filter. We derive
  velocity *from the smoothed coordinates* and never smooth the velocity signal
  again — double-smoothing would blunt the onset spike we threshold (plan §11).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    ARM_JOINTS, Config, DEFAULT_CONFIG, L_HIP, L_SHO, R_HIP, R_SHO, SIDES,
    L_ELB, R_ELB, L_WRI, R_WRI,
)

# Joints packed into the per-frame NN feature vector (6 arm joints).
_VECTOR_JOINTS = [L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI]
# 6 joints * (x,y,z) + 2 elbow angles + 2 wrist speeds + 2 forearm lens + stance
FEATURE_DIM = len(_VECTOR_JOINTS) * 3 + 2 + 2 + 2 + 1


# --------------------------------------------------------------------------
# Pure functions (easy to unit-test, no state).
# --------------------------------------------------------------------------
def normalize(lm: np.ndarray) -> np.ndarray:
    """Torso-relative coordinates.

    ``lm``: ``(33, 3)`` (or ``(33, 4)`` — extra cols pass through) in
    image-normalized coords. Centers on the mid-hip and scales by torso length
    (mid-shoulder <-> mid-hip), which is steadier under body yaw than shoulder
    width. Returns the same shape.
    """
    xyz = lm[:, :3]
    mid_hip = (xyz[L_HIP] + xyz[R_HIP]) * 0.5
    mid_sho = (xyz[L_SHO] + xyz[R_SHO]) * 0.5
    scale = np.linalg.norm(mid_sho - mid_hip) + 1e-6      # torso length
    out = lm.astype(np.float32, copy=True)
    out[:, :3] = (xyz - mid_hip) / scale
    return out


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex ``b`` in degrees, between segments b->a and b->c.

    Works in whatever dimensionality the inputs carry (pass 2-D slices for an
    image-plane angle, 3-D for a spatial one).
    """
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# --------------------------------------------------------------------------
# Per-frame features.
# --------------------------------------------------------------------------
@dataclass
class FrameFeatures:
    """Everything the spotter/classifier needs about one frame.

    Positions are torso-normalized; ``t`` is seconds; speeds are torso-lengths
    per second. ``elbow`` / ``wrist`` / ``vel`` / ``speed`` / ``forearm`` are
    keyed by ``"left"`` / ``"right"``.
    """

    t: float
    coords: np.ndarray                  # (33, 3) torso-normalized
    visibility: np.ndarray              # (33,)
    elbow: Dict[str, float]             # degrees, image-plane
    wrist: Dict[str, np.ndarray]        # (3,) normalized position
    vel: Dict[str, np.ndarray]          # (3,) torso/s
    speed: Dict[str, float]             # torso/s (image-plane magnitude)
    forearm: Dict[str, float]           # normalized image-plane elbow->wrist len

    def vector(self, stance_flag: float = 0.0) -> np.ndarray:
        """Flat per-frame feature vector of length ``FEATURE_DIM`` for the NN.

        ``stance_flag``: -1 rear-left / +1 rear-right (or 0 unknown) — a global
        signal repeated per frame so the temporal model sees it.
        """
        parts: List[float] = []
        for j in _VECTOR_JOINTS:
            parts.extend(self.coords[j].tolist())
        parts.extend([self.elbow["left"], self.elbow["right"]])
        parts.extend([self.speed["left"], self.speed["right"]])
        parts.extend([self.forearm["left"], self.forearm["right"]])
        parts.append(stance_flag)
        return np.asarray(parts, dtype=np.float32)


class FeatureExtractor:
    """Stateful: turns a landmark array + timestamp into ``FrameFeatures``.

    Holds the previous wrist positions (for velocity) and a short coordinate
    history (for median smoothing).
    """

    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self._coord_hist: Deque[np.ndarray] = deque(maxlen=max(1, cfg.smooth_window))
        self._prev_wrist: Optional[Dict[str, np.ndarray]] = None
        self._prev_t: Optional[float] = None

    def reset(self) -> None:
        self._coord_hist.clear()
        self._prev_wrist = None
        self._prev_t = None

    def _smooth(self, coords: np.ndarray) -> np.ndarray:
        """Element-wise median over the last ``smooth_window`` frames."""
        self._coord_hist.append(coords)
        if len(self._coord_hist) == 1:
            return coords
        return np.median(np.stack(self._coord_hist, axis=0), axis=0).astype(np.float32)

    def update(self, lm: np.ndarray, t: float) -> FrameFeatures:
        """Process one ``(33, 4)`` landmark array captured at time ``t`` (s)."""
        vis = lm[:, 3].copy()
        coords = self._smooth(normalize(lm)[:, :3])

        elbow: Dict[str, float] = {}
        wrist: Dict[str, np.ndarray] = {}
        forearm: Dict[str, float] = {}
        for side in SIDES:
            sho_i, elb_i, wri_i = ARM_JOINTS[side]
            # Image-plane (x, y) angle — robust to noisy z.
            elbow[side] = joint_angle(
                coords[sho_i, :2], coords[elb_i, :2], coords[wri_i, :2]
            )
            wrist[side] = coords[wri_i].copy()
            forearm[side] = float(
                np.linalg.norm(coords[wri_i, :2] - coords[elb_i, :2])
            )

        # Velocity from *smoothed* coords vs the previous frame.
        vel: Dict[str, np.ndarray] = {s: np.zeros(3, np.float32) for s in SIDES}
        speed: Dict[str, float] = {s: 0.0 for s in SIDES}
        if self._prev_wrist is not None and self._prev_t is not None:
            dt = max(t - self._prev_t, 1e-3)
            for side in SIDES:
                v = (wrist[side] - self._prev_wrist[side]) / dt
                vel[side] = v.astype(np.float32)
                # Threshold on image-plane speed (x, y) — the depth axis is the
                # one the camera measures worst, so we don't trust vz here.
                speed[side] = float(np.linalg.norm(v[:2]))

        self._prev_wrist = {s: wrist[s].copy() for s in SIDES}
        self._prev_t = t

        return FrameFeatures(
            t=t, coords=coords, visibility=vis, elbow=elbow,
            wrist=wrist, vel=vel, speed=speed, forearm=forearm,
        )


# --------------------------------------------------------------------------
# Ring buffer.
# --------------------------------------------------------------------------
class RingBuffer:
    """Time-bounded history of ``FrameFeatures`` (last ``buffer_seconds``)."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self._buf: Deque[FrameFeatures] = deque()

    def append(self, ff: FrameFeatures) -> None:
        self._buf.append(ff)
        horizon = ff.t - self.cfg.buffer_seconds
        while self._buf and self._buf[0].t < horizon:
            self._buf.popleft()

    def __len__(self) -> int:
        return len(self._buf)

    def __iter__(self):
        return iter(self._buf)

    @property
    def latest(self) -> Optional[FrameFeatures]:
        return self._buf[-1] if self._buf else None

    def window(self, t0: float, t1: float) -> List[FrameFeatures]:
        """All frames with ``t0 <= t <= t1`` (inclusive), time-ordered."""
        return [ff for ff in self._buf if t0 <= ff.t <= t1]

    def covers(self, t: float) -> bool:
        """True once the buffer holds a frame at/after ``t`` (window ready)."""
        return bool(self._buf) and self._buf[-1].t >= t


def window_to_array(window: List[FrameFeatures], stance_flag: float = 0.0) -> np.ndarray:
    """Stack a list of frames into a ``(T, FEATURE_DIM)`` array for the NN."""
    if not window:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([ff.vector(stance_flag) for ff in window], axis=0)
