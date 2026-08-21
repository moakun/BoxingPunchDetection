"""Calibration logic: derive spotter thresholds for this body + camera.

Hardcoded thresholds are the biggest source of "works for you standing there,
fails everywhere else" (plan §11). This measures two short phases —

* **guard**   : stand still in your guard  -> the wrist-speed noise floor and
                the resting elbow angle,
* **punch**   : throw straight punches     -> representative punch speed and
                elbow extension,

and picks ``spot_v_on`` / ``spot_v_off`` to sit above the noise but comfortably
below real punches, plus ``spot_elbow_loaded_deg`` and ``spot_min_extend_deg``.

The maths is kept separate from any capture loop so it is unit-testable: feed
frames to ``observe_guard`` / ``observe_punch``, then call ``compute``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .constants import Config, DEFAULT_CONFIG
from .features import FrameFeatures
from .stance import StanceDetector


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class CalibrationResult:
    ok: bool
    spot_v_on: float
    spot_v_off: float
    spot_elbow_loaded_deg: float
    spot_min_extend_deg: float
    lead_side: Optional[str]
    # diagnostics
    guard_noise: float
    punch_speed: float
    guard_elbow: float
    punch_elbow: float
    n_guard: int
    n_punch: int
    warnings: List[str] = field(default_factory=list)

    def apply(self, cfg: Config) -> Config:
        """Write the calibrated thresholds onto ``cfg`` (only if ``ok``)."""
        if self.ok:
            cfg.spot_v_on = round(self.spot_v_on, 3)
            cfg.spot_v_off = round(self.spot_v_off, 3)
            cfg.spot_elbow_loaded_deg = round(self.spot_elbow_loaded_deg, 1)
            cfg.spot_min_extend_deg = round(self.spot_min_extend_deg, 1)
        return cfg

    def summary(self) -> str:
        lines = [
            f"Calibration ({'OK' if self.ok else 'FALLBACK — keeping defaults'}):",
            f"  guard frames={self.n_guard}  punch frames={self.n_punch}  "
            f"lead={self.lead_side}",
            f"  guard noise={self.guard_noise:.2f}  punch speed={self.punch_speed:.2f} "
            f"(torso/s)",
            f"  guard elbow={self.guard_elbow:.0f}  punch elbow={self.punch_elbow:.0f} deg",
            f"  -> v_on={self.spot_v_on:.2f}  v_off={self.spot_v_off:.2f}  "
            f"loaded={self.spot_elbow_loaded_deg:.0f}  "
            f"min_extend={self.spot_min_extend_deg:.0f}",
        ]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


class Calibrator:
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self._stance = StanceDetector(cfg)
        # per-frame samples
        self._guard_speed: List[float] = []
        self._guard_elbow: List[float] = []
        self._punch_speed: List[float] = []
        self._punch_elbow: List[float] = []   # per-frame MAX elbow (extending arm)

    @staticmethod
    def _frame_speed(ff: FrameFeatures) -> float:
        return max(ff.speed["left"], ff.speed["right"])

    def observe_guard(self, ff: FrameFeatures) -> None:
        self._stance.update(ff)
        self._guard_speed.append(self._frame_speed(ff))
        self._guard_elbow.append(0.5 * (ff.elbow["left"] + ff.elbow["right"]))

    def observe_punch(self, ff: FrameFeatures) -> None:
        self._stance.update(ff)
        self._punch_speed.append(self._frame_speed(ff))
        self._punch_elbow.append(max(ff.elbow["left"], ff.elbow["right"]))

    def compute(self) -> CalibrationResult:
        cfg = self.cfg
        warnings: List[str] = []
        gs = np.asarray(self._guard_speed, dtype=np.float32)
        ps = np.asarray(self._punch_speed, dtype=np.float32)
        ge = np.asarray(self._guard_elbow, dtype=np.float32)
        pe = np.asarray(self._punch_elbow, dtype=np.float32)

        noise = float(np.percentile(gs, 95)) if gs.size else 0.0
        punch_speed = float(np.percentile(ps, 85)) if ps.size else 0.0
        guard_elbow = float(np.median(ge)) if ge.size else 60.0
        punch_elbow = float(np.percentile(pe, 85)) if pe.size else 160.0

        ok = True
        if gs.size < 5:
            warnings.append("very few guard frames")
        if ps.size < 5:
            warnings.append("very few punch frames")
        # Punches must be clearly faster than guard jitter to trust the numbers.
        if punch_speed <= noise * 1.2 or punch_speed <= 0:
            ok = False
            warnings.append("punches not distinct from guard — throw harder / more")

        # v_on: above the noise floor, below real punches.
        v_on = max(noise * 1.4, punch_speed * 0.45)
        if punch_speed > 0:
            v_on = min(v_on, punch_speed * 0.7)
        v_on = _clamp(v_on, 0.8, 8.0)
        v_off = _clamp(max(noise * 1.2, v_on * 0.5), 0.4, v_on * 0.9)

        loaded = _clamp(guard_elbow + 30.0, 90.0, 160.0)
        min_extend = _clamp(0.5 * (punch_elbow - guard_elbow), 15.0, 60.0)

        if not ok:      # fall back to the current config's thresholds
            v_on, v_off = cfg.spot_v_on, cfg.spot_v_off
            loaded, min_extend = cfg.spot_elbow_loaded_deg, cfg.spot_min_extend_deg

        return CalibrationResult(
            ok=ok, spot_v_on=v_on, spot_v_off=v_off,
            spot_elbow_loaded_deg=loaded, spot_min_extend_deg=min_extend,
            lead_side=self._stance.lead_side,
            guard_noise=noise, punch_speed=punch_speed,
            guard_elbow=guard_elbow, punch_elbow=punch_elbow,
            n_guard=int(gs.size), n_punch=int(ps.size), warnings=warnings,
        )
