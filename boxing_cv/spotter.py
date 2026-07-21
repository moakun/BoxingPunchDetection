"""Punch spotting — temporal segmentation via a per-arm state machine.

Most frames are guard/idle. Instead of classifying every frame, a small state
machine per arm emits an event only on a real throw, which kills false triggers
and hands the classifier a clean, centered window (plan §6).

    IDLE --(fast + elbow extending)--> EXTENDING
    EXTENDING --(extension turns over / stalls)--> emit PunchEvent, RETRACT
    RETRACT --(arm settles)--> IDLE

A per-arm refractory period stops one throw being counted twice. The zone
(head/body) is read from the wrist height at the peak; type is left to the
classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .constants import (
    Config, DEFAULT_CONFIG, L_SHO, R_SHO, SIDES,
)
from .features import FrameFeatures

# Elbow-angle drop (deg) from its running max that we read as "extension over".
_TURNOVER_DROP = 6.0

# State labels.
IDLE, EXTENDING, RETRACT = "IDLE", "EXTENDING", "RETRACT"


@dataclass
class PunchEvent:
    """One detected throw. ``type_`` / ``display`` are filled by the classifier."""

    side: str                       # "left" / "right" (physical, spotter frame)
    peak_t: float
    onset_t: float
    peak_wrist: np.ndarray          # (3,) normalized wrist pos at peak
    onset_wrist: np.ndarray         # (3,) normalized wrist pos at onset
    zone: str                       # "head" / "body"
    peak_elbow: float               # deg at max extension
    onset_elbow: float              # deg at onset

    # Filled downstream:
    type_: Optional[str] = None     # "straight" / "uppercut" / "hook"
    role: Optional[str] = None      # "lead" / "rear"
    display: Optional[str] = None   # boxing term, e.g. "jab", "body cross"
    nn_conf: Optional[float] = None


class ArmSpotter:
    """State machine for a single arm."""

    def __init__(self, side: str, cfg: Config = DEFAULT_CONFIG):
        self.side = side
        self.cfg = cfg
        self.state = IDLE
        self._prev_elbow: Optional[float] = None
        self._last_emit_t = -1e9
        self._reset_throw()

    def _reset_throw(self) -> None:
        self._onset_t = 0.0
        self._onset_elbow = 0.0
        self._onset_wrist = np.zeros(3, np.float32)
        self._max_elbow = -1.0
        self._peak_t = 0.0
        self._peak_wrist = np.zeros(3, np.float32)
        self._peak_sternum_y = 0.0

    def _zone_from(self, wrist_y: float, sternum_y: float) -> str:
        # Image y grows downward: wrist *below* the sternum (larger y) == body.
        return "body" if wrist_y > sternum_y else "head"

    def update(self, ff: FrameFeatures) -> Optional[PunchEvent]:
        cfg = self.cfg
        t = ff.t
        elbow = ff.elbow[self.side]
        speed = ff.speed[self.side]
        wrist = ff.wrist[self.side]
        rising = self._prev_elbow is None or elbow >= self._prev_elbow
        self._prev_elbow = elbow

        event: Optional[PunchEvent] = None

        if self.state == IDLE:
            fast = speed > cfg.spot_v_on
            loaded = elbow < cfg.spot_elbow_loaded_deg
            ready = (t - self._last_emit_t) >= cfg.spot_refractory_s
            if fast and rising and loaded and ready:
                self.state = EXTENDING
                self._onset_t = t
                self._onset_elbow = elbow
                self._onset_wrist = wrist.copy()
                self._max_elbow = elbow
                self._peak_t = t
                self._peak_wrist = wrist.copy()
                self._peak_sternum_y = self._sternum_y(ff)

        elif self.state == EXTENDING:
            if elbow > self._max_elbow:
                self._max_elbow = elbow
                self._peak_t = t
                self._peak_wrist = wrist.copy()
                self._peak_sternum_y = self._sternum_y(ff)

            net_gain = self._max_elbow - self._onset_elbow
            turned_over = elbow <= self._max_elbow - _TURNOVER_DROP
            stalled = speed < cfg.spot_v_off
            timed_out = (t - self._onset_t) > cfg.spot_max_throw_s

            if turned_over or stalled or timed_out:
                accepted = (
                    net_gain >= cfg.spot_min_extend_deg
                    and (self._peak_t - self._last_emit_t) >= cfg.spot_refractory_s
                )
                if accepted:
                    event = PunchEvent(
                        side=self.side,
                        peak_t=self._peak_t,
                        onset_t=self._onset_t,
                        peak_wrist=self._peak_wrist.copy(),
                        onset_wrist=self._onset_wrist.copy(),
                        zone=self._zone_from(self._peak_wrist[1], self._peak_sternum_y),
                        peak_elbow=self._max_elbow,
                        onset_elbow=self._onset_elbow,
                    )
                    self._last_emit_t = self._peak_t
                self.state = RETRACT
                self._reset_throw()

        elif self.state == RETRACT:
            # Arm has settled — ready to detect the next throw.
            if speed < cfg.spot_v_off:
                self.state = IDLE

        return event

    def _sternum_y(self, ff: FrameFeatures) -> float:
        mid_sho_y = 0.5 * (ff.coords[L_SHO, 1] + ff.coords[R_SHO, 1])
        # mid-hip is the origin after normalization, so the sternum sits at a
        # fraction of the way up toward the shoulders.
        return float(self.cfg.sternum_torso_frac * mid_sho_y)


class PunchSpotter:
    """Runs both arms; returns the events emitted on each frame."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self._arms: Dict[str, ArmSpotter] = {s: ArmSpotter(s, cfg) for s in SIDES}

    def update(self, ff: FrameFeatures) -> List[PunchEvent]:
        events: List[PunchEvent] = []
        for side in SIDES:
            ev = self._arms[side].update(ff)
            if ev is not None:
                events.append(ev)
        return events

    def state(self, side: str) -> str:
        return self._arms[side].state
