"""v1 classifier — hand-written rules over the punch window (plan §7).

Zero data, fully interpretable, ships day one. It also doubles as the candidate
generator for labeling in ``record.py``. Log its confusion matrix; it sets the
bar the neural net (v2) has to beat.

Decision (per event window):
* **Uppercut** — dominant *upward* wrist travel and the elbow stays partly
  flexed at the peak.
* **Hook** (only if enabled) — mostly *lateral* wrist travel with modest
  extension; self-occludes on a front cam, so off by default.
* **Straight** — otherwise (large elbow extension, roughly horizontal path).

Role (lead/rear) comes from the stance detector; zone (head/body) from the
spotter. The boxing term is assembled in ``naming``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .constants import Config, DEFAULT_CONFIG
from .features import FrameFeatures
from .naming import display_name
from .spotter import PunchEvent
from .stance import StanceDetector


class RuleClassifier:
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg

    def classify(
        self,
        event: PunchEvent,
        window: List[FrameFeatures],
        stance: Optional[StanceDetector] = None,
    ) -> PunchEvent:
        """Fill ``event.type_`` / ``role`` / ``display`` in place and return it."""
        cfg = self.cfg
        side = event.side

        # Vertical & horizontal wrist travel across the window (robust to the
        # exact peak frame). Image y grows downward, so "up" == decreasing y.
        onset_y = float(event.onset_wrist[1])
        onset_x = float(event.onset_wrist[0])
        if window:
            ys = np.array([ff.wrist[side][1] for ff in window], dtype=np.float32)
            xs = np.array([ff.wrist[side][0] for ff in window], dtype=np.float32)
            up_travel = onset_y - float(ys.min())          # rise above onset
            horiz_travel = float(np.abs(xs - onset_x).max())
        else:
            up_travel = onset_y - float(event.peak_wrist[1])
            horiz_travel = abs(float(event.peak_wrist[0]) - onset_x)

        partly_flexed = event.peak_elbow <= cfg.uppercut_max_elbow_deg

        if up_travel >= cfg.up_travel_thresh and partly_flexed:
            trajectory = "uppercut"
        elif (
            cfg.enable_hook
            and partly_flexed
            and horiz_travel >= up_travel
            and horiz_travel >= cfg.up_travel_thresh
        ):
            trajectory = "hook"
        else:
            trajectory = "straight"

        role = stance.role(side) if stance is not None else "?"

        event.type_ = trajectory
        event.role = role
        event.display = display_name(trajectory, role, event.zone)
        return event
