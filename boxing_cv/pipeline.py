"""The wired pipeline: pose -> features -> stance/spotter -> classify.

Kept UI-free so both ``main.py`` (live overlay) and ``record.py`` (data tool)
reuse the exact same detection path. Classification is finalized a beat after
the peak, once the ring buffer holds the symmetric ``[peak-before, peak+after]``
window the plan asks for (§6) — a ~200 ms latency you won't feel on a counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .classify_rules import RuleClassifier
from .constants import Config, DEFAULT_CONFIG
from .features import FeatureExtractor, FrameFeatures, RingBuffer
from .pose import PoseEstimator
from .spotter import PunchEvent, PunchSpotter
from .stance import StanceDetector


@dataclass
class _Pending:
    event: PunchEvent
    deadline_t: float          # buffer time at which the window is complete


@dataclass
class FrameResult:
    """What ``Pipeline.process`` returns for one frame."""
    lm: Optional[np.ndarray]
    ff: Optional[FrameFeatures]
    finalized: List[PunchEvent] = field(default_factory=list)


class Pipeline:
    def __init__(self, cfg: Config = DEFAULT_CONFIG, classifier=None):
        self.cfg = cfg
        self.pose = PoseEstimator(cfg)
        self.extractor = FeatureExtractor(cfg)
        self.buffer = RingBuffer(cfg)
        self.stance = StanceDetector(cfg)
        self.spotter = PunchSpotter(cfg)
        self.classifier = classifier if classifier is not None else RuleClassifier(cfg)

        self._pending: List[_Pending] = []
        self.total = 0
        self.counts: Dict[str, int] = {}

    # --------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        lm = self.pose.process(frame_bgr)
        if lm is None:
            # Person lost — flush anything still pending with what we have.
            finalized = self._flush_all()
            return FrameResult(lm=None, ff=None, finalized=finalized)

        ff = self.extractor.update(lm, t)
        self.buffer.append(ff)
        self.stance.update(ff)

        for ev in self.spotter.update(ff):
            self._pending.append(
                _Pending(event=ev, deadline_t=ev.peak_t + self.cfg.window_after_s)
            )

        finalized = self._finalize_ready(now_t=ff.t)
        return FrameResult(lm=lm, ff=ff, finalized=finalized)

    # --------------------------------------------------------------
    def _finalize_ready(self, now_t: float) -> List[PunchEvent]:
        done: List[PunchEvent] = []
        still: List[_Pending] = []
        for p in self._pending:
            if now_t >= p.deadline_t:
                done.append(self._classify(p.event))
            else:
                still.append(p)
        self._pending = still
        return done

    def _flush_all(self) -> List[PunchEvent]:
        done = [self._classify(p.event) for p in self._pending]
        self._pending = []
        return done

    def _classify(self, event: PunchEvent) -> PunchEvent:
        cfg = self.cfg
        window = self.buffer.window(
            event.peak_t - cfg.window_before_s, event.peak_t + cfg.window_after_s
        )
        self.classifier.classify(event, window, self.stance)
        self.total += 1
        name = event.display or event.type_ or "punch"
        self.counts[name] = self.counts.get(name, 0) + 1
        return event

    # --------------------------------------------------------------
    def reset_counts(self) -> None:
        self.total = 0
        self.counts.clear()

    def close(self) -> None:
        self.pose.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
