"""MediaPipe Pose wrapper.

Turns a BGR frame into a ``(33, 4)`` float array ``[x, y, z, visibility]`` in
image-normalized coordinates (x, y in [0, 1]; z relative to the mid-hip, smaller
== closer to the camera). Returns ``None`` when no person is found.

Uses the legacy ``mp.solutions.pose`` Solutions API on purpose: it auto-downloads
its model on first use (zero manual setup) and exposes the relative ``z`` the
plan relies on. The Tasks API (PoseLandmarker) is a drop-in future swap.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "pose.py needs opencv-python and mediapipe. Install with "
        "`pip install -r requirements.txt`."
    ) from exc

from .constants import Config, DEFAULT_CONFIG

NUM_LANDMARKS = 33


class PoseEstimator:
    """Thin, stateful wrapper around ``mp.solutions.pose.Pose``.

    Use as a context manager so the underlying graph is released::

        with PoseEstimator(cfg) as pose:
            lm = pose.process(frame_bgr)
    """

    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=cfg.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        # Keep the last raw result around for callers that want the proto.
        self.last_result = None

    def process(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run pose on one BGR frame.

        Returns a ``(33, 4)`` array ``[x, y, z, visibility]`` or ``None``.
        The frame is treated read-only (marked non-writeable for a small speed
        win, then restored).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True
        self.last_result = result

        if not result.pose_landmarks:
            return None

        lm = np.empty((NUM_LANDMARKS, 4), dtype=np.float32)
        for i, p in enumerate(result.pose_landmarks.landmark):
            lm[i] = (p.x, p.y, p.z, p.visibility)
        return lm

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
