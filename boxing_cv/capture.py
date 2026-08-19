"""Frame sources: webcam, video file, or image folder.

Every source yields ``(frame_bgr, t, dt)`` with ``t`` in seconds. The timestamp
semantics differ by source *on purpose* — kinematics depend on ``dt``:

* **Webcam**  -> wall-clock (``perf_counter``). Real-time; ``dt`` is actual
  elapsed time between grabs.
* **Video**   -> the video's own timeline (``frame_index / fps``). Deterministic,
  so velocities are correct no matter how fast the machine processes the file.
* **Images**  -> ``frame_index / source_fps`` (image files carry no timing).

For file/folder sources the preview is paced to the source timeline by default
so it looks natural; pass ``realtime=False`` (``--no-realtime``) to run flat out
for batch/eval. Pacing only ever *slows down* playback — the timestamp handed to
the pipeline is always the source time, so pacing never affects the kinematics.

Use ``open_source(spec, cfg)`` to pick the right one:

    open_source(None, cfg)        # webcam at cfg.camera_index
    open_source("1", cfg)         # webcam #1
    open_source("clip.mp4", cfg)  # video file
    open_source("frames/", cfg)   # folder of images
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .constants import Config, DEFAULT_CONFIG

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
Frame = np.ndarray


def _natural_key(name: str):
    """Sort key so ``frame_2`` precedes ``frame_10``."""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", name)]


class FrameSource:
    """Base class: timing, mirror, smoothed FPS, optional real-time pacing.

    Subclasses implement ``_read`` (next raw BGR frame or ``None`` at end),
    ``_now`` (source time of the current frame), and ``opened``.
    """

    is_live = False

    def __init__(self, cfg: Config = DEFAULT_CONFIG, realtime: bool = True):
        self.cfg = cfg
        self.realtime = realtime
        self._fps = 0.0
        self._i = 0

    # -- subclass hooks --------------------------------------------------
    def _read(self) -> Optional[Frame]:
        raise NotImplementedError

    def _now(self) -> float:
        raise NotImplementedError

    def opened(self) -> bool:
        raise NotImplementedError

    def release(self) -> None:
        pass

    # -- public ----------------------------------------------------------
    @property
    def fps(self) -> float:
        return self._fps

    def frames(self) -> Iterator[Tuple[Frame, float, float]]:
        prev_t: Optional[float] = None
        wall0: Optional[float] = None
        t0 = 0.0
        while True:
            frame = self._read()
            if frame is None:
                break
            t = self._now()
            if wall0 is None:
                wall0, t0 = time.perf_counter(), t
            dt = 0.0 if prev_t is None else max(t - prev_t, 0.0)
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
            if self.cfg.mirror:
                frame = cv2.flip(frame, 1)
            if self.realtime and not self.is_live:
                slack = (t - t0) - (time.perf_counter() - wall0)
                if slack > 0:
                    time.sleep(min(slack, 1.0))
            prev_t = t
            self._i += 1
            yield frame, t, dt

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class WebcamSource(FrameSource):
    """Live webcam via DirectShow (reliable backend on Windows)."""

    is_live = True

    def __init__(self, cfg: Config = DEFAULT_CONFIG, index: Optional[int] = None):
        super().__init__(cfg, realtime=False)     # live: hardware sets the pace
        idx = cfg.camera_index if index is None else index
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cfg.frame_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        if cfg.frame_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    def _read(self) -> Optional[Frame]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def _now(self) -> float:
        return time.perf_counter()

    def opened(self) -> bool:
        return self.cap.isOpened()

    def release(self) -> None:
        self.cap.release()


class VideoFileSource(FrameSource):
    """A video file, played on its own frame-rate timeline."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, path: str = "", realtime: bool = True):
        super().__init__(cfg, realtime=realtime)
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        valid = fps and fps > 1e-3 and not np.isnan(fps)
        self._src_fps = float(fps) if valid else cfg.source_fps

    def _read(self) -> Optional[Frame]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def _now(self) -> float:
        return self._i / self._src_fps

    def opened(self) -> bool:
        return self.cap.isOpened()

    def release(self) -> None:
        self.cap.release()


class ImageFolderSource(FrameSource):
    """A folder of images, treated as a sequence at ``cfg.source_fps``."""

    def __init__(self, cfg: Config = DEFAULT_CONFIG, folder: str = "", realtime: bool = True):
        super().__init__(cfg, realtime=realtime)
        self.folder = Path(folder)
        self.files: List[Path] = sorted(
            (p for p in self.folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS),
            key=lambda p: _natural_key(p.name),
        )
        self._src_fps = cfg.source_fps

    def _read(self) -> Optional[Frame]:
        if self._i >= len(self.files):
            return None
        return cv2.imread(str(self.files[self._i]))     # None if unreadable -> stops

    def _now(self) -> float:
        return self._i / self._src_fps

    def opened(self) -> bool:
        return len(self.files) > 0


def open_source(
    spec: Optional[str],
    cfg: Config = DEFAULT_CONFIG,
    realtime: Optional[bool] = None,
) -> FrameSource:
    """Pick a source from ``spec``.

    ``None`` or a digit -> webcam; an existing directory -> image folder; an
    existing file -> video. ``realtime=None`` means auto (paced for file/folder,
    irrelevant for the live webcam).
    """
    if spec is None:
        return WebcamSource(cfg)
    s = str(spec)
    if s.isdigit():
        return WebcamSource(cfg, index=int(s))
    p = Path(s)
    rt = True if realtime is None else realtime
    if p.is_dir():
        return ImageFolderSource(cfg, p, realtime=rt)
    if p.is_file():
        return VideoFileSource(cfg, p, realtime=rt)
    raise FileNotFoundError(
        f"--source {s!r} is not a webcam index, an existing file, or a folder."
    )


# Backwards-compatible alias for the original class name.
WebcamCapture = WebcamSource
