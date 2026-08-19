"""Tests for the frame-source layer (webcam / video / image folder).

Camera-free: exercises the video-file and image-folder sources with tiny files
written to a temp dir, checking frame counts, source-timeline timestamps, mirror,
and error handling.

Run directly:   python tests/test_sources.py
Or under pytest: pytest tests/test_sources.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxing_cv.constants import Config  # noqa: E402
from boxing_cv.capture import (  # noqa: E402
    ImageFolderSource, VideoFileSource, WebcamSource, open_source, _natural_key,
)


def _cfg(fps=30.0, mirror=False) -> Config:
    c = Config()
    c.source_fps = fps
    c.mirror = mirror
    return c


def test_natural_sort_orders_numbers():
    names = ["f10.png", "f2.png", "f1.png", "f100.png"]
    assert sorted(names, key=_natural_key) == ["f1.png", "f2.png", "f10.png", "f100.png"]


def test_image_folder_frame_count_and_timeline():
    tmp = Path(tempfile.mkdtemp())
    try:
        for i in range(5):
            img = np.full((48, 64, 3), i * 10, np.uint8)
            cv2.imwrite(str(tmp / f"frame_{i}.png"), img)
        src = open_source(str(tmp), _cfg(fps=30.0), realtime=False)
        assert isinstance(src, ImageFolderSource) and src.opened()
        frames = list(src.frames())

        assert len(frames) == 5
        ts = [t for _, t, _ in frames]
        assert np.allclose(ts, [i / 30.0 for i in range(5)])
        dts = [dt for _, _, dt in frames]
        assert dts[0] == 0.0
        assert all(abs(d - 1 / 30.0) < 1e-6 for d in dts[1:])
        assert abs(src.fps - 30.0) < 1e-3
        assert frames[0][0].shape == (48, 64, 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mirror_is_applied():
    tmp = Path(tempfile.mkdtemp())
    try:
        img = np.zeros((16, 16, 3), np.uint8)
        img[:, :8] = 255                       # left half white, right half black
        cv2.imwrite(str(tmp / "a.png"), img)
        src = open_source(str(tmp), _cfg(mirror=True), realtime=False)
        frame = next(iter(src.frames()))[0]
        # After a horizontal flip the left half should now be black.
        assert frame[:, :8].mean() < 10 and frame[:, 8:].mean() > 245
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bad_source_raises():
    try:
        open_source("Z:/definitely/not/here.mp4", _cfg())
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for a missing path")


def test_digit_spec_selects_webcam_type():
    # Construct the object without opening frames (no hardware read here).
    src = open_source("0", _cfg())
    try:
        assert isinstance(src, WebcamSource)
        assert src.is_live is True
    finally:
        src.release()


def test_video_file_source():
    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
        if not writer.isOpened():
            print("SKIP test_video_file_source (no mp4 encoder available)")
            return
        for i in range(10):
            writer.write(np.full((48, 64, 3), i * 20, np.uint8))
        writer.release()

        src = open_source(str(path), _cfg(), realtime=False)
        assert isinstance(src, VideoFileSource)
        if not src.opened():
            print("SKIP test_video_file_source (decoder could not open the file)")
            return
        frames = list(src.frames())
        assert len(frames) >= 8, f"got {len(frames)} frames"      # codecs may drop a couple
        assert abs(src.fps - 20.0) < 2.0, f"fps={src.fps}"
        ts = [t for _, t, _ in frames]
        assert ts == sorted(ts) and ts[0] == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _main():
    tests = [
        test_natural_sort_orders_numbers,
        test_image_folder_frame_count_and_timeline,
        test_mirror_is_applied,
        test_bad_source_raises,
        test_digit_spec_selects_webcam_type,
        test_video_file_source,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
