"""Tests for session logging + stats.

Camera-free: builds PunchEvents directly and checks the JSONL log, the summary
numbers (cadence, distributions), round segmentation (timed + manual), and reset.

Run directly:   python tests/test_session.py
Or under pytest: pytest tests/test_session.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxing_cv.session import SessionLog, format_summary  # noqa: E402
from boxing_cv.spotter import PunchEvent  # noqa: E402


def _ev(side="right", typ="straight", role="lead", zone="head", disp="jab",
        elbow=165.0, speed=6.0):
    e = PunchEvent(
        side=side, peak_t=0.0, onset_t=0.0,
        peak_wrist=np.zeros(3, np.float32), onset_wrist=np.zeros(3, np.float32),
        zone=zone, peak_elbow=elbow, onset_elbow=40.0, peak_speed=speed,
    )
    e.type_, e.role, e.display = typ, role, disp
    return e


def test_logging_summary_and_files():
    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "s.jsonl"
        log = SessionLog(path, round_len=0.0, source="test")
        for i, t in enumerate([0.0, 20.0, 40.0, 60.0]):
            log.tick(t)
            even = i % 2 == 0
            log.log(_ev(side="right" if even else "left",
                        role="lead" if even else "rear",
                        disp="jab" if even else "cross",
                        zone="head" if even else "body"),
                    t, "orthodox")

        s = log.summary()
        assert s["punches"] == 4
        assert abs(s["duration_s"] - 60.0) < 1e-6
        assert abs(s["ppm"] - 4.0) < 1e-6          # 4 punches over 60 s
        assert s["by_display"] == {"jab": 2, "cross": 2}
        assert s["by_side"] == {"right": 2, "left": 2}
        assert s["by_zone"] == {"head": 2, "body": 2}
        assert s["max_speed"] == 6.0

        summary_path = log.write_summary()
        log.close()
        assert path.exists() and summary_path.exists()

        kinds = [json.loads(ln)["kind"] for ln in path.read_text().splitlines()]
        assert kinds[0] == "session_start"
        assert kinds.count("punch") == 4
        assert kinds[-1] == "session_end"

        disk = json.loads(summary_path.read_text())
        assert disk["punches"] == 4
        assert isinstance(format_summary(s), str) and "4 punches" in format_summary(s)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_time_based_rounds():
    log = SessionLog(None, round_len=30.0)          # no disk
    for t in (10.0, 40.0, 70.0):                     # rounds 1, 2, 3
        log.tick(t)
        log.log(_ev(), t)
    rounds = {r["round"]: r["punches"] for r in log.summary()["rounds"]}
    assert rounds == {1: 1, 2: 1, 3: 1}, rounds


def test_manual_rounds_and_reset():
    log = SessionLog(None, round_len=0.0)
    log.tick(0.0)
    log.log(_ev(), 0.0)
    assert log.round == 1
    log.next_round()
    log.tick(5.0)
    log.log(_ev(), 5.0)
    assert log.round == 2

    s = log.summary()
    assert s["punches"] == 2
    assert {r["round"] for r in s["rounds"]} == {1, 2}

    log.reset()
    assert log.total == 0 and log.round == 1
    assert log.summary()["punches"] == 0


def test_round_time_left():
    log = SessionLog(None, round_len=60.0)
    log.tick(0.0)
    log.tick(15.0)
    assert abs(log.round_time_left - 45.0) < 1e-6
    untimed = SessionLog(None, round_len=0.0)
    untimed.tick(0.0)
    assert untimed.round_time_left is None


def _main():
    tests = [
        test_logging_summary_and_files,
        test_time_based_rounds,
        test_manual_rounds_and_reset,
        test_round_time_left,
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
