"""Tests for calibration logic and Config (de)serialization.

Camera-free: drives synthetic guard + jab frames through the FeatureExtractor into
the Calibrator and checks the derived thresholds are sane; also round-trips a
Config through JSON.

Run directly:   python tests/test_calibrate.py
Or under pytest: pytest tests/test_calibrate.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from boxing_cv.constants import Config  # noqa: E402
from boxing_cv.calibrate import Calibrator  # noqa: E402
from boxing_cv.features import FeatureExtractor  # noqa: E402
from test_synthetic import _pose, _throw_sequence, _JAB, DT  # noqa: E402


def _guard_frames(n=40, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        lm = _pose()
        lm[[15, 16, 13, 14], :2] += rng.normal(0, 0.001, (4, 2)).astype(np.float32)
        frames.append(lm)
    return frames


def _run(guard_frames, punch_frames):
    cfg = Config()
    ext = FeatureExtractor(cfg)
    calib = Calibrator(cfg)
    t = 0.0
    for lm in guard_frames:
        calib.observe_guard(ext.update(lm, t)); t += DT
    for lm in punch_frames:
        calib.observe_punch(ext.update(lm, t)); t += DT
    return calib.compute()


def test_calibration_derives_sane_thresholds():
    punch = []
    for _ in range(6):                       # six jabs in a row
        punch += _throw_sequence(_JAB)
    res = _run(_guard_frames(), punch)

    assert res.ok, res.warnings
    assert res.guard_noise < res.punch_speed, (res.guard_noise, res.punch_speed)
    # v_on sits strictly between guard jitter and real punch speed.
    assert res.guard_noise < res.spot_v_on < res.punch_speed
    assert 0.0 < res.spot_v_off < res.spot_v_on
    assert 90.0 <= res.spot_elbow_loaded_deg <= 160.0
    assert res.spot_min_extend_deg >= 15.0
    assert res.lead_side == "right"          # synthetic stance leads right


def test_apply_writes_only_when_ok():
    # Guard only (no punches) -> not distinct -> fallback keeps defaults.
    res = _run(_guard_frames(), [])
    assert res.ok is False
    cfg = Config()
    before = cfg.spot_v_on
    res.apply(cfg)
    assert cfg.spot_v_on == before          # unchanged on fallback


def test_apply_sets_fields_when_ok():
    punch = []
    for _ in range(6):
        punch += _throw_sequence(_JAB)
    res = _run(_guard_frames(), punch)
    cfg = Config()
    res.apply(cfg)
    assert cfg.spot_v_on == round(res.spot_v_on, 3)
    assert cfg.spot_min_extend_deg == round(res.spot_min_extend_deg, 1)


def test_config_json_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    try:
        c = Config()
        c.spot_v_on = 3.14
        c.spot_min_extend_deg = 27.0
        p = c.save(tmp / "cfg.json")
        assert p.exists()
        c2 = Config.load(p)
        assert c2 == c                        # dataclass equality
        assert c2.spot_v_on == 3.14
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_config_from_dict_rejects_unknown():
    try:
        Config.from_dict({"not_a_real_field": 1})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown config key")


def _main():
    tests = [
        test_calibration_derives_sane_thresholds,
        test_apply_writes_only_when_ok,
        test_apply_sets_fields_when_ok,
        test_config_json_roundtrip,
        test_config_from_dict_rejects_unknown,
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
