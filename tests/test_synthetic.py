"""Camera-free validation of the detection logic.

Synthesizes landmark trajectories for a jab (straight, lead), an uppercut, and a
guard fidget (hard negative), then runs them through the real pipeline modules
(FeatureExtractor -> PunchSpotter -> RuleClassifier) and checks the outcome.

Run directly (no pytest needed):   python tests/test_synthetic.py
Or under pytest:                    pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxing_cv.constants import (  # noqa: E402
    Config, NOSE, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP,
)
from boxing_cv.features import FeatureExtractor, joint_angle, normalize  # noqa: E402
from boxing_cv.spotter import PunchSpotter  # noqa: E402
from boxing_cv.classify_rules import RuleClassifier  # noqa: E402
from boxing_cv.stance import StanceDetector  # noqa: E402

FPS = 30.0
DT = 1.0 / FPS

# Base standing pose in image-normalized coords (x right, y down). Right side is
# nudged forward in z (more negative == closer) so stance -> right lead.
_BASE = {
    NOSE: (0.50, 0.20, 0.0),
    L_SHO: (0.42, 0.32, 0.10), R_SHO: (0.58, 0.32, -0.10),
    L_HIP: (0.45, 0.62, 0.10), R_HIP: (0.55, 0.62, -0.10),
    L_ELB: (0.40, 0.45, 0.0), L_WRI: (0.45, 0.34, 0.0),
    R_ELB: (0.60, 0.45, 0.0), R_WRI: (0.55, 0.34, 0.0),
}
# Right-arm targets at full punch extension.
_JAB = {R_ELB: (0.69, 0.325, 0.0), R_WRI: (0.80, 0.33, 0.0)}
_UPPERCUT = {R_ELB: (0.62, 0.30, 0.0), R_WRI: (0.60, 0.12, 0.0)}


def _pose(overrides=None):
    lm = np.zeros((33, 4), np.float32)
    lm[:, 3] = 1.0
    for i, (x, y, z) in _BASE.items():
        lm[i, :3] = (x, y, z)
    if overrides:
        for i, (x, y, z) in overrides.items():
            lm[i, :3] = (x, y, z)
    return lm


def _lerp(a, b, u):
    return {i: tuple(np.array(a.get(i, _BASE[i])) * (1 - u) + np.array(b[i]) * u)
            for i in b}


def _throw_sequence(peak_overrides, guard=8, extend=5):
    """guard -> extend to peak -> retract -> guard."""
    frames = [_pose() for _ in range(guard)]
    for k in range(1, extend + 1):
        frames.append(_pose(_lerp(_BASE, peak_overrides, k / extend)))
    for k in range(1, extend + 1):
        frames.append(_pose(_lerp(_BASE, peak_overrides, 1 - k / extend)))
    frames += [_pose() for _ in range(guard)]
    return frames


def _run(frames):
    cfg = Config()
    ext, spot = FeatureExtractor(cfg), PunchSpotter(cfg)
    stance, rules = StanceDetector(cfg), RuleClassifier(cfg)
    events = []
    t = 0.0
    for lm in frames:
        ff = ext.update(lm, t)
        stance.update(ff)
        for ev in spot.update(ff):
            window = [ff]  # single-frame window is enough for the rule check here
            rules.classify(ev, window, stance)
            events.append(ev)
        t += DT
    return events, stance


# -- pure-function sanity ---------------------------------------------------
def test_joint_angle_right_angle():
    a, b, c = np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0])
    assert abs(joint_angle(a, b, c) - 90.0) < 1e-4


def test_normalize_centers_midhip():
    lm = _pose()
    n = normalize(lm)
    mid_hip = (n[L_HIP, :3] + n[R_HIP, :3]) * 0.5
    assert np.allclose(mid_hip, 0.0, atol=1e-5)


# -- spotting + classification ---------------------------------------------
def test_jab_is_detected_as_straight_lead():
    events, stance = _run(_throw_sequence(_JAB))
    assert stance.lead_side == "right", f"stance lead={stance.lead_side}"
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    ev = events[0]
    assert ev.side == "right"
    assert ev.type_ == "straight", f"type={ev.type_}"
    assert ev.role == "lead", f"role={ev.role}"
    assert ev.display == "jab", f"display={ev.display}"


def test_uppercut_is_detected():
    events, _ = _run(_throw_sequence(_UPPERCUT))
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    assert events[0].type_ == "uppercut", f"type={events[0].type_}"


def test_guard_fidget_does_not_fire():
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(60):
        lm = _pose()
        lm[[R_WRI, L_WRI, R_ELB, L_ELB], :2] += rng.normal(0, 0.002, (4, 2)).astype(np.float32)
        frames.append(lm)
    events, _ = _run(frames)
    assert len(events) == 0, f"false triggers: {len(events)}"


def _main():
    tests = [
        test_joint_angle_right_angle,
        test_normalize_centers_midhip,
        test_jab_is_detected_as_straight_lead,
        test_uppercut_is_detected,
        test_guard_fidget_does_not_fire,
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
