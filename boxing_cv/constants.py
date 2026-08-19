"""Landmark indices, skeleton connections, and tunable configuration.

All thresholds live here so the whole system can be tuned from one place.
Coordinates used throughout the pipeline are *torso-normalized* (see
``features.normalize``): the mid-hip is the origin and one unit == one torso
length. That makes every distance/speed threshold below scale- and
distance-invariant. Image y grows *downward*, so "up" is the negative y
direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# MediaPipe Pose (BlazePose) landmark indices — subject-anatomical L/R.
# --------------------------------------------------------------------------
NOSE = 0
L_SHO, R_SHO = 11, 12
L_ELB, R_ELB = 13, 14
L_WRI, R_WRI = 15, 16
L_HIP, R_HIP = 23, 24

# Joints we actually use (keeps feature vectors small and interpretable).
UPPER_BODY = [NOSE, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP]

# Skeleton edges for the overlay (upper body only).
UPPER_BODY_CONNECTIONS = [
    (L_SHO, R_SHO),
    (L_SHO, L_ELB), (L_ELB, L_WRI),
    (R_SHO, R_ELB), (R_ELB, R_WRI),
    (L_SHO, L_HIP), (R_SHO, R_HIP),
    (L_HIP, R_HIP),
]

# Per-arm joint triples, keyed by side. Order: shoulder, elbow, wrist.
ARM_JOINTS = {
    "left":  (L_SHO, L_ELB, L_WRI),
    "right": (R_SHO, R_ELB, R_WRI),
}

SIDES = ("left", "right")


@dataclass
class Config:
    """Tunable parameters. Defaults are sane starting points — expect to tune
    ``spot_*`` live against your own webcam (see README §Tuning)."""

    # ---- capture ----
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    mirror: bool = True            # flip horizontally so it behaves like a mirror
    # Assumed fps for image-folder sources, and the fallback when a video file
    # doesn't report its own frame rate.
    source_fps: float = 30.0

    # ---- MediaPipe Pose ----
    model_complexity: int = 1      # 0 fastest / 1 balanced / 2 most accurate
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_landmark_visibility: float = 0.5   # below this a joint is "unreliable"

    # ---- smoothing ----
    # Light median smoothing on *coordinates* only (never on the velocity we
    # threshold — see plan §11). 1 disables it.
    smooth_window: int = 3

    # ---- ring buffer ----
    buffer_seconds: float = 1.5    # how much recent history to retain

    # ---- spotter (state machine), all in torso-units & seconds ----
    spot_v_on: float = 2.6         # wrist speed to *start* a throw (torso/s)
    spot_v_off: float = 1.2        # wrist speed to consider the throw finished
    spot_elbow_extend_deg: float = 148.0   # elbow angle counted as "extended"
    spot_elbow_loaded_deg: float = 125.0   # must start below this to be a punch
    spot_min_extend_deg: float = 22.0      # min elbow-angle gain over the throw
    spot_refractory_s: float = 0.18        # per-arm debounce
    spot_max_throw_s: float = 0.6          # abandon a throw that never peaks

    # ---- classification window (frames grabbed around the peak) ----
    window_before_s: float = 0.25
    window_after_s: float = 0.20

    # ---- rule classifier thresholds ----
    # Uppercut: net upward wrist travel during EXTENDING exceeds this (torso).
    up_travel_thresh: float = 0.28
    # Uppercut elbow stays partly flexed (peak angle below this).
    uppercut_max_elbow_deg: float = 150.0
    # Head/body divider along the torso axis, as a fraction from mid-hip(0) to
    # mid-shoulder(1). Sternum ~0.62; a fist peaking below it counts as body.
    sternum_torso_frac: float = 0.62

    # ---- stance ----
    stance_vote_frames: int = 20   # frames of z-depth voting before locking

    # ---- misc ----
    enable_hook: bool = False      # front-cam hooks self-occlude; off for v1

    # feature dimension is derived, not set by hand (see features.FEATURE_DIM)


DEFAULT_CONFIG = Config()
