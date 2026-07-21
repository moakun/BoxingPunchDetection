"""Map ``(trajectory, role, zone)`` to a boxing term at the presentation layer.

Keeping this out of the classifier is what makes the model stance-invariant: it
predicts *lead/rear + trajectory*, and only here does that become "jab" vs
"cross" using the detected stance (plan §2.1–2.2).
"""

from __future__ import annotations

from typing import Optional

# (trajectory, role) -> base boxing term (head-level).
_BASE = {
    ("straight", "lead"): "jab",
    ("straight", "rear"): "cross",
    ("uppercut", "lead"): "lead uppercut",
    ("uppercut", "rear"): "rear uppercut",
    ("hook", "lead"): "lead hook",
    ("hook", "rear"): "rear hook",
}


def display_name(
    trajectory: str,
    role: Optional[str],
    zone: str = "head",
) -> str:
    """Return the boxing name, e.g. ``"jab"``, ``"body cross"``, ``"uppercut"``.

    Falls back to the bare trajectory when the role (lead/rear) is unknown —
    e.g. before stance has been established.
    """
    if role in ("lead", "rear"):
        base = _BASE.get((trajectory, role), trajectory)
    else:
        base = trajectory
    return f"body {base}" if zone == "body" else base
