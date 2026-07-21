"""v2 classifier — temporal 1-D CNN with two heads (type, zone).

The **type** head is stance-neutral: it predicts trajectory *and* role jointly
(``straight_lead``, ``straight_rear``, ``uppercut_lead`` ...), because a stance
flag is fed in as a feature (see ``features.FrameFeatures.vector``). The **zone**
head predicts head/body. Both heads share a small conv trunk (plan §7 v2).

This file holds the model definition, a fixed-length resampler, and an inference
wrapper (`NNClassifier`) that plugs into the same `.classify(event, window,
stance)` slot as the rule classifier. Training/export lives in ``train.py``.

Only keep the NN if it clearly beats v1 on held-out data (plan §7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .constants import Config, DEFAULT_CONFIG
from .features import FEATURE_DIM, FrameFeatures, window_to_array
from .naming import display_name
from .spotter import PunchEvent

# Fixed temporal length the model sees (windows are resampled to this).
SEQ_LEN = 16

# Default class vocabularies (the trained model ships its own via a sidecar).
TYPE_CLASSES = [
    "straight_lead", "straight_rear",
    "uppercut_lead", "uppercut_rear",
    "hook_lead", "hook_rear",
]
ZONE_CLASSES = ["head", "body"]


# --------------------------------------------------------------------------
# Shared: fixed-length resampling of a variable-T window.
# --------------------------------------------------------------------------
def resample(seq: np.ndarray, target_t: int = SEQ_LEN) -> np.ndarray:
    """Linearly resample a ``(T, F)`` sequence to ``(target_t, F)`` along time."""
    seq = np.asarray(seq, dtype=np.float32)
    t = seq.shape[0]
    if t == 0:
        return np.zeros((target_t, FEATURE_DIM), dtype=np.float32)
    if t == target_t:
        return seq
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, target_t)
    return np.stack([np.interp(dst, src, seq[:, f]) for f in range(seq.shape[1])], axis=1
                    ).astype(np.float32)


def parse_type(name: str) -> Tuple[str, str]:
    """``"straight_lead"`` -> ``("straight", "lead")`` (role ``"?"`` if absent)."""
    if "_" in name:
        traj, role = name.rsplit("_", 1)
        if role in ("lead", "rear"):
            return traj, role
    return name, "?"


# --------------------------------------------------------------------------
# Model definition (import torch lazily so inference-via-ONNX needs no torch).
# --------------------------------------------------------------------------
def build_model(n_type: int, n_zone: int, in_dim: int = FEATURE_DIM):
    """Small multi-task 1-D CNN: conv trunk -> global pool -> two linear heads."""
    import torch.nn as nn

    class PunchNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Conv1d(in_dim, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 96, kernel_size=3, padding=1),
                nn.BatchNorm1d(96), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),          # global temporal pool
            )
            self.type_head = nn.Linear(96, n_type)
            self.zone_head = nn.Linear(96, n_zone)

        def forward(self, x):
            # x: (B, T, F) -> conv wants (B, F, T)
            z = self.trunk(x.transpose(1, 2)).squeeze(-1)
            return self.type_head(z), self.zone_head(z)

    return PunchNet()


# --------------------------------------------------------------------------
# Inference wrapper (ONNX first, torch .pt fallback).
# --------------------------------------------------------------------------
class NNClassifier:
    """Drop-in replacement for ``RuleClassifier`` backed by a trained model.

    ``model_path`` may be an ONNX file (run via onnxruntime) or a ``.pt`` state
    checkpoint (run via torch). A ``<model>.labels.json`` sidecar supplies the
    class vocabularies; falls back to the defaults above.
    """

    def __init__(self, model_path: str, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        self.path = Path(model_path)
        self.type_classes, self.zone_classes = self._load_labels()
        self._backend = "onnx" if self.path.suffix.lower() == ".onnx" else "torch"
        if self._backend == "onnx":
            self._init_onnx()
        else:
            self._init_torch()

    def _load_labels(self) -> Tuple[List[str], List[str]]:
        sidecar = self.path.with_suffix(".labels.json")
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
            return meta.get("type_classes", TYPE_CLASSES), meta.get("zone_classes", ZONE_CLASSES)
        return list(TYPE_CLASSES), list(ZONE_CLASSES)

    def _init_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install onnxruntime to run an .onnx model.") from exc
        self._sess = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self._in_name = self._sess.get_inputs()[0].name

    def _init_torch(self) -> None:
        import torch
        self._torch = torch
        self._model = build_model(len(self.type_classes), len(self.zone_classes))
        state = torch.load(self.path, map_location="cpu")
        self._model.load_state_dict(state)
        self._model.eval()

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """x: (1, T, F) -> (type_logits, zone_logits) as 1-D arrays."""
        if self._backend == "onnx":
            tl, zl = self._sess.run(None, {self._in_name: x})
            return np.asarray(tl)[0], np.asarray(zl)[0]
        with self._torch.no_grad():
            tl, zl = self._model(self._torch.from_numpy(x))
        return tl.numpy()[0], zl.numpy()[0]

    def classify(
        self,
        event: PunchEvent,
        window: List[FrameFeatures],
        stance=None,
    ) -> PunchEvent:
        flag = 0.0
        if stance is not None and stance.lead_side is not None:
            flag = 1.0 if stance.lead_side == "left" else -1.0
        seq = window_to_array(window, flag)
        x = resample(seq)[None, :, :].astype(np.float32)   # (1, T, F)

        type_logits, zone_logits = self._forward(x)
        tp = self._softmax(type_logits)
        ti = int(tp.argmax())
        zi = int(self._softmax(zone_logits).argmax())

        trajectory, role = parse_type(self.type_classes[ti])
        if role == "?" and stance is not None:
            role = stance.role(event.side)
        zone = self.zone_classes[zi]

        event.type_ = trajectory
        event.role = role
        event.zone = zone
        event.nn_conf = float(tp[ti])
        event.display = display_name(trajectory, role, zone)
        return event
