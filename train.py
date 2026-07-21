"""Train the v2 temporal classifier and export to ONNX (plan §7–8).

    python train.py --data data --epochs 40 --out model.onnx

Loads the labeled clips recorded by ``record.py``, trains the two-head 1-D CNN
(``boxing_cv.classify_nn.build_model``), reports per-head accuracy and a type
confusion matrix on a held-out split, saves the best checkpoint, and exports
ONNX + a ``<out>.labels.json`` sidecar for ``main.py --nn``.

Augmentation (§8): horizontal flip (which also flips the stance label — the big
one, it synthesizes the opposite stance for free), light coordinate jitter, and
a small temporal crop/warp.

NOTE: needs a recorded dataset to run. Aim for >=150 clean clips per class,
across angles and stances, and evaluate on a *different session* than you
trained on — a random split flatters the model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from boxing_cv.classify_nn import SEQ_LEN, build_model, resample
from boxing_cv.features import FEATURE_DIM

# Horizontal-flip permutation + sign flips for the per-frame feature vector.
# (See features.FrameFeatures.vector for the layout.)
_FLIP_PERM = np.array(
    [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 15, 16, 17, 12, 13, 14,
     19, 18, 21, 20, 23, 22, 24]
)
_NEG_X = [0, 3, 6, 9, 12, 15, 24]   # x-components of the 6 joints + stance flag


def flip_sequence(seq: np.ndarray) -> np.ndarray:
    """Mirror a ``(T, F)`` window: swap L/R, negate x, negate stance flag."""
    out = seq[:, _FLIP_PERM].copy()
    out[:, _NEG_X] *= -1.0
    return out


def flip_role(role: str) -> str:
    return {"lead": "rear", "rear": "lead"}.get(role, role)


def type_label(trajectory: str, role: str) -> str:
    return f"{trajectory}_{role}" if role in ("lead", "rear") else trajectory


# --------------------------------------------------------------------------
# Dataset.
# --------------------------------------------------------------------------
class ClipDataset:
    """Loads clips listed in ``<data>/index.csv``. Torch-free until wrapped."""

    def __init__(self, data_dir: Path):
        self.rows: List[dict] = []
        index = data_dir / "index.csv"
        if not index.exists():
            raise SystemExit(
                f"No dataset at {index}. Record clips first: python record.py"
            )
        with index.open(newline="") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        if not self.rows:
            raise SystemExit("index.csv is empty — record some clips first.")

    def load(self) -> List[dict]:
        """Materialize each clip's seq + labels into memory."""
        items = []
        for r in self.rows:
            npz = np.load(r["path"])
            seq = npz["seq"].astype(np.float32)
            if seq.shape[0] < 2:
                continue
            items.append({
                "seq": seq,
                "type": type_label(r["trajectory"], r["role"]),
                "zone": r["zone"],
            })
        return items


def build_vocab(items: List[dict]) -> Tuple[List[str], List[str]]:
    types = sorted({it["type"] for it in items})
    # Ensure both flip partners exist so augmentation never yields an OOV class.
    for t in list(types):
        if "_lead" in t:
            types.append(t.replace("_lead", "_rear"))
        elif "_rear" in t:
            types.append(t.replace("_rear", "_lead"))
    types = sorted(set(types))
    zones = sorted({it["zone"] for it in items}) or ["head", "body"]
    return types, zones


# --------------------------------------------------------------------------
# Torch training.
# --------------------------------------------------------------------------
def make_torch_dataset(items, type_classes, zone_classes, train: bool):
    import torch
    from torch.utils.data import Dataset

    t_idx = {c: i for i, c in enumerate(type_classes)}
    z_idx = {c: i for i, c in enumerate(zone_classes)}

    class _DS(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            it = items[i]
            seq = it["seq"]
            ty, zo = it["type"], it["zone"]

            if train:
                if np.random.rand() < 0.5:                      # hflip (+stance)
                    seq = flip_sequence(seq)
                    traj, _, role = ty.partition("_")
                    ty = type_label(traj, flip_role(role)) if role else ty
                if np.random.rand() < 0.7:                      # temporal crop
                    a = np.random.uniform(0.0, 0.15)
                    b = np.random.uniform(0.85, 1.0)
                    src = np.linspace(0, 1, seq.shape[0])
                    dst = np.linspace(a, b, SEQ_LEN)
                    seq = np.stack([np.interp(dst, src, seq[:, f])
                                    for f in range(seq.shape[1])], axis=1).astype(np.float32)
                else:
                    seq = resample(seq)
                seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)
            else:
                seq = resample(seq)

            return (
                torch.from_numpy(seq),
                torch.tensor(t_idx[ty]),
                torch.tensor(z_idx[zo]),
            )

    return _DS()


def confusion(y_true, y_pred, n) -> np.ndarray:
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the v2 punch classifier")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = ClipDataset(Path(args.data)).load()
    type_classes, zone_classes = build_vocab(items)
    print(f"{len(items)} clips | type classes: {type_classes} | zones: {zone_classes}")

    idx = np.random.permutation(len(items))
    n_val = max(1, int(len(items) * args.val_frac))
    val_items = [items[i] for i in idx[:n_val]]
    train_items = [items[i] for i in idx[n_val:]]
    print(f"train={len(train_items)} val={len(val_items)}")

    train_ds = make_torch_dataset(train_items, type_classes, zone_classes, train=True)
    val_ds = make_torch_dataset(val_items, type_classes, zone_classes, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    model = build_model(len(type_classes), len(zone_classes))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for seq, ty, zo in train_dl:
            opt.zero_grad()
            tl, zl = model(seq)
            loss = ce(tl, ty) + ce(zl, zo)
            loss.backward()
            opt.step()

        # Validation.
        model.eval()
        yt, yp, correct_zone, total = [], [], 0, 0
        with torch.no_grad():
            for seq, ty, zo in val_dl:
                tl, zl = model(seq)
                tp = tl.argmax(1)
                yt += ty.tolist(); yp += tp.tolist()
                correct_zone += int((zl.argmax(1) == zo).sum())
                total += len(zo)
        type_acc = float(np.mean(np.array(yt) == np.array(yp))) if total else 0.0
        zone_acc = correct_zone / total if total else 0.0
        if type_acc > best_acc:
            best_acc = type_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}  type_acc={type_acc:.3f}  zone_acc={zone_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nbest type_acc={best_acc:.3f}")
    print("type confusion (rows=true, cols=pred):")
    print("  classes:", type_classes)
    print(confusion(yt, yp, len(type_classes)))

    # Export.
    out = Path(args.out)
    ckpt = out.with_suffix(".pt")
    torch.save(model.state_dict(), ckpt)
    model.eval()
    dummy = torch.zeros(1, SEQ_LEN, FEATURE_DIM)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["seq"], output_names=["type_logits", "zone_logits"],
        dynamic_axes={"seq": {0: "batch"}},
        opset_version=17,
    )
    out.with_suffix(".labels.json").write_text(json.dumps(
        {"type_classes": type_classes, "zone_classes": zone_classes,
         "seq_len": SEQ_LEN, "feature_dim": FEATURE_DIM}, indent=2))
    print(f"saved {out}, {ckpt}, and labels sidecar.")


if __name__ == "__main__":
    main()
