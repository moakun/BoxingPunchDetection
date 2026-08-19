# Boxing Punch Detection

Real-time webcam system that estimates upper-body pose, spots when a punch is
thrown, and classifies each as a `(type, zone)` pair — with a live skeleton
overlay and punch counter.

Built from [`boxing_pose_cv_plan.md`](boxing_pose_cv_plan.md). This repo
implements **Phases 0–3** (a usable rule-based detector) plus the tooling for
**Phases 4–5** (data recorder + neural-net training/inference).

```
webcam → MediaPipe pose → normalized features + kinematics → ring buffer
       → per-arm punch spotter (state machine) → classifier → overlay + counter
```

## Quick start

```bash
pip install -r requirements.txt
python main.py                 # opens your webcam, starts detecting
```

Point a webcam at yourself. For the straight punches especially, stand at
**~30–45° to the camera**, not dead-on — a single camera measures the depth axis
worst, which is exactly where a front-on jab/cross travels (plan §2.3).

**Keys:** `q`/`Esc` quit · `r` reset counter

**Options:**

| Flag | Effect |
|---|---|
| `--source SPEC` | input source: webcam index, **video file**, or **image folder** (default: webcam) |
| `--camera N` | webcam index when no `--source` (default 0) |
| `--no-mirror` | disable the mirror flip |
| `--no-realtime` | for file/folder sources, process flat out instead of paced to the source fps |
| `--source-fps F` | assumed fps for image folders / videos with no fps tag (default 30) |
| `--complexity {0,1,2}` | pose model: 0 fastest … 2 most accurate (default 1) |
| `--hook` | enable the experimental hook rule (front-cam hooks are unreliable) |
| `--nn model.onnx` | use a trained v2 classifier instead of the rules |

### Input sources

Detection isn't tied to a live webcam — `--source` also accepts a recorded video
or a folder of extracted frames, which is how you get reproducible testing,
demos, and datasets built from existing footage:

```bash
python main.py --source sparring.mp4      # a video file
python main.py --source frames/           # a folder of images (uses --source-fps)
python record.py --source sparring.mp4    # label clips from footage
```

Non-webcam sources are timestamped on the **source's own timeline** (frame index
÷ fps), so wrist velocities and elbow rates are identical no matter how fast your
machine chews through the file. The preview is paced to that timeline by default;
`--no-realtime` runs as fast as possible for batch processing.

## What each phase gives you

| Phase | Command | Status |
|---|---|---|
| 0 Setup — pose overlay | `python main.py` | ✅ |
| 1 Features — live angle/speed readouts | on-screen in `main.py` | ✅ |
| 2 Spotter — counts punches, ignores guard | `python main.py` | ✅ |
| 3 Rule classifier — `(type, zone)` labels | `python main.py` | ✅ |
| 4 Data tool — auto-clip + keypress labeling | `python record.py` | ✅ tooling |
| 5 NN classifier — PyTorch → ONNX → live | `python train.py` → `--nn` | ✅ tooling |

Phase 3 is the shippable deliverable; 4–5 are the accuracy upgrade and need you
to record your own data.

## Module map (`boxing_cv/`)

| File | Role |
|---|---|
| `constants.py` | landmark indices, skeleton edges, **all tunable thresholds** (`Config`) |
| `pose.py` | MediaPipe Pose wrapper → `(33, 4)` `[x, y, z, visibility]` |
| `features.py` | `normalize`, `joint_angle`, per-frame kinematics, ring buffer |
| `stance.py` | orthodox/southpaw (lead side) from shoulder/hip depth |
| `spotter.py` | per-arm `IDLE→EXTENDING→RETRACT` state machine → `PunchEvent` |
| `classify_rules.py` | v1 rule-based `(trajectory, role, zone)` |
| `classify_nn.py` | v2 two-head 1-D CNN + ONNX/torch inference wrapper |
| `naming.py` | `(trajectory, role, zone)` → boxing term (jab, body cross, …) |
| `overlay.py` | skeleton, live readouts, counter, punch flash |
| `pipeline.py` | wires it all together (used by `main.py` and `record.py`) |
| `capture.py` | webcam loop with frame timing / FPS |

Entrypoints at the repo root: `main.py` (live app), `record.py` (data tool),
`train.py` (train + export).

## How classification works

Two orthogonal outputs, kept separate so the model stays stance-invariant
(plan §2.1–2.2):

- **type** = `trajectory` (`straight` / `uppercut` / `hook?`) × `role`
  (`lead` / `rear`). Which arm is *lead* comes from stance detection, so the
  same throw is a **jab** for an orthodox fighter and still correct for a
  southpaw. The `(role, trajectory)` pair is renamed to the boxing term only at
  the display layer.
- **zone** = `head` / `body`, from the fist's height at the peak relative to the
  sternum.

Straight vs uppercut is decided from the punch window: an uppercut has dominant
**upward** wrist travel with the elbow staying partly flexed; a straight has
large **elbow extension** along a roughly horizontal path. The spotter leans on
**elbow-extension angle** rather than hand displacement, because extension is
visible in the image plane even when the hand moves toward the camera (§2.3).

## Tuning

Every threshold lives in `boxing_cv/constants.py` (`Config`). The ones you'll
most likely adjust against your own camera/lighting:

- `spot_v_on` / `spot_v_off` — punch onset/offset wrist speed (torso-lengths/s).
  Raise `spot_v_on` if guard fidgets trigger; lower it if soft punches are
  missed.
- `spot_min_extend_deg` — minimum elbow-angle gain to count as a real throw.
- `spot_refractory_s` — per-arm debounce; raise if one punch double-counts.
- `up_travel_thresh` — how much upward travel flips a call to *uppercut*.
- `sternum_torso_frac` — the head/body divider.

Coordinates are torso-normalized (centered on the mid-hip, scaled by torso
length), so these thresholds are independent of where you stand or how far you
are from the camera.

## Recording data (Phase 4)

```bash
python record.py --angle 30 --out data
```

Runs the live detector; every spotted punch is frozen and queued. Label each with
one key: `s` straight · `u` uppercut · `h` hook · `z` flip head/body · `x`
discard. Clips save as `.npz` (normalized landmark sequence + features + label)
with an `index.csv`. For generalization, record **both stances**, **2–3 camera
angles**, varied lighting/clothing, and **hard negatives** (guard fidgets, feints
— hit `x`). Target ≥150 clean clips per class (plan §8).

## Training the neural net (Phase 5)

```bash
python train.py --data data --epochs 40 --out model.onnx
python main.py --nn model.onnx
```

Trains the two-head 1-D CNN with augmentation (horizontal flip — which also
flips the stance label and synthesizes the opposite stance for free — plus
coordinate jitter and temporal warp), reports per-head accuracy and a type
confusion matrix, and exports ONNX + a labels sidecar.

**Beat-the-baseline rule:** only switch to `--nn` if it clearly beats the rules
on a *held-out session* (different day/clothes/angle). A random split flatters
the model (§7).

## Testing

```bash
python tests/test_synthetic.py        # detection logic (camera-free)
python tests/test_sources.py          # webcam/video/image sources; or: pytest tests/
```

`test_synthetic.py` synthesizes jab / uppercut / guard-fidget landmark
trajectories and asserts the pipeline detects and classifies them correctly, with
no false triggers on the fidget. `test_sources.py` checks the video-file and
image-folder sources (frame counts, source-timeline timestamps, mirror). Both run
without a webcam.

## The mirror caveat

By default the frame is mirror-flipped so it behaves like a mirror. Under
mirroring, MediaPipe's anatomical left/right labels can swap, which is why the
system reports a **mirror-invariant lead side** and derives jab/cross from it —
those stay correct. The orthodox/southpaw *word* in the HUD is cosmetic and may
read inverted; use `--no-mirror` if you want the raw camera orientation.

## Known limitations (v1)

- **Front-on straights** are the hard case (monocular depth) — use an angled
  camera.
- **Hooks** self-occlude on a front view; off by default, low accuracy when on.
- Single person, upper body only; no defense, footwork, or force estimation.
