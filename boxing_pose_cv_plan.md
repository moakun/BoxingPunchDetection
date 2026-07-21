# Boxing Punch Detection — Project Plan

Real-time webcam system that estimates upper-body pose, spots when a punch is thrown, and classifies it by **type** and **target zone**.

---

## 1. Objective & Scope

**Goal:** From a single webcam feed, detect punch events and classify each as a `(type, zone)` pair, in real time (≥20 FPS), with an on-screen overlay and a live punch counter.

**In scope (v1):**
- Upper-body pose tracking (shoulders, elbows, wrists, hips, nose).
- Punch *spotting* — deciding *when* a punch happened (temporal segmentation).
- Punch *classification* — deciding *what* it was.

**Explicitly out of scope (v1):** opponent tracking, force/power estimation, foot movement, defense (slips/blocks), multi-person.

---

## 2. Design Decisions (read this before coding)

These correct problems in the naïve framing. Skipping them produces a demo that works for you standing in one spot and fails everywhere else.

### 2.1 Two output heads, not one flat label set
"Jab / cross / uppercut" are **punch types**; "body shot" is a **target zone**. They're orthogonal — a body jab is both. Model them separately:

- **Type head:** `{jab, cross, uppercut, hook?}` → better expressed stance-neutral as `{straight_lead, straight_rear, uppercut_lead, uppercut_rear, hook_lead?, hook_rear?}`
- **Zone head:** `{head, body}` — derived from the fist's terminal height relative to the torso (below the sternum landmark ≈ body).

Final display name = map `(lead/rear, trajectory)` → boxing term using the detected stance.

### 2.2 Classify lead/rear, not "jab/cross" directly
Jab = straight punch with the **lead** hand; cross = straight with the **rear** hand. Which hand leads flips between orthodox and southpaw. So:
1. Detect **stance** once per session (or continuously): whichever shoulder/hip is closer to the camera plane is the lead side.
2. Classify punches as lead/rear + trajectory.
3. Rename to jab/cross/etc. at the presentation layer.

This makes the model stance-invariant instead of memorizing "left arm = jab."

### 2.3 The monocular-depth trap (most important)
Facing the camera, **straight punches travel along the camera's depth (z) axis** — exactly the axis a single camera measures worst. MediaPipe's `z` is relative and noisy, so hand *displacement* is a bad onset signal for the two most common punches.

**Mitigations, in priority order:**
1. Use **elbow extension angle** (flexed ~40° → extended ~165°) as the primary straight-punch signal instead of hand travel. Extension is visible in the image plane even when the hand moves toward the camera.
2. Recommend camera at **~30–45° to the fighter**, not dead-on. This gives the straight punches a lateral component while still seeing both hands.
3. Use **apparent forearm foreshortening** (forearm pixel length shrinking as it points at the camera) as a secondary depth cue.

### 2.4 A punch is a temporal event, not a pose
You cannot classify a punch from one frame — a fully extended arm looks the same at the end of a jab and the end of a cross to the same target. Classification operates on a **short window of frames** (~0.3–0.5 s) around the detected peak. Any single-frame classifier is a dead end; don't build one.

### 2.5 Hook is optional on a front cam
The hook travels laterally and the fist/forearm **self-occlude** against the torso and head on a front view. Include it only after straights + uppercuts work, and expect lower accuracy. It's a candidate for the angled-camera setup.

---

## 3. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Capture | OpenCV (`cv2.VideoCapture`) | Standard, low-latency webcam access |
| Pose | **MediaPipe Pose (BlazePose)** | 33 landmarks, real-time on CPU, no training needed; gives relative `z` |
| Numerics | NumPy | Vectorized feature math |
| Classifier (v1) | Hand-written state machine + rules | Zero data, interpretable baseline |
| Classifier (v2) | **PyTorch** 1D-CNN or GRU on landmark sequences | Temporal action recognition |
| Viz/UI | OpenCV overlay | Skeleton + labels + counter |

**Why Python, not React Native:** the classifier-training loop (data, PyTorch, experiments) dominates the effort, and that ecosystem is Python. A browser build (MediaPipe Tasks JS + TF.js) is a valid *deployment* path later — plays to your JS background and gives a zero-install demo — but you'd still train in Python and export to ONNX/TF.js. Don't start in the browser; you'll fight the tooling during the hard part.

**Alternatives considered:** MoveNet (fewer landmarks, no z), OpenPose (GPU-heavy, overkill), YOLO-Pose (great but heavier than needed for single-person). MediaPipe wins on effort-to-result for this scope.

---

## 4. Pipeline Architecture

```
webcam frame
   │
   ▼
[Pose Estimator]  MediaPipe → 33 landmarks (x,y,z,visibility)
   │
   ▼
[Feature Extractor]  normalize → joint angles + kinematics (needs frame history)
   │
   ▼
[Ring Buffer]  last N frames of features
   │
   ▼
[Punch Spotter]  state machine on wrist velocity + elbow extension → emits (onset, peak, side)
   │
   ▼
[Classifier]  window around peak → (type, zone)
   │
   ▼
[Overlay + Counter + Logger]
```

Keep these as separate modules with clean interfaces — you'll swap the classifier (rules → NN) without touching anything upstream.

---

## 5. Feature Engineering

Raw pixel coords depend on where you stand and how far from the camera. **Normalize first or nothing generalizes.**

### 5.1 Scale- and translation-invariant coordinates
Center on the mid-hip; scale by torso length (mid-shoulder ↔ mid-hip), which is more stable under body yaw than shoulder width.

```python
import numpy as np

# MediaPipe indices
L_SHO, R_SHO, L_HIP, R_HIP = 11, 12, 23, 24
L_ELB, R_ELB, L_WRI, R_WRI = 13, 14, 15, 16

def normalize(lm: np.ndarray) -> np.ndarray:
    """lm: (33, 3) landmarks in image-normalized coords. Returns torso-relative coords."""
    mid_hip = (lm[L_HIP] + lm[R_HIP]) * 0.5
    mid_sho = (lm[L_SHO] + lm[R_SHO]) * 0.5
    scale = np.linalg.norm(mid_sho - mid_hip) + 1e-6      # torso length
    return (lm - mid_hip) / scale
```

### 5.2 Joint angles (vectorized)
Elbow extension is your workhorse feature. One function covers elbow, shoulder, etc.

```python
def joint_angle(a, b, c) -> float:
    """Angle at vertex b (degrees), between segments b→a and b→c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

# elbow_r = joint_angle(lm[R_SHO], lm[R_ELB], lm[R_WRI])
```

### 5.3 Kinematics (require frame history → compute in the ring buffer)
For each wrist, per frame, using fps-aware `dt`:
- **velocity** = `(pos[t] - pos[t-1]) / dt`
- **speed** = `‖velocity‖`
- **acceleration** = finite diff of velocity (optional; useful to distinguish snap vs push)

Per-frame feature vector (fed to the NN later):
`[normalized coords of 6 upper-body joints, both elbow angles, both wrist speeds, forearm pixel-lengths (foreshortening cue), stance flag]`.

---

## 6. Punch Spotting (temporal segmentation)

Don't classify every frame — most frames are guard/idle. A lightweight state machine per arm emits a punch event only on a real throw, which cuts false triggers and gives the classifier a clean window.

**Per-arm states:** `IDLE → LOADING → EXTENDING → PEAK → RETRACT → IDLE`

Transition logic (tune thresholds empirically):
- `IDLE → EXTENDING`: wrist speed > `V_ON` **and** elbow angle increasing.
- `EXTENDING → PEAK`: elbow angle reaches local max (extension stops increasing) or crosses `~150°`.
- On `PEAK`: emit event `{side, peak_frame_idx, zone = head|body from wrist height}`. Grab the window `[peak-W, peak+W]` from the ring buffer for classification.
- `PEAK → RETRACT → IDLE`: wrist speed reverses / drops below `V_OFF`.
- **Debounce:** enforce a refractory period (`~150 ms`) per arm so one punch isn't counted twice.

Straight vs uppercut is separable already at this stage by the **vertical direction of wrist travel** during EXTENDING (uppercut = net upward), which gives the rule-based classifier most of what it needs.

---

## 7. Classification: v1 → v2

### v1 — Rule-based (build this first, no data required)
Decide type from the window's kinematics:
- **Uppercut:** dominant upward wrist displacement + elbow stays partly flexed.
- **Straight:** large elbow extension + roughly horizontal wrist path.
- **Lead/rear:** which arm fired (from the spotter) mapped against detected stance.
- **Zone:** wrist `y` at peak relative to sternum midpoint.

This baseline is interpretable, ships in a day, and doubles as a **candidate generator for labeling** (§8). Log its confusion matrix — it sets the bar the NN must beat.

### v2 — Temporal neural net
Input: sequence `(T, F)` — T≈15 frames, F = per-frame feature dim. Two options:
- **1D-CNN over time:** fast, few params, strong baseline for short actions. *Start here.*
- **GRU/LSTM:** handles variable timing, marginally better, slower.

Two classification heads (type, zone) sharing a trunk → single multi-task loss. Keep it small (this runs live): a couple of conv blocks + global pool + two linear heads. Export to ONNX for inference to drop the training deps at runtime.

**Beat-the-baseline rule:** only keep the NN if it clearly outperforms v1 on held-out data. If it doesn't, your features or data are the problem, not the model.

---

## 8. Data Collection & Labeling (the real cost)

There's no public dataset for exactly this — you record your own. This is the bulk of the work, not the modeling.

**Tool to build:** a recording mode that runs the v1 spotter, auto-clips each detected punch window, and lets you label it with a keypress. Store per clip: the raw landmark sequence (not video — smaller, and it's what the model eats) + label + stance + camera-angle tag.

**Collection protocol for generalization:**
- Both stances (orthodox + southpaw, or flip the frame horizontally to synthesize the other).
- 2–3 camera angles (front, 30°, 45°) and distances.
- Vary lighting and clothing.
- Include **hard negatives:** guard fidgets, feints, retractions, reaching — so the spotter learns *not* to fire.
- Target ≥150–300 clean examples per class before expecting the NN to beat rules.

Augmentation (cheap, effective): horizontal flip (also flips stance label), small time-warp, jitter on landmark coords, drop low-visibility frames.

---

## 9. Phased Roadmap

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **0. Setup** | Webcam + MediaPipe skeleton overlay at ≥20 FPS | Landmarks track your arms smoothly |
| **1. Features** | `normalize`, angles, ring buffer with kinematics | Live elbow-angle + wrist-speed readout looks sane |
| **2. Spotter** | State machine emits punch events + counter | Counts your punches, ignores guard, no double-counts |
| **3. Rule classifier (v1)** | `(type, zone)` labels on screen | Correct on straights & uppercuts you throw deliberately |
| **4. Data tool** | Auto-clip + keypress labeling; dataset on disk | ≥150 labeled clips/class across angles & stances |
| **5. NN classifier (v2)** | PyTorch temporal model, ONNX export, live inference | Beats v1 on held-out set; combos handled |

Ship something usable at the end of **Phase 3**. Phases 4–5 are the accuracy upgrade.

---

## 10. Evaluation

Define these before Phase 3 so you're measuring, not vibing.
- **Per-class precision/recall + confusion matrix** on a held-out session (different day/clothes/angle from training).
- **Spotting metrics:** missed-punch rate and false-trigger rate during a 60 s guard-only clip.
- **Latency:** end-to-end ms/frame; must stay under the frame budget (≤50 ms for 20 FPS).
- **Combo test:** rapid 1-2 (jab-cross) — do both register, in order, without merging?

---

## 11. Failure Modes & Mitigations

| Failure | Cause | Mitigation |
|---|---|---|
| Straights misread | punch along camera z-axis | elbow-extension signal + angled camera (§2.3) |
| Hooks missed | self-occlusion on front cam | defer to angled setup; lower expectations |
| Double counts | no refractory period | per-arm debounce (~150 ms) |
| Fires on guard fidget | onset threshold too low | hard negatives in training + higher `V_ON` |
| Works only where you stand | unnormalized coords | torso-relative normalization (§5.1) |
| Combos merge into one | window too long / no reset | shorter window + force state reset on retract |
| Jitter on fast frames | landmark noise at speed | light temporal smoothing (e.g. 3-frame median) on coords, **not** on the velocity you threshold |

---

## 12. Suggested Structure

```
boxing-cv/
├── capture.py          # webcam loop, frame timing
├── pose.py             # MediaPipe wrapper → landmark array
├── features.py         # normalize, angles, kinematics, ring buffer
├── spotter.py          # per-arm punch state machine
├── classify_rules.py   # v1 rule-based classifier
├── classify_nn.py      # v2 model def + ONNX inference
├── stance.py           # orthodox/southpaw detection
├── record.py           # data collection + keypress labeling
├── train.py            # PyTorch training loop, augmentation
├── overlay.py          # skeleton, labels, counter rendering
├── main.py             # wires the pipeline
└── data/               # labeled landmark-sequence clips
```

---

## 13. Stretch Goals (post-v1)

- Combo recognition ("1-2", "1-2-3") via a sequence model over emitted punch events.
- Rough form feedback: elbow flare on hooks, dropped guard hand during a punch, over-extension.
- Cadence / output-rate stats (punches/min, type distribution) per round.
- Browser port (MediaPipe Tasks JS + ONNX Runtime Web) for a shareable zero-install demo.
- Power *proxy* from peak wrist speed × extension — clearly labeled as relative, not real force.
