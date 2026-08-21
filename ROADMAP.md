# Boxing Punch Detection — Roadmap

Prioritized improvements beyond the shipped v1. The base system (Phases 0–3 of
[`boxing_pose_cv_plan.md`](boxing_pose_cv_plan.md)) already does real-time pose,
punch spotting, and rule-based `(type, zone)` classification, with tooling for
data recording and NN training. This document tracks where it goes next.

**Effort legend:** **S** = a few hours · **M** = a day or two · **L** = multi-day.

---

## A. Make it real & measurable (foundation)

Do these first — they unblock everything else and turn "looks right" into numbers.

- [x] **1. Video / image-sequence input** (`--source clip.mp4`) — **S** ✅ *done*
  Reproducible testing, demos, and building datasets from existing footage; also
  removes the hard webcam dependency. `capture.py` is now a `FrameSource`
  interface (`WebcamSource` / `VideoFileSource` / `ImageFolderSource`) picked by
  `open_source(spec, cfg)`; non-webcam sources use the source's own timeline so
  kinematics are processing-speed-independent. Wired into `main.py` and
  `record.py` via `--source`; covered by `tests/test_sources.py`.
- [x] **2. Session logger + stats** — **S–M** ✅ *done*
  `session.py` (`SessionLog`) writes every punch to `sessions/<ts>.jsonl` (time,
  round, side, type, role, zone, display, peak elbow, peak speed, conf, stance)
  and emits an end-of-session + per-round summary (count, punches/min,
  type/zone/side distributions) to console and `.summary.json`. Live HUD shows
  round, round timer, and cadence; `--round-len` for timed rounds or `n` to
  advance manually. Added `peak_speed` to `PunchEvent`; covered by
  `tests/test_session.py`.
- [ ] **3. Evaluation harness** (`eval.py`) — **M**
  Implements plan §10: per-class precision/recall + confusion matrix on a
  held-out session, spotter miss/false-trigger rate on a guard-only clip, and a
  latency histogram (ms/frame). This is what honestly gates v1 → v2.
- [x] **4. Calibration routine** (`calibrate.py`) — **M** ✅ *done*
  Attacks the #1 fragility: thresholds were hardcoded. `calibrate.py` runs two
  prompted phases (guard, then straights), and `boxing_cv/calibrate.py`
  (`Calibrator`) derives `spot_v_on`/`spot_v_off`/`spot_elbow_loaded_deg`/
  `spot_min_extend_deg` for this body+camera, falling back to defaults if punches
  aren't distinct from guard. Saved as a JSON `Config`; loaded via `--config` in
  `main.py`/`record.py`/`calibrate.py`. Covered by `tests/test_calibrate.py`.

## B. Accuracy & robustness

- [ ] **5. Provisional labeling** — **S**
  Flash side+zone instantly at PEAK, refine type when the window closes — cuts
  the ~200 ms perceived latency to near zero.
- [ ] **6. Occlusion / low-visibility handling** — **S**
  When key joints drop below the visibility threshold, suppress spotting and
  flag it on the HUD instead of emitting garbage events.
- [ ] **7. Explicit stance / handedness** (`--stance`) — **S**
  Removes the mirror ambiguity for the orthodox/southpaw label; keep auto-detect
  as the default.
- [ ] **8. Reach-adaptive thresholds** — **M**
  Scale velocity thresholds by measured arm length so tall/short users behave
  identically.
- [ ] **9. Hook support on an angled camera** — **M**
  Finish the disabled hook rule and collect angled-camera data (front-cam hooks
  self-occlude; off by default today).

## C. New capabilities

- [ ] **10. Combo recognition** — **M**
  Sequence model / rules over emitted events → "1-2", "1-2-3", body-head combos
  with timing (plan §13).
- [ ] **11. Form feedback** — **M**
  Dropped guard hand during a punch, elbow flare, over-extension, slow
  retraction — on-screen coaching cues (§13).
- [ ] **12. Power / speed proxy** — **S**
  Peak wrist speed × extension, clearly labeled *relative*; per-punch and
  rolling max (§13).
- [ ] **13. Workout modes** — **M–L**
  Round timer, target-combo prompts, reaction drills, pace targets with audio
  cues.
- [ ] **14. Richer HUD** — **M**
  Punch-history strip, live type-distribution bar, combo display, confidence
  bars, per-round panel.

## D. ML workflow & data

- [ ] **15. Dataset stats & clip review** — **M**
  Counts per class/angle/stance, class-balance warnings, and stick-figure
  playback of saved clips to audit labels.
- [ ] **16. Session-based train/val split** — **S**
  A random split flatters the model (the README already warns). Split by
  recording session in `train.py`.
- [ ] **17. GRU/LSTM head + model comparison** — **M**
  Add the recurrent alternative from §7 and have `train.py` report both, keeping
  the winner.
- [ ] **18. Active-learning assist** — **M**
  In `record.py`, pre-fill labels with the NN's prediction and surface
  low-confidence clips first.

## E. Engineering & deployment

- [~] **19. Config file** (`--config my.json`) — **S** — *partly done (via #4)*
  `Config.save`/`load` (JSON) exist and `main.py`/`record.py`/`calibrate.py` take
  `--config`. Remaining: YAML support and exposing more fields as CLI overrides.
- [ ] **20. CI + quality gates** — **S**
  GitHub Actions running `tests/test_synthetic.py`, plus `ruff` (lint/format)
  and `mypy`.
- [ ] **21. Packaging** — **S**
  `pyproject.toml` with console entry points (`boxing-detect`, `boxing-record`,
  `boxing-train`).
- [ ] **22. Browser demo port** — **L**
  MediaPipe Tasks JS + ONNX Runtime Web for a zero-install shareable demo —
  train in Python, run in the browser (§13).
- [ ] **23. Two-camera / angled fusion** — **L**
  Genuinely fixes the monocular depth trap for front-on straights — highest
  accuracy ceiling, most work.

---

## Recommended sequence

Build the foundation before investing in the neural net or a browser port:

> **1** (video input) → **2** (logging/stats) → **4** (calibration) →
> **3** (eval harness) → then **5** + **6**.

That makes the project usable, self-tuning, and *measurable* — so when you later
invest in the neural net (17) or combos (10), you can prove they actually help.
