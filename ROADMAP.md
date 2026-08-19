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
- [ ] **2. Session logger + stats** — **S–M**
  The "Logger" box in the §4 pipeline diagram is unbuilt. Log every punch to
  JSONL/CSV (t, side, type, zone, conf, speed) → end-of-session and per-round
  summary (count, punches/min, type distribution).
- [ ] **3. Evaluation harness** (`eval.py`) — **M**
  Implements plan §10: per-class precision/recall + confusion matrix on a
  held-out session, spotter miss/false-trigger rate on a guard-only clip, and a
  latency histogram (ms/frame). This is what honestly gates v1 → v2.
- [ ] **4. Calibration routine** (`--calibrate`) — **M**
  Attacks the #1 fragility: thresholds are hardcoded in `constants.py`. A short
  guided capture ("guard… throw 5 jabs") auto-sets `V_ON`/`V_OFF`, guard-elbow
  baseline, stance, and the head/body divider for *this* body + camera.

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

- [ ] **19. Config file** (`--config my.yaml`) — **S**
  Persist tuning instead of editing `constants.py` or re-passing flags.
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
