"""Session logging + stats — the "Logger" box from the pipeline (plan §4, §13).

Writes every detected punch to a JSONL file (one record per line, append-safe)
and computes end-of-session and per-round summaries: total, punches/min
(cadence), and type/zone/role/side distributions.

Rounds:
* ``round_len > 0`` -> time-based rounds (round = elapsed // round_len + 1).
* ``round_len == 0`` -> single round; ``next_round()`` advances it manually.

The on-disk JSONL keeps a full history (session_start / punch / reset /
session_end markers); ``reset()`` starts a fresh stats segment in the same file
so the live HUD and the end summary reflect the current segment.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__
from .spotter import PunchEvent


def default_log_path(base: str = "sessions") -> Path:
    """Timestamped path like ``sessions/2026-08-20_143012.jsonl``."""
    return Path(base) / f"{datetime.now():%Y-%m-%d_%H%M%S}.jsonl"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class SessionLog:
    def __init__(
        self,
        path: Optional[Path] = None,
        round_len: float = 0.0,
        source: str = "webcam",
    ):
        self.round_len = round_len
        self.records: List[dict] = []
        self._start_t: Optional[float] = None
        self._now_t: Optional[float] = None
        self._manual_rounds = 0
        self._n = 0
        self.path = Path(path) if path else None
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")
            self._write({"kind": "session_start", "wall": _now_iso(),
                         "source": source, "round_len": round_len,
                         "version": __version__})

    # -- disk -----------------------------------------------------------
    def _write(self, obj: dict) -> None:
        if self._fh:
            self._fh.write(json.dumps(obj) + "\n")
            self._fh.flush()

    # -- clock / rounds -------------------------------------------------
    def tick(self, t: float) -> None:
        """Advance the session clock (call once per frame with source time)."""
        if self._start_t is None:
            self._start_t = t
        self._now_t = t

    @property
    def elapsed(self) -> float:
        if self._start_t is None or self._now_t is None:
            return 0.0
        return max(0.0, self._now_t - self._start_t)

    @property
    def round(self) -> int:
        if self.round_len > 0 and self._start_t is not None:
            return int(self.elapsed // self.round_len) + 1
        return 1 + self._manual_rounds

    @property
    def round_time_left(self) -> Optional[float]:
        """Seconds left in the current timed round, or ``None`` if untimed."""
        if self.round_len <= 0 or self._start_t is None:
            return None
        return self.round_len - (self.elapsed % self.round_len)

    def next_round(self) -> None:
        """Manually start the next round (only meaningful when untimed)."""
        if self.round_len <= 0:
            self._manual_rounds += 1

    # -- counters -------------------------------------------------------
    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def ppm(self) -> float:
        e = self.elapsed
        return (self.total / (e / 60.0)) if e > 0 else 0.0

    # -- logging --------------------------------------------------------
    def log(self, ev: PunchEvent, t: float, stance_name: str = "unknown") -> dict:
        self._n += 1
        rec = {
            "kind": "punch", "n": self._n, "t": round(t, 3), "wall": _now_iso(),
            "round": self.round, "side": ev.side, "type": ev.type_,
            "role": ev.role, "zone": ev.zone, "display": ev.display,
            "peak_elbow": round(ev.peak_elbow, 1),
            "peak_speed": round(ev.peak_speed, 2),
            "conf": None if ev.nn_conf is None else round(ev.nn_conf, 3),
            "stance": stance_name,
        }
        self.records.append(rec)
        self._write(rec)
        return rec

    def reset(self) -> None:
        """Clear live stats and start a fresh segment (history stays on disk)."""
        self._write({"kind": "reset", "wall": _now_iso()})
        self.records.clear()
        self._start_t = self._now_t = None
        self._manual_rounds = 0
        self._n = 0

    # -- summary --------------------------------------------------------
    def summary(self) -> dict:
        recs = self.records
        n = len(recs)
        dur = self.elapsed

        def dist(key: str) -> Dict[str, int]:
            return dict(Counter(r[key] for r in recs if r[key] is not None))

        by_round: Dict[int, List[dict]] = {}
        for r in recs:
            by_round.setdefault(r["round"], []).append(r)
        rounds = [
            {
                "round": rd,
                "punches": len(rr),
                "by_display": dict(Counter(x["display"] for x in rr)),
            }
            for rd, rr in sorted(by_round.items())
        ]

        return {
            "punches": n,
            "duration_s": round(dur, 1),
            "ppm": round(n / (dur / 60.0), 1) if dur > 0 else 0.0,
            "by_display": dist("display"),
            "by_type": dist("type"),
            "by_zone": dist("zone"),
            "by_role": dist("role"),
            "by_side": dist("side"),
            "max_speed": round(max((r["peak_speed"] for r in recs), default=0.0), 2),
            "rounds": rounds,
        }

    def write_summary(self) -> Optional[Path]:
        """Write ``<log>.summary.json`` and a session_end marker; return its path."""
        s = self.summary()
        if self._fh:
            self._write({"kind": "session_end", "wall": _now_iso(),
                         "punches": s["punches"], "duration_s": s["duration_s"],
                         "ppm": s["ppm"]})
        if not self.path:
            return None
        sp = self.path.with_suffix(".summary.json")
        sp.write_text(json.dumps(s, indent=2), encoding="utf-8")
        return sp

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "SessionLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def format_summary(s: dict) -> str:
    """Human-readable multi-line summary for the console."""
    def top(d: Dict[str, int]) -> str:
        return ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))

    lines = [
        f"Session: {s['punches']} punches in {s['duration_s']:.0f}s "
        f"({s['ppm']:.1f}/min)"
    ]
    if s["by_display"]:
        lines.append(f"  types: {top(s['by_display'])}")
    if s["by_zone"]:
        lines.append(f"  zones: {top(s['by_zone'])}")
    if s["by_side"]:
        lines.append(f"  hands: {top(s['by_side'])}")
    if s["max_speed"]:
        lines.append(f"  peak speed: {s['max_speed']:.1f} torso/s (relative)")
    if len(s["rounds"]) > 1:
        lines.append("  rounds:")
        for r in s["rounds"]:
            lines.append(f"    R{r['round']}: {r['punches']}  ({top(r['by_display'])})")
    return "\n".join(lines)
