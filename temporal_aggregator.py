"""
Temporal multi-frame aggregation with progressive bowling rules.

Collects per-cell OCR readings across sampled frames and resolves them into
a consistent, high-confidence scoreboard state using majority voting,
domain validation, and temporal persistence.
"""

import logging
from collections import Counter
from src import config
from src.ocr_processor import is_valid_bowling_shot, is_valid_cumulative_score

logger = logging.getLogger(__name__)


class TemporalAggregator:
    """Accumulate OCR results over time and resolve via majority voting."""

    def __init__(self, min_votes: int = None):
        self.min_votes = min_votes or config.MIN_VOTE_COUNT
        # _readings[key] = list of (timestamp, value)
        self._readings: dict = {}
        self._snapshots: list[dict] = []
        self._obs_count = 0

    # ── recording ────────────────────────────────────────────────────────────
    def record(self, timestamp: float, player_idx: int,
               frame_idx: int, cell_type: str, value: str):
        """Store one OCR observation."""
        if value:
            self._obs_count += 1
        key = (player_idx, frame_idx, cell_type)
        self._readings.setdefault(key, []).append((timestamp, value))

    def record_snapshot(self, timestamp: float, state: dict):
        """Record distinct scoreboard snapshot at timestamp."""
        if self._snapshots and self._snapshots[-1]["state"] == state:
            return
        self._snapshots.append({"timestamp_seconds": timestamp, "state": state})

    # ── aggregation ──────────────────────────────────────────────────────────
    def aggregate(self) -> dict:
        """
        Resolve all recorded readings into a final validated scoreboard state dict:
          { (player_idx, frame_idx, cell_type): best_value }
        """
        resolved = {}
        num_players = len(config.PLAYER_ROWS)

        # 1. Resolve raw non-score cell values (names, shot sub-boxes) by timestamp-weighted voting
        for key, observations in self._readings.items():
            p, f, c_type = key
            if c_type in ("score", "total"):
                continue
            valid_obs = [(ts, v) for ts, v in observations if v]
            if not valid_obs:
                resolved[key] = ""
                continue

            # Weight votes by timestamp (later frames carry higher weight as scoreboard updates)
            weights = {}
            for ts, val in valid_obs:
                w = 1.0 + ts * 0.1
                weights[val] = weights.get(val, 0.0) + w

            best_val = max(weights.items(), key=lambda x: x[1])[0]
            resolved[key] = best_val

        # 2. Combine sub-box shot readings (shot_1, shot_2, shot_3) into frame shot strings
        for p in range(num_players):
            for f in range(10):
                s1 = resolved.get((p, f, "shot_1"), "")
                s2 = resolved.get((p, f, "shot_2"), "")
                s3 = resolved.get((p, f, "shot_3"), "") if f == 9 else ""
                full_s = resolved.get((p, f, "shot"), "")

                combined_shot = self._build_frame_shot_string(f, s1, s2, s3, full_s)
                resolved[(p, f, "shot")] = combined_shot

        # 3. Post-process cumulative scores for progressive temporal consistency
        for p in range(num_players):
            prev_score = 0
            for f in range(10):
                obs = self._readings.get((p, f, "score"), [])
                valid_scores = [(ts, v) for ts, v in obs if v.isdigit()]

                weights = {}
                for ts, val in valid_scores:
                    v_int = int(val)
                    if v_int >= prev_score and v_int <= 300:
                        w = 1.0 + ts * 0.1
                        weights[val] = weights.get(val, 0.0) + w

                if weights:
                    best_score = max(weights.items(), key=lambda x: x[1])[0]
                    resolved[(p, f, "score")] = best_score
                    prev_score = int(best_score)
                else:
                    resolved[(p, f, "score")] = ""

            # Total score: match or validate against last completed cumulative score
            ttl_obs = [(ts, v) for ts, v in self._readings.get((p, 10, "total"), []) if v.isdigit()]
            if ttl_obs:
                weights = {}
                for ts, val in ttl_obs:
                    if int(val) >= prev_score:
                        w = 1.0 + ts * 0.1
                        weights[val] = weights.get(val, 0.0) + w
                if weights:
                    resolved[(p, 10, "total")] = max(weights.items(), key=lambda x: x[1])[0]
                else:
                    resolved[(p, 10, "total")] = str(prev_score) if prev_score > 0 else ""
            else:
                resolved[(p, 10, "total")] = str(prev_score) if prev_score > 0 else ""

        return resolved

    def _build_frame_shot_string(self, frame_idx: int, s1: str, s2: str, s3: str, full_s: str) -> str:
        """Combine individual shot sub-box OCRs into a clean bowling shot string."""
        s1 = s1.upper().strip()
        s2 = s2.upper().strip()
        s3 = s3.upper().strip()
        full_s = full_s.upper().strip()

        if frame_idx < 9:
            # Normalize 2-digit spare shots in full_s (e.g. 74 -> 7/, 91 -> 9/, 28 -> 2/, 18 -> 1/)
            if full_s and len(full_s) == 2 and full_s[0].isdigit() and full_s[1].isdigit():
                if int(full_s[0]) + int(full_s[1]) >= 10:
                    full_s = f"{full_s[0]}/"

            # Frames 1–9
            if s1 == "X":
                return "X"

            if s1 and s2:
                # If s1 is a digit and s2 is a digit/spare
                if s1.isdigit() and (s2.isdigit() or s2 == "/"):
                    if s2 == "/":
                        return f"{s1}/"
                    val1 = int(s1)
                    val2 = int(s2)
                    if val1 + val2 >= 10:
                        return f"{s1}/"
                    else:
                        return f"{s1}{s2}"
                if s1.isdigit() and s2 == "-":
                    return f"{s1}-"
                if s1 == "-" and s2.isdigit():
                    return f"-{s2}"
                if s1 == "-" and s2 == "-":
                    return "--"

            if full_s and is_valid_bowling_shot(full_s):
                return full_s

            if s1 and not s2:
                if s1 in ("X", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
                    return s1

            return s1 + s2
        else:
            # Frame 10 (can have 3 shots)
            shots = [s for s in (s1, s2, s3) if s]
            if shots:
                return "".join(shots)
            if full_s and is_valid_bowling_shot(full_s):
                return full_s
            return ""

    def get_snapshots(self) -> list[dict]:
        """Return distinct scoreboard state snapshots."""
        return list(self._snapshots)

    def get_observation_count(self) -> int:
        """Total number of individual OCR observations stored."""
        return self._obs_count
