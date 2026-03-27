"""ScoringSession: per-epoch sleep stage labels, auto-scoring, undo/redo, persistence."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sleep_tools.io import SleepRecording

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

STATE_COLORS: dict[str, str] = {
    "W": "#e3b341",  # amber  — Wake
    "N": "#58a6ff",  # blue   — NREM
    "R": "#3fb950",  # green  — REM
    "U": "#8b949e",  # gray   — Unscored
}

_VALID_STATES: frozenset[str] = frozenset({"W", "N", "R", "U"})

# ---------------------------------------------------------------------------
# Unit conversion: base V / V² → display units that match scope.py
#   delta_power: V² → µV²/Hz   (scale = 1e12 / bandwidth_Hz)
#   emg_rms:     V  → µV       (scale = 1e6)
#   td_ratio:    dimensionless
# ---------------------------------------------------------------------------
_DELTA_BW: float = 4.0 - 0.5          # BANDS["delta"] = (0.5, 4.0) Hz
_SCALE_DELTA: float = 1e12 / _DELTA_BW
_SCALE_EMG: float = 1e6
# td_ratio is already dimensionless — no conversion needed


# ---------------------------------------------------------------------------
# Threshold dataclass
# ---------------------------------------------------------------------------

@dataclass
class AutoScoreThresholds:
    """Threshold parameters for automatic sleep staging.

    All power thresholds are in **µV²/Hz** (matching the default display unit
    in the Scope oscilloscope).  EMG thresholds are in **µV**.
    The T:D ratio is dimensionless.

    Scoring order (Julia's protocol): Wake → NREM → REM.
    Later stages overwrite earlier ones, so REM can overwrite NREM.

    Defaults
    --------
    Wake : delta < 1200 µV²/Hz AND emg > 3 µV
    NREM : delta > 1000 µV²/Hz AND emg < 5 µV
    REM  : td_ratio > 4 AND emg < 3 µV
    """

    delta_wake: float = 1200.0  # µV²/Hz — below this → candidate Wake
    delta_nrem: float = 1000.0  # µV²/Hz — above this → candidate NREM
    emg_wake:   float = 3.0     # µV     — above this → Wake confirmed
    emg_nrem:   float = 5.0     # µV     — below this → NREM confirmed
    emg_rem:    float = 3.0     # µV     — below this → REM confirmed (atonia)
    td_rem:     float = 4.0     # dimensionless — above this → REM candidate


# ---------------------------------------------------------------------------
# ScoringSession
# ---------------------------------------------------------------------------

class ScoringSession:
    """Manages per-epoch sleep stage labels for one recording.

    Parameters
    ----------
    recording:
        Loaded :class:`~sleep_tools.io.SleepRecording`.
    epoch_len:
        Non-overlapping epoch length in seconds (default 5.0).

    Attributes
    ----------
    times : np.ndarray
        Centre times of each epoch in seconds (shape ``(n_epochs,)``).
    labels : np.ndarray
        Per-epoch state string (``"W"``, ``"N"``, ``"R"``, ``"U"``),
        shape ``(n_epochs,)``, dtype ``object``.
    thresholds : AutoScoreThresholds
        Thresholds used for the last ``auto_score()`` call.
    """

    _MAX_UNDO: int = 100

    def __init__(
        self,
        recording: SleepRecording,
        epoch_len: float = 5.0,
    ) -> None:
        self.recording = recording
        self.epoch_len = float(epoch_len)

        # Non-overlapping epoch centres: L/2, 3L/2, 5L/2, …
        self.times: np.ndarray = np.arange(
            self.epoch_len / 2.0,
            recording.duration,
            self.epoch_len,
            dtype=np.float64,
        )
        self.labels: np.ndarray = np.full(len(self.times), "U", dtype=object)

        self.thresholds: AutoScoreThresholds = AutoScoreThresholds()

        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # Index helpers
    # ------------------------------------------------------------------ #

    def epoch_index(self, t: float) -> int:
        """Convert a time in seconds to the nearest epoch index (0-based, clamped).

        For epoch_len=5: t in [0,5) → 0, t in [5,10) → 1, etc.
        """
        idx = int(t / self.epoch_len)
        return int(np.clip(idx, 0, len(self.times) - 1))

    # ------------------------------------------------------------------ #
    # Label assignment
    # ------------------------------------------------------------------ #

    def label_epoch(self, idx: int, state: str) -> None:
        """Assign *state* to a single epoch by index."""
        if state not in _VALID_STATES:
            raise ValueError(f"State {state!r} must be one of {sorted(_VALID_STATES)}")
        self._push_undo()
        self.labels[idx] = state

    def label_range(self, t_start: float, t_end: float, state: str) -> None:
        """Assign *state* to all epochs whose centres fall in [t_start, t_end]."""
        if state not in _VALID_STATES:
            raise ValueError(f"State {state!r} must be one of {sorted(_VALID_STATES)}")
        i0 = self.epoch_index(t_start)
        i1 = self.epoch_index(t_end)
        if i0 > i1:
            i0, i1 = i1, i0
        self._push_undo()
        self.labels[i0 : i1 + 1] = state

    def label_indices(self, i0: int, i1: int, state: str) -> None:
        """Assign *state* to epochs from index *i0* to *i1* inclusive."""
        if state not in _VALID_STATES:
            raise ValueError(f"State {state!r} must be one of {sorted(_VALID_STATES)}")
        lo = int(np.clip(min(i0, i1), 0, len(self.times) - 1))
        hi = int(np.clip(max(i0, i1), 0, len(self.times) - 1))
        self._push_undo()
        self.labels[lo : hi + 1] = state

    # ------------------------------------------------------------------ #
    # Auto-scoring (Julia's Spike2 protocol)
    # ------------------------------------------------------------------ #

    def auto_score(
        self,
        features: dict,
        thresholds: AutoScoreThresholds | None = None,
    ) -> None:
        """Apply threshold-based auto-scoring.

        Implements Julia's protocol:

        1. **Wake**: delta < ``delta_wake`` AND emg > ``emg_wake``
        2. **NREM**: delta > ``delta_nrem`` AND emg < ``emg_nrem``
        3. **REM**: td_ratio > ``td_rem`` AND emg < ``emg_rem``

        Later stages overwrite earlier ones (REM can overwrite NREM).
        Epochs not matching any rule remain ``"U"`` (unscored).

        Parameters
        ----------
        features:
            Dict returned by ``SleepAnalyzer.compute_all_features()``.
            Must contain keys ``times``, ``delta_power``, ``emg_rms``,
            ``td_ratio`` (all in base V / V² units).
        thresholds:
            Override thresholds.  If *None*, uses ``self.thresholds``.
        """
        if thresholds is None:
            thresholds = self.thresholds
        else:
            self.thresholds = thresholds

        feat_times = np.asarray(features["times"], dtype=np.float64)
        delta_base = np.asarray(features["delta_power"], dtype=np.float64)
        emg_base   = np.asarray(features["emg_rms"],    dtype=np.float64)
        td_vals    = np.asarray(features["td_ratio"],   dtype=np.float64)

        # Convert base units → display units (matching Scope spinbox units)
        delta_disp = delta_base * _SCALE_DELTA
        emg_disp   = emg_base   * _SCALE_EMG

        self._push_undo()
        new_labels = np.full(len(self.times), "U", dtype=object)

        for i, t_centre in enumerate(self.times):
            # Map this scoring epoch to the nearest feature sample
            k = int(np.argmin(np.abs(feat_times - t_centre)))

            d  = delta_disp[k]
            e  = emg_disp[k]
            td = td_vals[k]

            # Stage 1 — Wake (applied first; can be overwritten below)
            if d < thresholds.delta_wake and e > thresholds.emg_wake:
                new_labels[i] = "W"

            # Stage 2 — NREM (overwrites Wake or Unscored)
            if d > thresholds.delta_nrem and e < thresholds.emg_nrem:
                new_labels[i] = "N"

            # Stage 3 — REM (overwrites anything, including NREM)
            if td > thresholds.td_rem and e < thresholds.emg_rem:
                new_labels[i] = "R"

        self.labels = new_labels

    # ------------------------------------------------------------------ #
    # Undo / redo
    # ------------------------------------------------------------------ #

    def _push_undo(self) -> None:
        """Save current labels to undo stack before any mutation."""
        self._undo_stack.append(self.labels.copy())
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> bool:
        """Undo the last label change.  Returns ``True`` if the stack was non-empty."""
        if not self._undo_stack:
            return False
        self._redo_stack.append(self.labels.copy())
        self.labels = self._undo_stack.pop()
        return True

    def redo(self) -> bool:
        """Redo the last undone change.  Returns ``True`` if the stack was non-empty."""
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.labels.copy())
        self.labels = self._redo_stack.pop()
        return True

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def state_counts(self) -> dict[str, int]:
        """Return per-state epoch counts (always includes W/N/R/U keys)."""
        result: dict[str, int] = {"W": 0, "N": 0, "R": 0, "U": 0}
        unique, counts = np.unique(self.labels, return_counts=True)
        for s, c in zip(unique.tolist(), counts.tolist()):
            result[s] = c
        return result

    def state_durations(self) -> dict[str, float]:
        """Return per-state total duration in seconds."""
        return {s: c * self.epoch_len for s, c in self.state_counts().items()}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> Path:
        """Save labels and thresholds to a JSON file.

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        path = Path(path)
        data = {
            "animal_id":  self.recording.animal_id,
            "epoch_len":  self.epoch_len,
            "n_epochs":   len(self.times),
            "labels":     self.labels.tolist(),
            "thresholds": asdict(self.thresholds),
        }
        path.write_text(json.dumps(data, indent=2))
        return path.resolve()

    @classmethod
    def load(cls, path: str | Path, recording: SleepRecording) -> "ScoringSession":
        """Load a previously saved session from *path*.

        Parameters
        ----------
        path:
            JSON file written by :meth:`save`.
        recording:
            The matching :class:`~sleep_tools.io.SleepRecording` (must
            produce the same number of epochs as the saved session).

        Raises
        ------
        ValueError
            If the epoch count does not match.
        """
        data = json.loads(Path(path).read_text())
        session = cls(recording, epoch_len=float(data["epoch_len"]))
        loaded = np.asarray(data["labels"], dtype=object)
        if len(loaded) != len(session.times):
            raise ValueError(
                f"Saved session has {len(loaded)} epochs but recording "
                f"produces {len(session.times)} epochs at "
                f"epoch_len={session.epoch_len}s."
            )
        session.labels = loaded
        if data.get("thresholds"):
            session.thresholds = AutoScoreThresholds(**data["thresholds"])
        return session

    def to_csv(self, path: str | Path) -> Path:
        """Export the hypnogram to a CSV file (epoch_index, time_s, label).

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        path = Path(path)
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch_index", "time_s", "label"])
            for i, (t, lbl) in enumerate(zip(self.times, self.labels)):
                writer.writerow([i, f"{t:.4f}", lbl])
        return path.resolve()

    def __repr__(self) -> str:
        counts = self.state_counts()
        return (
            f"ScoringSession(animal={self.recording.animal_id!r}, "
            f"n_epochs={len(self.times)}, epoch_len={self.epoch_len}s, "
            f"W={counts['W']}, N={counts['N']}, R={counts['R']}, U={counts['U']})"
        )
