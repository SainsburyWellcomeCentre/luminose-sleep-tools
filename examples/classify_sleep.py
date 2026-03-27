"""classify_sleep.py — Stage 2: classify sleep stages from script or GUI.

Usage (script):
    conda run -n sleep-tools python examples/classify_sleep.py

Usage (GUI):
    Opens the Scope oscilloscope with the right-panel scoring UI.
"""
from pathlib import Path

from sleep_tools import (
    SleepRecording, SleepAnalyzer, ScoringSession, AutoScoreThresholds, save_to_h5
)

EDF = Path("example_data/luminose/LUMI-0013_2026-03-24_14_15_34_export.edf")
OUT_H5  = Path("output/LUMI-0013_classified.h5")
OUT_CSV = Path("output/LUMI-0013_hypnogram.csv")
OUT_JSON = Path("output/LUMI-0013_sleep_scores.json")

# ── 1. Load recording & compute features ─────────────────────────────────
rec = SleepRecording.from_edf(EDF, verbose=False)
print(f"Loaded: {rec}")

ana = SleepAnalyzer(rec, epoch_len=5.0)
features = ana.compute_all_features()
print(f"Features computed: {len(features['times'])} windows")

# ── 2. Auto-classify with Julia's protocol ───────────────────────────────
#   Scoring order: Wake → NREM → REM (later stages overwrite earlier)
#   All power thresholds in µV²/Hz; EMG threshold in µV
thresholds = AutoScoreThresholds(
    delta_wake = 1200.0,   # µV²/Hz — Wake: delta below this
    delta_nrem = 1000.0,   # µV²/Hz — NREM: delta above this
    emg_wake   = 3.0,      # µV     — Wake: EMG above this
    emg_nrem   = 5.0,      # µV     — NREM/REM: EMG below this
    emg_rem    = 3.0,      # µV     — REM: EMG below this (muscle atonia)
    td_rem     = 4.0,      # T:D ratio above this → REM candidate
)

session = ScoringSession(rec, epoch_len=5.0)
session.auto_score(features, thresholds)

counts = session.state_counts()
durs = session.state_durations()
print("\nAuto-classification results:")
for state in ["W", "N", "R", "U"]:
    pct = 100 * counts[state] / len(session.times)
    print(f"  {state}: {counts[state]} epochs  {durs[state]/60:.1f} min  ({pct:.1f}%)")

# ── 3. Manual corrections (scripted example) ─────────────────────────────
#   In the GUI you'd click the hypnogram; here we do it programmatically.
#   session.label_indices(i0, i1, 'W')  — epochs i0..i1 inclusive
#   session.label_range(t_start, t_end, 'N')  — by time in seconds
#   session.undo() / session.redo()

# ── 4. Save session, HDF5, and CSV ───────────────────────────────────────
OUT_H5.parent.mkdir(parents=True, exist_ok=True)

session.save(OUT_JSON)
print(f"\nSession saved → {OUT_JSON}")

session.to_csv(OUT_CSV)
print(f"Hypnogram CSV → {OUT_CSV}")

save_to_h5(rec, OUT_H5, analyzer=ana, session=session, overwrite=True)
print(f"HDF5 dataset  → {OUT_H5}")

# ── 5. Open interactive GUI (optional) ───────────────────────────────────
# Uncomment the lines below to launch the Scope GUI with scoring panel.
# After "Analyze Signals", the right sidebar shows CLASSIFICATION + LABELING panels.

# from sleep_tools import Scope
# scope = Scope(rec, ana)
# scope.show()   # opens GUI; scroll + click hypnogram; press W/N/R/U to label
