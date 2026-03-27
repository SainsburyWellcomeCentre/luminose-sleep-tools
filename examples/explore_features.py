"""Explore all computed features for a recording.

Demonstrates compute_all_features() and prints a numeric summary.

Run from the repo root:

    conda run -n sleep-tools python examples/explore_features.py
"""
from pathlib import Path
import numpy as np

from sleep_tools import SleepRecording, SleepAnalyzer

EDF = Path("example_data/luminose/LUMI-0013_2026-03-24_14_15_34_export.edf")

print("Loading …")
rec = SleepRecording.from_edf(EDF, verbose=False)
print(rec)
print()

ana = SleepAnalyzer(rec, epoch_len=5.0)

print("Computing features (this takes ~10–20 s for a 3-hour recording) …")
feats = ana.compute_all_features(eeg_channel="EEG1", emg_channel="EMG")
print()

for key, val in feats.items():
    if isinstance(val, np.ndarray):
        finite = val[np.isfinite(val)]
        print(f"  {key:20s}  shape={val.shape}  "
              f"min={finite.min():.3e}  max={finite.max():.3e}  "
              f"mean={finite.mean():.3e}")

print()
print("T:D ratio — 95th percentile:", np.nanpercentile(feats["td_ratio"], 95))
print("EMG RMS   — 95th percentile:", np.nanpercentile(feats["emg_rms"], 95))
