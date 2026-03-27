"""Save a curated HDF5 dataset from a sleep recording.

Demonstrates three usage patterns:

  1. Raw signals only (no features computed yet).
  2. Full features (analyzer passed in).
  3. Full features + manual labels.

Run from the repo root:

    conda run -n sleep-tools python examples/save_dataset.py
"""

from pathlib import Path

import h5py
import numpy as np

from sleep_tools import SleepRecording, SleepAnalyzer, save_to_h5

EDF = Path("example_data/luminose/LUMI-0013_2026-03-24_14_15_34_export.edf")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# ── 1. Load recording ────────────────────────────────────────────────────────
print("Loading recording …")
rec = SleepRecording.from_edf(EDF, verbose=False)
print(rec)

# ── 2. Save: raw signals only, no features computed ─────────────────────────
p1 = save_to_h5(rec, OUT / "dataset_no_features.h5", overwrite=True)
print(f"\n[1] Saved (no features): {p1}")

with h5py.File(p1) as f:
    print(f"    root attrs  : {dict(f.attrs)}")
    print(f"    /signals    : {list(f['signals'].keys())}")
    print(f"    /epochs/features/delta_power[:5]: "
          f"{f['epochs/features/delta_power'][:5]}")   # → all NaN

# ── 3. Save: with computed features ─────────────────────────────────────────
print("\nComputing features …")
ana = SleepAnalyzer(rec, epoch_len=5.0)

p2 = save_to_h5(rec, OUT / "dataset_with_features.h5", analyzer=ana, overwrite=True)
print(f"[2] Saved (with features): {p2}")

with h5py.File(p2) as f:
    n_epochs = f["epochs/times"].shape[0]
    delta = f["epochs/features/delta_power"][:]
    print(f"    n_epochs    : {n_epochs}")
    print(f"    delta_power : min={delta.min():.3e}  max={delta.max():.3e}  "
          f"nan={np.isnan(delta).sum()}")
    print(f"    labels[:5]  : {f['epochs/labels'][:5]}")   # → all "U"

# ── 4. Save: with features + fake labels ────────────────────────────────────
feat = ana.compute_all_features()
n = len(feat["times"])
# Assign dummy labels: first third Wake, second NREM, last third REM
dummy_labels = np.where(
    np.arange(n) < n // 3, "W",
    np.where(np.arange(n) < 2 * n // 3, "N", "R"),
)

p3 = save_to_h5(
    rec,
    OUT / "dataset_with_labels.h5",
    analyzer=ana,
    labels=dummy_labels,
    overwrite=True,
)
print(f"\n[3] Saved (features + labels): {p3}")

with h5py.File(p3) as f:
    labels = f["epochs/labels"][:]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"    label counts: {dict(zip(unique, counts))}")

# ── 5. Save: features only, no raw signals ──────────────────────────────────
p4 = save_to_h5(
    rec,
    OUT / "dataset_features_only.h5",
    analyzer=ana,
    include_raw_signals=False,
    overwrite=True,
)
print(f"\n[4] Saved (features only, no signals): {p4}")
print(f"    file size: {p4.stat().st_size / 1e6:.1f} MB")

print("\nDone. All files written to output/")
