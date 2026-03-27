"""Demo: oscilloscope viewer and video export.

Run:
    conda run -n sleep-tools python examples/scope_demo.py

The first call opens an interactive Qt window.
The second call renders output/LUMI-0013_scope.mp4 (requires ffmpeg).
"""
from pathlib import Path

from sleep_tools import SleepRecording, SleepAnalyzer, Scope

EDF = Path("example_data/luminose/LUMI-0013_2026-03-24_14_15_34_export.edf")

rec = SleepRecording.from_edf(EDF)
ana = SleepAnalyzer(rec, epoch_len=5.0)
scope = Scope(rec, ana)

# ── 1. Interactive oscilloscope ──────────────────────────────────────────────
# Uncomment to open the Qt window (blocking until closed):
scope.show(x_window=30.0)

# Custom signal selection:
# scope.show(["EEG1", "EMG", "delta_power", "td_ratio"], x_window=60.0)

# ── 2. Video export ──────────────────────────────────────────────────────────
# path = scope.make_video(
#     "output/LUMI-0013_scope.mp4",
#     signals=["EEG1", "EEG2", "EMG", "delta_power", "emg_rms", "td_ratio"],
#     t_start=0,
#     t_end=600,        # first 10 minutes
#     x_window=30.0,    # 30 s scrolling window
#     speed=60.0,       # 60 s recording → 1 s video
#     fps=30,
#     dpi=120,
# )
# print(f"Video saved: {path}")
