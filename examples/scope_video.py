"""Render a real-time scrolling video of an arbitrary 20 s epoch.

Run:
    conda run -n sleep-tools python examples/scope_video.py

Adjust T1 and T2 to select any 20 s window within the recording.
Output is saved to output/<animal_id>_scope_epoch.mp4 in the cwd.

Requires ffmpeg on PATH:
    conda install -c conda-forge ffmpeg
"""
from pathlib import Path

from sleep_tools import SleepRecording, SleepAnalyzer, Scope
from sleep_tools.scoring.state import ScoringSession, AutoScoreThresholds

EDF = Path("example_data/luminose/LUMI-0013_2026-03-24_14_15_34_export.edf")

# ── Epoch selection ───────────────────────────────────────────────────────────
T1 = 60.0   # start time in seconds
T2 = 80.0   # end time in seconds  (T2 - T1 = 20 s)

rec = SleepRecording.from_edf(EDF)
ana = SleepAnalyzer(rec, epoch_len=5.0)

# ── Auto-score the full recording ─────────────────────────────────────────────
features = ana.compute_all_features()
session = ScoringSession(rec, epoch_len=5.0)
session.auto_score(features, AutoScoreThresholds())

scope = Scope(rec, ana)

out = Path("output") / f"{rec.animal_id}_scope_epoch.mp4"

path = scope.make_video(
    out,
    signals=["EEG1", "EEG2", "EMG", "delta_power", "emg_rms", "td_ratio"],
    t_start=T1,
    t_end=T2,
    x_window=10.0,   # 10 s scrolling window within the 20 s clip
    speed=1.0,       # real-time: 1 s of recording → 1 s of video
    fps=30,
    dpi=150,
    session=session,         # adds colour-coded sleep stage strip
    show_hypnogram=True,     # True by default — pass False to omit
)
print(f"Saved: {path}")
