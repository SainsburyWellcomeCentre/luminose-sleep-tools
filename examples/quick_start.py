"""Quick-start example: load, analyse, and plot a sleep recording.

Run from the repo root with the sleep-tools conda env active:

    conda run -n sleep-tools python examples/quick_start.py

Or interactively:

    conda activate sleep-tools
    python examples/quick_start.py
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # change to "TkAgg" / "Qt5Agg" for interactive display
import matplotlib.pyplot as plt

from sleep_tools import SleepRecording, SleepAnalyzer, SleepVisualizer

# ── 1. Load recording (load a test edf file) ────────────────────────────────────
EDF = Path("example_data/luminose/LUMI-0013_2026-03-24_14_15_34_export.edf")

print("Loading recording …")
rec = SleepRecording.from_edf(EDF, verbose=False)
print(rec)

# ── 2. Print annotation summary ──────────────────────────────────────────────
if rec.annotations is not None:
    n_ttl = rec.annotations["Annotation"].str.startswith("TTL").sum()
    print(f"Annotations loaded: {len(rec.annotations)} rows, {n_ttl} TTL events")

# ── 3. Set up analyzer ───────────────────────────────────────────────────────
ana = SleepAnalyzer(rec, epoch_len=5.0)

# ── 4. Quick filter check ────────────────────────────────────────────────────
eeg = ana.filter_eeg("EEG1")
emg_filt = ana.filter_emg("EMG")
emg_rms = ana.emg_rms(emg_filt)
print(f"EEG filtered: {eeg.shape}, mean={eeg.mean():.3e}")
print(f"EMG RMS:      min={emg_rms.min():.3e}  max={emg_rms.max():.3e}")

# ── 5. Band power for a short segment ────────────────────────────────────────
t_seg, p_delta = ana.band_power(eeg, band=(0.5, 4.0))
_, p_theta = ana.band_power(eeg, band=(6.0, 10.0))
td = ana.td_ratio(p_delta, p_theta)
print(f"Spectrogram epochs: {len(t_seg)}")

# ── 6. Visualize ─────────────────────────────────────────────────────────────
viz = SleepVisualizer(rec, ana)

out = Path("output")
out.mkdir(exist_ok=True)

print("Plotting raw traces (first 60 s) …")
fig = viz.plot_raw_traces(t_start=0, t_end=60)
fig.savefig(out / "raw_traces.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Plotting spectrogram (EEG1) …")
fig = viz.plot_spectrogram("EEG1", freq_max=50.0)
fig.savefig(out / "spectrogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Plotting band power time series …")
fig = viz.plot_band_timeseries("EEG1")
fig.savefig(out / "band_power.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Plotting overview figure …")
fig = viz.plot_overview("EEG1")
fig.savefig(out / "overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nFigures saved to {out.resolve()}/")
