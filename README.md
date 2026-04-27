# sleep-tools

A Python package for inspecting, visualizing, transforming, and scoring rodent sleep recordings from EDF electrophysiology files acquired by the luminose team.

Follows Julia's Spike2 sleep-scoring protocol, reimplemented in open-source Python.

**[How to score a recording — step-by-step guide](how_to_score.md)**

## Features

- **Load** EDF recordings (EEG1, EEG2, EMG channels) and paired TSV annotation files
- **Analyse** signals: EEG high-pass filtering, EMG FIR band-pass + RMS envelope, delta/theta/alpha/beta/gamma band power, T:D ratio, spectrograms
- **Visualise** raw traces, per-band power time series, spectrograms, and a 4-panel overview figure
- **Oscilloscope** (`Scope.show`) — interactive scrollable Qt viewer for any combination of raw and derived signals, with auto-scaling time axis and per-channel amplitude controls

- **Video export** (`Scope.make_video`) — render a scrolling MP4 of any signals with configurable window, speed, and y-limits
- **Export** curated HDF5 datasets (`save_to_h5`) with raw signals, epoch features, and sleep-stage labels — NaN placeholders for any feature not yet computed, so the schema is always consistent
- **Score** sleep interactively via a Qt GUI: Wake, NREM, REM — auto-scoring with adjustable thresholds, manual epoch correction, undo/redo, save/load
- **Align** TTL sync triggers from a behavioral rig with `SyncAligner`

## Installation

```bash
mamba activate sleep-tools      # or conda activate sleep-tools
pip install -e ".[gui]"         # includes PySide6 for the oscilloscope GUI
```

> **Note:** `pip install -e .` (without `[gui]`) installs the core package only — signal processing, HDF5 export, and analysis work, but the oscilloscope viewer (`Scope.show`, `sleep-scope` CLI) requires PySide6 and will fail to import without it. Use `.[gui]` for the full experience.

For video export, also install `ffmpeg`:

```bash
conda install -c conda-forge ffmpeg
```

## Running the Viewer

After installation, launch the interactive scorer from the project root:

```bash
python run_scope.py                  # open empty — load a file from inside the viewer
python run_scope.py path/to/file.edf # load a specific recording on startup
```

Or use the installed CLI command (requires `pip install -e .`):

```bash
sleep-scope
sleep-scope path/to/file.edf
```

Or as a module:

```bash
python -m sleep_tools [path_to_edf]
```

## Quick Start

```python
from sleep_tools import SleepRecording, SleepAnalyzer, SleepVisualizer

rec = SleepRecording.from_edf("example_data/luminose/LUMI-0013_2026-03-24_14_15_34_export.edf")
print(rec)
# SleepRecording(animal='LUMI-0013', duration=11780.0s, channels=['EEG1', 'EEG2', 'EMG'], sfreq=400.0Hz)

ana = SleepAnalyzer(rec, epoch_len=5.0)
viz = SleepVisualizer(rec, ana)

# Raw signal traces (first 60 s)
fig = viz.plot_raw_traces(t_start=0, t_end=60)
fig.savefig("raw_traces.png", dpi=150, bbox_inches="tight")

# EEG spectrogram
fig = viz.plot_spectrogram(channel="EEG1", freq_max=50.0)
fig.savefig("spectrogram.png", dpi=150, bbox_inches="tight")

# 4-panel overview: spectrogram + delta/theta power + EMG RMS + T:D ratio
fig = viz.plot_overview(channel="EEG1")
fig.savefig("overview.png", dpi=150, bbox_inches="tight")
```

## Sleep Scoring

See **[how_to_score.md](how_to_score.md)** for a complete guide covering signal processing details and step-by-step scoring instructions.

```python
from sleep_tools import SleepRecording, SleepAnalyzer, Scope, ScoringSession

rec = SleepRecording.from_edf("path/to/recording_export.edf")
# eeg_channel=None (default) averages EEG1+EEG2; pass "EEG1" or "EEG2" to use one channel
ana = SleepAnalyzer(rec, epoch_len=5.0, eeg_channel=None)
features = ana.compute_all_features()

# Spike2-compatible analysis uses an explicit EEG channel and causal filtering.
spike2_ana = SleepAnalyzer(rec, epoch_len=5.0, eeg_channel="EEG2", profile="spike2")
spike2_features = spike2_ana.compute_all_features()

# Auto-score with default thresholds
session = ScoringSession(rec, epoch_len=5.0)
session.auto_score(features)
print(session)
# ScoringSession(animal='LUMI-0013', n_epochs=2356, epoch_len=5.0s,
#                W=823, N=1012, R=289, U=232)

# Save / export
session.save("LUMI-0013_sleep_scores.json")
session.to_csv("LUMI-0013_hypnogram.csv")

# Open the oscilloscope for interactive review and manual correction
scope = Scope(rec, ana)
scope.show()
```

The Scope window provides:
- Hypnogram strip with colour-coded epochs (amber = Wake, blue = NREM, green = REM)
- CLASSIFICATION panel: six threshold spinboxes + **Epoch length** field + **Reset Defaults** + Run Classification button
- **Epoch length field** — set the scoring window (seconds) before running classification; default 5.0 s
- **Reset Defaults** — one-click restore of all thresholds and epoch length to factory values
- **Draggable threshold lines** — dotted reference lines on δ-power, EMG RMS, and T:D axes; drag up/down to adjust thresholds live (spinboxes sync, and vice-versa)
- LABELING panel: state counts, W/N/R/U buttons, undo/redo, save/export
- Click or Shift+click epochs in the hypnogram to select; Cmd/Ctrl+click to toggle
- Press **W / N / R / U** to relabel selected epochs; **Cmd/Ctrl+Z/Y** to undo/redo
- **`?` button** — opens a step-by-step scoring guide with all keyboard shortcuts

## Oscilloscope Viewer

```python
from sleep_tools import SleepRecording, SleepAnalyzer, Scope

rec = SleepRecording.from_edf("path/to/recording_export.edf")
ana = SleepAnalyzer(rec, epoch_len=5.0)

scope = Scope(rec, ana)

# Open interactive window
scope.show()

# You can also start an empty scope and load files from the menu
# scope = Scope()
# scope.show()
```

The window features:
- **Horizontal scrollbar** — Scrub through the full recording. Mouse wheel also scrolls (proportional to delta — trackpad-friendly).
- **Play / Pause** — Transport button or `Space`. Speed slider (1×–100×, log-mapped).
- **Page navigation** — `<` / `>` buttons, or `[` / `]` keys, or PageUp / PageDown.
- **Fine scroll** — `←` / `→` keys (10 % of visible window per press).
- **Time Window** — Adjust visible width (0.1 – 3600 s) via the `⌛` menu.
- **Per-channel controls** — Amplitude spinbox, unit selector, and Optimize Scale button for each trace. `⊕` icon in the channel header centres that trace on its visible mean; `−` hides it.
- **Centre signals** — `⊕` button in the transport bar (or press `C`) centres all visible channels on their visible mean simultaneously. Per-channel `⊕` in the header row does the same for one channel. Clicking **Optimize Scale** resets that channel's offset back to zero baseline.
- **↕ Optimize All** — transport bar button (right of `⊕`) auto-scales all visible traces to fit the current window simultaneously.
- **Resizable panels** — Drag the splitter handles between the left channel panel, the centre canvas, and the right sidebar to resize them freely. The sidebar toggle (`☰`) collapses/restores the sidebar while remembering its last width.
- **Y-Axis Labels** — Shows current unit (e.g. `µV`, `µV²/Hz`); updates automatically.
- **Theme** — `☯` button toggles dark / light. Spinbox and combobox arrows adapt to the theme (white on dark, black on light).
- **? Help** — Leftmost transport button; step-by-step scoring guide with platform-aware shortcuts (Cmd on macOS).
- **EEG Channel** — `∿` button in the transport bar (right of `⊕`) opens a popup menu to select which EEG signal drives band-power and classification: **Average (EEG1+EEG2)** (default), **EEG1 only**, or **EEG2 only**. Button label updates live. Click **Analyze Signals** after changing to recompute features.
- **Analysis Profile** — `Std` / `Spike2` button beside the EEG selector chooses the feature pipeline. Spike2 mode disables hidden EEG averaging and auto-selects EEG2 when available.

## Video Export

```python
from sleep_tools import SleepRecording, SleepAnalyzer, Scope
from sleep_tools.scoring.state import ScoringSession

rec = SleepRecording.from_edf("path/to/recording_export.edf")
ana = SleepAnalyzer(rec, epoch_len=5.0)
ana.compute_all_features()

scope = Scope(rec, ana)

# Render a 1-minute MP4 covering the first hour at 60× speed, with hypnogram
# Option A: pass a ScoringSession object directly
session = ScoringSession.load("LUMI-0013_sleep_scores.json", rec)
path = scope.make_video(
    "output/my_recording.mp4",
    signals=["EEG1", "EEG2", "EMG", "delta_power", "td_ratio"],
    t_start=0,
    t_end=3600,          # 1-hour slice of recording
    x_window=30.0,       # 30 s visible window
    speed=60.0,          # 60 s of recording per second of video
    fps=30,
    dpi=150,
    session=session,     # adds colour-coded hypnogram strip with playhead
    show_hypnogram=True,
)
print(f"Saved: {path}")

# Option B: load sleep-stage labels directly from an HDF5 file
path = scope.make_video(
    "output/my_recording.mp4",
    signals=["EEG1", "EEG2", "EMG", "delta_power", "td_ratio"],
    speed=60.0,
    session_h5="output/LUMI-0013_scope.h5",  # labels + thresholds loaded automatically
    show_hypnogram=True,
)
print(f"Saved: {path}")

# Defaults: full recording, no hypnogram, output/LUMI-0013_scope.mp4
path = scope.make_video()
```

Requires `ffmpeg` on the system PATH (`conda install -c conda-forge ffmpeg`).

## Saving a Curated Dataset

```python
from sleep_tools import SleepRecording, SleepAnalyzer, save_to_h5

rec = SleepRecording.from_edf("path/to/recording_export.edf")
ana = SleepAnalyzer(rec, epoch_len=5.0)

# All features computed; labels default to "U" (unscored)
path = save_to_h5(rec, "output/dataset.h5", analyzer=ana, overwrite=True)

# Or save now, features as NaN — same schema, fill in later
path = save_to_h5(rec, "output/dataset_raw.h5", overwrite=True)
```

HDF5 layout: `/signals/{EEG1,EEG2,EMG}`, `/epochs/times`, `/epochs/labels`,
`/epochs/features/{delta_power,theta_power,alpha_power,beta_power,gamma_power,emg_rms,td_ratio}`,
`/annotations/` (if TSV loaded).

When saving with `session=...`, `/epochs/times` and `/epochs/features/*` always use the session epoch grid. If analyzer features were computed on a different timebase, they are interpolated onto the session epochs and the native analyzer time series is preserved under `/analysis/times` and `/analysis/features/*`.

Every dataset and group carries self-describing attributes so the file is readable without
the package:

| Location | Attributes written |
|----------|--------------------|
| `/` (root) | `animal_id`, `experiment_id`, `start_datetime`, `sfreq`, `duration_s`, `n_samples`, `channels`, `epoch_len`, `n_epochs`, `features_computed`, `analysis_profile`, `eeg_channel`, `feature_source`, `sleep_tools_version`, `band_definitions` (JSON), `saved_at` |
| `/signals/{ch}` | `unit="V"`, `sfreq` |
| `/epochs/times` | `units="s"`, `description` |
| `/epochs/labels` | `description` (label key: U/W/N/R) |
| `/epochs/features/` | `band_definitions` (JSON) |
| `/epochs/features/{name}` | `description`, `units`, `frequency_range_hz`, `sleep_relevance` |

## Recording Metadata

```python
meta = rec.metadata()
# {'animal_id': 'LUMI-0013', 'channels': ['EEG1', 'EEG2', 'EMG'],
#  'sfreq': 400.0, 'duration_s': 11780.0, 'n_samples': 4712000, ...}

sig = rec.signal_info()
# {'EEG1': {'unit': 'V', 'n_samples': 4712000, 'min': -0.0005, 'max': 0.0004, ...}, ...}
```

## Feature Columns

All 7 HDF5 feature columns are documented in `FEATURE_INFO` (exported from `sleep_tools`):

```python
from sleep_tools import FEATURE_INFO
for name, info in FEATURE_INFO.items():
    freq = info["frequency_range_hz"]
    print(f"{name}: {freq}  {info['units']}  — {info['sleep_relevance']}")
```

| Feature | Frequency range (Hz) | Units | Sleep relevance |
|---------|----------------------|-------|-----------------|
| `delta_power` | 0.5 – 4.0 | V²/Hz·Hz | Very high in NREM; primary NREM marker |
| `theta_power` | 6.0 – 10.0 | V²/Hz·Hz | Elevated in REM; numerator of T:D ratio |
| `alpha_power` | 8.0 – 13.0 | V²/Hz·Hz | Overlaps theta in rodents |
| `beta_power` | 13.0 – 30.0 | V²/Hz·Hz | Elevated during Wake/arousal |
| `gamma_power` | 30.0 – 100.0 | V²/Hz·Hz | High-frequency; elevated during arousal |
| `emg_rms` | 5.0 – 45.0 (EMG bandpass) | V | High in Wake; flat (atonia) in REM; low in NREM |
| `td_ratio` | theta / delta (derived) | dimensionless | Peak in REM; low in NREM; moderate in Wake |

In `standard` mode, band powers use Hann-windowed STFT (`scipy.signal.spectrogram`, `scaling="density"`) integrated over each band with `np.trapezoid`. EMG RMS uses an FIR bandpass (5–45 Hz, transition 1.8 Hz) followed by a causal exponential smoother (τ = 5 s). In `spike2` mode, EEG drift filtering is causal, EMG RMS uses an OSD4-like centred window, and band powers use an OSD4 `Pw(...)`-style approximation on a 0.1 s grid by default.

## Example Scripts

```bash
# Loads the longest EDF, prints feature stats, saves 4 figures to output/
conda run -n sleep-tools python examples/quick_start.py

# Prints a numeric summary of all computed features
conda run -n sleep-tools python examples/explore_features.py

# Saves HDF5 datasets demonstrating all save_to_h5 usage patterns
conda run -n sleep-tools python examples/save_dataset.py
```

## Running Tests

```bash
conda run -n sleep-tools python -m pytest tests/ -v
```

Tests use the real EDF/TSV files in `example_data/luminose/`.

## Data Format

| File | Description |
|------|-------------|
| `*_export.edf` | Electrophysiology recording (EEG1, EEG2, EMG @ 400 Hz) |
| `*_annotations.tsv` | Luminose annotation file with TTL sync events |
| `*_dataset.h5` | Curated HDF5 export: signals + epoch features + labels |
| `*_sleep_scores.json` | Saved scoring session (Stage 2) |
| `*_hypnogram.csv` | Exported hypnogram (Stage 2) |

## Sleep States

| State | EEG | EMG | Delta | T:D |
|-------|-----|-----|-------|-----|
| Wake | Low amp, high freq | High | ~0 | Moderate |
| NREM | High amp, low freq | Low | Very high | Low |
| REM | Low amp, high freq | Flat (atonia) | Low | High |

## Signal Processing

Mirrors Julia's Spike2 protocol:

| Signal | Processing |
|--------|-----------|
| EEG | IIR 2nd-order Butterworth LP at 0.5 Hz → subtract drift (≈ HP at 0.5 Hz) |
| EMG | FIR bandpass 5–45 Hz (transition 1.8 Hz) → exponential RMS envelope (τ = 5 s) |
| Band power | STFT with Hann window (default 5 s, 50 % overlap) |

For closest Spike2 compatibility:

```python
ana = SleepAnalyzer(rec, epoch_len=5.0, eeg_channel="EEG2", profile="spike2")
```

Spike2's exact `Pw(...)` band-power implementation is not public; the package exposes a compatible approximation based on the OSD4 script defaults and keeps the standard STFT pipeline as the backward-compatible default.

## Package Structure

```
sleep_tools/
├── __init__.py          # Public API
├── io.py                # EDF + TSV loading; SleepRecording dataclass; save_to_h5
├── analysis.py          # Signal filtering, band power, spectrogram, T:D ratio
├── visualization.py     # Matplotlib static figures (no GUI)
├── scope.py             # Scope: oscilloscope viewer + video export
└── scoring/
    ├── __init__.py      # Exports ScoringSession, AutoScoreThresholds, STATE_COLORS
    └── state.py         # Per-epoch labels, auto-scoring, undo/redo, save/load
```

## Requirements

- Python ≥ 3.11
- `mne ≥ 1.5`, `numpy`, `scipy`, `pandas`, `matplotlib`, `h5py ≥ 3.8` — installed by `pip install -e .`
- `PySide6 ≥ 6.10` — GUI/oscilloscope; installed by `pip install -e ".[gui]"`
- `ffmpeg` system binary — video export only; install via `conda install -c conda-forge ffmpeg`
