# CLAUDE.md

This file is automatically loaded by Claude Code and compatible LLM coding assistants.
For full project context, architecture, and implementation plan see [agent.md](agent.md).
For signal processing details and scoring workflow see [how_to_score.md](how_to_score.md).

## Quick Reference

- **Package**: `sleep_tools` — rodent sleep scoring in Python
- **Conda env**: `sleep-tools` (mamba); activate before running or installing
- **EDF I/O**: use `mne.io.read_raw_edf` (not pyedf)
- **GUI framework**: PySide6
- **Data**: example recordings in `example_data/luminose/`

## Coding Rules

- All parameters (epoch length, filter cutoffs, thresholds) must be runtime-configurable — never hard-coded
- Type hints on all public functions and class methods
- Tests go in `tests/`; use the example EDF/TSV data for integration tests
- Install with `pip install -e .` via `pyproject.toml`

## HDF5 Export

`save_to_h5(recording, path, *, analyzer=None, labels=None, epoch_len=5.0, include_raw_signals=True, overwrite=False)` — in `io.py`, exported from `sleep_tools`.

- Pass `analyzer` to include computed features; omit for NaN placeholders (schema is always identical).
- All 7 feature columns (`delta_power`, `theta_power`, `alpha_power`, `beta_power`, `gamma_power`, `emg_rms`, `td_ratio`) are always present — NaN when not computed.
- `labels` (per-epoch sleep stage strings) default to `"U"` (unscored) when omitted.
- Requires `h5py` (in `pyproject.toml` dependencies).

### HDF5 Metadata written per object

| Location | Attribute | Content |
|----------|-----------|---------|
| `/` (root) | `sleep_tools_version` | package version string |
| `/` | `band_definitions` | JSON — exact Hz ranges for all 5 EEG bands |
| `/signals/{ch}` | `unit`, `sfreq` | `"V"`, sampling frequency in Hz |
| `/epochs/times` | `units`, `description` | `"s"`, `"epoch centre times"` |
| `/epochs/labels` | `description` | stage label key |
| `/epochs/features/` (group) | `band_definitions` | JSON — same as root |
| `/epochs/features/{name}` | `description`, `units`, `frequency_range_hz`, `sleep_relevance` | from `FEATURE_INFO` |

## Feature Column Descriptions

`FEATURE_INFO` (exported from `sleep_tools`, defined in `analysis.py`) — dict keyed by feature name, each entry has `description`, `frequency_range_hz`, `units`, `sleep_relevance`.

| Feature | Frequency range (Hz) | Units | Sleep relevance |
|---------|----------------------|-------|-----------------|
| `delta_power` | 0.5 – 4.0 | V²/Hz · Hz | Very high in NREM; primary NREM marker |
| `theta_power` | 6.0 – 10.0 | V²/Hz · Hz | Elevated in REM; numerator of T:D ratio |
| `alpha_power` | 8.0 – 13.0 | V²/Hz · Hz | Overlaps theta in rodents; custom staging |
| `beta_power` | 13.0 – 30.0 | V²/Hz · Hz | Elevated during Wake/arousal |
| `gamma_power` | 30.0 – 100.0 | V²/Hz · Hz | High-frequency; elevated during arousal |
| `emg_rms` | 5.0 – 45.0 (EMG FIR bandpass) | V | High in Wake; flat (atonia) in REM; low in NREM |
| `td_ratio` | derived: theta / delta | dimensionless | Peak in REM; low in NREM; moderate in Wake |

All band powers use Hann-windowed STFT (`scipy.signal.spectrogram`, `scaling="density"`), integrated via `np.trapezoid` over the exact band edges in `BANDS`.

## SleepRecording Metadata Methods

- `recording.metadata()` → `dict` — `animal_id`, `channels`, `sfreq`, `duration_s`, `n_samples`, `has_annotations`, `n_annotations`, …
- `recording.signal_info()` → `dict[str, dict]` — per-channel stats: `{unit, n_samples, min, max, mean, std}` in Volts

## Oscilloscope & Video Export (`scope.py`)

`Scope(recording, analyzer=None)` — exported from `sleep_tools`.

Two methods:

### `show(signals=None, x_window=30.0)`
Opens a PySide6 Qt window.  Backend must be `QtAgg` (set at module level in `scope.py`).

- `signals`: list of channel/feature names (defaults: all raw channels + all derived if analyzer given)
- `x_window`: visible time window in seconds
- Controls: horizontal scrollbar, window-width spinbox, per-channel amplitude spinbox, Reset Y
- **Y-axis labels**: each channel axis shows `<channel name>\n(<unit>)` rotated horizontal on the left side of the plot; updates automatically when unit selector changes
- **Channel hide button**: `−` minus button per channel in the left panel; click to hide, re-add via **+ Add Channel**
- **Epoch multi-select on hypnogram**: plain click = single epoch; Shift+click = contiguous range from anchor; Ctrl+click = toggle individual epoch in/out of selection (non-contiguous); single undo entry covers all epochs in one assignment

### `make_video(output_path=None, *, signals, t_start, t_end, x_window, y_lims, fps, speed, figsize, dpi)`
Renders a scrolling MP4 video using `matplotlib.animation.FFMpegWriter`.

- Defaults output to `output/<animal_id>_scope.mp4` in cwd
- `speed`: recording-seconds per video-second (e.g. `speed=60` → 1-hour recording → 1-minute video)
- `y_lims`: dict `{signal_name: (ymin, ymax)}` in display units (µV for EEG/EMG)
- Requires `ffmpeg` on PATH; raises `RuntimeError` with install instructions if missing

### Signal names
| Name | Source | Unit |
|------|--------|------|
| `EEG1`, `EEG2`, `EMG` | raw EDF | µV |
| `delta_power` … `gamma_power` | analyzer | V²/Hz |
| `emg_rms` | analyzer | µV |
| `td_ratio` | analyzer | dimensionless |

## Current Stage

**Stage 1 complete** — `io.py` (incl. `save_to_h5`), `analysis.py`, `visualization.py`, `scope.py` (oscilloscope + video export) implemented and tested.

**Stage 2 complete** — `scoring/state.py` (`ScoringSession`, `AutoScoreThresholds`, `STATE_COLORS`), auto-scoring (Wake→NREM→REM thresholds), hypnogram strip in Scope, keyboard hotkeys (W/N/R/U, Ctrl+Z/Y), CLASSIFICATION + LABELING sidebar panels, save JSON / export CSV / save HDF5.

See `agent.md` → Stage Breakdown for what is complete and what is next.

## Channel Names

Raw EDF channels are auto-renamed on load:
- `EEG EEG1A-B` → `EEG1`
- `EEG EEG2A-B` → `EEG2`
- `EMG EMG` → `EMG`

Always use the short canonical names (`EEG1`, `EEG2`, `EMG`) in code.
