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
- **Main entry point**: `python run_scope.py` or `sleep-scope` (CLI after `pip install -e .`) or `python -m sleep_tools`

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

## HDF5 Session Reload

`ScoringSession.from_h5(path, recording)` — classmethod in `scoring/state.py`, exported from `sleep_tools`.

Loads epoch labels and scoring thresholds from a previously saved HDF5 file (written by `save_to_h5` with a session argument).

- Reads `epoch_len` from root attrs, labels from `/epochs/labels`, thresholds from `/epochs/thresholds` attrs (if present).
- Validates that epoch count matches the recording at the stored `epoch_len`; raises `ValueError` on mismatch.
- Thresholds are optional — if the HDF5 was saved without a session, defaults are used.
- In Scope: **"Load Session from H5..."** button in the CLASSIFICATION panel (always visible when a recording is loaded, no prior classification required).

```python
session = ScoringSession.from_h5("output/LUMI-0013_scope.h5", recording)
```

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
- Controls: horizontal scrollbar, window-width spinbox, per-channel amplitude spinbox
- **Y-axis labels**: each channel axis shows `<channel name>\n(<unit>)` rotated horizontal on the left side of the plot; updates automatically when unit selector changes
- **Channel hide button**: `−` minus button per channel in the left panel; click to hide, re-add via **+ Add Channel**
- **Epoch multi-select on hypnogram**: plain click = single epoch; Shift+click = contiguous range from anchor; Ctrl/Cmd+click = toggle individual epoch in/out of selection (non-contiguous); single undo entry covers all epochs in one assignment
- **Keyboard shortcuts**: Space (play/pause), `[`/`]` or PageUp/PageDown (page back/forward), `←`/`→` (fine scroll 10%), mouse wheel (proportional scroll), Ctrl/Cmd+Z/Y (undo/redo), Ctrl/Cmd+O (open folder), Ctrl/Cmd+E (open file), W/N/R/U (assign sleep stage), C (centre all signals on visible mean)
- **Centre signals**: `⊕` button in transport bar (or `C` key) centres all channels on their visible mean; per-channel `⊕` icon in the channel header row (right of label, left of `−`) centres that channel only; clicking **Optimize Scale** resets centering for that channel to zero baseline
- **Resizable panels**: all three panels (left channel controls, centre canvas, right sidebar) are separated by draggable `QSplitter` handles; drag to resize; sidebar toggle (☰) collapses/restores with saved width
- **Draggable threshold lines**: after Run Classification, dotted reference lines on δ-power, EMG RMS, and T:D ratio can be dragged vertically; spinboxes update live and vice-versa
- **? help button**: leftmost transport button; shows step-by-step scoring instructions (keyboard shortcuts are platform-aware: Cmd on macOS, Ctrl elsewhere)

### `make_video(output_path=None, *, signals, t_start, t_end, x_window, y_lims, fps, speed, figsize, dpi, session=None, session_h5=None, show_hypnogram=True)`
Renders a scrolling MP4 video using `matplotlib.animation.FFMpegWriter`.

- Defaults output to `output/<animal_id>_scope.mp4` in cwd
- `speed`: recording-seconds per video-second (e.g. `speed=60` → 1-hour recording → 1-minute video)
- `y_lims`: dict `{signal_name: (ymin, ymax)}` in display units (µV for EEG/EMG)
- `session`: pass a `ScoringSession` to include a sleep-stage hypnogram strip
- `session_h5`: path to an HDF5 file saved by `save_to_h5`; loads the session automatically via `ScoringSession.from_h5` when `session` is `None` — convenience alternative to constructing a `ScoringSession` explicitly
- `show_hypnogram`: `True` by default; renders colour-coded hypnogram row (W/N/R/U) beneath signal traces when a session is available (via `session` or `session_h5`); strip spans the full recording with a white vertical playhead tracking the current scroll position
- Requires `ffmpeg` on PATH; raises `RuntimeError` with install instructions if missing

### Signal names
| Name | Source | Unit |
|------|--------|------|
| `EEG1`, `EEG2`, `EMG` | raw EDF | µV |
| `delta_power` … `gamma_power` | analyzer | V²/Hz |
| `emg_rms` | analyzer | µV |
| `td_ratio` | analyzer | dimensionless |

## TTL Events

`recording.ttl_events()` → `dict[str, dict[str, np.ndarray]]` — parses the paired TSV annotations and extracts TTL pulse times.

- Keys are TTL names (e.g. `"TTL 3"`); values have `"rise"` and `"fall"` float64 arrays of times in **seconds from recording start**.
- Deduplicates cross-channel entries (TSV logs each event once per EEG/EMG channel at identical timestamps).
- Returns `{}` if no TSV is loaded or no TTL rows are found.
- `save_to_h5` automatically writes `/ttl_events/{TTL_N}/rise_times` and `fall_times` when TTL data is present.
- **TSV is fully optional** — if no `*_annotations.tsv` file exists alongside the EDF, the recording loads normally, `ttl_events()` returns `{}`, the TTL panel is hidden in Scope, and HDF5 export simply omits the `/ttl_events/` group. No errors are raised.

### TTL display in Scope

When a recording with TTL events is loaded, a **TTL EVENTS** panel appears in the sidebar with:
- **Show Strips** — semi-transparent amber spans (`axvspan`) from rise to next fall across all channel axes (default on)
- **Rising Edges** — dotted green vertical lines at each rising edge (default off)
- **Falling Edges** — dotted red vertical lines at each falling edge (default off)

Overlays are redrawn on every scroll/play tick and removed cleanly on each redraw.

## Current Stage

**Stage 1 complete** — `io.py` (incl. `save_to_h5`), `analysis.py`, `visualization.py`, `scope.py` (oscilloscope + video export) implemented and tested.

**Stage 2 complete** — `scoring/state.py` (`ScoringSession`, `AutoScoreThresholds`, `STATE_COLORS`), auto-scoring (Wake→NREM→REM thresholds), hypnogram strip in Scope, keyboard hotkeys (W/N/R/U, Ctrl/Cmd+Z/Y, Space, `[`/`]`, arrows, Ctrl/Cmd+O/E), CLASSIFICATION + LABELING sidebar panels, draggable threshold lines, `?` help dialog, save JSON / export CSV / save HDF5, reload scored session from JSON or HDF5 (`ScoringSession.from_h5`).

**Stage 3 in progress** — TTL event parsing (`recording.ttl_events()`), HDF5 export of TTL times, TTL overlay panel in Scope (strips, rising/falling edge markers).

See `agent.md` → Stage Breakdown for what is complete and what is next.

## Channel Names

Raw EDF channels are auto-renamed on load:
- `EEG EEG1A-B` → `EEG1`
- `EEG EEG2A-B` → `EEG2`
- `EMG EMG` → `EMG`

Always use the short canonical names (`EEG1`, `EEG2`, `EMG`) in code.
