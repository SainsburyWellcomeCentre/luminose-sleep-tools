# Sleep Tools — Agent Context & Implementation Plan

## Project Overview

`sleep-tools` is a Python package for inspecting, visualizing, transforming, and scoring rodent sleep recordings acquired in harris sleep lab. It ingests EDF electrophysiology files and paired TSV annotation files produced by the Luminose recording system, and provides:

- **Stage 1**: Signal analysis and visualization (raw traces, frequency bands, spectrograms)
- **Stage 2**: Interactive GUI for manual sleep staging
- **Stage 3**: Sync-trigger alignment with a behavioral rig (e.g. Bpod)

The workflow mirrors Julia's Spike2 sleep-scoring protocol (see `Julia's Sleep Score protocol for Spike2.md`) but runs as a Python package.

See `how_to_score.md` for a detailed guide covering exact signal processing math, auto-scoring algorithm, and step-by-step scoring workflow — written for sleep neuroscientists.

---

## Data Format

### EDF files
- Channels: `EEG1`, `EEG2`, `EMG`
- Naming: `<AnimalID>_<datetime>_export.edf`
- Loaded via `mne` (installed in conda env `sleep-tools`)

### TSV annotation files
- Named: `<AnimalID>_<datetime>_annotations.tsv`
- Header rows (rows 1–5): `Experiment ID`, `Animal ID`, `Researcher`, `Directory path`, blank
- Data columns: `Number`, `Start Time`, `End Time`, `Time From Start`, `Channel`, `Annotation`
- Annotation values: `Started Recording`, `TTL 3: Rise`, `TTL 3: Fall`
- `Time From Start` is in seconds from recording start
- TTL pulses appear simultaneously on all three channels (EEG1, EEG2, EMG) — deduplicate by taking one channel or unique timestamps

---

## Sleep Staging Criteria (Julia's Protocol)

| State | EEG | EMG | Delta Power | T:D Ratio |
|-------|-----|-----|-------------|-----------|
| Wake  | Low amplitude, high frequency | High | ~0 | Moderate |
| NREM  | High amplitude, low frequency | Low | Very high | Low |
| REM   | Low amplitude, high frequency (like Wake) | Near flat (muscle atonia) | Low | High (peak) |

**Scoring sequence rules:**
- REM always follows NREM
- REM always ends with Wake (not directly back to NREM — but this policy should be configurable)
- Wake cannot transition directly to REM (must go through NREM first)
- EMG "flinches" during REM do not count as Wake

**Scoring order:** Wake classified first → NREM → REM (REM hardest to catch automatically)

---

## Signal Processing Pipeline (mirroring Spike2 protocol)

### EEG Processing
1. Compute DC/drift channel: IIR low-pass Butterworth 2nd order, cutoff 0.5 Hz → `EEGlow`
2. Subtract drift: `EEGfilt = EEGorig − EEGlow` (effectively a high-pass at 0.5 Hz)

### EMG Processing
1. FIR bandpass: 5–45 Hz, transition gap 1.8 Hz → `EMGfilt`
2. Compute RMS envelope with time constant 5 s

### Power Features
- Window: 5 s epoch (configurable); Hann window; 50% overlap
- STFT via `scipy.signal.spectrogram(scaling="density")`; band power integrated with `np.trapezoid`
- **Delta**: 0.5–4.0 Hz — primary NREM marker
- **Theta**: 6.0–10.0 Hz — elevated in REM; numerator of T:D ratio
- **Alpha**: 8.0–13.0 Hz — overlaps theta in rodents
- **Beta**: 13.0–30.0 Hz — elevated in Wake/arousal
- **Gamma**: 30.0–100.0 Hz — high-frequency arousal
- **EMG RMS**: FIR bandpass 5–45 Hz → exponential smoother (τ = 5 s); high in Wake, flat in REM
- **T:D ratio**: theta_power / (delta_power + 1e-12) — peak in REM, low in NREM

`FEATURE_INFO` (exported from `sleep_tools`) documents frequency ranges, units, and sleep relevance for all 7 feature columns.

### Automatic Scoring Thresholds
- Wake: `EMG_rms > emg_thresh` AND `delta < delta_thresh`
- NREM: `EMG_rms < emg_thresh` AND `delta > delta_thresh`
- REM: `EMG_rms < emg_thresh` AND `td_ratio > td_thresh`
- Thresholds set interactively by user

---

## Package Architecture

```
sleep_tools/
├── __init__.py
├── io.py               # EDF + TSV loading; SleepRecording dataclass
├── analysis.py         # Signal processing: filtering, band power, spectrogram
├── visualization.py    # Matplotlib-based static plots (no GUI)
├── scope.py            # Scope: PySide6 oscilloscope viewer + MP4 video export
├── scoring/
│   ├── __init__.py
│   ├── gui.py          # Main Qt GUI window
│   ├── panels.py       # Reusable Qt widgets (trace panel, power panel, stage bar)
│   └── state.py        # ScoringSession: epoch labels, undo stack, save/load
└── sync.py             # TTL event extraction, edge detection, Bpod alignment
```

### Key Classes

#### `SleepRecording` (io.py)
- Attributes: `raw` (mne.io.Raw), `annotations` (DataFrame), `animal_id`, `experiment_id`, `start_datetime`
- `classmethod from_edf(edf_path, tsv_path=None)` — auto-discovers paired TSV if not given
- `@property channels` — returns list of channel names
- `@property duration` — total recording duration in seconds
- `@property sfreq` — sampling frequency in Hz
- `metadata()` → `dict` — animal_id, experiment_id, start_datetime, channels, sfreq, duration_s, n_samples, has_annotations, n_annotations
- `signal_info()` → `dict[str, dict]` — per-channel stats: unit ("V"), n_samples, min, max, mean, std

#### `save_to_h5` (io.py)
Module-level function; also exported from `sleep_tools`.

`save_to_h5(recording, path, *, analyzer=None, labels=None, epoch_len=5.0, include_raw_signals=True, overwrite=False) → Path`

HDF5 layout:
```
/                           ← root attrs: animal_id, sfreq, epoch_len, saved_at,
                              sleep_tools_version, band_definitions (JSON)
/signals/{EEG1,EEG2,EMG}    ← raw arrays, float32, gzip-compressed (optional)
                              dataset attrs: unit="V", sfreq
/epochs/times               ← epoch-centre times, float64; attrs: units="s", description
/epochs/labels              ← per-epoch stage labels, variable-length UTF-8 ("U" = unscored)
                              attrs: description (label key)
/epochs/features/           ← group attr: band_definitions (JSON)
    delta_power             ← 7 fixed float64 columns, NaN when not computed
    theta_power               each dataset attrs: description, units,
    alpha_power               frequency_range_hz (float array or "derived"),
    beta_power                sleep_relevance
    gamma_power
    emg_rms
    td_ratio
/annotations/               ← TSV columns (if annotations loaded)
```

Key design: all 7 feature datasets are **always present** even when `analyzer=None`, so downstream code can rely on a fixed schema. Missing features → NaN arrays, not absent keys. All datasets and groups carry self-describing attrs so the file is readable without the package.

#### `SleepAnalyzer` (analysis.py)
- `__init__(recording: SleepRecording, epoch_len=5.0, eeg_channel: str | None = None)`
  - `eeg_channel`: `"EEG1"`, `"EEG2"`, or `None` (default → auto-average both channels when both present)
- `filter_eeg(channel=None, hp_cutoff=0.5)` → filtered signal array; when `channel=None` uses instance `eeg_channel`
- `filter_emg(channel='EMG', bp_low=5, bp_high=45)` → filtered EMG array
- `emg_rms(signal, time_constant=5.0)` → RMS envelope
- `band_power(signal, fs, band, window=5.0, overlap=0.5)` → time-series of band power
- `td_ratio(delta_power, theta_power)` → element-wise ratio
- `spectrogram(signal, fs, freq_max=50, window=5.0, overlap=0.5)` → (times, freqs, Sxx)
- `compute_all_features(eeg_channel=None)` → dict of all band powers, EMG RMS, T:D ratio (cached per channel+window+overlap)

#### `Scope` (scope.py)
- `__init__(recording: SleepRecording | None = None, analyzer: SleepAnalyzer | None = None)`
- `show(signals=None, x_window=30.0)` — opens PySide6 oscilloscope window (blocking)
  - Qt backend (`QtAgg`) set at module import
  - **Action Bar (Left)**: Gear icon toggles the control panel to maximize plot area.
  - **File Menu**: Load recordings or folders directly from the GUI.
  - **Amplitude Controls**:
    - **Optimize View**: Auto-scale visible signals to fit current window.
    - **Auto**: Continuously optimize amplitude during playback.
    - **Reset All**: Scale to full-recording 99th-percentile.
    - Per-channel spinboxes for manual y-scale.
  - **Time Unit Selector**: `auto`, `s`, `m`, `h`.
  - Stacked axes, shared x, horizontal scrollbar.
  - **Fix**: Masking logic includes boundary points to ensure continuous traces.
  - **Fix**: Replaced "Sans-serif" with a robust font stack to avoid Qt font population overhead.
  - **Y-axis labels**: Each channel axis displays its current unit (e.g. `µV`, `µV²/Hz`); label updates automatically when the unit selector changes.
  - **Channel visibility button**: Per-channel `−` hide button, right-aligned in channel label row (label has `stretch=1`). **Do NOT use `setFixedSize` on this button** — the global stylesheet sets `padding:4px 8px` for QToolButton; a 20px fixed width leaves only 4px of text space, clipping the "−" completely. Instead, omit `setFixedSize` and let the button size naturally, overriding only `color`, `font-size`, and `padding` in the widget-level stylesheet with the `QToolButton { ... }` selector.
- `make_video(output_path=None, *, signals, t_start, t_end, x_window, y_lims, fps, speed, figsize, dpi, session=None, show_hypnogram=True)` → `Path`
  - Default output: `output/<animal_id>_scope.mp4` in cwd
  - Uses `matplotlib.animation.FuncAnimation` + `FFMpegWriter`
  - `speed` = recording-seconds per video-second
  - `y_lims` dict overrides auto-scale per signal (display units)
  - `session`: `ScoringSession` — when provided with `show_hypnogram=True`, renders a colour-coded hypnogram strip (W/N/R/U) beneath signal traces with a vertical playhead tracking scroll position
  - Raises `RuntimeError` with install hint if ffmpeg missing

#### `SleepVisualizer` (visualization.py)
- `__init__(recording, analyzer)`
- `plot_raw_traces(t_start, t_end, channels=None)` → matplotlib Figure
- `plot_band_timeseries(channel='EEG1', bands=None)` → Figure with subplots per band
- `plot_spectrogram(channel='EEG1', freq_max=50)` → Figure
- `plot_scoring_overview(session)` → hypnogram + features + spectrogram aligned on time axis

#### `ScoringSession` (scoring/state.py)
- `__init__(recording, epoch_len=5.0, states=None)`
- Default states: `{'W': 'Wake', 'N': 'NREM', 'R': 'REM', 'U': 'Unscored'}`
- `states` is user-extensible dict
- `labels`: numpy array of epoch state labels, initialized to 'U'
- `label_range(t_start, t_end, state)` — assign state to epoch range
- `auto_score(thresholds)` — apply threshold-based auto-scoring
- `undo()` / `redo()` — command stack
- `save(path)` / `load(path)` — serialize to JSON or CSV
- `from_h5(path, recording)` — classmethod; reload epoch labels and thresholds from a previously saved HDF5 file

#### `SleepScorerGUI` (scoring/gui.py)
- Built with PyQt5/PySide6 (prefer PySide6 for licensing)
- Main window layout:
  - Top panel: raw EEG1 + EEG2 + EMG traces (scrollable by time)
  - Middle panel: delta, theta power + EMG RMS + T:D ratio time series
  - Bottom panel: spectrogram (EEG1 by default)
  - Right panel: hypnogram (vertical time axis aligned to traces)
  - Control sidebar:
    - Epoch length slider (1–30 s)
    - Smoothing time constant slider (1–10 s)
    - Per-state threshold sliders: `emg_thresh`, `delta_thresh`, `td_thresh`
    - Channel selector (EEG1 / EEG2)
    - EEG frequency range sliders
    - Auto-score button
    - State buttons (hotkeys: W, N, R; + user-defined)
    - Undo/Redo buttons
    - Save / Load session buttons
- Interaction: click or drag on hypnogram to select epoch range → press state hotkey to label
- Current epoch highlighted across all panels

#### `SyncAligner` (sync.py)
- `__init__(recording)`
- `extract_ttl_events(edge='both')` → DataFrame of {`time_from_start`, `edge`, `channel`}
- `deduplicate_channels()` — collapse multi-channel duplicate entries to unique timestamps
- `detect_pulses()` → list of (rise_time, fall_time, duration) tuples
- `align_to_bpod(bpod_timestamps)` → time-offset correction (stub for Stage 3)
- `plot_events(ax=None)` → overlay TTL events on a matplotlib axis

---

## Environment & Dependencies

- **Conda env**: `sleep-tools` (tested with mamba, but should be able to work with either conda or .venv)
- **Core**: `mne`, `numpy`, `scipy`, `pandas`, `matplotlib`, `h5py`
- **GUI**: `PySide6`
- **EDF I/O**: `mne` handles EDF natively via `mne.io.read_raw_edf` (preferred over `pyedf`)
- **Optional**: `pyEDFlib` if raw byte-level EDF access needed

## Development Conventions

- Python ≥ 3.10
- Package installable via `pip install -e .` using `pyproject.toml`
- Type hints throughout
- Tests in `tests/` using `pytest`; use real example data for integration tests
- No hard-coded paths; all file discovery is relative or passed explicitly
- Epoch length and all processing parameters are always runtime-configurable, never hard-coded

---

## Stage Breakdown

### Stage 1 — Analysis & Visualization (no GUI) ✅ COMPLETE
**Goal**: Python objects that can be used in notebooks or scripts to explore a recording.

Tasks:
1. `io.py`: load EDF with mne, parse TSV header/data, expose clean API ✅
2. `analysis.py`: EEG/EMG filters, band power (STFT), spectrogram, T:D ratio ✅
3. `visualization.py`: raw traces, per-band time series, spectrogram, 4-panel overview ✅
4. `io.py` — `save_to_h5()`: curated HDF5 export with fixed schema, NaN for missing features ✅
5. `scope.py` — `Scope.show()`: interactive PySide6 oscilloscope viewer ✅
6. `scope.py` — `Scope.make_video()`: scrolling MP4 export via ffmpeg ✅

Notes:
- EDF channel names (`EEG EEG1A-B`, `EEG EEG2A-B`, `EMG EMG`) are auto-renamed to `EEG1`, `EEG2`, `EMG`
- TSV auto-discovered from EDF filename (replaces `_export.edf` → `_annotations.tsv`)
- `SleepAnalyzer.compute_all_features()` caches result; call `invalidate_cache()` to reset
- 22/22 integration tests passing against `LUMI-0013_2026-03-24_14_15_34_export.edf`
- Example scripts: `examples/quick_start.py`, `examples/explore_features.py`, `examples/save_dataset.py`
- Dark-theme design tokens in `scope.py`: `_BG`, `_PANEL`, `_BORDER`, `_ACCENT`, `_TEXT`, `_GRID`, `_SIG_COLORS`

Acceptance: can run `SleepRecording.from_edf(path)`, call analysis methods, produce figures, export HDF5, open oscilloscope, and render a video. ✅

### Stage 2 — Interactive GUI Scorer ✅ COMPLETE
**Goal**: A standalone Qt window for manual and semi-automated sleep staging.

Tasks:
1. `scoring/state.py`: epoch label array, undo/redo (100-step stack), save JSON, load JSON, to_csv ✅
2. `scoring/__init__.py`: exports ScoringSession, AutoScoreThresholds, STATE_COLORS ✅
3. Auto-score via `ScoringSession.auto_score(features, thresholds)` (Wake→NREM→REM) ✅
4. Hypnogram strip in Scope figure (pcolormesh, color-coded W/N/R/U) ✅
5. Click to select epoch, Shift+click to extend selection range ✅
6. W/N/R/U keyboard hotkeys for state assignment; Ctrl/Cmd+Z/Y for undo/redo ✅
7. CLASSIFICATION sidebar panel: 6 threshold spinboxes + **epoch length field** + **Reset Defaults** + Run Classification button ✅
8. LABELING sidebar panel: state counts, manual W/N/R/U buttons, undo/redo, save/export ✅
9. Save session JSON, export hypnogram CSV, save HDF5 (with thresholds in /epochs/thresholds) ✅
10. Load session from JSON (`ScoringSession.load`) ✅
10b. Load session from HDF5 (`ScoringSession.from_h5`): reads labels + thresholds from `/epochs/labels` and `/epochs/thresholds`; "Load Session from H5..." button in CLASSIFICATION panel (available before running classification) ✅
11. `save_to_h5` updated: `session` parameter; labels + epoch_len taken from session; thresholds stored as HDF5 attributes ✅
12. Draggable threshold reference lines: dotted lines on δ-power/EMG RMS/T:D axes; drag to update spinbox + session.thresholds live; spinbox edits also move lines ✅
13. Full keyboard navigation: Space (play/pause), `[`/`]` + PageUp/Down (page), `←`/`→` (fine scroll), mouse-wheel (proportional scroll), Ctrl/Cmd+O (open folder), Ctrl/Cmd+E (open file) ✅
14. `?` help button (transport bar, leftmost): scrollable step-by-step dialog; modifier key names are platform-aware (Cmd on macOS, Ctrl elsewhere) ✅
15. Fusion Qt style set at app init: ensures spinbox/combobox arrows use palette colors (white in dark theme, black in light theme) ✅
16. **Centre signals**: `⊕` transport button + `C` key centres all visible channels on their visible-window mean (stored as per-channel y-offset in `_y_offsets`); per-channel `⊕` icon in channel header row centres that channel only; **Optimize Scale** resets offset to 0 ✅
17. **Resizable panels via QSplitter**: left channel panel, centre canvas, and right sidebar are separated by draggable `QSplitter` handles (`_inner_splitter`, `_outer_splitter`); `time_row` uses a `_time_spacer` widget whose width tracks the inner splitter so the x-axis label stays canvas-centred; `_toggle_sidebar` uses `setSizes` to save/restore sidebar width ✅
18. **↕ Optimize All**: transport bar button (right of `⊕`); calls `_optimize_amplitude(local=True)` to auto-scale all visible traces to the current window in one click ✅
19. **Epoch length field** (`_epoch_len_spin`): `QDoubleSpinBox` in CLASSIFICATION panel, range 0.5–60.0 s, default 5.0 s; applied to `analyzer.epoch_len` before `compute_all_features()` on Run Classification ✅
20. **Reset Defaults** button: in CLASSIFICATION panel; calls `_on_reset_thr_defaults()` to restore all 6 threshold spinboxes and epoch length to `AutoScoreThresholds()` factory values ✅

Key classes:
- `ScoringSession(recording, epoch_len=5.0)` — exported from `sleep_tools`
- `AutoScoreThresholds` — default: delta_wake=1200 µV²/Hz, delta_nrem=1000 µV²/Hz, emg_wake=3 µV, emg_nrem=5 µV, emg_rem=3 µV, td_rem=4
- `STATE_COLORS` — dict mapping W/N/R/U to hex color strings

Acceptance: open a recording → Analyze Signals → Run Classification → adjust thresholds → re-run → manually fix epochs → Save HDF5. ✅

Example script: `examples/classify_sleep.py`

### Stage 3 — Sync Trigger Alignment ✅ COMPLETE
**Goal**: Extract TTL events from TSV, identify rise/fall edges, align to Bpod trial timestamps.

Tasks:
1. `sync.py`: parse TTL events, deduplicate channels, detect pulse trains ✅
2. Bpod alignment stub: accept Bpod timestamp array, compute offset, return aligned event times ✅
3. Visualization: overlay TTL events on trace panels ✅

Acceptance: given a TSV file, correctly extract all Rise/Fall times; alignment stub runs without error.

Notes:
- `SyncAligner(recording)` is exported from `sleep_tools`.
- `extract_ttl_events(edge="both")` preserves per-channel TTL rows.
- `deduplicate_channels()` collapses simultaneous EEG1/EEG2/EMG TTL rows.
- `detect_pulses()` pairs rise/fall edges into `(rise_time, fall_time, duration)`.
- `align_to_bpod()` computes a constant median offset between TTL rises and Bpod timestamps.
- `plot_events()` overlays deduplicated TTL edges on any Matplotlib axis.

---

## Spike2 Compatibility & HDF5 Consistency Plan

This section captures the investigation from
`example_data/comparison_with_spike2_to_debug_inconsistencies/` so a future
session can continue without re-deriving the findings.

### Comparison Findings

Files inspected:
- Spike2 export: `sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_0to300s-using_spike2.mat`
- sleep-tools export: `sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47_using_sleeptools.h5`
- Raw EDF: `sub-003_ses-001_recording-001_0001_2026-03-25_16_13_47.edf`

Observed facts:
- Raw signals match exactly after unit conversion. Spike2 MAT stores raw data in `uV`; sleep-tools/MNE stores raw data in `V`. Multiplying H5/EDF values by `1e6` gives correlation approximately 1.0 for EEG1, EEG2, and EMG.
- Spike2 filtered EEG in the MAT file uses `EEG2`, not EEG1 and not an EEG1+EEG2 average. Numerically, `EEGFilt == EEG2 - EEGLow`.
- Spike2 `EEGLow` matches a causal 2nd-order Butterworth low-pass (`scipy.signal.lfilter`) at 0.5 Hz. Current sleep-tools uses zero-phase `signal.sosfiltfilt` in `SleepAnalyzer.filter_eeg`, so the filtered signal shape differs even when using the same channel.
- Spike2 power traces are not the same algorithm as current sleep-tools STFT power. The MAT comments show `Power in band: 0.0 - 4.0 Hz; tc: 5.0s` for delta and `6.0 - 10.0 Hz; tc: 5.0s` for theta, with an output interval of about 0.1 s. Current sleep-tools uses Hann-windowed STFT band power with default 5 s windows and 50% overlap.
- Spike2 `EMGFilt` in the MAT file is positive and envelope-like, roughly `20-58 uV`, so compare it to sleep-tools `emg_rms`, not to the centered FIR band-passed waveform returned by `filter_emg`.
- The inspected H5 is internally inconsistent: root `epoch_len = 10.0`, `/epochs/times` and `/epochs/labels` length `17547`, but every `/epochs/features/*` dataset has length `70187`. This happens because `save_to_h5(..., analyzer=..., session=...)` computes analyzer features first, then replaces `times` with `session.times` without resampling or rebuilding the feature arrays.

### Design Principle

Do not silently change current outputs. Add explicit, runtime-configurable
analysis profiles:

- `standard`: current sleep-tools behavior, backward-compatible.
- `spike2`: compatibility mode matching Julia's Spike2 workflow as closely as practical.

All parameters must remain runtime-configurable, per `CLAUDE.md`: epoch length,
filter cutoffs, filter phase, EEG channel, band definitions, smoothing time
constants, output interval, and thresholds.

### Implementation Order

1. **Fix HDF5 feature/epoch alignment first** ✅
   - Add a helper in `io.py`, e.g. `_align_features_to_epoch_times(features, epoch_times)`.
   - When `session is not None`, always write `/epochs/features/*` with the same length as `session.times`.
   - If analyzer feature times differ from session times, interpolate scalar feature arrays onto `session.times`.
   - Consider preserving the original analyzer timebase under `/analysis/times` and `/analysis/features/{feature}`.
   - Add root/group attrs such as `analysis_profile`, `eeg_channel`, `feature_timebase`, and `feature_source` (`native_epoch`, `interpolated_from_analysis`, etc.).

2. **Fix analyzer cache correctness** ✅
   - Current `compute_all_features()` cache keys only track overlap, window, and EEG channel.
   - Cache identity must include profile, EEG filter config, EMG filter config, band-power method, band definitions, EMG RMS time constant, epoch/window/output interval, and selected channel.
   - Prefer a frozen config object over loose cache fields.
   - Ensure GUI changes to `analyzer.epoch_len` cannot reuse stale feature arrays.

3. **Add explicit analysis profiles** ✅
   - Suggested API:
     ```python
     ana = SleepAnalyzer(rec, epoch_len=5.0, eeg_channel="EEG2", profile="spike2")
     ```
   - Suggested config concepts:
     ```python
     AnalysisProfile = Literal["standard", "spike2"]

     @dataclass(frozen=True)
     class EEGFilterConfig:
         cutoff_hz: float = 0.5
         order: int = 2
         phase: Literal["zero", "causal"] = "zero"

     @dataclass(frozen=True)
     class BandPowerConfig:
         method: Literal["stft", "spike2_smooth"] = "stft"
         output_interval: float | None = None
         smoothing_time_constant: float = 5.0
         window: float = 5.0
         overlap: float = 0.5
     ```

4. **Implement Spike2-compatible EEG filtering** ✅
   - Keep current zero-phase filtering as the `standard` default.
   - Add causal filtering with `signal.lfilter` for `profile="spike2"`.
   - Spike2-compatible defaults: 2nd-order Butterworth low-pass, cutoff 0.5 Hz, causal phase.
   - Make EEG channel selection explicit. Spike2 protocol says to choose the best EEG channel; avoid hidden averaging in Spike2 mode.

5. **Clarify EMG waveform vs envelope**
   - Keep `filter_emg()` as centered FIR band-passed waveform.
   - Keep/use `emg_rms()` as the positive envelope for scoring.
   - In GUI and docs, label these separately as `EMG filtered` and `EMG RMS`/`EMG envelope`.
   - In Spike2 validation, compare Spike2 `EMGFilt` to sleep-tools `emg_rms`.

6. **Implement Spike2-like band power** ✅ approximation implemented; validation still pending
   - Keep existing STFT implementation as `method="stft"`.
   - Add a Spike2-like method, e.g. `band_power_spike2_like()`.
   - Defaults for Spike2 compatibility:
     - delta band `0.0-4.0 Hz` to match MAT comments
     - theta band `6.0-10.0 Hz`
     - smoothing time constant `5.0 s`
     - output interval `0.1 s`
     - OSD4-derived defaults: EEG target sample rate `512 Hz`, Hann FFT size `256`, chunked FFT implementation to avoid dense full-night spectrogram allocation
   - Validate against the MAT export before claiming exact compatibility. If Spike2's proprietary "Power in band" details cannot be matched exactly, document this as a Spike2-compatible approximation with measured error bounds.

8. **Inspect OSD4 Spike2 script** ✅
   - `OSD.s2cx` is a sampling configuration: EEG channels at ~1024 Hz, EMG channels at ~5 kHz, output sequencer `GH_OSD3.pls`.
   - `OSD4.s2s` is the analysis script: EMG uses Spike2 RMS process with `5 s (= +/-5 s)` default; EEG is interpolated to ~512 Hz before band power; band power virtual channels use `Pw(eegch, stepsz, low, high)` sampled every `0.1 s`, then Smooth channel process with default `tc=5 s`.
   - sleep-tools `profile="spike2"` now uses centred RMS for EMG and an OSD4 `Pw(...)`-style chunked Hann FFT approximation rather than the earlier causal bandpass-square approximation.

7. **Update Scope GUI** ✅
   - Add an Analysis Profile control: `Standard` vs `Spike2-compatible`.
   - Make EEG channel selection prominent in Spike2 mode.
   - Consider disabling `Average (EEG1+EEG2)` in Spike2 mode unless explicitly selected.
   - When saving H5, include selected profile/channel/config attrs.
   - Ensure classification uses features aligned to the scoring session epoch length/timebase.

### Tests To Add

Core unit/integration tests:
- `test_filter_eeg_standard_zero_phase_matches_sosfiltfilt` ✅
- `test_filter_eeg_spike2_causal_matches_lfilter` ✅
- `test_filter_eeg_spike2_uses_selected_channel_not_average` ✅ via explicit-channel requirement
- `test_compute_all_features_cache_changes_with_profile` ✅
- `test_compute_all_features_cache_changes_with_epoch_len` ✅
- `test_save_to_h5_session_features_match_epoch_count` ✅
- `test_save_to_h5_preserves_analysis_timeseries_when_feature_timebase_differs` ✅
- `test_scoring_session_from_h5_roundtrip_after_feature_alignment` ✅

Optional slow validation tests using the comparison folder:
- Raw MAT vs EDF/H5 signal agreement after `1e6` unit conversion.
- Spike2 `EEGLow` vs causal low-pass of EEG2.
- Spike2 `EEGFilt` vs `EEG2 - causal_lowpass(EEG2)`.
- Spike2 `EMGFilt` vs sleep-tools `emg_rms`.
- Spike2 delta/theta/T:D vs Spike2-profile sleep-tools outputs on the 0.1 s grid.
- H5 consistency: every `/epochs/features/*` length equals `/epochs/times` length, labels length equals epoch count, and profile/channel attrs exist.

Mark large-data comparison tests as slow or optional so normal CI does not
depend on the large local EDF/MAT/H5 files.

### Validation Utility

Add a repeatable validation script:

```text
scripts/validate_spike2_comparison.py
```

Suggested inputs:
- EDF path
- Spike2 MAT path
- sleep-tools H5 path or output path

Suggested outputs:
- Printed metrics table
- Optional CSV metrics
- Optional PDF overlay plot

Suggested metrics:
- Raw signal unit conversion slope/correlation
- EEG low-pass agreement
- EEG filtered agreement
- EMG RMS/envelope agreement
- Delta/theta/T:D agreement
- H5 schema and shape consistency

### Documentation Updates

After implementation, update:

- `CLAUDE.md`
  - Add analysis profiles.
  - Correct the current claim that zero-phase filtering exactly matches Spike2.
  - Document H5 feature alignment rules.

- `agent.md`
  - Keep this plan updated as tasks are completed.
  - Move completed items into the stage breakdown when done.

- `how_to_score.md`
  - Explain `standard` vs `spike2` modes.
  - Clarify EEG channel selection.
  - Clarify EMG filtered waveform vs EMG RMS/envelope.
  - Document units: raw stored in `V`, display in `uV`, display power in `uV^2`.

- `README.md`
  - Add a short Spike2-compatible example:
    ```python
    ana = SleepAnalyzer(rec, eeg_channel="EEG2", profile="spike2")
    ```
  - Add an H5 schema note for `/epochs/features` vs optional `/analysis/features`.

Status: `CLAUDE.md`, `how_to_score.md`, `README.md`, and `agent.md` updated for analysis profiles, H5 feature alignment, and Stage 3 sync alignment. Dedicated Spike2 MAT validation utility remains pending.

**Post-session fix (2026-04-27):** Scope `_on_open_folder` and `_load` now glob `*.edf` instead of `*_export.edf`, so any EDF file (including the comparison `large_file_*.edf`) appears in the folder list and loads correctly. TSV auto-discovery in `io.py` gained a generic stem-based fallback so non-Luminose EDFs can also pick up paired TSVs if present. The comparison EDF (no paired TSV) loads cleanly with `ttl_events()` returning `{}`.

### Recommended Completion Order

1. H5 feature/epoch alignment.
2. Analyzer cache-key correctness.
3. Analysis profile plumbing.
4. Causal Spike2 EEG filtering.
5. Spike2-like band power.
6. Stage 3 sync alignment.
7. GUI controls for profile/channel selection.
8. Optional comparison validation script.
9. README, `how_to_score.md`, `CLAUDE.md`, and `agent.md` updates.

Note: on this workspace's case-insensitive filesystem, `AGENT.md` and
`agent.md` resolve to the same path. Keep `agent.md` as the canonical file.

---

## File Naming Conventions

| File type | Pattern |
|-----------|---------|
| EDF recording | `<AnimalID>_<YYYY-MM-DD>_<HH_MM_SS>_export.edf` |
| TSV annotations | `<AnimalID>_<YYYY-MM-DD>_<HH_MM_SS>_annotations.tsv` |
| Curated HDF5 dataset | `<AnimalID>_<YYYY-MM-DD>_<HH_MM_SS>_dataset.h5` |
| Scoring session | `<AnimalID>_<YYYY-MM-DD>_<HH_MM_SS>_sleep_scores.json` |
| Exported hypnogram | `<AnimalID>_<YYYY-MM-DD>_<HH_MM_SS>_hypnogram.csv` |

---

## Open Questions / Design Decisions

1. **Single vs. dual EEG channel**: ✅ Resolved — `SleepAnalyzer(eeg_channel=None/"EEG1"/"EEG2")`; GUI dropdown in the RECORDING panel selects channel before "Analyze Signals"; default is average of both.
2. **Epoch alignment**: epochs aligned to session start barcodes from bpod, followed by the first TTL event.
3. **REM→NREM policy**: configurable (some labs allow REM→NREM directly, others do not) - TODO: find proof
4. **Auto-score conflict resolution**: when EMG and delta thresholds give contradictory results, flag as "Ambiguous"
5. **Bpod timestamp format**: unknown until Stage 3 data is available — design `align_to_bpod` as a plugin interface
6. **Unclassified target**: protocol targets ~50% unclassified, ~10% ambiguous in first pass
