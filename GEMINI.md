# GEMINI.md

## Project Overview

**sleep-tools** is a Python package designed for inspecting, visualizing, transforming, and scoring rodent sleep recordings. It processes EDF (European Data Format) electrophysiology files typically produced by the Luminose system. The package implements Julia's Spike2 sleep-scoring protocol in an open-source Python environment.

### Main Technologies
- **Python >= 3.11**
- **MNE-Python**: Used for EDF and TSV annotation loading.
- **NumPy & SciPy**: Core signal processing (filtering, STFT, RMS envelopes).
- **Pandas**: Data manipulation for annotations and features.
- **Matplotlib**: Static visualizations and video frame rendering.
- **h5py**: High-performance storage of curated datasets in HDF5 format.
- **PySide6**: Interactive Qt-based "Scope" viewer.
- **FFMpeg**: System-level dependency for video export.

---

## Key Components & Architecture

The project is organized into modular components within the `sleep_tools/` package:

- **`SleepRecording` (`io.py`)**: The primary data container. Loads EDF recordings (EEG1, EEG2, EMG) and paired TSV annotations. Provides metadata and signal statistics.
- **`SleepAnalyzer` (`analysis.py`)**: Implements the signal processing pipeline.
    - EEG high-pass filtering (Butterworth + drift subtraction).
    - EMG FIR band-pass filtering + causal exponential RMS envelope.
    - Band power calculation (Delta, Theta, Alpha, Beta, Gamma) via Hann-windowed STFT.
    - Theta-to-Delta (T:D) ratio calculation.
- **`ScoringSession` (`scoring/state.py`)**: Manages per-epoch sleep stage labels (Wake, NREM, REM, Unscored).
    - Implements Julia's Spike2 protocol with threshold-based auto-scoring.
    - Provides undo/redo functionality for manual stage assignment.
    - Calculates state statistics and durations.
    - Persists session state (labels + thresholds) to JSON.
- **`SleepVisualizer` (`visualization.py`)**: Generates static Matplotlib figures, including raw traces, spectrograms, and a 4-panel overview (Spectrogram + Delta/Theta + EMG RMS + T:D Ratio).
- **`Scope` (`scope.py`)**: An interactive, scrollable Qt viewer ("Oscilloscope") for real-time inspection of raw and derived signals.
    - Supports interactive sleep staging with a synchronized hypnogram strip.
    - Includes a threshold-based classification panel.
    - Handles rendering scrolling MP4 videos (requires system `ffmpeg`).
- **HDF5 Export (`io.py:save_to_h5`)**: Saves raw signals, computed epoch features, and sleep labels into a self-describing HDF5 schema.

---

## Setup and Development

### Environment Setup
The project uses `mamba` (or `conda`) for environment management.

```bash
# Create the environment
mamba env create -f environment.yml

# Activate the environment
mamba activate sleep-tools

# Install in editable mode
pip install -e .
```

### Key Commands
- **Install**: `pip install -e .`
- **Install with GUI/Dev extras**: `pip install -e ".[gui,dev]"`
- **Run Tests**: `pytest tests/ -v`
- **Run Examples**: `python examples/scope_viewer.py`

### Development Conventions
- **Type Hinting**: Mandatory for all new functions and classes.
- **Docstrings**: Follow the NumPy/SciPy style guide.
- **Signal Processing**: Always maintain consistency with the Spike2 protocol (e.g., 5s epochs, specific filter cutoffs).
- **Testing**: New features must be accompanied by tests in the `tests/` directory. Use the provided `example_data/` for integration tests.

---

## Interactive Scoring & Hotkeys

The `Scope` viewer supports a streamlined workflow for sleep staging:

1. **Auto-Score**: Use the "Classification" panel to set thresholds and run the initial pass.
2. **Review**: Scroll through the recording while observing the synchronized hypnogram.
3. **Manual Edit**: Select epochs on the hypnogram and assign stages using buttons or hotkeys.
4. **Save**: Export the session to `.json` or the final hypnogram to `.csv`.

### Hotkeys
| Key | Action |
| :--- | :--- |
| **W** | Assign **Wake** to selected epoch(s) |
| **N** | Assign **NREM** to selected epoch(s) |
| **R** | Assign **REM** to selected epoch(s) |
| **U** | Assign **Unscored** to selected epoch(s) |
| **C** | Centre all visible signals on their visible-window mean |
| **Ctrl + Z** | Undo last label change |
| **Ctrl + Y** | Redo last undone change |
| **Space** | Play / Pause playback |
| **← / →** | Fine scroll (10 % of visible window) |
| **[ / ]** or **PageUp / PageDown** | Previous / Next page |

### Panel Controls
- **Centre signals** — `⊕` in the transport bar (or `C`) centres all channels on their current-window mean. The `⊕` icon in each channel's header row centres that channel individually. **Optimize Scale** resets both scale and offset for that channel.
- **Resizable panels** — Drag the splitter handles between the left channel-control panel, the centre canvas, and the right sidebar. The **☰** sidebar toggle collapses/restores the sidebar remembering its last width.

---

## Signal Processing Details

The package mirrors a specific rodent sleep scoring protocol:

| Signal | Processing Step | Parameters |
| :--- | :--- | :--- |
| **EEG** | High-pass / Drift Removal | 2nd-order Butterworth LP @ 0.5Hz, subtract from original |
| **EMG** | Band-pass Filter | FIR 5–45 Hz (transition 1.8 Hz) |
| **EMG** | RMS Envelope | Causal exponential smoother (τ = 5 s) |
| **Bands** | Power Integration | Hann-windowed STFT (5s window, 50% overlap), trapezoidal integration |

### Default Frequency Bands
- **Delta**: 0.5 – 4.0 Hz
- **Theta**: 6.0 – 10.0 Hz
- **Alpha**: 8.0 – 13.0 Hz
- **Beta**: 13.0 – 30.0 Hz
- **Gamma**: 30.0 – 100.0 Hz

### Auto-Scoring Thresholds (Default)
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| **Delta (Wake)** | < 1200 µV²/Hz | Delta power below this is candidate Wake |
| **Delta (NREM)** | > 1000 µV²/Hz | Delta power above this is candidate NREM |
| **EMG (Wake)** | > 3 µV | EMG above this confirms Wake |
| **EMG (NREM)** | < 5 µV | EMG below this confirms NREM |
| **EMG (REM)** | < 3 µV | EMG below this confirms REM (atonia) |
| **T:D Ratio (REM)** | > 4.0 | Theta:Delta ratio above this is candidate REM |

---

## Data Formats

- **EDF (`*_export.edf`)**: Raw electrophysiology data (usually 400 Hz).
- **TSV (`*_annotations.tsv`)**: Luminose-format sync triggers and event markers.
- **JSON (`*_sleep_scores.json`)**: Scoring session state, including per-epoch labels and the thresholds used.
- **CSV (`*_hypnogram.csv`)**: Exported hypnogram (columns: `epoch_index`, `time_s`, `label`).
- **HDF5 (`*.h5`)**: Curated datasets containing:
    - `/signals/`: Raw time-series data.
    - `/epochs/`: Features and sleep labels (U=Unscored, W=Wake, N=NREM, R=REM).
    - Extensive root and group attributes for self-documentation.

---

## Examples

Check the `examples/` directory for common workflows:
- `quick_start.py`: Basic loading, analysis, and static visualization.
- `classify_sleep.py`: Programmatic auto-scoring and session saving.
- `explore_features.py`: Numeric summary of computed sleep features.
- `save_dataset.py`: Demonstrates HDF5 export patterns.
- `scope_viewer.py`: Launching the interactive Qt window with scoring tools.
