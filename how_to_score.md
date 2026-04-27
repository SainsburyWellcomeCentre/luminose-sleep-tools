# How to Score Rodent Sleep with sleep-tools

This guide is for harris lab focks who want to understand exactly how the package processes signals and how to score a recording from start to finish.

---

## Table of Contents

1. [Overview](#overview)
2. [Derived Signals — What the Analyzer Computes](#derived-signals)
   - [EEG (high-pass filtered)](#eeg-high-pass-filtered)
   - [EMG (FIR band-pass filtered)](#emg-fir-band-pass-filtered)
   - [EMG RMS envelope](#emg-rms-envelope)
   - [Band powers (delta, theta, alpha, beta, gamma)](#band-powers)
   - [Theta-to-Delta ratio (T:D)](#theta-to-delta-ratio)
3. [Sleep State Criteria](#sleep-state-criteria)
4. [Auto-Scoring Algorithm](#auto-scoring-algorithm)
5. [Step-by-Step Scoring Workflow](#step-by-step-scoring-workflow)
6. [Threshold Reference](#threshold-reference)
7. [Saving and Exporting](#saving-and-exporting)

---

## Overview

`sleep-tools` reimplements Julia's Spike2 sleep-scoring protocol in Python. The workflow is:

1. **Load** an EDF recording (EEG1, EEG2, EMG channels).
2. **Compute features**: filtered EEG, filtered EMG + RMS envelope, five band power time series, T:D ratio.
3. **Auto-score** all epochs using amplitude thresholds (Wake → NREM → REM).
4. **Inspect and correct** epochs interactively in the Scope viewer.
5. **Save** the session to JSON and/or export a hypnogram CSV.

All parameters (epoch length, EEG channel, filter cutoffs, thresholds) are configurable at runtime — nothing is hard-coded.

---

## Derived Signals

`SleepAnalyzer.compute_all_features()` returns a dict with these keys:

| Key | Shape | Units | Description |
|-----|-------|-------|-------------|
| `times` | `(n_epochs,)` | s | Centre time of each feature window (0.1 s grid) |
| `eeg_filtered` | `(n_samples,)` | V | High-passed EEG (full sample-rate) |
| `emg_filtered` | `(n_samples,)` | V | FIR band-passed EMG (full sample-rate) |
| `delta_power` | `(n_epochs,)` | V²/Hz·Hz | Delta band power per epoch |
| `theta_power` | `(n_epochs,)` | V²/Hz·Hz | Theta band power per epoch |
| `alpha_power` | `(n_epochs,)` | V²/Hz·Hz | Alpha band power per epoch |
| `beta_power` | `(n_epochs,)` | V²/Hz·Hz | Beta band power per epoch |
| `gamma_power` | `(n_epochs,)` | V²/Hz·Hz | Gamma band power per epoch |
| `emg_rms` | `(n_epochs,)` | V | EMG RMS envelope, resampled to epoch centres |
| `td_ratio` | `(n_epochs,)` | dimensionless | theta_power / delta_power |

All values are in SI base units (V, V²/Hz·Hz). The Scope viewer converts to display units (µV, µV²/Hz) for the spinboxes and hypnogram; the auto-scorer does the same conversion internally so threshold values entered in the GUI match what you see on screen.

The pipeline closely follows the Spike2 scoring protocol used in the lab (OSD4 script). Two separate methods are used depending on the purpose:

- **Scoring features** (`compute_all_features`): causal EEG filter, centred EMG RMS, Hann-windowed FFT band powers on a fine time grid. Optimised for epoch-level classification.
- **Spectrogram / visualisation** (`spectrogram`, `band_power`): STFT with a long Hann window. Provides higher frequency resolution for visual inspection.

---

### EEG (high-pass filtered)

**Goal**: remove slow DC drift (< 0.5 Hz) while preserving sleep-relevant oscillations.

**Implementation** (`filter_eeg`, `analysis.py`):

```
1. Design a causal 2nd-order Butterworth low-pass filter at 0.5 Hz:
   b, a = scipy.signal.butter(2, 0.5, btype='low', fs=sfreq)

2. Apply it forward-only to estimate the DC drift:
   drift = scipy.signal.lfilter(b, a, raw_eeg)

3. Subtract drift from original:
   eeg_filtered = raw_eeg − drift
```

The causal (forward-only) filter matches the Spike2 protocol, which builds a virtual `EEGlow` channel with a real-time IIR filter and subtracts it from the original. Using subtraction rather than a direct high-pass keeps the pipeline conceptually aligned with the Spike2 workflow.

---

### EMG (FIR band-pass filtered)

**Goal**: isolate muscle activity (5–45 Hz), rejecting cardiac artifacts below 5 Hz and high-frequency noise above 45 Hz.

**Implementation** (`filter_emg`, `analysis.py:156`):

```
1. Compute number of FIR taps (must be odd):
   numtaps = int(sfreq / transition_width)   # transition_width = 1.8 Hz by default
   numtaps += 1 - (numtaps % 2)              # force odd

   At 400 Hz: numtaps = int(400 / 1.8) = 222 → 223 taps

2. Design band-pass FIR (Kaiser-windowed least-squares):
   fir = scipy.signal.firwin(numtaps, [5.0, 45.0], pass_zero=False, fs=sfreq)

3. Zero-phase filtering:
   emg_filtered = scipy.signal.filtfilt(fir, [1.0], raw_emg)
```

`filtfilt` applies the FIR twice (forward and reverse), giving zero phase shift and doubling the effective filter order. The 1.8 Hz transition width produces a sharp roll-off with acceptable ringing for these band edges.

---

### EMG RMS Envelope

**Goal**: convert the band-passed EMG into a slowly-varying power envelope for epoch-by-epoch comparison.

**Implementation** (`emg_rms`, `analysis.py`):

```
1. Square the band-passed signal:
   squared = emg_filtered²

2. Apply a centred uniform sliding window of width 2τ (τ = 5 s default):
   mean_sq[n] = mean(squared[n − τ·sfreq : n + τ·sfreq])
   Implemented as: scipy.ndimage.uniform_filter1d(squared, size=2·τ·sfreq)

3. Take square root:
   emg_rms_signal = sqrt(mean_sq)
```

The total window width is 10 s (±5 s), matching the Spike2 OSD4 protocol which describes this as a "5 s (=±5 s)" RMS smoother. The result is computed at the full sample rate, then resampled to epoch centre times via linear interpolation (`np.interp`) before being stored in the features dict.

---

### Band Powers

**Goal**: quantify spectral energy in five physiologically meaningful frequency bands for every epoch.

**Bands** (`BANDS`, `analysis.py:10`):

| Band | Range (Hz) | Sleep relevance |
|------|-----------|-----------------|
| Delta | 0 – 4.0 | Primary NREM marker; very high in slow-wave sleep |
| Theta | 6.0 – 10.0 | Elevated in REM; numerator of T:D ratio |
| Alpha | 8.0 – 13.0 | Overlaps theta in rodents; available for custom staging |
| Beta | 13.0 – 30.0 | Elevated during Wake and arousal |
| Gamma | 30.0 – 100.0 | High-frequency arousal; elevated during active wake |

Delta starts at 0 Hz to include all slow-wave power below 4 Hz, matching the scoring protocol. Any residual DC is suppressed by the upstream EEG high-pass filter.

Note: alpha and theta overlap in the 8–10 Hz range. This is intentional — rodent theta peaks around 7–9 Hz, which falls in both bands. Both are kept for flexibility.

**Scoring feature implementation** (`_band_power_smoothed`, `analysis.py`):

```
1. Resample EEG to 512 Hz (matches the Spike2 OSD4 script default).

2. Slide a 256-point Hann window every 0.1 s:
   freq resolution = 512 / 256 = 2 Hz per bin
   time resolution = 0.1 s output grid

3. For each window, compute one-sided FFT power and sum bins in band (f_lo, f_hi).

4. Apply zero-phase exponential smoothing (τ = 5 s) to the resulting time series.
```

**Why this method for scoring?**
The 2 Hz frequency resolution is coarser than an STFT on the raw signal, but the fine 0.1 s time grid and 5 s smoothing give a stable, epoch-aligned feature that closely follows the Spike2 OSD4 protocol (`Pw(...)` function with default settings).

**Spectrogram / visualisation** uses `spectrogram()` (STFT, Hann window, `scaling='density'`), which gives 0.2 Hz frequency resolution at 5 s windows — better suited for visual inspection of spectral structure.

---

### Theta-to-Delta Ratio

**Goal**: a single scalar that peaks in REM (high theta, low delta), is low in NREM (high delta, low theta), and is moderate in Wake.

**Implementation** (`td_ratio`, `analysis.py:265`):

```
td_ratio = theta_power / (delta_power + ε)
ε = 1×10⁻¹²   # prevents division by zero
```

The epsilon guard ensures numerical stability when delta_power approaches zero (e.g., during high-frequency epochs). At typical signal amplitudes (delta_power ~ 10⁻¹¹ V²), epsilon is negligible.

---

## Sleep State Criteria

| State | Code | EEG amplitude | EEG frequency | EMG | Delta power | T:D ratio |
|-------|------|--------------|--------------|-----|-------------|-----------|
| Wake | `W` | Low | High | High | Near zero | Moderate |
| NREM | `N` | High | Low | Low | Very high | Low |
| REM | `R` | Low | High | Flat (atonia) | Low | Peak |
| Unscored | `U` | — | — | — | — | — |

**Key discriminators**:
- **EMG** is the primary Wake vs. sleep discriminator. High EMG → Wake; low EMG → NREM or REM.
- **Delta power** discriminates NREM from Wake and REM. Very high delta → NREM.
- **T:D ratio** discriminates REM from NREM when EMG is low. High T:D with low EMG → REM.

**Important**: EMG "flinches" (brief spikes) during REM do not constitute waking. In practice, the 5 s epoch length averages over brief transients and the τ = 5 s RMS time constant further smooths them.

**State colors** used in the hypnogram strip:

| State | Color | Hex |
|-------|-------|-----|
| Wake | Amber | `#e3b341` |
| NREM | Blue | `#58a6ff` |
| REM | Green | `#3fb950` |
| Unscored | Gray | `#8b949e` |

---

## Auto-Scoring Algorithm

Auto-scoring is implemented in `ScoringSession.auto_score()` (`scoring/state.py:166`).

**Steps**:

```
For each scoring epoch i (centre time t_centre):

  1. Find the nearest feature window k:
     k = argmin |feat_times − t_centre|

  2. Convert base units to display units:
     delta_µV² = delta_power[k] × (1e12 / 4.0)   # 4.0 Hz = delta bandwidth (0–4 Hz)
     emg_µV    = emg_rms[k]    × 1e6

  3. Apply rules in order (later rules overwrite earlier ones):

     Wake:  delta_µV² < delta_wake  AND  emg_µV > emg_wake  → label = "W"
     NREM:  delta_µV² > delta_nrem  AND  emg_µV < emg_nrem  → label = "N"
     REM:   td_ratio[k] > td_rem    AND  emg_µV < emg_rem   → label = "R"

     If no rule fires → label = "U" (unscored)
```

**Rule ordering**: Wake is tested first, then NREM overwrites, then REM overwrites. This means REM can overwrite an epoch initially labeled as NREM, which is correct behavior (REM epochs have low delta and low EMG — both rules could fire, but REM wins).

**Default thresholds** (`AutoScoreThresholds`):

| Parameter | Default | Units | Rule |
|-----------|---------|-------|------|
| `delta_wake` | 1200 | µV²/Hz | delta < this → Wake candidate |
| `delta_nrem` | 1000 | µV²/Hz | delta > this → NREM candidate |
| `emg_wake` | 3 | µV | emg > this confirms Wake |
| `emg_nrem` | 5 | µV | emg < this confirms NREM |
| `emg_rem` | 3 | µV | emg < this confirms REM atonia |
| `td_rem` | 4 | dimensionless | T:D > this → REM candidate |

These defaults are starting points. You will almost certainly need to adjust them for each animal and recording session.

**Threshold overlap**: `delta_wake=1200` and `delta_nrem=1000` intentionally overlap. Epochs with delta between 1000 and 1200 µV²/Hz could be classified as either Wake or NREM depending on EMG. This is a deliberate "grey zone" — epochs in this range are more likely to be correctly classified by their EMG value alone.

---

## Step-by-Step Scoring Workflow

### 1. Load the recording

```python
from sleep_tools import SleepRecording, SleepAnalyzer, Scope

rec = SleepRecording.from_edf("LUMI-0013_2026-03-24_14_15_34_export.edf")
# TSV annotations are auto-discovered from the EDF filename

print(rec)
# SleepRecording(animal='LUMI-0013', duration=11780.0s,
#                channels=['EEG1', 'EEG2', 'EMG'], sfreq=400.0Hz)
```

### 2. Compute features

```python
ana = SleepAnalyzer(rec, epoch_len=5.0)
features = ana.compute_all_features()

# Features are cached — a second call returns immediately
print(features.keys())
# dict_keys(['times', 'eeg_filtered', 'emg_filtered',
#            'delta_power', 'theta_power', 'alpha_power',
#            'beta_power', 'gamma_power', 'emg_rms', 'td_ratio'])

print(f"n_epochs: {len(features['times'])}")       # e.g. 2356
print(f"epoch dt: {features['times'][1]:.1f} s")   # e.g. 2.5 s (centre of first epoch)
```

### 3. Open the oscilloscope and run auto-scoring

```python
scope = Scope(rec, ana)
scope.show()
```

In the Scope window:

1. **Analyze Signals** — click in the sidebar to compute features (if not already done).
2. **CLASSIFICATION panel** — review the six threshold spinboxes. Adjust if needed:
   - Increase `delta_nrem` if too many noisy Wake epochs are being called NREM.
   - Increase `td_rem` if Wake epochs with moderate theta are being called REM.
3. **Epoch length** — set the scoring window (seconds) in the `Epoch length` field (default 5.0 s). This is applied to the analyzer before features are computed.
4. **Reset Defaults** — click to restore all six thresholds and the epoch length to factory values (`AutoScoreThresholds()`).
5. **Run Classification** — click to auto-score all epochs. The hypnogram strip at the top of the figure updates immediately.
6. Inspect the result: look for obvious errors (e.g., isolated NREM epochs surrounded by Wake, REM epochs at the start of the recording before any NREM).

### 4. Adjust thresholds and re-run

You can re-run classification as many times as needed. Each run replaces all labels and pushes one entry onto the undo stack (Ctrl+Z to revert).

Typical threshold adjustments:
- **Too many unscored epochs**: lower `delta_nrem` or `emg_nrem` thresholds slightly.
- **Wake epochs leaking into NREM**: raise `delta_nrem`.
- **REM not being detected**: lower `td_rem` or `emg_rem`.
- **REM false positives during Wake**: raise `td_rem` or lower `emg_rem`.

### 5. Manual epoch correction

To correct individual epochs or ranges:

- **Click** on the hypnogram strip to select an epoch (highlighted in white).
- **Shift+click** to extend the selection to a range.
- Press **W**, **N**, **R**, or **U** to assign the state to the selected epoch(s).
- Alternatively, use the **W / N / R / U** buttons in the LABELING panel.
- **Ctrl+Z** / **Ctrl+Y** to undo / redo (100-step stack).

Scrolling workflow:
- Use the horizontal scrollbar to scan through the recording.
- The **window width** spinbox sets how many seconds are visible at once (try 60–300 s for overview, 15–30 s for detailed inspection).
- The hypnogram strip at the top always shows the full recording; the shaded region shows the current window.

Centering and scaling signals:
- Press **C** (or click **⊕** in the transport bar) to centre all visible signals on their current-window mean — useful when a DC offset or drift pushes a trace out of view.
- Click the **⊕** icon in a channel's header row to centre only that channel.
- Click **↕** in the transport bar (right of **⊕**) to auto-scale all visible traces to fit the current window at once.
- Click **Optimize Scale** to reset both the amplitude scale and the offset for that channel.

Resizing panels:
- Drag the splitter handle between the left channel-control panel and the canvas to widen or narrow the controls.
- Drag the splitter handle between the canvas and the right sidebar to give more room to either panel.
- The **☰** button collapses or restores the sidebar; its last width is remembered.

### 6. Inspect feature traces while scoring

In the oscilloscope, you can display any combination of:

| Signal name | What it shows |
|-------------|---------------|
| `EEG1` | Raw (high-pass filtered) EEG, channel 1 |
| `EEG2` | Raw (high-pass filtered) EEG, channel 2 |
| `EMG` | Raw (FIR band-pass filtered) EMG |
| `delta_power` | Delta band power time series |
| `theta_power` | Theta band power time series |
| `emg_rms` | EMG RMS envelope |
| `td_ratio` | Theta-to-Delta ratio |

Use the **+ Add Channel** button to add or restore signals. Use the **−** button next to a channel name to hide it.

Recommended layout for scoring:
- EEG1 (raw trace — look at amplitude and frequency)
- EMG (raw trace — look at tonic level vs. transient spikes)
- delta_power (should be very high in NREM, near zero in Wake/REM)
- td_ratio (should peak in REM)

### 7. Save the session

From the **LABELING panel**:
- **Save Session** → JSON file (stores all labels, epoch length, thresholds used)
- **Export CSV** → hypnogram CSV: `epoch_index, time_s, label`
- **Save HDF5** → full curated dataset with signals, features, and labels

Or programmatically:

```python
from sleep_tools import ScoringSession

session = ScoringSession(rec, epoch_len=5.0)
session.auto_score(features)

# Inspect counts
print(session)
# ScoringSession(animal='LUMI-0013', n_epochs=2356, epoch_len=5.0s,
#                W=823, N=1012, R=289, U=232)

print(session.state_durations())
# {'W': 4115.0, 'N': 5060.0, 'R': 1445.0, 'U': 1160.0}  # seconds

# Save
session.save("LUMI-0013_sleep_scores.json")
session.to_csv("LUMI-0013_hypnogram.csv")

# Reload later
session2 = ScoringSession.load("LUMI-0013_sleep_scores.json", rec)
```

### 8. Export an HDF5 dataset

```python
from sleep_tools import save_to_h5

# With features and labels from a scoring session
path = save_to_h5(rec, "output/LUMI-0013_dataset.h5",
                  analyzer=ana,
                  session=session,
                  overwrite=True)
```

The HDF5 file contains everything needed for downstream analysis without the package:
- `/signals/{EEG1,EEG2,EMG}` — raw signal arrays (float32, gzip compressed)
- `/epochs/times` — epoch centre times (float64, seconds)
- `/epochs/labels` — per-epoch state strings (U/W/N/R)
- `/epochs/features/{delta_power, theta_power, alpha_power, beta_power, gamma_power, emg_rms, td_ratio}` — all features, NaN if not computed
- `/analysis/times` and `/analysis/features/*` — native analyzer time series when session epochs required interpolation
- `/annotations/` — TTL event times from TSV (if loaded)

Every dataset and group carries self-describing attributes (`unit`, `description`, `frequency_range_hz`, `sleep_relevance`, `band_definitions`). Root attrs include `analysis_profile`, `eeg_channel`, and `feature_source`, so downstream code can see whether epoch features are native or interpolated from the analyzer grid.

---

## Threshold Reference

Default thresholds in `AutoScoreThresholds` and their rationale:

| Threshold | Default | Rationale |
|-----------|---------|-----------|
| `delta_wake = 1200 µV²/Hz` | Delta below this → Wake candidate. Set high enough to exclude NREM slow waves, low enough to catch waking epochs with residual slow activity. |
| `delta_nrem = 1000 µV²/Hz` | Delta above this → NREM candidate. The 200 µV²/Hz gap below `delta_wake` creates a grey zone where EMG decides. |
| `emg_wake = 3 µV` | EMG above this confirms Wake. Set at the noise floor of typical Wake-level muscle activity. |
| `emg_nrem = 5 µV` | EMG below this confirms NREM. Slightly higher than `emg_wake` to allow for low-level tonic EMG in light NREM. |
| `emg_rem = 3 µV` | EMG below this confirms REM atonia. Same level as `emg_wake` — REM atonia should be as flat as or flatter than NREM. |
| `td_rem = 4` | T:D ratio above this → REM candidate. At default band definitions, REM typically shows T:D > 5; setting the threshold at 4 provides a margin. |

These defaults were derived from the Spike2 scoring experience described in Julia's protocol. Adjust per-animal based on visual inspection of the feature traces in the oscilloscope.

---

## Saving and Exporting

| Output | Method | Format |
|--------|--------|--------|
| Scoring session | `session.save(path)` | JSON — labels, thresholds, epoch_len, animal_id |
| Hypnogram | `session.to_csv(path)` | CSV — epoch_index, time_s, label |
| Full dataset | `save_to_h5(rec, path, analyzer=ana, labels=session.labels)` | HDF5 |
| Figures | `fig.savefig(path, dpi=150, bbox_inches='tight')` | PNG/PDF |
| Video | `scope.make_video(path, signals=[...], t_start=0, t_end=3600, speed=60, session=session)` | MP4 with hypnogram |

Load a previously saved session:

```python
session = ScoringSession.load("LUMI-0013_sleep_scores.json", rec)
```

The epoch count must match the recording at the same `epoch_len` — a `ValueError` is raised if it does not.
