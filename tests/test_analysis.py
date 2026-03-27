"""Integration tests for sleep_tools.analysis."""
import numpy as np

from sleep_tools import SleepAnalyzer, SleepRecording


def test_filter_eeg_shape(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    eeg = analyzer.filter_eeg("EEG1")
    n_raw = recording.raw.n_times
    assert eeg.shape == (n_raw,)


def test_filter_eeg_removes_dc(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    """Filtered signal should have near-zero mean (DC removed)."""
    eeg = analyzer.filter_eeg("EEG1")
    assert abs(eeg.mean()) < 1e-7


def test_filter_emg_shape(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    emg = analyzer.filter_emg("EMG")
    assert emg.shape == (recording.raw.n_times,)


def test_emg_rms_non_negative(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    emg_filt = analyzer.filter_emg("EMG")
    rms = analyzer.emg_rms(emg_filt)
    assert np.all(rms >= 0)
    assert rms.shape == emg_filt.shape


def test_band_power_returns_positive(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    eeg = analyzer.filter_eeg("EEG1")
    times, power = analyzer.band_power(eeg, band=(0.5, 4.0))
    assert len(times) == len(power)
    assert np.all(power >= 0)


def test_td_ratio_shape(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    eeg = analyzer.filter_eeg("EEG1")
    _, delta = analyzer.band_power(eeg, (0.5, 4.0))
    _, theta = analyzer.band_power(eeg, (6.0, 10.0))
    td = analyzer.td_ratio(delta, theta)
    assert td.shape == delta.shape
    assert np.all(np.isfinite(td))


def test_spectrogram_shape(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    eeg = analyzer.filter_eeg("EEG1")
    times, freqs, Sxx = analyzer.spectrogram(eeg, freq_max=50.0)
    assert freqs[-1] <= 50.0
    assert Sxx.shape == (len(freqs), len(times))


def test_compute_all_features_keys(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    feats = analyzer.compute_all_features()
    expected = {
        "times", "eeg_filtered", "emg_filtered",
        "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
        "emg_rms", "td_ratio",
    }
    assert expected <= feats.keys()


def test_compute_all_features_cached(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    f1 = analyzer.compute_all_features()
    f2 = analyzer.compute_all_features()
    assert f1 is f2  # same object — cached
