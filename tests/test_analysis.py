"""Integration tests for sleep_tools.analysis."""
import numpy as np
import pytest
from scipy import signal

from sleep_tools import SleepAnalyzer, SleepRecording


def test_filter_eeg_shape(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    eeg = analyzer.filter_eeg("EEG1")
    assert eeg.shape == (recording.raw.n_times,)


def test_filter_eeg_removes_dc(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    """Filtered signal should have near-zero mean (DC removed)."""
    eeg = analyzer.filter_eeg("EEG1")
    assert abs(eeg.mean()) < 1e-7


def test_filter_eeg_causal_matches_lfilter(recording: SleepRecording) -> None:
    """filter_eeg uses a causal Butterworth drift subtraction."""
    analyzer = SleepAnalyzer(recording, epoch_len=5.0, eeg_channel="EEG1")
    observed = analyzer.filter_eeg()
    data = analyzer._get_channel_data("EEG1")
    b, a = signal.butter(2, 0.5, btype="low", fs=recording.sfreq)
    expected = data - signal.lfilter(b, a, data)
    np.testing.assert_allclose(observed, expected)


def test_filter_eeg_auto_selects_channel(recording: SleepRecording) -> None:
    """When eeg_channel is None, auto-selects without error."""
    analyzer = SleepAnalyzer(recording, epoch_len=5.0)
    eeg = analyzer.filter_eeg()
    assert eeg.shape == (recording.raw.n_times,)


def test_filter_emg_shape(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    emg = analyzer.filter_emg("EMG")
    assert emg.shape == (recording.raw.n_times,)


def test_emg_rms_non_negative(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    """Centred RMS envelope is non-negative and same length as input."""
    emg_filt = analyzer.filter_emg("EMG")
    rms = analyzer.emg_rms(emg_filt)
    assert np.all(rms >= 0)
    assert rms.shape == emg_filt.shape


def test_band_power_returns_positive(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    """STFT band_power (used for visualization) returns positive values."""
    eeg = analyzer.filter_eeg("EEG1")
    times, power = analyzer.band_power(eeg, band=(0.0, 4.0))
    assert len(times) == len(power)
    assert np.all(power >= 0)


def test_td_ratio_shape(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    eeg = analyzer.filter_eeg("EEG1")
    _, delta = analyzer.band_power(eeg, (0.0, 4.0))
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


def test_compute_all_features_cache_invalidated_on_param_change(
    recording: SleepRecording,
) -> None:
    """Different output_interval → different cache entry."""
    analyzer = SleepAnalyzer(recording, epoch_len=5.0, eeg_channel="EEG1")
    f1 = analyzer.compute_all_features(output_interval=0.1)
    f2 = analyzer.compute_all_features(output_interval=0.5)
    assert f1 is not f2
    assert len(f1["times"]) != len(f2["times"])


def test_delta_band_starts_at_zero(recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
    """Delta band is 0–4 Hz (not 0.5–4 Hz)."""
    from sleep_tools.analysis import BANDS
    assert BANDS["delta"] == (0.0, 4.0)
    feats = analyzer.compute_all_features()
    assert feats["band_definitions"]["delta"] == (0.0, 4.0)
