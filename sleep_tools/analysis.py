"""Signal processing: EEG/EMG filtering, band power, spectrogram, T:D ratio."""

import numpy as np
from scipy import signal

from sleep_tools.io import SleepRecording


# Default frequency bands (Hz)
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (6.0, 10.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

# EMG filter defaults (Hz) — kept here so FEATURE_INFO can reference them.
_EMG_BP_LOW: float = 5.0
_EMG_BP_HIGH: float = 45.0

# Descriptions for each of the 7 HDF5 feature columns.
# ``frequency_range_hz`` is the exact band passed to the STFT power integrator
# (or the EMG bandpass edges for emg_rms, or None for derived scalars).
FEATURE_INFO: dict[str, dict] = {
    "delta_power": {
        "description": "Delta band power: area under the Hann-windowed STFT PSD.",
        "frequency_range_hz": BANDS["delta"],
        "units": "V²/Hz · Hz",
        "sleep_relevance": (
            "Very high during NREM (slow-wave sleep); "
            "primary NREM marker; near zero during Wake."
        ),
    },
    "theta_power": {
        "description": "Theta band power: area under the Hann-windowed STFT PSD.",
        "frequency_range_hz": BANDS["theta"],
        "units": "V²/Hz · Hz",
        "sleep_relevance": (
            "Elevated during REM sleep; "
            "numerator of the Theta-to-Delta (T:D) ratio."
        ),
    },
    "alpha_power": {
        "description": "Alpha band power: area under the Hann-windowed STFT PSD.",
        "frequency_range_hz": BANDS["alpha"],
        "units": "V²/Hz · Hz",
        "sleep_relevance": (
            "Overlaps theta in rodents; "
            "available for custom staging rules or cross-validation."
        ),
    },
    "beta_power": {
        "description": "Beta band power: area under the Hann-windowed STFT PSD.",
        "frequency_range_hz": BANDS["beta"],
        "units": "V²/Hz · Hz",
        "sleep_relevance": (
            "Elevated during arousal and Wake; "
            "reflects high-frequency cortical activity."
        ),
    },
    "gamma_power": {
        "description": "Gamma band power: area under the Hann-windowed STFT PSD.",
        "frequency_range_hz": BANDS["gamma"],
        "units": "V²/Hz · Hz",
        "sleep_relevance": (
            "High-frequency oscillations; "
            "elevated during active arousal and sensory processing."
        ),
    },
    "emg_rms": {
        "description": (
            "EMG root-mean-square envelope: "
            f"FIR bandpass {_EMG_BP_LOW}–{_EMG_BP_HIGH} Hz (transition 1.8 Hz) "
            "followed by a causal single-pole exponential smoother (τ = 5 s)."
        ),
        "frequency_range_hz": (_EMG_BP_LOW, _EMG_BP_HIGH),
        "units": "V",
        "sleep_relevance": (
            "High during Wake; near-flat (muscle atonia) during REM; "
            "low during NREM.  Primary Wake vs. sleep discriminator."
        ),
    },
    "td_ratio": {
        "description": (
            "Theta-to-Delta power ratio: "
            "theta_power / (delta_power + ε),  ε = 1 × 10⁻¹²."
        ),
        "frequency_range_hz": None,
        "units": "dimensionless",
        "sleep_relevance": (
            "Peak during REM (high theta, low delta); "
            "low during NREM (high delta, low theta); "
            "moderate during Wake."
        ),
    },
}


class SleepAnalyzer:
    """Signal processing for a :class:`~sleep_tools.io.SleepRecording`.

    All parameters (epoch length, filter cutoffs, time constants) are
    runtime-configurable — nothing is hard-coded.

    Parameters
    ----------
    recording:
        The loaded recording.
    epoch_len:
        Default epoch length in seconds used by band-power and spectrogram
        computations when no explicit ``window`` is supplied.
    """

    def __init__(self, recording: SleepRecording, epoch_len: float = 5.0) -> None:
        self.recording = recording
        self.epoch_len = epoch_len
        self._features: dict | None = None
        self._overlap_cached: float | None = None
        self._window_cached: float | None = None

    # ------------------------------------------------------------------ #
    # EEG / EMG filtering
    # ------------------------------------------------------------------ #

    def _resolve_eeg_channels(self) -> list[str]:
        """Return available EEG channel names (EEG1, EEG2) from the recording."""
        available = self.recording.raw.ch_names
        return [ch for ch in ("EEG1", "EEG2") if ch in available]

    def filter_eeg(
        self,
        channel: str | None = None,
        hp_cutoff: float = 0.5,
    ) -> np.ndarray:
        """High-pass EEG by subtracting a low-pass DC/drift estimate.

        Mirrors the Spike2 protocol:
        1. IIR 2nd-order Butterworth low-pass at *hp_cutoff* Hz → drift.
        2. Return ``original − drift``.

        When *channel* is ``None`` (the default), the channel is resolved
        automatically:

        - Both EEG1 and EEG2 present → average the two signals before filtering.
        - Only one present → use that channel.
        - Neither present → print a message and block until Enter is pressed,
          then raise ``RuntimeError``.

        Parameters
        ----------
        channel:
            Explicit channel name (e.g. ``"EEG1"``), or ``None`` for
            automatic resolution.
        hp_cutoff:
            Low-pass cutoff for the drift estimate (= effective high-pass
            corner), in Hz.

        Returns
        -------
        np.ndarray
            Filtered signal, shape ``(n_samples,)``.
        """
        if channel is not None:
            data = self._get_channel_data(channel)
        else:
            eeg_channels = self._resolve_eeg_channels()
            if len(eeg_channels) == 0:
                print(
                    "No EEG channels (EEG1 or EEG2) found in the recording. "
                    "Please check your data, then press Enter to continue..."
                )
                input()
                raise RuntimeError("No EEG channels available for analysis.")
            elif len(eeg_channels) == 1:
                print(f"Only {eeg_channels[0]} found; using it for EEG analysis.")
                data = self._get_channel_data(eeg_channels[0])
            else:
                data = (
                    self._get_channel_data("EEG1") + self._get_channel_data("EEG2")
                ) / 2.0

        fs = self.recording.sfreq
        sos = signal.butter(2, hp_cutoff, btype="low", fs=fs, output="sos")
        drift = signal.sosfiltfilt(sos, data)
        return data - drift

    def filter_emg(
        self,
        channel: str = "EMG",
        bp_low: float = 5.0,
        bp_high: float = 45.0,
        transition_width: float = 1.8,
    ) -> np.ndarray:
        """FIR band-pass filter for EMG.

        Parameters
        ----------
        channel:
            Channel name.
        bp_low, bp_high:
            Pass-band edges in Hz.
        transition_width:
            FIR transition bandwidth in Hz (controls filter length).

        Returns
        -------
        np.ndarray
            Filtered signal, shape ``(n_samples,)``.
        """
        data = self._get_channel_data(channel)
        fs = self.recording.sfreq
        numtaps = int(fs / transition_width)
        numtaps += 1 - (numtaps % 2)  # ensure odd
        fir = signal.firwin(numtaps, [bp_low, bp_high], pass_zero=False, fs=fs)
        return signal.filtfilt(fir, [1.0], data)

    def emg_rms(
        self,
        emg_signal: np.ndarray,
        time_constant: float = 5.0,
    ) -> np.ndarray:
        """Compute RMS envelope using a causal exponential moving average.

        Uses ``scipy.signal.lfilter`` for efficiency (no Python loop).

        Parameters
        ----------
        emg_signal:
            EMG signal array (typically the band-passed signal).
        time_constant:
            Time constant of the exponential smoother, in seconds.

        Returns
        -------
        np.ndarray
            RMS envelope, same shape as *emg_signal*.
        """
        fs = self.recording.sfreq
        alpha = 1.0 - np.exp(-1.0 / (fs * time_constant))
        squared = emg_signal.astype(np.float64) ** 2
        # Single-pole IIR: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
        ema = signal.lfilter([alpha], [1.0, -(1.0 - alpha)], squared)
        return np.sqrt(np.maximum(ema, 0.0))

    # ------------------------------------------------------------------ #
    # Frequency-domain features
    # ------------------------------------------------------------------ #

    def band_power(
        self,
        eeg_signal: np.ndarray,
        band: tuple[float, float],
        window: float | None = None,
        overlap: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Band-limited power time series via STFT.

        Parameters
        ----------
        eeg_signal:
            1-D EEG signal array.
        band:
            ``(low_hz, high_hz)`` frequency band.
        window:
            Analysis window length in seconds (defaults to ``epoch_len``).
        overlap:
            Fractional overlap between successive windows (0–1).

        Returns
        -------
        times:
            Centre times of each window, in seconds.
        power:
            Band power at each window centre.
        """
        if window is None:
            window = self.epoch_len
        fs = self.recording.sfreq
        nperseg = int(window * fs)
        noverlap = int(nperseg * overlap)

        freqs, times, Sxx = signal.spectrogram(
            eeg_signal.astype(np.float64),
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
            scaling="density",
        )

        f_lo, f_hi = band
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        power = np.trapezoid(Sxx[mask, :], freqs[mask], axis=0)
        return times, power

    def td_ratio(
        self,
        delta_power: np.ndarray,
        theta_power: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Element-wise theta-to-delta ratio.

        Parameters
        ----------
        delta_power, theta_power:
            Power arrays of the same length.
        eps:
            Small constant added to the denominator to avoid division by zero.
        """
        return theta_power / (delta_power + eps)

    def spectrogram(
        self,
        eeg_signal: np.ndarray,
        freq_max: float = 50.0,
        window: float | None = None,
        overlap: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a power spectrogram up to *freq_max* Hz.

        Parameters
        ----------
        eeg_signal:
            1-D EEG signal.
        freq_max:
            Discard frequencies above this value.
        window:
            Window length in seconds.
        overlap:
            Fractional overlap.

        Returns
        -------
        times:
            Window centre times in seconds.
        freqs:
            Frequency axis up to *freq_max* Hz.
        Sxx:
            Power spectral density, shape ``(n_freqs, n_times)``.
        """
        if window is None:
            window = self.epoch_len
        fs = self.recording.sfreq
        nperseg = int(window * fs)
        noverlap = int(nperseg * overlap)

        freqs, times, Sxx = signal.spectrogram(
            eeg_signal.astype(np.float64),
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
            scaling="density",
        )

        mask = freqs <= freq_max
        return times, freqs[mask], Sxx[mask, :]

    # ------------------------------------------------------------------ #
    # Convenience: compute everything at once
    # ------------------------------------------------------------------ #

    def compute_all_features(
        self,
        eeg_channel: str | None = None,
        emg_channel: str = "EMG",
        hp_cutoff: float = 0.5,
        bp_low: float = 5.0,
        bp_high: float = 45.0,
        emg_time_constant: float = 5.0,
        epoch_window: float | None = None,
        overlap: float = 0.5,
        bands: dict[str, tuple[float, float]] | None = None,
    ) -> dict:
        """Compute all band powers, EMG RMS, and T:D ratio in one call.

        The result is cached; a second call with the same instance returns
        the cached dict immediately, unless *overlap* or *epoch_window*
        differs from the cached ones.

        Parameters
        ----------
        eeg_channel:
            Explicit EEG channel name, or ``None`` (default) to auto-resolve:
            average EEG1+EEG2 when both present, use whichever is present
            when only one exists, or raise ``RuntimeError`` (after prompting)
            if neither is found.

        Returns
        -------
        dict with keys:
            ``times``, ``eeg_filtered``, ``emg_filtered``,
            ``delta_power``, ``theta_power``, ``alpha_power``,
            ``beta_power``, ``gamma_power``,
            ``emg_rms``, ``td_ratio``.
        """
        if epoch_window is None:
            epoch_window = self.epoch_len

        if (self._features is not None and
            self._overlap_cached == overlap and
            self._window_cached == epoch_window):
            return self._features

        if bands is None:
            bands = BANDS

        eeg = self.filter_eeg(eeg_channel, hp_cutoff)
        emg_filt = self.filter_emg(emg_channel, bp_low, bp_high)
        emg_envelope = self.emg_rms(emg_filt, emg_time_constant)

        times: np.ndarray | None = None
        powers: dict[str, np.ndarray] = {}
        for name, band in bands.items():
            t, p = self.band_power(eeg, band, epoch_window, overlap)
            times = t
            powers[name] = p

        assert times is not None

        # Resample EMG RMS from sample-rate grid → epoch-centre times
        fs = self.recording.sfreq
        sample_times = np.arange(len(emg_envelope)) / fs
        emg_rms_epochs = np.interp(times, sample_times, emg_envelope)

        td = self.td_ratio(powers.get("delta", np.ones_like(times)), powers.get("theta", np.ones_like(times)))

        self._features = {
            "times": times,
            "eeg_filtered": eeg,
            "emg_filtered": emg_filt,
            **{f"{k}_power": v for k, v in powers.items()},
            "emg_rms": emg_rms_epochs,
            "td_ratio": td,
        }
        self._overlap_cached = overlap
        self._window_cached = epoch_window
        return self._features

    def invalidate_cache(self) -> None:
        """Clear the cached features (call after changing parameters)."""
        self._features = None
        self._overlap_cached = None
        self._window_cached = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_channel_data(self, channel: str) -> np.ndarray:
        """Return a 1-D float64 array for *channel*."""
        idx = self.recording.raw.ch_names.index(channel)
        data, _ = self.recording.raw[idx]
        return data[0].astype(np.float64)
