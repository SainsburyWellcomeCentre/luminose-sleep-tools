"""Signal processing: EEG/EMG filtering, band power, spectrogram, T:D ratio."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from sleep_tools.io import SleepRecording


# Frequency bands used for sleep staging (Hz).
# Delta starts at 0 to include all slow-wave power below 4 Hz.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.0, 4.0),
    "theta": (6.0, 10.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

# EMG bandpass edges — kept here so FEATURE_INFO can reference them.
_EMG_BP_LOW: float = 5.0
_EMG_BP_HIGH: float = 45.0

# Descriptions for all 7 HDF5 feature columns.
FEATURE_INFO: dict[str, dict] = {
    "delta_power": {
        "description": (
            "Delta band power (0–4 Hz): Hann-windowed FFT, smoothed in time."
        ),
        "frequency_range_hz": BANDS["delta"],
        "units": "V²/Hz · Hz",
        "sleep_relevance": (
            "Very high during NREM (slow-wave sleep); "
            "primary NREM marker; near zero during Wake."
        ),
    },
    "theta_power": {
        "description": (
            "Theta band power (6–10 Hz): Hann-windowed FFT, smoothed in time."
        ),
        "frequency_range_hz": BANDS["theta"],
        "units": "V²/Hz · Hz",
        "sleep_relevance": (
            "Elevated during REM sleep; "
            "numerator of the Theta-to-Delta (T:D) ratio."
        ),
    },
    "alpha_power": {
        "description": (
            "Alpha band power (8–13 Hz): Hann-windowed FFT, smoothed in time."
        ),
        "frequency_range_hz": BANDS["alpha"],
        "units": "V²/Hz · Hz",
        "sleep_relevance": (
            "Overlaps theta in rodents; "
            "available for custom staging rules or cross-validation."
        ),
    },
    "beta_power": {
        "description": (
            "Beta band power (13–30 Hz): Hann-windowed FFT, smoothed in time."
        ),
        "frequency_range_hz": BANDS["beta"],
        "units": "V²/Hz · Hz",
        "sleep_relevance": (
            "Elevated during arousal and Wake; "
            "reflects high-frequency cortical activity."
        ),
    },
    "gamma_power": {
        "description": (
            "Gamma band power (30–100 Hz): Hann-windowed FFT, smoothed in time."
        ),
        "frequency_range_hz": BANDS["gamma"],
        "units": "V²/Hz · Hz",
        "sleep_relevance": (
            "High-frequency oscillations; "
            "elevated during active arousal and sensory processing."
        ),
    },
    "emg_rms": {
        "description": (
            f"EMG root-mean-square envelope: "
            f"FIR bandpass {_EMG_BP_LOW}–{_EMG_BP_HIGH} Hz (transition 1.8 Hz) "
            "followed by a centred ±5 s uniform-window RMS smoother."
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


@dataclass(frozen=True)
class _FeatureCacheKey:
    eeg_channel: str | None
    emg_channel: str
    hp_cutoff: float
    eeg_filter_order: int
    bp_low: float
    bp_high: float
    emg_transition_width: float
    emg_time_constant: float
    output_interval: float
    smoothing_tau: float
    fft_size: int
    target_sfreq: float
    bands: tuple[tuple[str, float, float], ...]


class SleepAnalyzer:
    """Signal processing for a :class:`~sleep_tools.io.SleepRecording`.

    All parameters (epoch length, filter cutoffs, time constants) are
    runtime-configurable — nothing is hard-coded.

    Parameters
    ----------
    recording:
        The loaded recording.
    epoch_len:
        Epoch length in seconds, used by :meth:`band_power` and
        :meth:`spectrogram` when no explicit window is supplied.
    eeg_channel:
        Which EEG channel to use (e.g. ``"EEG1"`` or ``"EEG2"``).
        When ``None``, the first available channel is selected automatically
        (``EEG1`` preferred over ``EEG2``).
    """

    def __init__(
        self,
        recording: SleepRecording,
        epoch_len: float = 5.0,
        eeg_channel: str | None = None,
    ) -> None:
        self.recording = recording
        self.epoch_len = float(epoch_len)
        self.eeg_channel = eeg_channel
        self._features: dict | None = None
        self._feature_cache_key: _FeatureCacheKey | None = None

    # ------------------------------------------------------------------ #
    # EEG / EMG filtering
    # ------------------------------------------------------------------ #

    def _auto_eeg_channel(self) -> str:
        """Return 'average' when both EEG channels exist, else the one that does."""
        ch_names = self.recording.raw.ch_names
        has1 = "EEG1" in ch_names
        has2 = "EEG2" in ch_names
        if has1 and has2:
            return "average"
        if has1:
            return "EEG1"
        if has2:
            return "EEG2"
        raise RuntimeError(
            "No EEG channel (EEG1 or EEG2) found in the recording."
        )

    def _resolve_eeg_channel(self, channel: str | None) -> str:
        """Resolve an explicit or instance-level EEG channel, auto-selecting if needed."""
        return channel or self.eeg_channel or self._auto_eeg_channel()

    def _get_eeg_data(self, channel: str) -> np.ndarray:
        """Return EEG signal; averages EEG1+EEG2 when *channel* is ``'average'``."""
        if channel == "average":
            ch_names = self.recording.raw.ch_names
            available = [ch for ch in ("EEG1", "EEG2") if ch in ch_names]
            if not available:
                raise RuntimeError("No EEG channel (EEG1 or EEG2) found in the recording.")
            return np.mean([self._get_channel_data(ch) for ch in available], axis=0)
        return self._get_channel_data(channel)

    def filter_eeg(
        self,
        channel: str | None = None,
        hp_cutoff: float = 0.5,
        order: int = 2,
    ) -> np.ndarray:
        """High-pass EEG by subtracting a causal low-pass drift estimate.

        Steps:
        1. Design a causal 2nd-order Butterworth low-pass at *hp_cutoff* Hz.
        2. Apply it forward-only (no phase correction) to estimate the DC drift.
        3. Subtract: ``eeg_filtered = raw − drift``.

        This matches the Spike2 scoring protocol, which builds a virtual
        ``EEGlow`` channel with an IIR filter and then subtracts it.

        Parameters
        ----------
        channel:
            EEG channel name, ``'average'`` to average EEG1+EEG2, or ``None``
            to defer to :attr:`eeg_channel` or auto-selection.
        hp_cutoff:
            Low-pass cutoff (= effective high-pass corner), in Hz.
        order:
            Butterworth filter order.

        Returns
        -------
        np.ndarray
            Filtered signal, shape ``(n_samples,)``.
        """
        ch = self._resolve_eeg_channel(channel)
        data = self._get_eeg_data(ch)
        fs = self.recording.sfreq
        b, a = signal.butter(order, hp_cutoff, btype="low", fs=fs)
        drift = signal.lfilter(b, a, data)
        return data - drift

    def filter_emg(
        self,
        channel: str = "EMG",
        bp_low: float = 5.0,
        bp_high: float = 45.0,
        transition_width: float = 1.8,
    ) -> np.ndarray:
        """FIR band-pass filter for EMG (5–45 Hz by default).

        Parameters
        ----------
        channel:
            EMG channel name.
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
        """Centred RMS envelope using a uniform sliding window.

        Computes the square root of the mean-squared signal over a window of
        ``2 * time_constant`` seconds, centred on each sample.  This gives a
        symmetric smoother with no phase lag.

        Parameters
        ----------
        emg_signal:
            Band-passed EMG signal array.
        time_constant:
            Half-width of the smoothing window in seconds (total width = 2×).

        Returns
        -------
        np.ndarray
            RMS envelope, same shape as *emg_signal*.
        """
        fs = self.recording.sfreq
        window_samples = max(1, int(round(2.0 * time_constant * fs)))
        squared = emg_signal.astype(np.float64, copy=False) ** 2
        mean_sq = ndimage.uniform_filter1d(squared, size=window_samples, mode="nearest")
        return np.sqrt(np.maximum(mean_sq, 0.0))

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
        """Band-limited power via STFT (high frequency resolution).

        Intended for visualization (e.g. computing a single band over a
        specific window).  For scoring features, :meth:`compute_all_features`
        uses a time-resolved smoothed estimate instead.

        Parameters
        ----------
        eeg_signal:
            1-D EEG signal array.
        band:
            ``(low_hz, high_hz)`` frequency band.
        window:
            Analysis window length in seconds (defaults to :attr:`epoch_len`).
        overlap:
            Fractional overlap between successive windows (0–1).

        Returns
        -------
        times:
            Centre times of each window, in seconds.
        power:
            Band power at each window centre (V²/Hz·Hz).
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
        return times, np.trapezoid(Sxx[mask, :], freqs[mask], axis=0)

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
        """Power spectrogram via STFT (high frequency resolution).

        Parameters
        ----------
        eeg_signal:
            1-D EEG signal.
        freq_max:
            Discard frequencies above this value (Hz).
        window:
            Window length in seconds (defaults to :attr:`epoch_len`).
        overlap:
            Fractional overlap between successive windows.

        Returns
        -------
        times:
            Window centre times in seconds.
        freqs:
            Frequency axis up to *freq_max* Hz.
        Sxx:
            Power spectral density (V²/Hz), shape ``(n_freqs, n_times)``.
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
    # Internal: time-resolved band power for scoring features
    # ------------------------------------------------------------------ #

    def _band_power_smoothed(
        self,
        eeg_signal: np.ndarray,
        band: tuple[float, float],
        output_interval: float = 0.1,
        smoothing_tau: float = 5.0,
        fft_size: int = 256,
        target_sfreq: float = 512.0,
        chunk_size: int = 20_000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Time-resolved band power for scoring.

        Resamples EEG to *target_sfreq*, applies a Hann-windowed FFT every
        *output_interval* seconds (coarse frequency resolution, fine time
        resolution), then smooths with a zero-phase exponential filter
        (τ = *smoothing_tau* seconds).

        Frequency resolution = target_sfreq / fft_size (e.g. 512/256 = 2 Hz).
        """
        source_fs = self.recording.sfreq
        x = eeg_signal.astype(np.float64, copy=False)
        if not np.isclose(source_fs, target_sfreq):
            n_target = int(round(len(x) * target_sfreq / source_fs))
            x = signal.resample(x, n_target)
            fs = float(target_sfreq)
        else:
            fs = float(source_fs)

        hop = max(1, int(round(output_interval * fs)))
        n = len(x)
        if n < fft_size:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

        starts = np.arange(0, n - fft_size + 1, hop, dtype=np.int64)
        times = starts.astype(np.float64) / fs
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / fs)
        f_lo, f_hi = band
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(mask):
            return times, np.full(len(times), np.nan, dtype=np.float64)

        win = signal.windows.hann(fft_size, sym=False)
        scale = 1.0 / np.sum(win) ** 2
        one_sided = np.ones(len(freqs), dtype=np.float64)
        if fft_size % 2 == 0:
            one_sided[1:-1] = 2.0
        else:
            one_sided[1:] = 2.0

        power = np.empty(len(starts), dtype=np.float64)
        for i0 in range(0, len(starts), chunk_size):
            sl = starts[i0: i0 + chunk_size]
            frames = np.empty((len(sl), fft_size), dtype=np.float64)
            for row, start in enumerate(sl):
                frames[row] = x[start: start + fft_size]
            frames *= win
            spec = np.fft.rfft(frames, axis=1)
            ps = (np.abs(spec) ** 2) * scale * one_sided
            power[i0: i0 + len(sl)] = ps[:, mask].sum(axis=1)

        if smoothing_tau > 0.0 and power.size > 3:
            sr = 1.0 / output_interval
            alpha = 1.0 - np.exp(-1.0 / (sr * smoothing_tau))
            power = signal.filtfilt(
                [alpha], [1.0, -(1.0 - alpha)], power, method="gust"
            )
        return times, np.maximum(power, 0.0)

    # ------------------------------------------------------------------ #
    # Convenience: compute all features at once
    # ------------------------------------------------------------------ #

    def compute_all_features(
        self,
        eeg_channel: str | None = None,
        emg_channel: str = "EMG",
        hp_cutoff: float = 0.5,
        eeg_filter_order: int = 2,
        bp_low: float = 5.0,
        bp_high: float = 45.0,
        emg_transition_width: float = 1.8,
        emg_time_constant: float = 5.0,
        output_interval: float = 0.1,
        smoothing_tau: float = 5.0,
        fft_size: int = 256,
        target_sfreq: float = 512.0,
        bands: dict[str, tuple[float, float]] | None = None,
    ) -> dict:
        """Compute all band powers, EMG RMS, and T:D ratio in one call.

        The result is cached; a second call with identical parameters returns
        the cached dict immediately.

        EEG processing:
            Causal 2nd-order Butterworth high-pass (drift subtraction at
            *hp_cutoff* Hz).

        EMG processing:
            FIR bandpass *bp_low*–*bp_high* Hz → centred ±*emg_time_constant* s
            RMS envelope.

        Band powers:
            Hann-windowed FFT every *output_interval* seconds (EEG resampled
            to *target_sfreq*, FFT size *fft_size*), then zero-phase
            exponential smoothing (τ = *smoothing_tau* s).

        Parameters
        ----------
        eeg_channel:
            EEG channel name, or ``None`` to use :attr:`eeg_channel` or
            auto-select.
        emg_channel:
            EMG channel name.
        hp_cutoff:
            EEG drift-filter cutoff in Hz.
        eeg_filter_order:
            Butterworth order for the EEG drift filter.
        bp_low, bp_high:
            EMG bandpass edges in Hz.
        emg_transition_width:
            FIR transition bandwidth in Hz.
        emg_time_constant:
            Half-width of the EMG RMS smoothing window in seconds.
        output_interval:
            Time step between successive band-power samples in seconds.
        smoothing_tau:
            Exponential smoothing time constant for band powers in seconds.
        fft_size:
            FFT size for band-power computation.
        target_sfreq:
            EEG is resampled to this rate before the FFT.
        bands:
            Frequency band definitions.  Defaults to :data:`BANDS`.

        Returns
        -------
        dict with keys:
            ``times``, ``eeg_filtered``, ``emg_filtered``,
            ``delta_power``, ``theta_power``, ``alpha_power``,
            ``beta_power``, ``gamma_power``,
            ``emg_rms``, ``td_ratio``,
            ``band_definitions``, ``eeg_channel``, ``feature_config``.
        """
        if bands is None:
            bands = BANDS

        _eeg_ch = eeg_channel if eeg_channel is not None else self.eeg_channel

        cache_key = _FeatureCacheKey(
            eeg_channel=_eeg_ch,
            emg_channel=emg_channel,
            hp_cutoff=float(hp_cutoff),
            eeg_filter_order=int(eeg_filter_order),
            bp_low=float(bp_low),
            bp_high=float(bp_high),
            emg_transition_width=float(emg_transition_width),
            emg_time_constant=float(emg_time_constant),
            output_interval=float(output_interval),
            smoothing_tau=float(smoothing_tau),
            fft_size=int(fft_size),
            target_sfreq=float(target_sfreq),
            bands=tuple(
                (name, float(lo), float(hi))
                for name, (lo, hi) in sorted(bands.items())
            ),
        )
        if self._features is not None and self._feature_cache_key == cache_key:
            return self._features

        resolved_ch = self._resolve_eeg_channel(_eeg_ch)
        eeg = self.filter_eeg(resolved_ch, hp_cutoff=hp_cutoff, order=eeg_filter_order)
        emg_filt = self.filter_emg(
            emg_channel,
            bp_low=bp_low,
            bp_high=bp_high,
            transition_width=emg_transition_width,
        )
        emg_envelope = self.emg_rms(emg_filt, emg_time_constant)

        times: np.ndarray | None = None
        powers: dict[str, np.ndarray] = {}
        for name, band in bands.items():
            t, p = self._band_power_smoothed(
                eeg,
                band,
                output_interval=output_interval,
                smoothing_tau=smoothing_tau,
                fft_size=fft_size,
                target_sfreq=target_sfreq,
            )
            times = t
            powers[name] = p

        assert times is not None

        fs = self.recording.sfreq
        sample_times = np.arange(len(emg_envelope)) / fs
        emg_rms_at_times = np.interp(times, sample_times, emg_envelope)

        td = self.td_ratio(
            powers.get("delta", np.ones_like(times)),
            powers.get("theta", np.ones_like(times)),
        )

        self._features = {
            "times": times,
            "eeg_filtered": eeg,
            "emg_filtered": emg_filt,
            **{f"{k}_power": v for k, v in powers.items()},
            "emg_rms": emg_rms_at_times,
            "td_ratio": td,
            "band_definitions": {k: (float(lo), float(hi)) for k, (lo, hi) in bands.items()},
            "eeg_channel": resolved_ch,
            "feature_config": {
                "eeg_channel": resolved_ch,
                "bands": {k: [float(lo), float(hi)] for k, (lo, hi) in bands.items()},
            },
        }
        self._feature_cache_key = cache_key
        return self._features

    def invalidate_cache(self) -> None:
        """Clear the cached features (call after changing parameters)."""
        self._features = None
        self._feature_cache_key = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_channel_data(self, channel: str) -> np.ndarray:
        """Return a 1-D float64 array for *channel*."""
        idx = self.recording.raw.ch_names.index(channel)
        data, _ = self.recording.raw[idx]
        return data[0].astype(np.float64)
