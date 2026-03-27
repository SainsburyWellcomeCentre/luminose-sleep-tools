"""Matplotlib-based static plots for sleep recordings (no GUI)."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.colors import LogNorm

from sleep_tools.io import SleepRecording
from sleep_tools.analysis import SleepAnalyzer, BANDS


class SleepVisualizer:
    """Produce matplotlib figures from a recording + analyzer.

    All methods return a :class:`matplotlib.figure.Figure` so the caller
    can display, save, or further annotate it.

    Parameters
    ----------
    recording:
        Loaded :class:`~sleep_tools.io.SleepRecording`.
    analyzer:
        Corresponding :class:`~sleep_tools.analysis.SleepAnalyzer`.
    """

    def __init__(self, recording: SleepRecording, analyzer: SleepAnalyzer) -> None:
        self.recording = recording
        self.analyzer = analyzer

    # ------------------------------------------------------------------ #
    # Raw traces
    # ------------------------------------------------------------------ #

    def plot_raw_traces(
        self,
        t_start: float = 0.0,
        t_end: float | None = None,
        channels: list[str] | None = None,
        figsize: tuple[float, float] = (14, 6),
    ) -> matplotlib.figure.Figure:
        """Plot raw (un-filtered) channel traces for a time window.

        Parameters
        ----------
        t_start, t_end:
            Start and end of the window in seconds.  *t_end* defaults to
            ``t_start + 60``.
        channels:
            List of channel names to plot.  Defaults to all channels.
        figsize:
            Figure size ``(width, height)`` in inches.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if t_end is None:
            t_end = t_start + 60.0
        if channels is None:
            channels = self.recording.channels

        raw = self.recording.raw
        sfreq = self.recording.sfreq
        t_start = float(np.clip(t_start, 0, self.recording.duration))
        t_end = float(np.clip(t_end, t_start, self.recording.duration))

        n_ch = len(channels)
        fig, axes = plt.subplots(n_ch, 1, figsize=figsize, sharex=True)
        if n_ch == 1:
            axes = [axes]

        for ax, ch in zip(axes, channels):
            idx = raw.ch_names.index(ch)
            data, times = raw[idx, int(t_start * sfreq):int(t_end * sfreq)]
            ax.plot(times, data[0] * 1e6, lw=0.6, color="steelblue")
            ax.set_ylabel(f"{ch}\n(µV)", fontsize=8)
            ax.margins(x=0)

        axes[-1].set_xlabel("Time (s)")
        animal = self.recording.animal_id
        fig.suptitle(
            f"{animal} — Raw traces  [{t_start:.1f}–{t_end:.1f} s]",
            fontsize=10,
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # Band power time series
    # ------------------------------------------------------------------ #

    def plot_band_timeseries(
        self,
        channel: str = "EEG1",
        bands: dict[str, tuple[float, float]] | None = None,
        hp_cutoff: float = 0.5,
        window: float | None = None,
        overlap: float = 0.5,
        figsize: tuple[float, float] = (14, 10),
    ) -> matplotlib.figure.Figure:
        """Plot per-band power time series as stacked subplots.

        Parameters
        ----------
        channel:
            EEG channel to analyse.
        bands:
            Dict mapping band names → ``(low_hz, high_hz)``.
            Defaults to :data:`~sleep_tools.analysis.BANDS`.
        hp_cutoff:
            High-pass corner used when filtering the EEG.
        window:
            Analysis window in seconds (defaults to ``analyzer.epoch_len``).
        overlap:
            Fractional window overlap.
        figsize:
            Figure size ``(width, height)`` in inches.
        """
        if bands is None:
            bands = BANDS
        if window is None:
            window = self.analyzer.epoch_len

        eeg = self.analyzer.filter_eeg(channel, hp_cutoff)

        n = len(bands)
        fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
        if n == 1:
            axes = [axes]

        colors = plt.cm.tab10.colors
        for ax, (name, band), color in zip(axes, bands.items(), colors):
            times, power = self.analyzer.band_power(eeg, band, window, overlap)
            ax.semilogy(times, power, lw=0.8, color=color)
            ax.set_ylabel(f"{name}\n({band[0]}–{band[1]} Hz)", fontsize=8)
            ax.margins(x=0)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(
            f"{self.recording.animal_id} — Band power  [{channel}]",
            fontsize=10,
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # Spectrogram
    # ------------------------------------------------------------------ #

    def plot_spectrogram(
        self,
        channel: str = "EEG1",
        freq_max: float = 50.0,
        hp_cutoff: float = 0.5,
        window: float | None = None,
        overlap: float = 0.5,
        db_range: float = 40.0,
        figsize: tuple[float, float] = (14, 4),
    ) -> matplotlib.figure.Figure:
        """Plot a power spectrogram (log-colour) for one EEG channel.

        Parameters
        ----------
        channel:
            EEG channel name.
        freq_max:
            Upper frequency limit in Hz.
        hp_cutoff:
            High-pass corner for EEG pre-filtering.
        window:
            STFT window length in seconds.
        overlap:
            Fractional overlap.
        db_range:
            Colour range in dB (from peak downwards).
        figsize:
            Figure size ``(width, height)`` in inches.
        """
        if window is None:
            window = self.analyzer.epoch_len

        eeg = self.analyzer.filter_eeg(channel, hp_cutoff)
        times, freqs, Sxx = self.analyzer.spectrogram(eeg, freq_max, window, overlap)

        # Clip to finite positive values for log scaling
        Sxx = np.where(Sxx > 0, Sxx, np.nan)
        vmax = np.nanmax(Sxx)
        vmin = vmax * 10 ** (-db_range / 10)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(
            times,
            freqs,
            Sxx,
            norm=LogNorm(vmin=max(vmin, 1e-30), vmax=vmax),
            cmap="inferno",
            shading="auto",
        )
        fig.colorbar(im, ax=ax, label="PSD (V²/Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.suptitle(
            f"{self.recording.animal_id} — Spectrogram  [{channel}]",
            fontsize=10,
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # Overview figure
    # ------------------------------------------------------------------ #

    def plot_overview(
        self,
        channel: str = "EEG1",
        hp_cutoff: float = 0.5,
        bp_low: float = 5.0,
        bp_high: float = 45.0,
        emg_time_constant: float = 5.0,
        freq_max: float = 50.0,
        window: float | None = None,
        overlap: float = 0.5,
        figsize: tuple[float, float] = (14, 12),
    ) -> matplotlib.figure.Figure:
        """4-panel overview: spectrogram, band powers, EMG RMS, T:D ratio.

        All panels share the same time axis.

        Parameters
        ----------
        channel:
            EEG channel for spectrogram and band powers.
        hp_cutoff:
            EEG high-pass corner (Hz).
        bp_low, bp_high:
            EMG band-pass edges (Hz).
        emg_time_constant:
            EMG RMS time constant (s).
        freq_max:
            Upper frequency for spectrogram (Hz).
        window:
            Analysis window (s).
        overlap:
            Fractional overlap.
        figsize:
            Figure size ``(width, height)`` in inches.
        """
        feats = self.analyzer.compute_all_features(
            eeg_channel=channel,
            emg_channel="EMG",
            hp_cutoff=hp_cutoff,
            bp_low=bp_low,
            bp_high=bp_high,
            emg_time_constant=emg_time_constant,
            epoch_window=window,
            overlap=overlap,
        )

        eeg = feats["eeg_filtered"]
        times = feats["times"]
        if window is None:
            window = self.analyzer.epoch_len
        spec_times, freqs, Sxx = self.analyzer.spectrogram(eeg, freq_max, window, overlap)

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # --- Spectrogram ---
        ax = axes[0]
        Sxx_plot = np.where(Sxx > 0, Sxx, np.nan)
        vmax = np.nanmax(Sxx_plot)
        db_range = 40.0
        vmin = vmax * 10 ** (-db_range / 10)
        ax.pcolormesh(
            spec_times,
            freqs,
            Sxx_plot,
            norm=LogNorm(vmin=max(vmin, 1e-30), vmax=vmax),
            cmap="inferno",
            shading="auto",
        )
        ax.set_ylabel("Freq (Hz)", fontsize=8)
        ax.set_title(f"{self.recording.animal_id} — Overview [{channel}]", fontsize=10)

        # --- Delta & Theta power ---
        ax = axes[1]
        ax.semilogy(times, feats["delta_power"], lw=0.8, label="Delta (0.5–4 Hz)", color="royalblue")
        ax.semilogy(times, feats["theta_power"], lw=0.8, label="Theta (6–10 Hz)", color="darkorange")
        ax.set_ylabel("Power\n(V²/Hz)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.margins(x=0)

        # --- EMG RMS ---
        ax = axes[2]
        ax.plot(times, feats["emg_rms"] * 1e6, lw=0.8, color="firebrick")
        ax.set_ylabel("EMG RMS\n(µV)", fontsize=8)
        ax.margins(x=0)

        # --- T:D ratio ---
        ax = axes[3]
        ax.plot(times, feats["td_ratio"], lw=0.8, color="seagreen")
        ax.set_ylabel("T:D ratio", fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.margins(x=0)

        fig.tight_layout()
        return fig
