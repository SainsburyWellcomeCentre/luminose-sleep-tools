"""Matplotlib-based static plots for sleep recordings (no GUI)."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.axes import Axes as _Axes
from matplotlib.colors import LogNorm

from sleep_tools.io import SleepRecording
from sleep_tools.analysis import SleepAnalyzer, BANDS

# ---------------------------------------------------------------------------
# Shared aesthetic constants
# ---------------------------------------------------------------------------

_BASE_STYLE: dict = {
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.edgecolor":     "#c8c8c8",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#ebebeb",
    "grid.linewidth":     0.5,
    "axes.labelsize":     8,
    "axes.labelcolor":    "#444444",
    "axes.titlesize":     9,
    "axes.titlecolor":    "#222222",
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "xtick.color":        "#666666",
    "ytick.color":        "#666666",
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "font.size":          8,
    "text.color":         "#333333",
}

# Per-channel colours for raw trace plots (cycles if more channels)
_TRACE_COLORS = ["#3B7DD8", "#48B89F", "#D9634C", "#9B59B6", "#F0B429"]

# Frequency-ordered palette: cool (low) → warm (high)
_BAND_COLORS = [
    "#3B6FD4",   # delta   – cobalt blue
    "#2EAAA0",   # theta   – teal
    "#6AB04C",   # alpha   – grass green
    "#F0B429",   # sigma / beta – golden amber
    "#E8543A",   # beta / gamma – coral
    "#9B59B6",   # gamma   – violet
]

# Greek symbols for each frequency band (keyed by BANDS names)
_BAND_SYMBOLS: dict[str, str] = {
    "delta": "δ",
    "theta": "θ",
    "alpha": "α",
    "beta":  "β",
    "gamma": "γ",
}


def _draw_band_overlays(ax: _Axes, freq_max: float) -> None:
    """Draw band boundary lines and Greek labels on a spectrogram axis.

    Each band's upper edge is a grey dashed line.  Labels are placed at the
    midpoint between consecutive drawn lines so theta and alpha (which have
    overlapping f_lo/f_hi definitions) each stay inside their own visual
    region.  Labels sit inside a semi-transparent dark pill.
    """
    prev_line_hz = 0.0
    for (name, (f_lo, f_hi)), color in zip(BANDS.items(), _BAND_COLORS):
        if f_lo >= freq_max:
            continue
        f_hi_clip = min(f_hi, freq_max)
        # Grey dashed boundary at the upper edge of this band
        if f_hi_clip < freq_max:
            ax.axhline(
                f_hi_clip,
                color="#c8c8c8",
                lw=1.0,
                alpha=0.9,
                linestyle="--",
                zorder=3,
            )
        # Label at the midpoint between the previous line and this band's upper
        # line — guaranteed to stay between the two drawn separators regardless
        # of overlapping band definitions (e.g. theta 6–10 vs alpha 8–13 Hz)
        f_label = (prev_line_hz + f_hi_clip) / 2
        symbol = _BAND_SYMBOLS.get(name, name)
        ax.text(
            0.008,
            f_label,
            symbol,
            transform=ax.get_yaxis_transform(),  # x: axes coords, y: data coords
            fontsize=7.5,
            color=color,
            va="center",
            ha="left",
            fontweight="bold",
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="#111111",
                edgecolor="none",
                alpha=0.55,
            ),
        )
        prev_line_hz = f_hi_clip


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

        with plt.rc_context(_BASE_STYLE):
            fig, axes = plt.subplots(
                n_ch, 1, figsize=figsize, sharex=True, constrained_layout=True
            )
            if n_ch == 1:
                axes = [axes]

            for i, (ax, ch) in enumerate(zip(axes, channels)):
                color = _TRACE_COLORS[i % len(_TRACE_COLORS)]
                idx = raw.ch_names.index(ch)
                data, times = raw[idx, int(t_start * sfreq):int(t_end * sfreq)]
                ax.plot(times, data[0] * 1e6, lw=0.7, color=color)
                ax.set_ylabel(f"{ch}\n(µV)", fontsize=8)
                ax.margins(x=0)
                ax.yaxis.set_label_coords(-0.06, 0.5)

            axes[-1].set_xlabel("Time (s)")
            animal = self.recording.animal_id
            fig.suptitle(
                f"{animal}  ·  Raw traces  [{t_start:.1f}–{t_end:.1f} s]",
                fontsize=10,
                fontweight="medium",
                color="#222222",
                x=0.5,
                y=1.01,
            )

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

        with plt.rc_context(_BASE_STYLE):
            fig, axes = plt.subplots(
                n, 1, figsize=figsize, sharex=True, constrained_layout=True
            )
            if n == 1:
                axes = [axes]

            for ax, (name, band), color in zip(axes, bands.items(), _BAND_COLORS):
                times, power = self.analyzer.band_power(eeg, band, window, overlap)
                ax.semilogy(times, power, lw=0.8, color=color)
                ax.set_ylabel(f"{name}\n({band[0]}–{band[1]} Hz)", fontsize=8)
                ax.margins(x=0)
                # Shade under the curve for readability
                ax.fill_between(times, power, power.min(), alpha=0.12, color=color)

            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(
                f"{self.recording.animal_id}  ·  Band power  [{channel}]",
                fontsize=10,
                fontweight="medium",
                color="#222222",
                x=0.5,
                y=1.01,
            )

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

        Sxx = np.where(Sxx > 0, Sxx, np.nan)
        vmax = np.nanmax(Sxx)
        vmin = vmax * 10 ** (-db_range / 10)

        with plt.rc_context(_BASE_STYLE):
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            im = ax.pcolormesh(
                times,
                freqs,
                Sxx,
                norm=LogNorm(vmin=max(vmin, 1e-30), vmax=vmax),
                cmap="viridis",
                shading="auto",
                rasterized=True,
            )
            cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
            cb.set_label("PSD (V²/Hz)", fontsize=7, color="#444444")
            cb.ax.tick_params(labelsize=6, colors="#666666")

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            ax.spines["left"].set_color("#c8c8c8")
            ax.spines["bottom"].set_color("#c8c8c8")
            ax.grid(False)   # grid looks cluttered on spectrogram
            _draw_band_overlays(ax, freq_max)

            fig.suptitle(
                f"{self.recording.animal_id}  ·  Spectrogram  [{channel}]",
                fontsize=10,
                fontweight="medium",
                color="#222222",
                x=0.5,
                y=1.02,
            )

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
        )

        eeg = feats["eeg_filtered"]
        times = feats["times"]
        if window is None:
            window = self.analyzer.epoch_len
        spec_times, freqs, Sxx = self.analyzer.spectrogram(eeg, freq_max, window, overlap)

        with plt.rc_context(_BASE_STYLE):
            fig, axes = plt.subplots(
                4, 1, figsize=figsize, sharex=True, constrained_layout=True,
                gridspec_kw={"height_ratios": [2, 1.2, 1, 1]},
            )

            # ── Spectrogram ───────────────────────────────────────────────
            ax = axes[0]
            Sxx_plot = np.where(Sxx > 0, Sxx, np.nan)
            vmax = np.nanmax(Sxx_plot)
            db_range = 40.0
            vmin = vmax * 10 ** (-db_range / 10)
            im = ax.pcolormesh(
                spec_times,
                freqs,
                Sxx_plot,
                norm=LogNorm(vmin=max(vmin, 1e-30), vmax=vmax),
                cmap="viridis",
                shading="auto",
                rasterized=True,
            )
            cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.008)
            cb.set_label("PSD (V²/Hz)", fontsize=6, color="#444444")
            cb.ax.tick_params(labelsize=5, colors="#666666")
            ax.set_ylabel("Freq (Hz)", fontsize=8)
            ax.grid(False)
            ax.spines["left"].set_color("#c8c8c8")
            ax.spines["bottom"].set_color("#c8c8c8")
            _draw_band_overlays(ax, freq_max)
            ax.set_title(
                f"{self.recording.animal_id}  ·  Overview  [{channel}]",
                fontsize=10,
                fontweight="medium",
                color="#222222",
                pad=6,
            )

            # ── Delta & Theta power ───────────────────────────────────────
            ax = axes[1]
            ax.semilogy(
                times, feats["delta_power"],
                lw=0.9, label="delta  0.5–4 Hz", color="#3B6FD4",
            )
            ax.semilogy(
                times, feats["theta_power"],
                lw=0.9, label="theta  6–10 Hz", color="#E8A838",
            )
            ax.fill_between(times, feats["delta_power"],
                            feats["delta_power"].min(), alpha=0.10, color="#3B6FD4")
            ax.fill_between(times, feats["theta_power"],
                            feats["theta_power"].min(), alpha=0.10, color="#E8A838")
            ax.set_ylabel("Power\n(V²/Hz)", fontsize=8)
            ax.legend(
                fontsize=6.5, loc="upper right",
                frameon=True, framealpha=0.85, edgecolor="#dddddd",
            )
            ax.margins(x=0)

            # ── EMG RMS ───────────────────────────────────────────────────
            ax = axes[2]
            ax.plot(times, feats["emg_rms"] * 1e6, lw=0.85, color="#C0392B")
            ax.fill_between(times, feats["emg_rms"] * 1e6, 0, alpha=0.12, color="#C0392B")
            ax.set_ylabel("EMG RMS\n(µV)", fontsize=8)
            ax.margins(x=0)

            # ── T:D ratio ─────────────────────────────────────────────────
            ax = axes[3]
            ax.plot(times, feats["td_ratio"], lw=0.85, color="#2EAAA0")
            ax.fill_between(times, feats["td_ratio"], 0, alpha=0.12, color="#2EAAA0")
            ax.set_ylabel("T:D ratio", fontsize=8)
            ax.set_xlabel("Time (s)")
            ax.margins(x=0)

        return fig
