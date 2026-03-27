"""Integration tests for sleep_tools.visualization (no display needed)."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.figure
import pytest

from sleep_tools import SleepRecording, SleepAnalyzer, SleepVisualizer


@pytest.fixture(scope="module")
def viz(recording: SleepRecording, analyzer: SleepAnalyzer) -> SleepVisualizer:
    return SleepVisualizer(recording, analyzer)


def test_plot_raw_traces_returns_figure(viz: SleepVisualizer) -> None:
    fig = viz.plot_raw_traces(t_start=0, t_end=30)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_raw_traces_subset_channels(viz: SleepVisualizer) -> None:
    fig = viz.plot_raw_traces(t_start=0, t_end=30, channels=["EEG1"])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_band_timeseries_returns_figure(viz: SleepVisualizer) -> None:
    fig = viz.plot_band_timeseries(channel="EEG1")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_spectrogram_returns_figure(viz: SleepVisualizer) -> None:
    fig = viz.plot_spectrogram(channel="EEG1", freq_max=50.0)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_overview_returns_figure(viz: SleepVisualizer) -> None:
    fig = viz.plot_overview(channel="EEG1")
    assert isinstance(fig, matplotlib.figure.Figure)
