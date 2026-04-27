"""Tests for TTL sync event extraction and alignment."""

import numpy as np

from sleep_tools import SleepRecording, SyncAligner


def test_extract_ttl_events_preserves_channel_rows(recording: SleepRecording) -> None:
    aligner = SyncAligner(recording)
    events = aligner.extract_ttl_events()
    assert {"time_from_start", "edge", "channel", "ttl"} <= set(events.columns)
    assert not events.empty
    assert set(events["edge"].unique()) <= {"rise", "fall"}


def test_deduplicate_channels_collapses_simultaneous_edges(
    recording: SleepRecording,
) -> None:
    aligner = SyncAligner(recording)
    events = aligner.extract_ttl_events()
    deduped = aligner.deduplicate_channels(events)
    assert len(deduped) < len(events)
    assert "n_channels" in deduped.columns
    assert deduped["n_channels"].max() >= 2


def test_detect_pulses_and_align_to_bpod(recording: SleepRecording) -> None:
    aligner = SyncAligner(recording)
    pulses = aligner.detect_pulses()
    assert pulses
    rise_times = np.asarray([p[0] for p in pulses[:3]])
    aligned = aligner.align_to_bpod(rise_times + 2.5)
    assert len(aligned) == len(rise_times)
    np.testing.assert_allclose(aligned["offset"].to_numpy(), 2.5)
