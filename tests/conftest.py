"""Shared fixtures for sleep_tools tests."""
from pathlib import Path

import pytest

from sleep_tools import SleepRecording, SleepAnalyzer

# Use the longest EDF file for integration tests
DATA_DIR = Path(__file__).parent.parent / "example_data" / "luminose"
LONG_EDF = DATA_DIR / "LUMI-0013_2026-03-24_14_15_34_export.edf"


@pytest.fixture(scope="session")
def recording() -> SleepRecording:
    """Load the full recording once per test session."""
    return SleepRecording.from_edf(LONG_EDF, verbose=False)


@pytest.fixture(scope="session")
def analyzer(recording: SleepRecording) -> SleepAnalyzer:
    return SleepAnalyzer(recording, epoch_len=5.0)
