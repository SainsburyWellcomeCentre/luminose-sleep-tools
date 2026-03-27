"""sleep_tools — rodent sleep scoring from EDF recordings."""

from sleep_tools.io import SleepRecording, save_to_h5
from sleep_tools.analysis import SleepAnalyzer, BANDS, FEATURE_INFO
from sleep_tools.visualization import SleepVisualizer
from sleep_tools.scope import Scope
from sleep_tools.scoring import ScoringSession, AutoScoreThresholds, STATE_COLORS

__all__ = [
    "SleepRecording",
    "SleepAnalyzer",
    "SleepVisualizer",
    "Scope",
    "BANDS",
    "FEATURE_INFO",
    "save_to_h5",
    "ScoringSession",
    "AutoScoreThresholds",
    "STATE_COLORS",
]
