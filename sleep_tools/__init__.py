"""sleep_tools — rodent sleep scoring from EDF recordings."""

__version__ = "0.1.0"

from sleep_tools.io import SleepRecording, save_to_h5
from sleep_tools.analysis import BANDS, FEATURE_INFO, SleepAnalyzer
from sleep_tools.visualization import SleepVisualizer
from sleep_tools.scope import Scope
from sleep_tools.scoring import ScoringSession, AutoScoreThresholds, STATE_COLORS
from sleep_tools.sync import SyncAligner

__all__ = [
    "__version__",
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
    "SyncAligner",
]
