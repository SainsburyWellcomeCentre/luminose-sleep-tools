#!/usr/bin/env python3
"""
Simple script to launch the interactive Sleep Scope viewer.

Usage:
    python examples/scope_viewer.py [path_to_edf]
"""

import os
import sys
from pathlib import Path

# ── Early Initialization (fixes segfaults on macOS/darwin) ────────────
if sys.platform == "darwin":
    os.environ.setdefault("QT_API", "pyside6")
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")


def main() -> None:
    # Optional: pass an EDF path as the first argument
    edf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    from sleep_tools import SleepRecording, SleepAnalyzer, Scope

    # 1. Create a Scope instance
    #    You can optionally pass a recording and analyzer here, 
    #    or load them from the file menu inside the app.
    if edf_path and edf_path.exists():
        print(f"Loading {edf_path}...")
        recording = SleepRecording.from_edf(edf_path)
        analyzer = SleepAnalyzer(recording)
        # analyzer.compute_all_features() # Optional: pre-compute
        scope = Scope(recording, analyzer=analyzer)
    else:
        print("No file provided. Opening Scope with file-picker available in the menu.")
        scope = Scope()

    # 2. Show the interactive window
    #    This blocks until the window is closed.
    scope.show()


if __name__ == "__main__":
    main()
