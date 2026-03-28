"""Launch the interactive Sleep Scope viewer.

Entry points:
    python -m sleep_tools [path_to_edf]
    sleep-scope [path_to_edf]          (after pip install -e .)
    python run_scope.py [path_to_edf]  (from project root)
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    from pathlib import Path

    # Must be set before PySide6 / Qt is initialised (lazy in scope.py)
    if sys.platform == "darwin":
        os.environ.setdefault("QT_API", "pyside6")
        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

    from sleep_tools import SleepRecording, SleepAnalyzer, Scope

    edf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    if edf_path and edf_path.exists():
        print(f"Loading {edf_path}…")
        recording = SleepRecording.from_edf(edf_path)
        analyzer = SleepAnalyzer(recording)
        scope = Scope(recording, analyzer=analyzer)
    else:
        if edf_path:
            print(f"File not found: {edf_path}", file=sys.stderr)
        else:
            print("No file provided — use Ctrl/Cmd+O or E inside the viewer to load a recording.")
        scope = Scope()

    scope.show()


if __name__ == "__main__":
    main()
