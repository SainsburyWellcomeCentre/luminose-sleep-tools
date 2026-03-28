#!/usr/bin/env python3
"""Launch the interactive Sleep Scope viewer.

Usage:
    python run_scope.py [path_to_edf]

If no path is given the viewer opens empty — load a recording via
Ctrl/Cmd+O (folder) or Ctrl/Cmd+E (file) from within the viewer.
"""

from sleep_tools.__main__ import main

if __name__ == "__main__":
    main()
