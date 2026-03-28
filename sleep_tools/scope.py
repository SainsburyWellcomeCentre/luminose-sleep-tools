"""Interactive oscilloscope-style viewer and video exporter for sleep recordings.

Two public methods on :class:`Scope`:

* ``show()``       — opens a scrollable Qt oscilloscope window (PySide6)
* ``make_video()`` — renders a scrolling MP4 video to an output folder

Qt and matplotlib-Qt imports are **lazy** (deferred to call time) so the
module can be imported in headless / notebook environments without errors.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from sleep_tools.scoring.state import ScoringSession

import numpy as np
import matplotlib.ticker as mticker

from sleep_tools.io import SleepRecording
from sleep_tools.analysis import SleepAnalyzer, BANDS

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

@dataclass
class Theme:
    """Color palette for the Scope viewer."""
    name:     str
    bg:       str
    panel:    str
    border:   str
    accent:   str
    text:     str
    muted:    str
    grid:     str
    signals:  list[str]


DARK_THEME = Theme(
    name="dark",
    bg="#0d1117",
    panel="#161b22",
    border="#21262d",
    accent="#58a6ff",
    text="#e6edf3",
    muted="#8b949e",
    grid="#1c2128",
    signals=[
        "#58a6ff", "#bc8cff", "#ff7b72", "#3fb950", "#e3b341",
        "#f78166", "#d2a8ff", "#79c0ff", "#ffa657", "#7ee787",
    ]
)

LIGHT_THEME = Theme(
    name="light",
    bg="#ffffff",
    panel="#f6f8fa",
    border="#d0d7de",
    accent="#0969da",
    text="#1f2328",
    muted="#656d76",
    grid="#f0f0f0",
    signals=[
        "#0969da", "#8250df", "#cf222e", "#1a7f37", "#9a6700",
        "#d1242f", "#8250df", "#0969da", "#bc4c00", "#1a7f37",
    ]
)

# Signal category mapping for unit selection
_SIG_CATEGORIES = {
    "EEG1": "voltage", "EEG2": "voltage", "EMG": "voltage",
    "emg_rms": "voltage",
    "delta_power": "power", "theta_power": "power", "alpha_power": "power",
    "beta_power": "power", "gamma_power": "power",
    "td_ratio": "ratio"
}

_UNITS_BY_CAT = {
    "voltage": ["µV", "mV", "V"],
    "power": ["µV²/Hz", "mV²/Hz", "V²/Hz", "µV²", "mV²", "V²"],
    "ratio": [""]
}

def _get_signal_scale(name: str, unit: str) -> float:
    """Calculate scale factor from base (V or V²) to display unit."""
    if not unit: return 1.0
    
    # Base scales for voltage/power prefixes
    v_scales = {"V": 1.0, "mV": 1e3, "µV": 1e6}
    p_scales = {"V²": 1.0, "mV²": 1e6, "µV²": 1e12}
    
    if unit in v_scales: return v_scales[unit]
    if unit in p_scales: return p_scales[unit]
    
    if "²/Hz" in unit:
        prefix = unit.replace("²/Hz", "")
        p_scale = p_scales.get(f"{prefix}²", 1.0)
        
        # Integrated Power (V²) -> Mean PSD (V²/Hz) requires dividing by Bandwidth
        bw = 1.0
        band_name = name.replace("_power", "")
        if band_name in BANDS:
            lo, hi = BANDS[band_name]
            bw = hi - lo
        return p_scale / bw
        
    return 1.0

# Human-readable label + unit for each possible signal name
# Note: Default units are now µV and µV²/Hz as per user request
_SIG_META: dict[str, tuple[str, str]] = {
    # name         : (display label, default unit)
    "EEG1":          ("EEG 1",         "µV"),
    "EEG2":          ("EEG 2",         "µV"),
    "EMG":           ("EMG",           "µV"),
    "delta_power":   ("δ power",       "µV²/Hz"),
    "theta_power":   ("θ power",       "µV²/Hz"),
    "alpha_power":   ("α power",       "µV²/Hz"),
    "beta_power":    ("β power",       "µV²/Hz"),
    "gamma_power":   ("γ power",       "µV²/Hz"),
    "emg_rms":       ("EMG RMS",       "µV"),
    "td_ratio":      ("T:D ratio",     ""),
}

# Keys returned by compute_all_features() that are plottable
_DERIVED_KEYS = {
    "delta_power", "theta_power", "alpha_power",
    "beta_power",  "gamma_power", "emg_rms",    "td_ratio",
}

_RAW_CHANNELS = {"EEG1", "EEG2", "EMG"}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class _SignalData:
    """One plottable signal trace."""
    name:   str
    times:  np.ndarray   # seconds from recording start
    values: np.ndarray   # in display units
    label:  str
    unit:   str
    color:  str
    color_index: int = 0
    y_half: float = 0.0  # symmetric ±y_half axis limit
    base_values: np.ndarray | None = None # values in base units (V or V²)


# ---------------------------------------------------------------------------
# Pure helpers (no Qt / matplotlib GUI)
# ---------------------------------------------------------------------------

def _auto_y_half(values: np.ndarray, t_range: tuple[float, float] | None = None, times: np.ndarray | None = None) -> float:
    """Symmetric y-axis half-range based on the 99th-percentile absolute value."""
    if t_range is not None and times is not None:
        t0, t1 = t_range
        idx = np.logical_and(times >= t0, times <= t1)
        v = values[idx] if np.any(idx) else values
    else:
        v = values
        
    p = float(np.nanpercentile(np.abs(v), 99))
    return max(p * 1.2, 1e-12)


def _format_time(seconds: float, unit: str = "auto") -> str:
    """Format seconds as a compact, human-readable string."""
    if unit == "s" or (unit == "auto" and seconds < 60):
        return f"{seconds:.1f}s"
    if unit == "m" or (unit == "auto" and seconds < 3600):
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m {int(s)}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m"


def _make_time_formatter(x_window: float, unit: str = "auto") -> mticker.FuncFormatter:
    """Return a tick formatter that auto-scales or uses fixed units."""
    if unit == "s" or (unit == "auto" and x_window <= 120):
        def fmt(x, _pos): return f"{x:.0f}s"
    elif unit == "m" or (unit == "auto" and x_window <= 7200):
        def fmt(x, _pos): return f"{int(x // 60)}:{int(x % 60):02d}"
    else:
        def fmt(x, _pos): return f"{int(x // 3600)}h{int((x % 3600) // 60):02d}"
    return mticker.FuncFormatter(fmt)


def _apply_ax_style(ax, sig: _SignalData, theme: Theme) -> None:
    """Apply theme styling to a matplotlib Axes."""
    ax.set_facecolor(theme.bg)
    for spine in ax.spines.values():
        spine.set_color(theme.border)
    ax.tick_params(colors=theme.muted, labelsize=9)
    ax.grid(True, color=theme.grid, linewidth=0.5, linestyle="-", alpha=0.5)
    unit_label = sig.unit if sig.unit else "a.u."
    ylabel = f"{sig.label}\n({unit_label})"
    ax.set_ylabel(ylabel, color=theme.muted, fontsize=8, labelpad=2, rotation=0,
                  ha="right", va="center", multialignment="center")


def _pair_ttl_visible(
    rise_times: np.ndarray,
    fall_times: np.ndarray,
    t0: float,
    t1: float,
):
    """Yield clipped ``(r, f)`` pairs for TTL HIGH periods that overlap ``[t0, t1]``.

    Each rise is paired with the immediately following fall.  Periods that
    start before ``t0`` or extend past ``t1`` are clipped to the window.
    """
    for r in rise_times:
        later = fall_times[fall_times > r]
        f = float(later[0]) if len(later) > 0 else float("inf")
        # Overlap test: [r, f] overlaps [t0, t1] iff r < t1 and f > t0
        if r < t1 and f > t0:
            yield float(max(r, t0)), float(min(f, t1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Scope:
    """Interactive oscilloscope viewer and video renderer for a sleep recording.

    Parameters
    ----------
    recording:
        Loaded :class:`~sleep_tools.io.SleepRecording`.
    analyzer:
        Optional :class:`~sleep_tools.analysis.SleepAnalyzer`.  Required
        to display derived features (band powers, EMG RMS, T:D ratio).
    """

    def __init__(
        self,
        recording: SleepRecording | None = None,
        analyzer: SleepAnalyzer | None = None,
    ) -> None:
        self.recording = recording
        self.analyzer  = analyzer

    # ------------------------------------------------------------------
    # 1. Interactive oscilloscope
    # ------------------------------------------------------------------

    def show(
        self,
        signals: Sequence[str] | None = None,
        x_window: float = 30.0,
    ) -> None:
        """Open the interactive oscilloscope window (blocks until closed)."""
        import matplotlib
        # Use lowercase qtagg and only set if not already using a Qt backend
        current_backend = matplotlib.get_backend().lower()
        if "qt" not in current_backend:
            try:
                matplotlib.use("qtagg")
            except Exception:
                pass

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QScrollBar, QLabel, QDoubleSpinBox, QPushButton, QSizePolicy,
            QGroupBox, QGridLayout, QSpacerItem, QCheckBox, QScrollArea,
            QSlider, QComboBox, QToolButton, QMenu, QFileDialog, QFrame,
            QListWidget, QListWidgetItem, QInputDialog, QAbstractSpinBox,
        )
        from PySide6.QtCore import Qt, QTimer, QSize, QPoint
        from PySide6.QtGui import QPalette, QColor, QGuiApplication, QAction, QIcon, QActionGroup

        # ── Deferred imports ──────────────────────────────────────────
        from sleep_tools.scoring.state import (
            ScoringSession, AutoScoreThresholds, STATE_COLORS,
        )
        import matplotlib.colors as _mcolors
        from matplotlib.patches import Rectangle as _Rect, Patch as _Patch

        # ── Data Preparation Helper ───────────────────────────────────
        def prepare_data(rec, ana, requested):
            if rec is None: return []
            if requested is None:
                requested = list(rec.channels)
                if ana: requested += list(_DERIVED_KEYS)
            return self._prepare_with(rec, ana, list(requested))

        _SCROLLBAR_RES = 10_000

        class _ScopeWindow(QMainWindow):
            def __init__(self_w, rec, ana) -> None:
                super().__init__()
                self_w._recording = rec
                self_w._analyzer = ana
                self_w._requested_signals = signals
                self_w._x_window = float(x_window)
                self_w._t0 = 0.0
                self_w._time_unit = "auto"
                self_w._playing = False
                self_w._opt_dynamic = False
                self_w._play_speed = 10 ** ((50 - 1) / 50.0)  # ~10x at slider=50
                self_w._theme = DARK_THEME
                self_w._folder_path = None
                self_w._files_in_folder = []
                
                # Internal widget caches
                self_w._yscale_spins = {}
                self_w._lockable_widgets: list = []  # disabled during playback

                # ── Scoring state ──────────────────────────────────────
                self_w._session: ScoringSession | None = None
                self_w._sel_indices: set[int] = set()   # selected epoch indices
                self_w._sel_anchor:  int | None = None  # anchor for shift+click range
                self_w._hypno_ax = None
                # Spinbox refs so _on_run_classification can read them
                self_w._thr_spins: dict = {}
                # click event connection id (stored to avoid double-binding)
                self_w._hypno_cid: int | None = None
                # dirty flag: rebuild full hypnogram only when content changed
                self_w._hypno_dirty: bool = True
                # whether to draw threshold reference lines after classification
                self_w._thr_lines_active: bool = False
                # Draggable threshold lines state
                self_w._thr_line_refs: list = []  # (line, text, spin_key, ax, display_to_val_factor)
                self_w._drag_line = None
                self_w._drag_text = None
                self_w._drag_spin_key: str | None = None
                self_w._drag_display_to_val: float | None = None
                self_w._drag_cids: list = []

                # ── TTL event display state ────────────────────────────
                self_w._ttl_show_strips: bool = True   # semi-transparent rise→fall spans
                self_w._ttl_show_rise:   bool = False  # dotted lines at rising edges
                self_w._ttl_show_fall:   bool = False  # dotted lines at falling edges
                self_w._ttl_artists: list = []          # matplotlib artists for cleanup
                self_w._ttl_data: dict = {}             # cached from recording.ttl_events()
                # Per-axis collection references for in-place updates (avoids rebuilding on each draw)
                self_w._ttl_strip_colls: list = []
                self_w._ttl_rise_colls: list = []
                self_w._ttl_fall_colls: list = []
                # Pre-sorted TTL arrays (computed once in _rebuild_ttl_artists)
                self_w._ttl_strips_r: np.ndarray = np.array([])
                self_w._ttl_strips_f: np.ndarray = np.array([])
                self_w._ttl_rises_all: np.ndarray = np.array([])
                self_w._ttl_falls_all: np.ndarray = np.array([])

                self_w.setWindowTitle("Sleep Scope")
                self_w.resize(1280, 800)
                
                self_w._init_data(rec, ana, self_w._requested_signals)
                self_w._build_ui()
                self_w._apply_theme(DARK_THEME)
                self_w._rebuild_figure()
                self_w._draw(0.0)

                self_w._timer = QTimer()
                self_w._timer.setInterval(50)  # 20 fps
                self_w._timer.timeout.connect(self_w._on_play_tick)

            def _init_data(self_w, rec, ana, requested):
                self_w._recording = rec
                self_w._analyzer = ana
                self_w._signal_data = prepare_data(rec, ana, requested)
                self_w._y_scales = {s.name: s.y_half for s in self_w._signal_data}
                self_w._visible = {s.name for s in self_w._signal_data}
                # Keep _vis_signals in sync so stale scrollbar events don't KeyError
                self_w._vis_signals = list(self_w._signal_data)
                # Cache TTL events once per recording load (avoid per-frame parsing)
                self_w._ttl_data = rec.ttl_events() if rec is not None else {}
                if rec:
                    self_w.setWindowTitle(f"Sleep Scope — {rec.animal_id} ({rec.start_datetime})")
                else:
                    self_w.setWindowTitle("Sleep Scope — No Data")

            def _apply_theme(self_w, theme) -> None:
                self_w._theme = theme
                pal = QPalette()
                for role, col in [
                    (QPalette.ColorRole.Window, theme.bg), (QPalette.ColorRole.WindowText, theme.text),
                    (QPalette.ColorRole.Base, theme.panel), (QPalette.ColorRole.AlternateBase, theme.border),
                    (QPalette.ColorRole.Text, theme.text), (QPalette.ColorRole.Button, theme.border),
                    (QPalette.ColorRole.ButtonText, theme.text), (QPalette.ColorRole.Highlight, theme.accent),
                ]: pal.setColor(role, QColor(col))
                self_w.setPalette(pal)
                
                fonts = "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
                self_w.setStyleSheet(f"""
                    QWidget {{ background:{theme.bg}; color:{theme.text}; font-family:{fonts}; font-size:12px; }}
                    QGroupBox {{ border:1px solid {theme.border}; border-radius:4px; margin-top:10px; padding-top:10px; }}
                    QGroupBox::title {{ color:{theme.muted}; subcontrol-origin:margin; left:8px; font-size:10px; }}
                    QScrollBar:horizontal {{ background:{theme.panel}; height:10px; border-radius:5px; }}
                    QScrollBar::handle:horizontal {{ background:{theme.border}; border-radius:5px; min-width:20px; }}
                    QScrollBar::handle:horizontal:hover {{ background:{theme.accent}; }}
                    QPushButton, QToolButton {{ background:{theme.panel}; border:1px solid {theme.border}; border-radius:4px; padding:4px 8px; }}
                    QPushButton:hover, QToolButton:hover {{ border-color:{theme.accent}; }}
                    QMenu {{ background:{theme.panel}; border:1px solid {theme.border}; padding:4px; }}
                    QMenu::item:selected {{ background:{theme.accent}; color:{theme.bg}; }}
                    QDoubleSpinBox, QSpinBox {{
                        background:{theme.panel}; color:{theme.text};
                        border:1px solid {theme.border}; border-radius:3px;
                        padding:2px 4px; min-height:26px;
                        selection-background-color:{theme.accent};
                    }}
                    QDoubleSpinBox:focus, QSpinBox:focus {{ border-color:{theme.accent}; }}
                    QDoubleSpinBox::up-button, QSpinBox::up-button,
                    QDoubleSpinBox::down-button, QSpinBox::down-button {{
                        width:22px; background:{theme.border};
                    }}
                    QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
                    QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
                        background:{theme.accent};
                    }}
                    QComboBox {{
                        background:{theme.panel}; color:{theme.text};
                        border:1px solid {theme.border}; border-radius:3px; padding:1px 4px;
                    }}
                    QComboBox::drop-down {{ width:14px; border-left:1px solid {theme.border}; }}
                    QComboBox QAbstractItemView {{ background:{theme.panel}; color:{theme.text}; }}
                """)
                
                # Update sidebar background if it exists (avoids it staying dark in light mode)
                if hasattr(self_w, "_sidebar"):
                    self_w._sidebar.setStyleSheet(f"background:{theme.panel}; border-left:1px solid {theme.border};")
                
                # Update labels with muted style
                if hasattr(self_w, "_time_label"):
                    self_w._time_label.setStyleSheet(f"color:{theme.muted}; font-size:10px;")
                if hasattr(self_w, "_speed_label"):
                    self_w._speed_label.setStyleSheet(f"color:{theme.muted}; font-size:10px;")

            def _build_ui(self_w) -> None:
                t = self_w._theme
                root = QWidget()
                self_w.setCentralWidget(root)
                main_layout = QHBoxLayout(root)
                main_layout.setContentsMargins(0, 0, 0, 0)
                main_layout.setSpacing(0)

                # ── Main Content Area ──────────────────────────────────
                content = QWidget()
                cl = QVBoxLayout(content)
                cl.setContentsMargins(10, 10, 10, 10)
                cl.setSpacing(4)

                # Canvas + Left Controls
                cv_row = QHBoxLayout()
                
                # Left sidebar for per-channel controls (aligned with Y-labels)
                self_w._left_panel = QWidget()
                self_w._left_layout = QVBoxLayout(self_w._left_panel)
                self_w._left_layout.setContentsMargins(0, 50, 0, 50)
                self_w._left_layout.setSpacing(10)

                left_scroll = QScrollArea()
                left_scroll.setWidget(self_w._left_panel)
                left_scroll.setWidgetResizable(True)
                left_scroll.setFixedWidth(160)
                left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                left_scroll.setFrameShape(QFrame.Shape.NoFrame)
                cv_row.addWidget(left_scroll)

                self_w._canvas = FigureCanvas(Figure(facecolor=t.bg))
                # Forward canvas key events to the window so our keyPressEvent
                # handles Space/arrows/brackets even when the canvas has focus.
                self_w._canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
                self_w._canvas.installEventFilter(self_w)
                cv_row.addWidget(self_w._canvas, stretch=1)
                cl.addLayout(cv_row, stretch=1)

                # X-Axis area
                time_row = QHBoxLayout()
                time_row.setContentsMargins(160, 0, 0, 0) # Offset for left panel
                time_row.addStretch()
                self_w._time_label = QLabel("Time (m)")
                time_row.addWidget(self_w._time_label)
                
                self_w._time_btn = QToolButton()
                self_w._time_btn.setText("⌛")
                self_w._time_btn.setFixedSize(24, 24)
                self_w._time_btn.setToolTip("Click to change time window and units")
                self_w._time_btn.clicked.connect(self_w._show_time_menu)
                time_row.addWidget(self_w._time_btn)
                time_row.addStretch()
                cl.addLayout(time_row)

                # Transport Row
                transport = QWidget()
                tl = QHBoxLayout(transport)
                tl.setContentsMargins(0, 0, 0, 0)
                
                help_btn = QToolButton()
                help_btn.setText("?")
                help_btn.setFixedSize(32, 32)
                help_btn.setToolTip("How to score sleep — step by step")
                help_btn.clicked.connect(self_w._show_help_dialog)
                tl.addWidget(help_btn)

                self_w._play_btn = QToolButton()
                self_w._play_btn.setText("▶")
                self_w._play_btn.setFixedSize(32, 32)
                self_w._play_btn.setToolTip("Play / Pause  [Space]")
                self_w._play_btn.clicked.connect(self_w._on_play_pause)
                tl.addWidget(self_w._play_btn)

                self_w._speed_label = QLabel("~10x")
                self_w._speed_label.setFixedWidth(40)
                tl.addWidget(self_w._speed_label)

                self_w._speed_slider = QSlider(Qt.Orientation.Horizontal)
                self_w._speed_slider.setRange(1, 100)  # log-mapped: 1→1x, 50→~10x, 100→100x
                self_w._speed_slider.setValue(50)
                self_w._speed_slider.setFixedWidth(120)
                self_w._speed_slider.setToolTip("Playback speed (1×–100×, logarithmic)")
                self_w._speed_slider.valueChanged.connect(self_w._on_speed_changed)
                tl.addWidget(self_w._speed_slider)

                self_w._scrollbar = QScrollBar(Qt.Orientation.Horizontal)
                self_w._scrollbar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                self_w._scrollbar.valueChanged.connect(self_w._on_scroll)
                tl.addWidget(self_w._scrollbar, stretch=1)

                # Pagination
                self_w._prev_btn = QToolButton()
                self_w._prev_btn.setText("<")
                self_w._prev_btn.setFixedSize(32, 32)
                self_w._prev_btn.setToolTip("Previous page  [ or PageUp ]")
                self_w._prev_btn.clicked.connect(self_w._on_prev_page)
                tl.addWidget(self_w._prev_btn)

                self_w._next_btn = QToolButton()
                self_w._next_btn.setText(">")
                self_w._next_btn.setFixedSize(32, 32)
                self_w._next_btn.setToolTip("Next page  ] or PageDown")
                self_w._next_btn.clicked.connect(self_w._on_next_page)
                tl.addWidget(self_w._next_btn)

                # Theme Toggle
                self_w._theme_btn = QToolButton()
                self_w._theme_btn.setText("☯")
                self_w._theme_btn.setFixedSize(32, 32)
                self_w._theme_btn.setToolTip("Toggle dark / light theme")
                self_w._theme_btn.clicked.connect(self_w._on_toggle_theme)
                tl.addWidget(self_w._theme_btn)

                # Sidebar Toggle
                self_w._sidebar_btn = QToolButton()
                self_w._sidebar_btn.setText("☰")
                self_w._sidebar_btn.setFixedSize(32, 32)
                self_w._sidebar_btn.setCheckable(True)
                self_w._sidebar_btn.setChecked(True)
                self_w._sidebar_btn.clicked.connect(self_w._toggle_sidebar)
                tl.addWidget(self_w._sidebar_btn)

                cl.addWidget(transport)
                main_layout.addWidget(content, stretch=1)

                # ── Sidebar ────────────────────────────────────────────
                self_w._sidebar = QFrame()
                self_w._sidebar.setFixedWidth(260)
                self_w._sidebar.setStyleSheet(f"background:{t.panel}; border-left:1px solid {t.border};")
                sl = QVBoxLayout(self_w._sidebar)
                sl.setContentsMargins(0, 0, 0, 0)
                
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setFrameShape(QFrame.Shape.NoFrame)
                self_w._sidebar_content = QWidget()
                self_w._sidebar_layout = QVBoxLayout(self_w._sidebar_content)
                self_w._sidebar_layout.setContentsMargins(12, 12, 12, 12)
                self_w._sidebar_layout.setSpacing(16)
                
                self_w._populate_sidebar()
                
                scroll.setWidget(self_w._sidebar_content)
                sl.addWidget(scroll)
                main_layout.addWidget(self_w._sidebar)

                self_w._update_scrollbar_range()

            def _populate_sidebar(self_w):
                while self_w._sidebar_layout.count():
                    item = self_w._sidebar_layout.takeAt(0)
                    if item.widget(): item.widget().deleteLater()
                
                l = self_w._sidebar_layout
                t = self_w._theme

                # Recording / Files
                rec_box = QGroupBox("RECORDING")
                rl = QVBoxLayout(rec_box)
                f_btn = QPushButton("Open File...")
                f_btn.clicked.connect(self_w._on_open_file)
                d_btn = QPushButton("Open Folder...")
                d_btn.clicked.connect(self_w._on_open_folder)
                rl.addWidget(f_btn); rl.addWidget(d_btn)
                
                if self_w._folder_path:
                    fl_lbl = QLabel(f"Files in {self_w._folder_path.name}:")
                    fl_lbl.setStyleSheet(f"color:{t.muted}; font-size:10px;")
                    rl.addWidget(fl_lbl)
                    self_w._file_list_widget = QListWidget()
                    self_w._file_list_widget.setFixedHeight(120)
                    for f in self_w._files_in_folder:
                        self_w._file_list_widget.addItem(f.name)
                    self_w._file_list_widget.itemDoubleClicked.connect(self_w._on_file_selected)
                    rl.addWidget(self_w._file_list_widget)

                if self_w._recording:
                    a_btn = QPushButton("Analyze Signals")
                    a_btn.clicked.connect(self_w._on_analyze)
                    a_btn.setStyleSheet(f"background:{t.accent}; color:{t.bg}; font-weight:bold;")
                    rl.addWidget(a_btn)
                l.addWidget(rec_box)

                # ── TTL EVENTS panel (only when triggers are present) ───
                if self_w._recording and self_w._ttl_data:
                    ttl_box = QGroupBox("TTL EVENTS")
                    ttl_l = QVBoxLayout(ttl_box)
                    ttl_l.setSpacing(4)

                    # Summary line: total unique rise/fall counts across all TTL types
                    total_r = sum(len(ev["rise"]) for ev in self_w._ttl_data.values())
                    total_f = sum(len(ev["fall"]) for ev in self_w._ttl_data.values())
                    info_lbl = QLabel(f"TTL: {total_r} rise,  {total_f} fall")
                    info_lbl.setStyleSheet(f"color:{t.muted}; font-size:10px;")
                    ttl_l.addWidget(info_lbl)

                    def _make_ttl_cb(label: str, attr: str, checked: bool, tip: str):
                        cb = QCheckBox(label)
                        cb.setChecked(checked)
                        cb.setToolTip(tip)
                        # NoFocus: clicking does NOT steal keyboard focus, so Space
                        # continues to trigger play/pause via the window keyPressEvent.
                        cb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                        def _on_toggle(v, _attr=attr):
                            setattr(self_w, _attr, v)
                            self_w._rebuild_ttl_artists()
                            self_w._canvas.draw_idle()
                        cb.toggled.connect(_on_toggle)
                        return cb

                    for _lbl, _attr, _chk, _tip in [
                        ("TTL (rise → fall)", "_ttl_show_strips", self_w._ttl_show_strips,
                         "Semi-transparent span across each channel for every TTL HIGH period."),
                        ("Rising Edges",      "_ttl_show_rise",   self_w._ttl_show_rise,
                         "Dotted vertical line at each rising edge."),
                        ("Falling Edges",     "_ttl_show_fall",   self_w._ttl_show_fall,
                         "Dotted vertical line at each falling edge."),
                    ]:
                        _cb = _make_ttl_cb(_lbl, _attr, _chk, _tip)
                        ttl_l.addWidget(_cb)
                        self_w._lockable_widgets.append(_cb)

                    l.addWidget(ttl_box)

                # ── CLASSIFICATION panel ───────────────────────────────
                if self_w._recording:
                    cls_box = QGroupBox("CLASSIFICATION")
                    cls_l = QVBoxLayout(cls_box)
                    cls_l.setSpacing(6)

                    # Read defaults from session thresholds (or factory defaults)
                    thr = (
                        self_w._session.thresholds
                        if self_w._session
                        else AutoScoreThresholds()
                    )
                    self_w._thr_spins = {}

                    thresh_cfg = [
                        ("delta_wake", "Wake: delta <",  thr.delta_wake, 0, 1e7,   "µV²/Hz"),
                        ("delta_nrem", "NREM: delta >",  thr.delta_nrem, 0, 1e7,   "µV²/Hz"),
                        ("emg_wake",   "Wake: EMG >",    thr.emg_wake,   0, 1e6,   "µV"),
                        ("emg_nrem",   "NREM: EMG <",    thr.emg_nrem,   0, 1e6,   "µV"),
                        ("emg_rem",    "REM: EMG <",     thr.emg_rem,    0, 1e6,   "µV"),
                        ("td_rem",     "REM: T:D >",     thr.td_rem,     0, 1000,  ""),
                    ]

                    thr_grid = QGridLayout()
                    thr_grid.setSpacing(3)
                    for row, (key, label_text, default, lo, hi, unit) in enumerate(thresh_cfg):
                        lbl = QLabel(label_text)
                        lbl.setStyleSheet(f"color:{t.muted}; font-size:10px;")
                        spin = QDoubleSpinBox()
                        spin.setRange(lo, hi)
                        spin.setDecimals(1)
                        spin.setValue(default)
                        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
                        spin.setFixedWidth(96)
                        self_w._thr_spins[key] = spin
                        if self_w._thr_lines_active:
                            spin.valueChanged.connect(
                                lambda _, k=key: self_w._on_thr_spin_changed(k)
                            )
                        unit_lbl = QLabel(unit)
                        unit_lbl.setStyleSheet(f"color:{t.muted}; font-size:9px;")
                        thr_grid.addWidget(lbl,      row, 0)
                        thr_grid.addWidget(spin,     row, 1)
                        thr_grid.addWidget(unit_lbl, row, 2)
                    cls_l.addLayout(thr_grid)

                    if not self_w._analyzer:
                        note = QLabel("Run 'Analyze Signals' first")
                        note.setStyleSheet(f"color:{t.muted}; font-size:9px; font-style:italic;")
                        cls_l.addWidget(note)

                    run_btn = QPushButton("Run Classification")
                    run_btn.setEnabled(self_w._analyzer is not None)
                    run_btn.setStyleSheet(
                        f"background:{t.accent}; color:{t.bg}; font-weight:bold; padding:5px;"
                        if self_w._analyzer
                        else f"background:{t.border}; color:{t.muted}; padding:5px;"
                    )
                    run_btn.clicked.connect(self_w._on_run_classification)
                    cls_l.addWidget(run_btn)
                    l.addWidget(cls_box)

                # ── LABELING panel (only when session exists) ──────────
                if self_w._session:
                    lab_box = QGroupBox("LABELING")
                    lab_l = QVBoxLayout(lab_box)
                    lab_l.setSpacing(6)

                    # State counts
                    counts = self_w._session.state_counts()
                    durs = self_w._session.state_durations()
                    total_scored = sum(v for k, v in counts.items() if k != "U")
                    total_e = len(self_w._session.times)

                    counts_grid = QGridLayout()
                    counts_grid.setSpacing(2)
                    for row, state in enumerate(["W", "N", "R", "U"]):
                        c = counts[state]
                        d = durs[state]
                        color = STATE_COLORS[state]
                        sl = QLabel(f"  {state}")
                        sl.setStyleSheet(
                            f"color:{color}; font-weight:bold; font-size:11px;"
                        )
                        pct = 100 * c / total_e if total_e else 0
                        cl = QLabel(f"{c} epochs  {d/60:.1f} min  ({pct:.0f}%)")
                        cl.setStyleSheet(f"color:{t.muted}; font-size:10px;")
                        counts_grid.addWidget(sl, row, 0)
                        counts_grid.addWidget(cl, row, 1)
                    lab_l.addLayout(counts_grid)

                    sep = QFrame()
                    sep.setFrameShape(QFrame.Shape.HLine)
                    sep.setStyleSheet(f"color:{t.border};")
                    lab_l.addWidget(sep)

                    # Selected epoch info
                    sel_sorted = sorted(self_w._sel_indices)
                    if sel_sorted:
                        lo_i = sel_sorted[0]
                        hi_i = sel_sorted[-1]
                        n_sel = len(sel_sorted)
                        t0s = self_w._session.times[lo_i] - self_w._session.epoch_len / 2
                        t1s = self_w._session.times[hi_i] + self_w._session.epoch_len / 2
                        lbls = {str(self_w._session.labels[i]) for i in sel_sorted}
                        cur_lbl = "/".join(sorted(lbls))
                        sel_txt = (
                            f"{n_sel} epoch(s) selected  [{cur_lbl}]\n"
                            f"t = {t0s:.1f}s – {t1s:.1f}s"
                        )
                    else:
                        sel_txt = (
                            "Click hypnogram to select\n"
                            "Ctrl+click: multi  Shift+click: range"
                        )

                    sel_info = QLabel(sel_txt)
                    sel_info.setStyleSheet(f"color:{t.muted}; font-size:10px;")
                    sel_info.setWordWrap(True)
                    lab_l.addWidget(sel_info)

                    # State assignment buttons (W / N / R / U) + hotkey hint
                    state_row = QHBoxLayout()
                    for state in ["W", "N", "R", "U"]:
                        sbtn = QPushButton(state)
                        sbtn.setFixedWidth(44)
                        sbtn.setToolTip(f"Assign '{state}' to selected epoch(s)  [hotkey: {state}]")
                        sbtn.setStyleSheet(
                            f"background:{STATE_COLORS[state]}; color:#0d1117; "
                            f"font-weight:bold; font-size:13px; border-radius:3px;"
                        )
                        sbtn.clicked.connect(lambda _, s=state: self_w._on_assign_state(s))
                        state_row.addWidget(sbtn)
                    lab_l.addLayout(state_row)

                    # Undo / Redo
                    ur_row = QHBoxLayout()
                    undo_btn = QPushButton("↩ Undo")
                    undo_btn.setToolTip("Undo last label change  [Ctrl+Z]")
                    undo_btn.clicked.connect(self_w._on_undo)
                    redo_btn = QPushButton("↪ Redo")
                    redo_btn.setToolTip("Redo last undone change  [Ctrl+Y]")
                    redo_btn.clicked.connect(self_w._on_redo)
                    ur_row.addWidget(undo_btn)
                    ur_row.addWidget(redo_btn)
                    lab_l.addLayout(ur_row)

                    sep2 = QFrame()
                    sep2.setFrameShape(QFrame.Shape.HLine)
                    sep2.setStyleSheet(f"color:{t.border};")
                    lab_l.addWidget(sep2)

                    # Load session from JSON
                    load_sess_btn = QPushButton("Load Session...")
                    load_sess_btn.clicked.connect(self_w._on_load_session)
                    lab_l.addWidget(load_sess_btn)

                    # Save / Export
                    save_json_btn = QPushButton("Save Session (JSON)...")
                    save_json_btn.clicked.connect(self_w._on_save_session)
                    lab_l.addWidget(save_json_btn)

                    save_h5_btn = QPushButton("Save HDF5...")
                    save_h5_btn.setStyleSheet(
                        f"background:{t.accent}; color:{t.bg}; font-weight:bold; padding:5px;"
                    )
                    save_h5_btn.clicked.connect(self_w._on_save_h5)
                    lab_l.addWidget(save_h5_btn)

                    csv_btn = QPushButton("Export Hypnogram CSV...")
                    csv_btn.clicked.connect(self_w._on_export_csv)
                    lab_l.addWidget(csv_btn)

                    l.addWidget(lab_box)

                l.addStretch()

                # Bottom controls
                clear_btn = QPushButton("Clear All Channels")
                clear_btn.clicked.connect(self_w._on_clear_channels)
                l.addWidget(clear_btn)

                # Apply play-lock state to any newly created widgets
                self_w._set_controls_enabled(not self_w._playing)

            def _show_time_menu(self_w):
                menu = QMenu(self_w)
                
                # Window Width Submenu
                win_menu = menu.addMenu("Visible Window")
                for val in [5, 10, 30, 60]:
                    act = win_menu.addAction(f"{val}s")
                    act.triggered.connect(lambda _, v=val: self_w._on_window_changed(v))
                
                # Custom Window
                custom_act = menu.addAction("Custom Window...")
                custom_act.triggered.connect(self_w._on_custom_window)
                
                menu.addSeparator()
                
                # Units Submenu
                unit_menu = menu.addMenu("Time Units")
                group = QActionGroup(menu)
                for u in ["auto", "s", "m", "h"]:
                    act = unit_menu.addAction(u.capitalize())
                    act.setCheckable(True)
                    act.setChecked(self_w._time_unit == u)
                    act.triggered.connect(lambda _, unit=u: self_w._on_time_unit_changed(unit))
                    group.addAction(act)
                
                menu.exec(self_w._time_btn.mapToGlobal(QPoint(0, 0)))

            def _toggle_sidebar(self_w):
                self_w._sidebar.setVisible(self_w._sidebar_btn.isChecked())

            def _update_scrollbar_range(self_w):
                if not self_w._recording:
                    self_w._scrollbar.setRange(0, 0)
                else:
                    dur = self_w._recording.duration
                    max_v = max(0, int((dur - self_w._x_window) * _SCROLLBAR_RES))
                    self_w._scrollbar.setRange(0, max_v)
                    self_w._scrollbar.setPageStep(int(self_w._x_window * _SCROLLBAR_RES))

            def _rebuild_figure(self_w) -> None:
                t = self_w._theme
                fig = self_w._canvas.figure
                fig.clear()
                fig.patch.set_facecolor(t.bg)
                
                # Clear left panel
                while self_w._left_layout.count():
                    item = self_w._left_layout.takeAt(0)
                    if item.widget(): item.widget().deleteLater()
                self_w._yscale_spins = {}
                self_w._lockable_widgets = []  # reset on each figure rebuild

                self_w._vis_signals = [s for s in self_w._signal_data if s.name in self_w._visible]
                n = len(self_w._vis_signals)
                self_w._hypno_ax = None
                self_w._hypno_dirty = True  # axes recreated; force full redraw

                has_session = self_w._session is not None

                if n == 0 and not has_session:
                    self_w._axes = []; self_w._lines = []
                    self_w._canvas.draw_idle(); return

                n_rows = n + (1 if has_session else 0)
                if n_rows == 0:
                    self_w._axes = []; self_w._lines = []
                    self_w._canvas.draw_idle(); return

                if has_session:
                    height_ratios = [1.0] * n + [0.22]
                    all_axes = fig.subplots(
                        n_rows, 1, sharex=True,
                        gridspec_kw={"hspace": 0.05, "height_ratios": height_ratios}
                    )
                    all_axes_list = [all_axes] if n_rows == 1 else list(all_axes)
                    self_w._axes = all_axes_list[:n]
                    self_w._hypno_ax = all_axes_list[n]
                else:
                    raw_axes = fig.subplots(n, 1, sharex=True, gridspec_kw={"hspace": 0.05})
                    self_w._axes = [raw_axes] if n == 1 else list(raw_axes)

                self_w._lines = []
                # Adjust left margin to give room for channel-name y-axis labels
                fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.08)

                for ax, sig in zip(self_w._axes, self_w._vis_signals):
                    _apply_ax_style(ax, sig, t)
                    (line,) = ax.plot([], [], lw=0.7, color=sig.color)
                    self_w._lines.append(line)
                    
                    # Create per-channel control block in left panel
                    ctrl = QWidget()
                    cll = QVBoxLayout(ctrl)
                    cll.setContentsMargins(5, 5, 5, 5)
                    cll.setSpacing(2)
                    
                    row1 = QHBoxLayout()
                    lbl = QLabel(sig.label)
                    lbl.setStyleSheet(f"color:{sig.color}; font-weight:bold; font-size:11px;")
                    row1.addWidget(lbl)
                    row1.addSpacing(6)

                    x_btn = QToolButton()
                    x_btn.setText("−")
                    x_btn.setStyleSheet(
                        f"QToolButton {{ color:{t.text}; font-size:13px;"
                        f" font-weight:bold; padding:1px 5px; }}"
                    )
                    x_btn.setAutoRaise(False)
                    x_btn.setToolTip("Hide channel")
                    x_btn.clicked.connect(lambda _, n=sig.name: self_w._on_channel_removed(n))
                    row1.addWidget(x_btn)
                    row1.addStretch()
                    cll.addLayout(row1)
                    
                    # Amplitude Spinbox + Unit Selector
                    spin_row = QHBoxLayout()
                    spin = QDoubleSpinBox()
                    spin.setRange(1e-12, 1e12); spin.setDecimals(1)
                    spin.setValue(self_w._y_scales[sig.name])
                    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
                    spin.valueChanged.connect(lambda v, n=sig.name: self_w._on_yscale_changed(n, v))
                    self_w._yscale_spins[sig.name] = spin
                    spin_row.addWidget(spin, stretch=1)

                    unit_combo = QComboBox()
                    u_cat = _SIG_CATEGORIES.get(sig.name, "ratio")
                    u_list = _UNITS_BY_CAT.get(u_cat, [""])
                    unit_combo.addItems(u_list)
                    unit_combo.setCurrentText(sig.unit)
                    unit_combo.setFixedWidth(60)
                    unit_combo.currentTextChanged.connect(lambda u, n=sig.name: self_w._on_signal_unit_changed(n, u))
                    spin_row.addWidget(unit_combo)
                    cll.addLayout(spin_row)

                    opt_btn = QToolButton()
                    opt_btn.setText("Optimize Scale")
                    opt_btn.setStyleSheet("font-size:10px; padding:2px;")
                    opt_btn.clicked.connect(lambda _, n=sig.name: self_w._optimize_channel(n))
                    cll.addWidget(opt_btn)

                    self_w._left_layout.addWidget(ctrl)
                    # Track for play-lock
                    self_w._lockable_widgets.extend([spin, unit_combo, x_btn, opt_btn])
                
                # Add "Add Channel" button at the bottom of left panel
                if self_w._recording:
                    self_w._left_layout.addStretch()
                    add_btn = QToolButton()
                    add_btn.setText("+ Add Channel")
                    add_btn.setStyleSheet(f"color:{t.accent}; font-weight:bold; padding:5px; margin-top:10px;")
                    add_btn.clicked.connect(self_w._show_add_channel_menu)
                    self_w._left_layout.addWidget(add_btn)

                # ── Hypnogram axis styling ──────────────────────────────
                if self_w._hypno_ax is not None:
                    ha = self_w._hypno_ax
                    ha.set_facecolor(t.bg)
                    ha.set_yticks([])
                    ha.set_ylabel("Stage", color=t.muted, fontsize=7, labelpad=2)
                    for spine in ha.spines.values():
                        spine.set_color(t.border)
                    ha.tick_params(colors=t.muted, labelsize=8)
                    # Connect mouse click (disconnect first to avoid double-binding)
                    if self_w._hypno_cid is not None:
                        try:
                            self_w._canvas.mpl_disconnect(self_w._hypno_cid)
                        except Exception:
                            pass
                    self_w._hypno_cid = self_w._canvas.mpl_connect(
                        "button_press_event", self_w._on_canvas_click
                    )

                self_w._time_label.setText(f"Time ({self_w._time_unit})")
                self_w._draw(self_w._t0)
                self_w._rebuild_ttl_artists()
                if self_w._thr_lines_active and self_w._session is not None:
                    self_w._draw_threshold_lines()
                    self_w._connect_thr_drag_events()
                self_w._set_controls_enabled(not self_w._playing)
                self_w._canvas.draw_idle()

            def _set_controls_enabled(self_w, enabled: bool) -> None:
                """Enable or disable interactive controls (called when play state changes)."""
                for w in self_w._lockable_widgets:
                    try:
                        w.setEnabled(enabled)
                    except RuntimeError:
                        pass  # widget already deleted during a figure rebuild

            def _on_signal_unit_changed(self_w, name, new_unit):
                sig = next((s for s in self_w._signal_data if s.name == name), None)
                if not sig or sig.base_values is None: return
                
                # Update unit and recalculate values from base_values
                old_unit = sig.unit
                sig.unit = new_unit
                scale = _get_signal_scale(sig.name, new_unit)
                sig.values = sig.base_values * scale
                
                # Update y_scale to maintain relative zoom if possible
                old_scale = _get_signal_scale(sig.name, old_unit)
                self_w._y_scales[sig.name] *= (scale / old_scale)
                
                # Trigger rebuild to update spinbox suffixes and labels
                self_w._rebuild_figure()

            def _draw(self_w, t0: float) -> None:
                if not self_w._recording: return
                if not self_w._axes and self_w._hypno_ax is None: return
                dur = self_w._recording.duration
                t0 = float(np.clip(t0, 0, max(0, dur - self_w._x_window)))
                self_w._t0 = t0
                t1 = t0 + self_w._x_window
                
                if self_w._opt_dynamic and self_w._playing:
                    for sig in self_w._vis_signals:
                        yh = _auto_y_half(sig.values, (t0, t1), sig.times)
                        self_w._y_scales[sig.name] = yh
                        if sig.name in self_w._yscale_spins:
                            self_w._yscale_spins[sig.name].setValue(yh)

                for ax, sig, line in zip(self_w._axes, self_w._vis_signals, self_w._lines):
                    yh = self_w._y_scales[sig.name]
                    i0 = np.searchsorted(sig.times, t0, side="left")
                    i1 = np.searchsorted(sig.times, t1, side="right")
                    line.set_data(sig.times[max(0, i0-1):i1+1], sig.values[max(0, i0-1):i1+1])
                    ax.set_xlim(t0, t1)
                    ax.set_ylim(-yh, yh)
                
                fmt = _make_time_formatter(self_w._x_window, self_w._time_unit)
                if self_w._axes:
                    self_w._axes[-1].xaxis.set_major_formatter(fmt)
                self_w._time_label.setText(f"{self_w._x_window:.1f}s | {self_w._time_unit}")

                if self_w._hypno_ax is not None:
                    if self_w._hypno_dirty:
                        self_w._draw_hypnogram()
                    else:
                        # sharex=True keeps xlim in sync; just update the viewport line
                        self_w._hypno_ax.set_xlim(t0, t1)

                # Update TTL overlays to only show the visible slice (O(log N + K))
                if self_w._ttl_data:
                    self_w._refresh_ttl_visible(t0, t1)

                self_w._canvas.draw_idle()

            def _on_scroll(self_w, value: int):
                self_w._draw(value / _SCROLLBAR_RES)

            def _on_play_pause(self_w):
                self_w._playing = not self_w._playing
                self_w._play_btn.setText("⏸" if self_w._playing else "▶")
                self_w._set_controls_enabled(not self_w._playing)
                if self_w._playing: self_w._timer.start()
                else: self_w._timer.stop()

            def _on_play_tick(self_w):
                if not self_w._playing:
                    return
                if not self_w._recording:
                    self_w._on_play_pause()
                    return
                tick_s = self_w._timer.interval() / 1000.0  # timer interval in seconds
                dt = self_w._play_speed * tick_s
                new_t0 = self_w._t0 + dt
                max_t0 = max(0.0, self_w._recording.duration - self_w._x_window)
                at_end = new_t0 >= max_t0
                new_t0 = min(new_t0, max_t0)
                # Update scrollbar position without re-triggering _on_scroll → _draw
                self_w._scrollbar.blockSignals(True)
                self_w._scrollbar.setValue(int(new_t0 * _SCROLLBAR_RES))
                self_w._scrollbar.blockSignals(False)
                self_w._draw(new_t0)
                if at_end:
                    self_w._on_play_pause()

            def _on_speed_changed(self_w, v: int) -> None:
                # Logarithmic mapping: slider 1→1×, 50→~10×, 100→100×
                self_w._play_speed = 10 ** ((v - 1) / 50.0)
                spd = self_w._play_speed
                text = f"{spd:.1f}x" if spd < 10 else f"{spd:.0f}x"
                self_w._speed_label.setText(text)

            def _on_window_changed(self_w, v):
                self_w._x_window = float(v)
                self_w._update_scrollbar_range()
                self_w._draw(self_w._t0)

            def _on_custom_window(self_w):
                val, ok = QInputDialog.getDouble(
                    self_w, "Custom Window", "Window width (seconds):",
                    self_w._x_window, 0.1, 3600.0, 1
                )
                if ok:
                    self_w._on_window_changed(val)

            def _on_prev_page(self_w):
                new_t0 = max(0.0, self_w._t0 - self_w._x_window)
                self_w._scrollbar.setValue(int(new_t0 * _SCROLLBAR_RES))

            def _on_next_page(self_w):
                if not self_w._recording: return
                dur = self_w._recording.duration
                new_t0 = min(dur - self_w._x_window, self_w._t0 + self_w._x_window)
                self_w._scrollbar.setValue(int(max(0, new_t0) * _SCROLLBAR_RES))

            def _show_add_channel_menu(self_w):
                if not self_w._recording: return
                menu = QMenu(self_w)
                available = list(self_w._recording.channels)
                if self_w._analyzer: available += list(_DERIVED_KEYS)
                
                added_any = False
                for name in available:
                    if name in self_w._visible: continue
                    label, _ = _SIG_META.get(name, (name, ""))
                    act = menu.addAction(label)
                    act.triggered.connect(lambda _, n=name: self_w._on_channel_added(n))
                    added_any = True
                
                if not added_any:
                    menu.addAction("No more channels available").setEnabled(False)
                
                # Show menu at the button position
                menu.exec(QGuiApplication.focusWindow().cursor().pos())

            def _on_time_unit_changed(self_w, u):
                self_w._time_unit = u
                self_w._rebuild_figure()

            def _on_channel_added(self_w, name):
                self_w._visible.add(name)
                if name not in {s.name for s in self_w._signal_data}:
                    # Merge requested signals
                    req = list(self_w._requested_signals or [])
                    if name not in req: req.append(name)
                    self_w._requested_signals = req
                    self_w._init_data(self_w._recording, self_w._analyzer, req)
                
                self_w._rebuild_figure()
                self_w._populate_sidebar()

            def _on_channel_removed(self_w, name):
                self_w._visible.discard(name)
                if self_w._requested_signals and name in self_w._requested_signals:
                    self_w._requested_signals.remove(name)
                self_w._rebuild_figure()
                self_w._populate_sidebar()

            def _on_clear_channels(self_w):
                self_w._recording = None
                self_w._analyzer = None
                self_w._signal_data = []
                self_w._requested_signals = None
                self_w._visible = set()
                self_w._y_scales = {}
                self_w._t0 = 0.0
                self_w._session = None
                self_w._sel_indices = set()
                self_w._sel_anchor = None
                self_w._hypno_ax = None
                self_w._ttl_data = {}
                self_w._ttl_artists = []
                self_w._ttl_strip_colls = []
                self_w._ttl_rise_colls = []
                self_w._ttl_fall_colls = []
                self_w._ttl_strips_r = np.array([])
                self_w._ttl_strips_f = np.array([])
                self_w._ttl_rises_all = np.array([])
                self_w._ttl_falls_all = np.array([])
                self_w.setWindowTitle("Sleep Scope — No Data")
                self_w._update_scrollbar_range()
                if hasattr(self_w, "_scrollbar"):
                    self_w._scrollbar.setValue(0)
                self_w._rebuild_figure()
                self_w._populate_sidebar()

            def _on_yscale_changed(self_w, name, v):
                self_w._y_scales[name] = float(v)
                self_w._draw(self_w._t0)

            def _optimize_channel(self_w, name):
                sig = next((s for s in self_w._signal_data if s.name == name), None)
                if not sig: return
                tr = (self_w._t0, self_w._t0 + self_w._x_window)
                yh = _auto_y_half(sig.values, tr, sig.times)
                self_w._y_scales[name] = yh
                if name in self_w._yscale_spins:
                    self_w._yscale_spins[name].setValue(yh)
                self_w._draw(self_w._t0)

            def _optimize_amplitude(self_w, local: bool = True, redraw: bool = True):
                tr = (self_w._t0, self_w._t0 + self_w._x_window) if local else None
                for sig in self_w._signal_data:
                    yh = _auto_y_half(sig.values, tr, sig.times)
                    self_w._y_scales[sig.name] = yh
                if redraw: self_w._rebuild_figure()

            def _on_opt_dyn_toggled(self_w, c): self_w._opt_dynamic = c

            def _on_open_file(self_w):
                p, _ = QFileDialog.getOpenFileName(self_w, "Open EDF", "", "EDF (*.edf)")
                if p: self_w._load(Path(p))

            def _on_open_folder(self_w):
                p = QFileDialog.getExistingDirectory(self_w, "Open Folder")
                if not p: return
                self_w._folder_path = Path(p)
                self_w._files_in_folder = sorted(list(self_w._folder_path.glob("*_export.edf")))
                self_w._populate_sidebar()

            def _on_file_selected(self_w, item):
                if self_w._folder_path:
                    p = self_w._folder_path / item.text()
                    self_w._load(p)

            def _on_analyze(self_w):
                if self_w._recording:
                    self_w._analyzer = SleepAnalyzer(self_w._recording)
                    self_w._init_data(self_w._recording, self_w._analyzer, self_w._requested_signals)
                    self_w._populate_sidebar()
                    self_w._rebuild_figure()

            def _load(self_w, p: Path):
                if p.is_dir():
                    edfs = list(p.glob("*_export.edf"))
                    if not edfs: return
                    p = edfs[0]

                if self_w._playing:
                    self_w._on_play_pause()
                self_w._t0 = 0.0

                # Clear any existing session when loading a new file
                self_w._session = None
                self_w._sel_indices = set()
                self_w._sel_anchor = None
                self_w._hypno_ax = None

                rec = SleepRecording.from_edf(p)
                self_w._requested_signals = None # Reset selection for new file
                self_w._init_data(rec, None, None)
                self_w._populate_sidebar()
                self_w._update_scrollbar_range()
                if hasattr(self_w, "_scrollbar"):
                    self_w._scrollbar.setValue(0)
                self_w._rebuild_figure()

            # ================================================================
            # Scoring methods
            # ================================================================

            def _draw_hypnogram(self_w) -> None:
                """Redraw the hypnogram strip using pcolormesh (O(1) draw calls)."""
                ax = self_w._hypno_ax
                if ax is None or self_w._session is None:
                    return

                session = self_w._session
                n = len(session.labels)
                L = session.epoch_len
                th = self_w._theme

                ax.cla()
                ax.set_facecolor(th.bg)
                ax.set_yticks([])
                ax.set_ylabel("Stage", color=th.muted, fontsize=7, labelpad=2)
                for spine in ax.spines.values():
                    spine.set_color(th.border)

                if n == 0:
                    return

                # ── pcolormesh: n+1 x-edges, 2 y-edges, Z shape (1, n) ──
                # Epoch k spans [k*L, (k+1)*L]; centres at (k+0.5)*L
                x_edges = np.arange(n + 1, dtype=np.float64) * L
                y_edges = np.array([0.0, 1.0])

                state_order = ["W", "N", "R", "U"]
                state_to_int = {s: i for i, s in enumerate(state_order)}
                color_vals = np.array(
                    [state_to_int.get(str(lbl), 3) for lbl in session.labels],
                    dtype=np.float64,
                ).reshape(1, n)

                cmap = _mcolors.ListedColormap(
                    [STATE_COLORS["W"], STATE_COLORS["N"],
                     STATE_COLORS["R"], STATE_COLORS["U"]]
                )
                ax.pcolormesh(
                    x_edges, y_edges, color_vals,
                    cmap=cmap, vmin=-0.5, vmax=3.5,
                    shading="flat",
                )

                # ── Highlight selected epochs ─────────────────────────────
                for idx in self_w._sel_indices:
                    ax.add_patch(_Rect(
                        (idx * L, 0.0), L, 1.0,
                        linewidth=0, facecolor="white", alpha=0.3, zorder=5,
                    ))
                # Draw a border around the contiguous blocks within the selection
                if self_w._sel_indices:
                    sel_sorted = sorted(self_w._sel_indices)
                    lo = sel_sorted[0]
                    hi = sel_sorted[-1]
                    ax.add_patch(_Rect(
                        (lo * L, 0.0), (hi - lo + 1) * L, 1.0,
                        linewidth=1.5, edgecolor="white", facecolor="none", zorder=6,
                    ))

                # ── Restore x-limits to match current scroll position ────
                ax.set_xlim(self_w._t0, self_w._t0 + self_w._x_window)
                ax.set_ylim(0, 1)

                # ── Legend ───────────────────────────────────────────────
                handles = [
                    _Patch(facecolor=STATE_COLORS[s], label=s, linewidth=0)
                    for s in state_order
                ]
                ax.legend(
                    handles=handles, loc="upper right", fontsize=6,
                    framealpha=0.5, ncol=4, handlelength=0.9, handleheight=0.9,
                    borderpad=0.3, labelspacing=0.2, columnspacing=0.5,
                    labelcolor=th.text,
                )
                self_w._hypno_dirty = False

            def _on_canvas_click(self_w, event) -> None:
                """Handle mouse click on the hypnogram strip to select epoch(s).

                Plain click  : select single epoch (clears previous selection)
                Shift+click  : extend selection as contiguous range from anchor
                Ctrl+click   : toggle individual epoch in/out of selection
                """
                if self_w._session is None:
                    return
                if event.inaxes is not self_w._hypno_ax:
                    return
                if event.xdata is None:
                    return

                clicked_idx = self_w._session.epoch_index(event.xdata)

                modifiers = QApplication.keyboardModifiers()
                shift_held = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                ctrl_held  = bool(modifiers & Qt.KeyboardModifier.ControlModifier)

                if ctrl_held:
                    # Toggle individual epoch without clearing existing selection
                    if clicked_idx in self_w._sel_indices:
                        self_w._sel_indices.discard(clicked_idx)
                    else:
                        self_w._sel_indices.add(clicked_idx)
                    self_w._sel_anchor = clicked_idx
                elif shift_held and self_w._sel_anchor is not None:
                    # Contiguous range from anchor to clicked
                    lo = min(self_w._sel_anchor, clicked_idx)
                    hi = max(self_w._sel_anchor, clicked_idx)
                    self_w._sel_indices = set(range(lo, hi + 1))
                else:
                    # Plain click: single epoch
                    self_w._sel_indices = {clicked_idx}
                    self_w._sel_anchor  = clicked_idx

                self_w._hypno_dirty = True
                self_w._draw(self_w._t0)
                self_w._populate_sidebar()

            def _on_run_classification(self_w) -> None:
                """Read threshold spinboxes, run auto-score, rebuild figure."""
                if not self_w._recording or not self_w._analyzer:
                    return

                features = self_w._analyzer.compute_all_features()

                thr = AutoScoreThresholds(
                    delta_wake = self_w._thr_spins["delta_wake"].value(),
                    delta_nrem = self_w._thr_spins["delta_nrem"].value(),
                    emg_wake   = self_w._thr_spins["emg_wake"].value(),
                    emg_nrem   = self_w._thr_spins["emg_nrem"].value(),
                    emg_rem    = self_w._thr_spins["emg_rem"].value(),
                    td_rem     = self_w._thr_spins["td_rem"].value(),
                )

                session = ScoringSession(
                    self_w._recording,
                    epoch_len=self_w._analyzer.epoch_len,
                )
                session.auto_score(features, thr)
                self_w._session = session
                self_w._sel_indices = set()
                self_w._sel_anchor  = None

                # Activate threshold reference lines, then rebuild
                self_w._thr_lines_active = True
                self_w._rebuild_figure()
                self_w._populate_sidebar()

            def _on_assign_state(self_w, state: str) -> None:
                """Assign *state* to the currently selected epoch(s)."""
                if self_w._session is None or not self_w._sel_indices:
                    return
                # Single undo entry for the whole multi-epoch assignment
                self_w._session._push_undo()
                for idx in self_w._sel_indices:
                    self_w._session.labels[idx] = state
                self_w._hypno_dirty = True
                self_w._draw(self_w._t0)
                self_w._populate_sidebar()

            def _draw_threshold_lines(self_w) -> None:
                """Draw draggable dotted reference lines at classification thresholds.

                Lines can be dragged vertically; the corresponding spinbox in the
                CLASSIFICATION panel updates live and vice-versa.
                """
                # Remove previously drawn lines and labels
                for line_obj, text_obj, _, _, _ in self_w._thr_line_refs:
                    try: line_obj.remove()
                    except Exception: pass
                    try: text_obj.remove()
                    except Exception: pass
                self_w._thr_line_refs = []

                if self_w._session is None:
                    return
                thr = self_w._session.thresholds

                _delta_bw = 3.5  # BANDS["delta"] = (0.5, 4.0)
                _delta_to_base = _delta_bw / 1e12   # µV²/Hz → V²
                _emg_to_base = 1.0 / 1e6            # µV → V

                for ax, sig in zip(self_w._axes, self_w._vis_signals):
                    lines_spec: list = []  # (display_val, color, label, spin_key, factor)

                    if sig.name == "delta_power":
                        scale = _get_signal_scale(sig.name, sig.unit)
                        factor = 1.0 / (_delta_to_base * scale)  # display → µV²/Hz
                        lines_spec = [
                            (thr.delta_wake * _delta_to_base * scale,
                             STATE_COLORS["W"], f"W: δ<{thr.delta_wake:.0f}",
                             "delta_wake", factor),
                            (thr.delta_nrem * _delta_to_base * scale,
                             STATE_COLORS["N"], f"N: δ>{thr.delta_nrem:.0f}",
                             "delta_nrem", factor),
                        ]
                    elif sig.name == "emg_rms":
                        scale = _get_signal_scale(sig.name, sig.unit)
                        factor = 1.0 / (_emg_to_base * scale)   # display → µV
                        lines_spec = [
                            (thr.emg_wake * _emg_to_base * scale,
                             STATE_COLORS["W"], f"W: EMG>{thr.emg_wake:.1f}",
                             "emg_wake", factor),
                            (thr.emg_nrem * _emg_to_base * scale,
                             STATE_COLORS["N"], f"N: EMG<{thr.emg_nrem:.1f}",
                             "emg_nrem", factor),
                            (thr.emg_rem  * _emg_to_base * scale,
                             STATE_COLORS["R"], f"R: EMG<{thr.emg_rem:.1f}",
                             "emg_rem", factor),
                        ]
                    elif sig.name == "td_ratio":
                        lines_spec = [
                            (thr.td_rem, STATE_COLORS["R"], f"R: T:D>{thr.td_rem:.1f}",
                             "td_rem", 1.0),
                        ]

                    for val, color, label, spin_key, factor in lines_spec:
                        line = ax.axhline(
                            val, linestyle=":", linewidth=1.5,
                            color=color, alpha=0.85, zorder=3,
                        )
                        txt = ax.text(
                            0.995, val, label,
                            transform=ax.get_yaxis_transform(),
                            color=color, fontsize=7, va="bottom", ha="right",
                            alpha=0.85, clip_on=True,
                        )
                        self_w._thr_line_refs.append((line, txt, spin_key, ax, factor))

            # ── Draggable threshold lines ──────────────────────────────

            def _connect_thr_drag_events(self_w) -> None:
                """Bind/refresh mouse events for dragging threshold reference lines."""
                for cid in self_w._drag_cids:
                    try: self_w._canvas.mpl_disconnect(cid)
                    except Exception: pass
                self_w._drag_cids = []
                self_w._drag_line = None
                self_w._drag_text = None
                self_w._drag_spin_key = None
                self_w._drag_display_to_val = None
                if not self_w._thr_line_refs:
                    return
                self_w._drag_cids = [
                    self_w._canvas.mpl_connect("button_press_event",   self_w._on_thr_press),
                    self_w._canvas.mpl_connect("motion_notify_event",  self_w._on_thr_move),
                    self_w._canvas.mpl_connect("button_release_event", self_w._on_thr_release),
                ]

            def _on_thr_press(self_w, event) -> None:
                if event.button != 1 or event.inaxes is None or event.ydata is None:
                    return
                if event.inaxes is self_w._hypno_ax:
                    return
                ax = event.inaxes
                bbox = ax.get_window_extent()
                ylim = ax.get_ylim()
                if ylim[1] == ylim[0] or bbox.height == 0:
                    return
                PIXEL_TOL = 8

                def data_to_px(y: float) -> float:
                    return bbox.y0 + (y - ylim[0]) / (ylim[1] - ylim[0]) * bbox.height

                click_px = data_to_px(event.ydata)
                best_dist = float("inf")
                best: tuple | None = None
                for line, txt, spin_key, line_ax, factor in self_w._thr_line_refs:
                    if line_ax is not ax:
                        continue
                    dist = abs(data_to_px(float(line.get_ydata()[0])) - click_px)
                    if dist < PIXEL_TOL and dist < best_dist:
                        best_dist = dist
                        best = (line, txt, spin_key, factor)
                if best:
                    self_w._drag_line, self_w._drag_text, self_w._drag_spin_key, self_w._drag_display_to_val = best

            def _on_thr_move(self_w, event) -> None:
                if self_w._drag_line is None or event.inaxes is None or event.ydata is None:
                    return
                spin_key = self_w._drag_spin_key
                factor = self_w._drag_display_to_val
                if spin_key is None or factor is None:
                    return
                y_new = event.ydata

                # Clamp to spinbox range
                if spin_key in self_w._thr_spins:
                    spin = self_w._thr_spins[spin_key]
                    thr_val = max(spin.minimum(), min(spin.maximum(), y_new * factor))
                    y_new = thr_val / factor
                else:
                    thr_val = y_new * factor

                # Move line
                self_w._drag_line.set_ydata([y_new, y_new])

                # Update text label
                _label_fmt = {
                    "delta_wake": f"W: δ<{thr_val:.0f}",
                    "delta_nrem": f"N: δ>{thr_val:.0f}",
                    "emg_wake":   f"W: EMG>{thr_val:.1f}",
                    "emg_nrem":   f"N: EMG<{thr_val:.1f}",
                    "emg_rem":    f"R: EMG<{thr_val:.1f}",
                    "td_rem":     f"R: T:D>{thr_val:.1f}",
                }
                if self_w._drag_text is not None:
                    self_w._drag_text.set_position((0.995, y_new))
                    self_w._drag_text.set_text(_label_fmt.get(spin_key, ""))

                # Update spinbox silently (no re-draw loop)
                if spin_key in self_w._thr_spins:
                    self_w._thr_spins[spin_key].blockSignals(True)
                    self_w._thr_spins[spin_key].setValue(thr_val)
                    self_w._thr_spins[spin_key].blockSignals(False)

                # Keep session thresholds in sync
                if self_w._session is not None:
                    thr = self_w._session.thresholds
                    kw = {
                        "delta_wake": thr.delta_wake, "delta_nrem": thr.delta_nrem,
                        "emg_wake":   thr.emg_wake,   "emg_nrem":   thr.emg_nrem,
                        "emg_rem":    thr.emg_rem,     "td_rem":     thr.td_rem,
                    }
                    if spin_key in kw:
                        kw[spin_key] = thr_val
                    self_w._session.thresholds = AutoScoreThresholds(**kw)

                self_w._canvas.draw_idle()

            def _on_thr_release(self_w, _event) -> None:
                self_w._drag_line = None
                self_w._drag_text = None
                self_w._drag_spin_key = None
                self_w._drag_display_to_val = None

            def _on_thr_spin_changed(self_w, key: str) -> None:
                """Redraw threshold lines when a spinbox is edited manually."""
                if not self_w._thr_lines_active or self_w._session is None:
                    return
                thr = self_w._session.thresholds
                kw = {
                    "delta_wake": thr.delta_wake, "delta_nrem": thr.delta_nrem,
                    "emg_wake":   thr.emg_wake,   "emg_nrem":   thr.emg_nrem,
                    "emg_rem":    thr.emg_rem,     "td_rem":     thr.td_rem,
                }
                if key in self_w._thr_spins and key in kw:
                    kw[key] = self_w._thr_spins[key].value()
                self_w._session.thresholds = AutoScoreThresholds(**kw)
                self_w._draw_threshold_lines()
                self_w._canvas.draw_idle()

            # ── TTL overlays ───────────────────────────────────────────

            def _rebuild_ttl_artists(self_w) -> None:
                """Create persistent (initially empty) TTL overlay collections and cache data.

                Called once after ``_rebuild_figure()`` and on checkbox toggle.
                Collections are created empty and filled per-frame by
                ``_refresh_ttl_visible()`` — only the slice visible in the
                current x-window is ever passed to the renderer, keeping
                scroll/play performance O(visible events) instead of O(total).
                """
                from matplotlib.collections import PolyCollection as _PolyC, LineCollection as _LineC

                for art in self_w._ttl_artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                self_w._ttl_artists = []
                self_w._ttl_strip_colls = []
                self_w._ttl_rise_colls = []
                self_w._ttl_fall_colls = []

                if not self_w._axes or not self_w._ttl_data:
                    self_w._ttl_strips_r = np.array([])
                    self_w._ttl_strips_f = np.array([])
                    self_w._ttl_rises_all = np.array([])
                    self_w._ttl_falls_all = np.array([])
                    return
                if not (self_w._ttl_show_strips or self_w._ttl_show_rise or self_w._ttl_show_fall):
                    return

                _STRIP_COLOR = "#e3b341"   # amber  — TTL HIGH period fill
                _RISE_COLOR  = "#3fb950"   # green  — rising-edge marker
                _FALL_COLOR  = "#ff7b72"   # red    — falling-edge marker

                # ── Pre-compute and cache sorted arrays (once per rebuild) ──
                all_rise = np.sort(np.concatenate(
                    [ev["rise"] for ev in self_w._ttl_data.values()]
                ))
                all_fall = np.sort(np.concatenate(
                    [ev["fall"] for ev in self_w._ttl_data.values()]
                ))
                self_w._ttl_rises_all = all_rise
                self_w._ttl_falls_all = all_fall

                # Pair each rise with its immediately following fall
                strips_r, strips_f = [], []
                for r in all_rise:
                    later = all_fall[all_fall > r]
                    f = float(later[0]) if len(later) > 0 else float("inf")
                    if np.isfinite(f):
                        strips_r.append(float(r))
                        strips_f.append(float(f))
                self_w._ttl_strips_r = np.array(strips_r)
                self_w._ttl_strips_f = np.array(strips_f)

                has_strips = self_w._ttl_show_strips and len(strips_r) > 0
                has_rise   = self_w._ttl_show_rise   and len(all_rise) > 0
                has_fall   = self_w._ttl_show_fall   and len(all_fall) > 0

                # ── Create one empty collection per axis per type ──────────
                for ax in self_w._axes:
                    xform = ax.get_xaxis_transform()  # x=data coords, y=axes coords

                    if has_strips:
                        coll = _PolyC(
                            [], facecolors=_STRIP_COLOR, alpha=0.15,
                            linewidth=0, zorder=1, transform=xform,
                        )
                        ax.add_collection(coll)
                        self_w._ttl_strip_colls.append(coll)
                        self_w._ttl_artists.append(coll)
                    else:
                        self_w._ttl_strip_colls.append(None)

                    if has_rise:
                        coll = _LineC(
                            [], colors=_RISE_COLOR,
                            linestyles=":", linewidth=1.0, alpha=0.85,
                            zorder=2, transform=xform,
                        )
                        ax.add_collection(coll)
                        self_w._ttl_rise_colls.append(coll)
                        self_w._ttl_artists.append(coll)
                    else:
                        self_w._ttl_rise_colls.append(None)

                    if has_fall:
                        coll = _LineC(
                            [], colors=_FALL_COLOR,
                            linestyles=":", linewidth=1.0, alpha=0.85,
                            zorder=2, transform=xform,
                        )
                        ax.add_collection(coll)
                        self_w._ttl_fall_colls.append(coll)
                        self_w._ttl_artists.append(coll)
                    else:
                        self_w._ttl_fall_colls.append(None)

                # Populate for the current scroll position
                self_w._refresh_ttl_visible(self_w._t0, self_w._t0 + self_w._x_window)

            def _refresh_ttl_visible(self_w, t0: float, t1: float) -> None:
                """Update TTL collections to only contain events in [t0, t1].

                Called on every ``_draw`` tick — uses ``np.searchsorted`` so the
                cost is O(log N + K) where K is the number of visible events.
                """
                if not (self_w._ttl_strip_colls or self_w._ttl_rise_colls or self_w._ttl_fall_colls):
                    return

                # ── Strips (overlap: r < t1 and f > t0) ───────────────────
                if self_w._ttl_strip_colls and len(self_w._ttl_strips_r):
                    sr = self_w._ttl_strips_r
                    sf = self_w._ttl_strips_f
                    mask = (sr < t1) & (sf > t0)
                    vis_r = sr[mask]
                    vis_f = sf[mask]
                    strip_verts = [
                        [(r, 0), (f, 0), (f, 1), (r, 1)]
                        for r, f in zip(vis_r.tolist(), vis_f.tolist())
                    ]
                    for coll in self_w._ttl_strip_colls:
                        if coll is not None:
                            coll.set_paths(strip_verts)

                # ── Rising edges ───────────────────────────────────────────
                if self_w._ttl_rise_colls and len(self_w._ttl_rises_all):
                    ar = self_w._ttl_rises_all
                    i0 = int(np.searchsorted(ar, t0, side="left"))
                    i1 = int(np.searchsorted(ar, t1, side="right"))
                    vis = ar[i0:i1]
                    segs = [[(float(t), 0), (float(t), 1)] for t in vis]
                    for coll in self_w._ttl_rise_colls:
                        if coll is not None:
                            coll.set_segments(segs)

                # ── Falling edges ──────────────────────────────────────────
                if self_w._ttl_fall_colls and len(self_w._ttl_falls_all):
                    af = self_w._ttl_falls_all
                    i0 = int(np.searchsorted(af, t0, side="left"))
                    i1 = int(np.searchsorted(af, t1, side="right"))
                    vis = af[i0:i1]
                    segs = [[(float(t), 0), (float(t), 1)] for t in vis]
                    for coll in self_w._ttl_fall_colls:
                        if coll is not None:
                            coll.set_segments(segs)

            # ── Help dialog ────────────────────────────────────────────

            def _show_help_dialog(self_w) -> None:
                from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea, QWidget
                dlg = QDialog(self_w)
                dlg.setWindowTitle("How to Score Sleep")
                dlg.setFixedWidth(460)
                dlg.setMaximumHeight(600)
                outer = QVBoxLayout(dlg)
                outer.setContentsMargins(0, 0, 0, 0)

                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setFrameShape(QFrame.Shape.NoFrame)
                inner = QWidget()
                layout = QVBoxLayout(inner)
                layout.setContentsMargins(16, 16, 16, 8)
                layout.setSpacing(10)

                t = self_w._theme
                # On macOS, Qt maps Cmd to ControlModifier — show the right key name
                _mod = "Cmd" if sys.platform == "darwin" else "Ctrl"
                steps = [
                    ("1  Load a recording",
                     f"<b>{_mod}+E</b> — open file  |  <b>{_mod}+O</b> — open folder<br>"
                     "Or use Open File… / Open Folder… in the RECORDING sidebar panel."),
                    ("2  Analyze signals",
                     "Click <b>Analyze Signals</b> to compute EEG power bands, EMG RMS, and the T:D ratio. Required before classification."),
                    ("3  Set thresholds &amp; classify",
                     "Adjust Wake / NREM / REM thresholds in the <b>CLASSIFICATION</b> panel, then click <b>Run Classification</b>. "
                     "Dotted lines appear on the δ-power, EMG RMS, and T:D ratio traces."),
                    ("4  Adjust thresholds interactively",
                     "Drag any dotted reference line up or down — the spinbox updates live. "
                     "Or edit the spinbox directly and the line moves."),
                    ("5  Review the hypnogram",
                     "Click an epoch in the colour strip to select it.<br>"
                     "• <b>Shift+click</b> for a contiguous range<br>"
                     f"• <b>{_mod}+click</b> to toggle individual epochs"),
                    ("6  Label epochs",
                     f"<b>W / N / R / U</b> — assign Wake, NREM, REM, or Unscored to selected epochs.<br>"
                     f"<b>{_mod}+Z / {_mod}+Y</b> — undo / redo."),
                    ("7  Navigate",
                     "<b>Space</b> — play / pause<br>"
                     "<b>[ / ]</b> or <b>PageUp / PageDown</b> — page back / forward<br>"
                     "<b>← / →</b> — fine scroll (10 % of window)<br>"
                     "<b>Mouse wheel</b> — scroll signals left / right"),
                    ("8  Save results",
                     "<b>Save Session (JSON)</b> to resume later, <b>Export Hypnogram CSV</b>, "
                     "or <b>Save HDF5</b> for downstream analysis."),
                ]

                for title, body in steps:
                    title_lbl = QLabel(f"<b style='font-size:12px'>{title}</b>")
                    body_lbl = QLabel(f"<span style='color:{t.muted}'>{body}</span>")
                    body_lbl.setWordWrap(True)
                    body_lbl.setTextFormat(Qt.TextFormat.RichText)
                    layout.addWidget(title_lbl)
                    layout.addWidget(body_lbl)

                layout.addStretch()
                scroll.setWidget(inner)
                outer.addWidget(scroll)

                close_btn = QPushButton("Got it")
                close_btn.setFixedHeight(34)
                close_btn.clicked.connect(dlg.accept)
                outer.addWidget(close_btn)
                dlg.exec()

            def _on_toggle_theme(self_w) -> None:
                """Switch between dark and light themes."""
                new_theme = LIGHT_THEME if self_w._theme.name == "dark" else DARK_THEME
                self_w._apply_theme(new_theme)
                self_w._rebuild_figure()
                self_w._populate_sidebar()

            def _on_undo(self_w) -> None:
                if self_w._session and self_w._session.undo():
                    self_w._hypno_dirty = True
                    self_w._draw(self_w._t0)
                    self_w._populate_sidebar()

            def _on_redo(self_w) -> None:
                if self_w._session and self_w._session.redo():
                    self_w._hypno_dirty = True
                    self_w._draw(self_w._t0)
                    self_w._populate_sidebar()

            def _on_save_h5(self_w) -> None:
                if not self_w._recording or not self_w._session:
                    return
                default_name = (
                    f"{self_w._recording.animal_id}_"
                    f"{self_w._recording.start_datetime}_dataset.h5"
                    if self_w._recording.start_datetime
                    else f"{self_w._recording.animal_id}_dataset.h5"
                )
                path, _ = QFileDialog.getSaveFileName(
                    self_w, "Save HDF5 Dataset", default_name, "HDF5 (*.h5)"
                )
                if not path:
                    return
                from sleep_tools.io import save_to_h5
                save_to_h5(
                    self_w._recording,
                    path,
                    analyzer=self_w._analyzer,
                    session=self_w._session,
                    overwrite=True,
                )

            def _on_export_csv(self_w) -> None:
                if not self_w._session:
                    return
                default_name = (
                    f"{self_w._session.recording.animal_id}_hypnogram.csv"
                )
                path, _ = QFileDialog.getSaveFileName(
                    self_w, "Export Hypnogram CSV", default_name, "CSV (*.csv)"
                )
                if path:
                    self_w._session.to_csv(path)

            def _on_save_session(self_w) -> None:
                if not self_w._session:
                    return
                default_name = (
                    f"{self_w._session.recording.animal_id}_"
                    f"{self_w._session.recording.start_datetime}_sleep_scores.json"
                    if self_w._session.recording.start_datetime
                    else f"{self_w._session.recording.animal_id}_sleep_scores.json"
                )
                path, _ = QFileDialog.getSaveFileName(
                    self_w, "Save Scoring Session", default_name, "JSON (*.json)"
                )
                if path:
                    self_w._session.save(path)

            def _on_load_session(self_w) -> None:
                if not self_w._recording:
                    return
                path, _ = QFileDialog.getOpenFileName(
                    self_w, "Load Scoring Session", "", "JSON (*.json)"
                )
                if not path:
                    return
                try:
                    self_w._session = ScoringSession.load(path, self_w._recording)
                    self_w._sel_indices = set()
                    self_w._sel_anchor  = None
                    self_w._rebuild_figure()
                    self_w._populate_sidebar()
                except Exception as exc:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.warning(self_w, "Load Error", str(exc))

            def eventFilter(self_w, obj, event):  # type: ignore[override]
                """Forward canvas key/wheel events so our handlers always fire."""
                from PySide6.QtCore import QEvent
                if obj is self_w._canvas:
                    if event.type() == QEvent.Type.KeyPress:
                        self_w.keyPressEvent(event)
                        return True
                    if event.type() == QEvent.Type.Wheel:
                        # Disable wheel scroll while playing — let the play timer drive position.
                        if self_w._playing:
                            return True
                        # Proportional scroll: 120 units = one mouse-wheel notch = 15% of window.
                        # Trackpad sends many small deltas so it accumulates naturally and stops
                        # immediately when the gesture ends (no fixed jumps).
                        delta = event.angleDelta().y()
                        step = (delta / 120.0) * self_w._x_window * 0.15
                        new_t0 = self_w._t0 - step   # +delta → scroll left
                        new_t0 = max(0.0, new_t0)
                        if self_w._recording:
                            max_t0 = max(0.0, self_w._recording.duration - self_w._x_window)
                            new_t0 = min(max_t0, new_t0)
                        self_w._scrollbar.setValue(int(new_t0 * _SCROLLBAR_RES))
                        return True
                return super().eventFilter(obj, event)

            def keyPressEvent(self_w, event) -> None:  # type: ignore[override]
                """Keyboard shortcuts for playback, navigation, and staging."""
                key  = event.key()
                mods = event.modifiers()
                Ctrl = Qt.KeyboardModifier.ControlModifier

                if key == Qt.Key.Key_Z and (mods & Ctrl):
                    self_w._on_undo(); return
                if key == Qt.Key.Key_Y and (mods & Ctrl):
                    self_w._on_redo(); return
                if key == Qt.Key.Key_O and (mods & Ctrl):
                    self_w._on_open_folder(); return
                if key == Qt.Key.Key_E and (mods & Ctrl):
                    self_w._on_open_file(); return

                # Play / pause
                if key == Qt.Key.Key_Space:
                    self_w._on_play_pause(); return

                # Page navigation: [ / ] and PageUp / PageDown
                if key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_PageUp):
                    self_w._on_prev_page(); return
                if key in (Qt.Key.Key_BracketRight, Qt.Key.Key_PageDown):
                    self_w._on_next_page(); return

                # Fine scroll: ← / → (10 % of visible window)
                if key == Qt.Key.Key_Left:
                    step = self_w._x_window * 0.1
                    new_t0 = max(0.0, self_w._t0 - step)
                    self_w._scrollbar.setValue(int(new_t0 * _SCROLLBAR_RES))
                    return
                if key == Qt.Key.Key_Right and self_w._recording:
                    step = self_w._x_window * 0.1
                    max_t0 = max(0.0, self_w._recording.duration - self_w._x_window)
                    new_t0 = min(max_t0, self_w._t0 + step)
                    self_w._scrollbar.setValue(int(new_t0 * _SCROLLBAR_RES))
                    return

                # Sleep stage assignment (W / N / R / U)
                if self_w._session is not None:
                    _state_keys = {
                        Qt.Key.Key_W: "W",
                        Qt.Key.Key_N: "N",
                        Qt.Key.Key_R: "R",
                        Qt.Key.Key_U: "U",
                    }
                    if key in _state_keys:
                        self_w._on_assign_state(_state_keys[key])
                        return

                super().keyPressEvent(event)

        from PySide6.QtCore import QObject as _QObject
        from PySide6.QtWidgets import QAbstractSpinBox as _QSpin, QLineEdit as _QLine

        class _NavKeyFilter(_QObject):
            """App-level filter that routes navigation keys to the scope window.

            Intercepts Left/Right/Space/brackets/PageUp/PageDown regardless of
            which widget currently has focus, UNLESS an input widget (spinbox,
            line-edit) has focus — so those widgets still work normally.
            """
            _NAV_KEYS = {
                Qt.Key.Key_Left, Qt.Key.Key_Right,
                Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight,
                Qt.Key.Key_PageUp, Qt.Key.Key_PageDown,
                Qt.Key.Key_Space,
            }

            def __init__(self, win: "_ScopeWindow") -> None:
                super().__init__(win)  # parent=win → auto-deleted with window
                self._win = win

            def eventFilter(self, obj, event):  # type: ignore[override]
                from PySide6.QtCore import QEvent
                if event.type() == QEvent.Type.KeyPress:
                    if event.key() in self._NAV_KEYS:
                        focused = QApplication.focusWidget()
                        # Let spinboxes / line-edits handle their own keys
                        if not isinstance(focused, (_QSpin, _QLine)):
                            self._win.keyPressEvent(event)
                            return True
                return False

        app: QApplication = QApplication.instance() or QApplication(sys.argv)  # type: ignore[assignment]
        # Fusion style draws spinbox/combobox arrows using QPalette ButtonText color,
        # which gives white arrows on dark backgrounds and black arrows on light ones.
        app.setStyle("Fusion")
        win = _ScopeWindow(self.recording, self.analyzer)
        win.show()
        _nav_filter = _NavKeyFilter(win)
        app.installEventFilter(_nav_filter)
        app.exec()

    # ------------------------------------------------------------------
    # 2. Video export
    # ------------------------------------------------------------------

    def make_video(
        self,
        output_path: str | Path | None = None,
        *,
        signals: Sequence[str] | None = None,
        t_start: float = 0.0,
        t_end: float | None = None,
        x_window: float = 10.0,
        y_lims: dict[str, tuple[float, float]] | None = None,
        fps: int = 30,
        speed: float = 1.0,
        figsize: tuple[float, float] = (12, 8),
        dpi: int = 100,
        session: "ScoringSession | None" = None,
        show_hypnogram: bool = True,
    ) -> Path:
        """Render a scrolling MP4 video of the specified signals.

        Parameters
        ----------
        output_path:
            Destination path. Defaults to output/<animal_id>_scope.mp4.
        signals:
            List of channel/feature names.
        t_start, t_end:
            Start and end of the clip in seconds.
        x_window:
            Visible width of the scrolling window in seconds.
        y_lims:
            Dict mapping signal names to (ymin, ymax).
        fps:
            Frames per second.
        speed:
            Recording-seconds per video-second.
        figsize:
            Figure size in inches.
        dpi:
            Dots per inch.
        session:
            :class:`~sleep_tools.scoring.state.ScoringSession` with per-epoch
            sleep stage labels.  When provided together with
            ``show_hypnogram=True``, a colour-coded hypnogram strip is rendered
            below the signal traces showing the full recording timeline with a
            vertical playhead indicating the current scroll position.
        show_hypnogram:
            If ``True`` (default) and *session* is supplied, append a hypnogram
            strip at the bottom of the video.  Ignored when *session* is
            ``None``.
        """
        if self.recording is None:
            raise ValueError("No recording loaded.")

        if output_path is None:
            output_path = Path("output") / f"{self.recording.animal_id}_scope.mp4"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if signals is None:
            signals = list(self.recording.channels)
            if self.analyzer:
                signals += list(_DERIVED_KEYS)

        if t_end is None:
            t_end = self.recording.duration

        import matplotlib.pyplot as plt
        import matplotlib.colors as _mcolors
        from matplotlib.animation import FFMpegWriter
        from sleep_tools.scoring.state import STATE_COLORS

        sig_data = self._prepare_with(self.recording, self.analyzer, list(signals))
        n = len(sig_data)
        if n == 0:
            raise ValueError("No valid signals found.")

        # ── Layout: signal rows + optional hypnogram row ──────────────────────
        draw_hyp = show_hypnogram and session is not None
        if draw_hyp:
            height_ratios = [1.0] * n + [0.25]
            fig, all_axes = plt.subplots(
                n + 1, 1, sharex=False,
                figsize=figsize,
                gridspec_kw={"hspace": 0.05, "height_ratios": height_ratios},
            )
            axes = list(all_axes[:n])
            hyp_ax = all_axes[n]
        else:
            fig, raw_axes = plt.subplots(
                n, 1, sharex=True, figsize=figsize,
                gridspec_kw={"hspace": 0.05},
            )
            axes = [raw_axes] if n == 1 else list(raw_axes)
            hyp_ax = None

        lines = []
        for ax, sig in zip(axes, sig_data):
            _apply_ax_style(ax, sig, DARK_THEME)
            (line,) = ax.plot([], [], lw=0.7, color=sig.color)
            lines.append(line)
            if y_lims and sig.name in y_lims:
                ax.set_ylim(y_lims[sig.name])
            else:
                ax.set_ylim(-sig.y_half, sig.y_half)

        # ── Hypnogram strip (static pcolormesh, scrolls with signal window) ──
        if draw_hyp:
            th = DARK_THEME
            n_epochs = len(session.labels)  # type: ignore[union-attr]
            L = session.epoch_len            # type: ignore[union-attr]
            total_dur = self.recording.duration

            hyp_ax.set_facecolor(th.bg)
            hyp_ax.set_yticks([])
            hyp_ax.set_ylabel("Stage", color=th.muted, fontsize=7, labelpad=2)
            for spine in hyp_ax.spines.values():
                spine.set_color(th.border)
            hyp_ax.tick_params(colors=th.muted, labelsize=7)
            hyp_ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{int(v)}s")
            )

            x_edges = np.arange(n_epochs + 1, dtype=np.float64) * L
            y_edges = np.array([0.0, 1.0])
            state_order = ["W", "N", "R", "U"]
            state_to_int = {s: i for i, s in enumerate(state_order)}
            color_vals = np.array(
                [state_to_int.get(str(lbl), 3) for lbl in session.labels],  # type: ignore[union-attr]
                dtype=np.float64,
            ).reshape(1, n_epochs)
            cmap = _mcolors.ListedColormap(
                [STATE_COLORS["W"], STATE_COLORS["N"],
                 STATE_COLORS["R"], STATE_COLORS["U"]]
            )
            hyp_ax.pcolormesh(
                x_edges, y_edges, color_vals,
                cmap=cmap, vmin=-0.5, vmax=3.5, shading="flat",
            )
            hyp_ax.set_xlim(t_start, t_start + x_window)
            hyp_ax.set_ylim(0.0, 1.0)

            # Legend
            from matplotlib.patches import Patch as _Patch
            handles = [
                _Patch(facecolor=STATE_COLORS[s], label=s, linewidth=0)
                for s in state_order
            ]
            hyp_ax.legend(
                handles=handles, loc="upper right", fontsize=6,
                framealpha=0.5, ncol=4, handlelength=0.9, handleheight=0.9,
                borderpad=0.3, labelspacing=0.2, columnspacing=0.5,
                labelcolor=th.text,
            )

        # ── Animation ────────────────────────────────────────────────────────
        dt = speed / fps
        total_frames = int((t_end - t_start) * fps / speed)

        writer = FFMpegWriter(fps=fps, metadata=dict(title="Sleep Scope"))

        n_frames_report = max(1, total_frames // 20)
        with writer.saving(fig, str(output_path), dpi):
            for i in range(total_frames):
                t0 = t_start + i * dt
                t1 = t0 + x_window

                for ax, sig, line in zip(axes, sig_data, lines):
                    i0 = max(0, np.searchsorted(sig.times, t0, side="left") - 1)
                    i1 = min(len(sig.times), np.searchsorted(sig.times, t1, side="right") + 1)
                    line.set_data(sig.times[i0:i1], sig.values[i0:i1])
                    ax.set_xlim(t0, t1)

                if hyp_ax is not None:
                    hyp_ax.set_xlim(t0, t1)

                writer.grab_frame()
                if (i + 1) % n_frames_report == 0 or i == total_frames - 1:
                    pct = (i + 1) / total_frames * 100
                    print(f"\rRendering video: {pct:.0f}% ({i+1}/{total_frames} frames)", end="", flush=True)
        print()

        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_with(
        self,
        recording: SleepRecording,
        analyzer: SleepAnalyzer | None,
        names: list[str]
    ) -> list[_SignalData]:
        """Build _SignalData objects using a specific recording/analyzer."""
        raw_arr, raw_times = recording.raw.get_data(return_times=True)
        ch_names = recording.raw.ch_names
        
        derived: dict | None = None
        if analyzer is not None and any(n in _DERIVED_KEYS for n in names):
            derived = analyzer.compute_all_features(overlap=0.9)
            
        result: list[_SignalData] = []
        theme_colors = DARK_THEME.signals

        for i, name in enumerate(names):
            color_idx = i % len(theme_colors)
            color = theme_colors[color_idx]
            
            # Use defaults from _SIG_META
            label, unit = _SIG_META.get(name, (name, ""))
            scale = _get_signal_scale(name, unit)

            if name in _RAW_CHANNELS and name in ch_names:
                idx = ch_names.index(name)
                base_values = raw_arr[idx]
                times = raw_times
            elif name in _DERIVED_KEYS and derived is not None and name in derived:
                times = derived["times"]
                base_values = derived[name]
            else:
                continue

            # Display values are base * scale
            values = np.asarray(base_values * scale, dtype=np.float64)
            
            result.append(_SignalData(
                name=name,
                times=np.asarray(times, dtype=np.float64),
                values=values,
                label=label,
                unit=unit,
                color=color,
                color_index=color_idx,
                y_half=_auto_y_half(values),
                base_values=np.asarray(base_values, dtype=np.float64)
            ))

        return result

    def _prepare(self, names: list[str]) -> list[_SignalData]:
        """Legacy helper for make_video (uses self.recording/self.analyzer)."""
        if self.recording is None:
            return []
        return self._prepare_with(self.recording, self.analyzer, names)
