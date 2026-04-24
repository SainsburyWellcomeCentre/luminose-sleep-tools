"""EDF + TSV loading; SleepRecording dataclass; HDF5 export."""
from __future__ import annotations

import datetime
import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import mne
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sleep_tools.analysis import SleepAnalyzer
    from sleep_tools.scoring.state import ScoringSession


# Feature columns always written to HDF5 (NaN when not computed).
_EPOCH_FEATURE_KEYS: tuple[str, ...] = (
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "emg_rms",
    "td_ratio",
)


# Map raw EDF channel names → canonical short names used throughout the package.
# Users can override by passing channel_rename to from_edf().
_DEFAULT_CHANNEL_RENAME: dict[str, str] = {
    "EEG EEG1A-B": "EEG1",
    "EEG EEG2A-B": "EEG2",
    "EMG EMG": "EMG",
}


@dataclass
class SleepRecording:
    """Container for one EDF recording plus its paired TSV annotations."""

    raw: mne.io.Raw
    """MNE Raw object with canonical channel names (EEG1, EEG2, EMG)."""

    annotations: pd.DataFrame | None
    """Parsed TSV data rows (columns: Number, Start Time, End Time,
    Time From Start, Channel, Annotation). None if no TSV was found."""

    animal_id: str
    experiment_id: str
    start_datetime: str
    """Recording start as a string (from TSV header or filename)."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_edf(
        cls,
        edf_path: str | Path,
        tsv_path: str | Path | None = None,
        preload: bool = True,
        channel_rename: dict[str, str] | None = None,
        verbose: bool = False,
    ) -> Self:
        """Load a recording from an EDF file (+ optional TSV annotations).

        Parameters
        ----------
        edf_path:
            Path to ``*_export.edf``.
        tsv_path:
            Path to ``*_annotations.tsv``.  If *None* the paired TSV is
            discovered automatically from the EDF filename.
        preload:
            If True, load the signal data into memory immediately.
        channel_rename:
            Override the default channel name mapping.  Keys are raw EDF
            channel names; values are the canonical names to use.
        verbose:
            Passed through to ``mne.io.read_raw_edf``.
        """
        edf_path = Path(edf_path)
        raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=verbose)

        rename = channel_rename if channel_rename is not None else _DEFAULT_CHANNEL_RENAME
        rename = {k: v for k, v in rename.items() if k in raw.ch_names}
        if rename:
            raw.rename_channels(rename)

        # Auto-discover TSV
        if tsv_path is None:
            candidate = edf_path.parent / edf_path.name.replace("_export.edf", "_annotations.tsv")
            tsv_path = candidate if (candidate.suffix == ".tsv" and candidate.exists()) else None

        annotations: pd.DataFrame | None = None
        animal_id = ""
        experiment_id = ""
        start_datetime = ""

        if tsv_path is not None:
            annotations, meta = _parse_tsv(Path(tsv_path))
            animal_id = meta.get("Animal ID", "")
            experiment_id = meta.get("Experiment ID", "")
            start_datetime = meta.get("start_datetime", "")

        # Fallback: parse animal_id and datetime from filename
        if not animal_id:
            m = re.match(
                r"([^_]+(?:_[^_]+)*)_(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})",
                edf_path.stem,
            )
            if m:
                animal_id = m.group(1)
                start_datetime = m.group(2)

        return cls(
            raw=raw,
            annotations=annotations,
            animal_id=animal_id,
            experiment_id=experiment_id,
            start_datetime=start_datetime,
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def channels(self) -> list[str]:
        """List of canonical channel names."""
        return list(self.raw.ch_names)

    @property
    def duration(self) -> float:
        """Total recording duration in seconds."""
        return float(self.raw.times[-1])

    @property
    def sfreq(self) -> float:
        """Sampling frequency in Hz."""
        return float(self.raw.info["sfreq"])

    # ------------------------------------------------------------------ #
    # Metadata & signal information
    # ------------------------------------------------------------------ #

    def metadata(self) -> dict:
        """Return a dict of recording-level metadata.

        Keys
        ----
        animal_id, experiment_id, start_datetime:
            Parsed from the TSV header or EDF filename.
        channels:
            Canonical channel names after renaming.
        sfreq:
            Sampling frequency in Hz.
        duration_s:
            Total recording duration in seconds.
        n_samples:
            Total number of samples per channel.
        has_annotations:
            True if a paired TSV was loaded and contains data rows.
        n_annotations:
            Number of annotation rows (0 if no TSV).
        """
        return {
            "animal_id": self.animal_id,
            "experiment_id": self.experiment_id,
            "start_datetime": self.start_datetime,
            "channels": self.channels,
            "sfreq": self.sfreq,
            "duration_s": self.duration,
            "n_samples": self.raw.n_times,
            "has_annotations": (
                self.annotations is not None and not self.annotations.empty
            ),
            "n_annotations": (
                len(self.annotations) if self.annotations is not None else 0
            ),
        }

    def signal_info(self) -> dict[str, dict]:
        """Return per-channel signal statistics.

        Requires the recording to have been loaded with ``preload=True``
        (the default).  Accessing this property on a non-preloaded recording
        will trigger a full data load.

        Returns
        -------
        dict
            Keys are canonical channel names (``"EEG1"``, ``"EEG2"``,
            ``"EMG"``).  Each value is a dict with:

            ``unit``
                Physical unit (always ``"V"`` for these channels).
            ``n_samples``
                Number of samples in the channel.
            ``min``, ``max``, ``mean``, ``std``
                Basic amplitude statistics (float, in Volts).
        """
        info: dict[str, dict] = {}
        for ch in self.channels:
            idx = self.raw.ch_names.index(ch)
            data, _ = self.raw[idx]
            arr = data[0].astype(np.float64)
            info[ch] = {
                "unit": "V",
                "n_samples": int(arr.size),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }
        return info

    def ttl_events(self) -> dict[str, dict[str, np.ndarray]]:
        """Extract TTL pulse events from the paired TSV annotations.

        The Luminose TSV logs each TTL edge once per recording channel
        (EEG1, EEG2, EMG) at the same timestamp; this method deduplicates
        those entries so each event appears exactly once.

        Returns
        -------
        dict
            Keys are TTL names (e.g. ``"TTL 3"``).  Each value is a dict
            with keys ``"rise"`` and ``"fall"``, both float64 arrays of
            event times in **seconds from recording start** (the
            ``"Time From Start"`` TSV column), sorted ascending.
            Returns an empty dict if no annotations are loaded or no TTL
            rows are found.
        """
        if self.annotations is None or self.annotations.empty:
            return {}
        ann = self.annotations
        if "Annotation" not in ann.columns or "Time From Start" not in ann.columns:
            return {}

        ttl_mask = ann["Annotation"].str.match(r"TTL\s+\d+:\s+(Rise|Fall)", na=False)
        ttl_df = ann.loc[ttl_mask, ["Time From Start", "Annotation"]].drop_duplicates()
        if ttl_df.empty:
            return {}

        parsed = ttl_df["Annotation"].str.extract(r"(TTL\s+\d+):\s+(Rise|Fall)")
        ttl_df = ttl_df.copy()
        ttl_df["_name"] = parsed[0].str.strip()
        ttl_df["_edge"] = parsed[1]

        result: dict[str, dict[str, np.ndarray]] = {}
        for ttl_name, grp in ttl_df.groupby("_name"):
            result[str(ttl_name)] = {
                "rise": np.sort(
                    grp.loc[grp["_edge"] == "Rise", "Time From Start"].to_numpy(dtype=float)
                ),
                "fall": np.sort(
                    grp.loc[grp["_edge"] == "Fall", "Time From Start"].to_numpy(dtype=float)
                ),
            }
        return result

    def __repr__(self) -> str:
        return (
            f"SleepRecording(animal={self.animal_id!r}, "
            f"duration={self.duration:.1f}s, "
            f"channels={self.channels}, "
            f"sfreq={self.sfreq}Hz)"
        )


# ------------------------------------------------------------------ #
# TSV parsing helpers
# ------------------------------------------------------------------ #

def _parse_tsv(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse a Luminose annotation TSV file.

    The file has 4 metadata rows (key\\tvalue), one or two blank rows,
    then a tab-separated data table.

    Returns
    -------
    data_df:
        DataFrame with columns ``Number``, ``Start Time``, ``End Time``,
        ``Time From Start``, ``Channel``, ``Annotation``.
    meta:
        Dict with ``Experiment ID``, ``Animal ID``, ``Researcher``,
        ``Directory path``, and ``start_datetime``.
    """
    meta: dict[str, str] = {}

    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

    # Extract metadata from leading rows until the first blank line or
    # until a line starting with "Number" (the data header).
    for line in lines[:6]:
        stripped = line.rstrip("\n")
        if not stripped or stripped.startswith("Number\t"):
            break
        parts = stripped.split("\t")
        if len(parts) >= 2 and parts[0]:
            meta[parts[0]] = parts[1]

    # Locate the data header row
    header_idx: int | None = None
    for i, line in enumerate(lines):
        if line.startswith("Number\t"):
            header_idx = i
            break

    if header_idx is None:
        return pd.DataFrame(), meta

    # Grab start_datetime from the first data row (the "Started Recording" row)
    if header_idx + 1 < len(lines):
        first_row = lines[header_idx + 1].split("\t")
        if len(first_row) > 1:
            meta["start_datetime"] = first_row[1].strip()

    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=header_idx,
        dtype=str,
    )
    df.columns = [c.strip() for c in df.columns]

    # Coerce numeric columns
    if "Number" in df.columns:
        df["Number"] = pd.to_numeric(df["Number"], errors="coerce")
    if "Time From Start" in df.columns:
        df["Time From Start"] = pd.to_numeric(df["Time From Start"], errors="coerce")

    return df, meta


# ------------------------------------------------------------------ #
# HDF5 export
# ------------------------------------------------------------------ #

def save_to_h5(
    recording: SleepRecording,
    path: str | Path,
    *,
    analyzer: SleepAnalyzer | None = None,
    session: ScoringSession | None = None,
    labels: np.ndarray | None = None,
    epoch_len: float = 5.0,
    include_raw_signals: bool = True,
    overwrite: bool = False,
) -> Path:
    """Save a curated dataset to HDF5.

    Parameters
    ----------
    recording:
        The loaded recording.
    path:
        Output file path (``*.h5``).
    analyzer:
        If provided, ``compute_all_features()`` is called and band powers,
        EMG RMS, and T:D ratio are stored.  If *None*, all feature datasets
        are written as NaN arrays so the file schema is always identical
        regardless of what has been computed.
    session:
        A :class:`~sleep_tools.scoring.ScoringSession`.  When provided,
        labels and ``epoch_len`` are taken from the session, and the
        scoring thresholds are stored as attributes on ``/epochs``.
        Takes priority over the *labels* argument.
    labels:
        1-D array of per-epoch sleep-stage labels (e.g. ``"W"``, ``"N"``,
        ``"R"``, ``"U"``).  Length must equal the number of epoch-centre
        times.  If *None* (and no *session*), every epoch is set to
        ``"U"`` (unscored).
    epoch_len:
        Epoch length in seconds.  Used **only** when *analyzer* is *None*
        and no *session* is given, to build the non-overlapping epoch time
        axis.  When *analyzer* or *session* is provided their ``epoch_len``
        is used instead.
    include_raw_signals:
        Write raw signal arrays to ``/signals/``.  Set *False* to keep
        file size small when only features are needed.
    overwrite:
        If *False* (default) and *path* exists, raise ``FileExistsError``.

    Returns
    -------
    Path
        Absolute path to the written file.

    HDF5 layout
    -----------
    ::

        /                           ← root attrs: animal_id, sfreq, epoch_len,
                                      sleep_tools_version, band_definitions, …
        /signals/EEG1               ← raw signal arrays, float32, gzip-compressed
        /signals/EEG2                 attrs: unit="V", sfreq
        /signals/EMG
        /epochs/times               ← epoch-centre times (float64); attrs: units="s"
        /epochs/labels              ← per-epoch stage labels (UTF-8); attrs: description
        /epochs/features/           ← attrs: band_definitions (JSON)
            delta_power             ← (n_epochs,) float64; NaN if not computed
            theta_power               each dataset has attrs: description, units,
            alpha_power               frequency_range_hz, sleep_relevance
            beta_power
            gamma_power
            emg_rms
            td_ratio
        /annotations/               ← TSV annotation columns (if annotations loaded)
            Number
            Start_Time
            End_Time
            Time_From_Start
            Channel
            Annotation
        /ttl_events/                ← TTL pulse times (if TTL rows present in TSV)
            TTL_3/                    one sub-group per TTL name
                rise_times          ← float64 array, seconds from recording start
                fall_times          ← float64 array, seconds from recording start
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 export. Install it with: pip install h5py"
        ) from exc

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass overwrite=True to replace it."
        )

    # ── Epoch times & features ────────────────────────────────────────────
    feat_arrays: dict[str, np.ndarray]
    if analyzer is not None:
        feat = analyzer.compute_all_features()
        times: np.ndarray = feat["times"]
        used_epoch_len = analyzer.epoch_len
        feat_arrays = {
            key: feat.get(key, np.full(len(times), np.nan))
            for key in _EPOCH_FEATURE_KEYS
        }
    else:
        used_epoch_len = epoch_len
        # Non-overlapping epoch centres: 0.5·L, 1.5·L, 2.5·L, …
        times = np.arange(used_epoch_len / 2.0, recording.duration, used_epoch_len)
        feat_arrays = {key: np.full(len(times), np.nan) for key in _EPOCH_FEATURE_KEYS}

    # ── Session overrides epoch_len / times / labels ──────────────────────
    if session is not None:
        used_epoch_len = session.epoch_len
        times = session.times
        # Rebuild NaN feature arrays to match session epoch count if no analyzer
        if analyzer is None:
            feat_arrays = {key: np.full(len(times), np.nan) for key in _EPOCH_FEATURE_KEYS}

    n_epochs = len(times)

    # ── Labels ────────────────────────────────────────────────────────────
    if session is not None:
        # Session takes priority over raw labels argument
        label_arr: np.ndarray = np.asarray(session.labels, dtype=object)
    elif labels is not None:
        if len(labels) != n_epochs:
            raise ValueError(
                f"labels length ({len(labels)}) must match "
                f"number of epochs ({n_epochs})."
            )
        label_arr = np.asarray(labels, dtype=object)
    else:
        label_arr = np.full(n_epochs, "U", dtype=object)

    # ── Write ─────────────────────────────────────────────────────────────
    str_dt = h5py.string_dtype(encoding="utf-8")

    def _str_array(arr: np.ndarray) -> list[bytes]:
        """Convert a string/object array to a list of UTF-8 bytes for h5py."""
        return [str(v).encode("utf-8") for v in arr]

    # Lazy import to avoid circular dependency (analysis.py imports SleepRecording).
    from sleep_tools.analysis import FEATURE_INFO, BANDS  # noqa: PLC0415

    try:
        _pkg_version = importlib.metadata.version("sleep-tools")
    except importlib.metadata.PackageNotFoundError:
        _pkg_version = "unknown"

    with h5py.File(path, "w") as f:
        # Root-level metadata
        f.attrs["animal_id"] = recording.animal_id
        f.attrs["experiment_id"] = recording.experiment_id
        f.attrs["start_datetime"] = recording.start_datetime
        f.attrs["sfreq"] = recording.sfreq
        f.attrs["n_samples"] = recording.raw.n_times
        f.attrs["duration_s"] = recording.duration
        f.attrs["n_channels"] = len(recording.channels)
        f.attrs["channels"] = recording.channels
        f.attrs["epoch_len"] = used_epoch_len
        f.attrs["n_epochs"] = n_epochs
        f.attrs["features_computed"] = analyzer is not None
        f.attrs["sleep_tools_version"] = _pkg_version
        f.attrs["band_definitions"] = json.dumps(
            {k: list(v) for k, v in BANDS.items()}
        )
        f.attrs["saved_at"] = (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

        # /signals/
        if include_raw_signals:
            sig_grp = f.create_group("signals")
            for ch in recording.channels:
                idx = recording.raw.ch_names.index(ch)
                data, _ = recording.raw[idx]
                ds = sig_grp.create_dataset(
                    ch,
                    data=data[0].astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                ds.attrs["unit"] = "V"
                ds.attrs["sfreq"] = recording.sfreq

        # /epochs/
        ep_grp = f.create_group("epochs")
        times_ds = ep_grp.create_dataset("times", data=times)
        times_ds.attrs["units"] = "s"
        times_ds.attrs["description"] = "epoch centre times"
        labels_ds = ep_grp.create_dataset(
            "labels",
            data=_str_array(label_arr),
            dtype=str_dt,
        )
        labels_ds.attrs["description"] = (
            "per-epoch sleep-stage label; 'U'=unscored, 'W'=Wake, 'N'=NREM, 'R'=REM"
        )

        # Store scoring thresholds when a session with auto-score results is provided
        if session is not None:
            from dataclasses import asdict as _asdict
            thr_grp = ep_grp.create_group("thresholds")
            thr_grp.attrs["units"] = (
                "delta_* in µV²/Hz; emg_* in µV; td_rem dimensionless"
            )
            for k, v in _asdict(session.thresholds).items():
                thr_grp.attrs[k] = float(v)

        feat_grp = ep_grp.create_group("features")
        feat_grp.attrs["band_definitions"] = json.dumps(
            {k: list(v) for k, v in BANDS.items()}
        )
        for key in _EPOCH_FEATURE_KEYS:
            ds = feat_grp.create_dataset(
                key,
                data=feat_arrays[key].astype(np.float64),
            )
            info = FEATURE_INFO.get(key, {})
            ds.attrs["description"] = info.get("description", "")
            ds.attrs["units"] = info.get("units", "")
            ds.attrs["sleep_relevance"] = info.get("sleep_relevance", "")
            freq = info.get("frequency_range_hz")
            if freq is not None:
                ds.attrs["frequency_range_hz"] = list(freq)
            else:
                ds.attrs["frequency_range_hz"] = "derived"

        # /annotations/ — one dataset per TSV column
        if recording.annotations is not None and not recording.annotations.empty:
            ann_grp = f.create_group("annotations")
            for col in recording.annotations.columns:
                col_data = recording.annotations[col]
                ds_name = col.replace(" ", "_")
                if pd.api.types.is_numeric_dtype(col_data):
                    ann_grp.create_dataset(
                        ds_name,
                        data=col_data.to_numpy(dtype=np.float64, na_value=np.nan),
                    )
                else:
                    ann_grp.create_dataset(
                        ds_name,
                        data=_str_array(col_data.fillna("").astype(str).to_numpy()),
                        dtype=str_dt,
                    )

        # /ttl_events/ — deduplicated TTL pulse times per TTL name
        ttl_data = recording.ttl_events()
        if ttl_data:
            ttl_grp = f.create_group("ttl_events")
            ttl_grp.attrs["units"] = "s"
            ttl_grp.attrs["description"] = (
                "TTL pulse event times (seconds from recording start); "
                "deduplicated across recording channels"
            )
            for ttl_name, edges in ttl_data.items():
                safe = ttl_name.replace(" ", "_")
                ch_grp = ttl_grp.create_group(safe)
                r_ds = ch_grp.create_dataset("rise_times", data=edges["rise"])
                r_ds.attrs["units"] = "s"
                r_ds.attrs["description"] = "Rising-edge times (s from recording start)"
                f_ds = ch_grp.create_dataset("fall_times", data=edges["fall"])
                f_ds.attrs["units"] = "s"
                f_ds.attrs["description"] = "Falling-edge times (s from recording start)"

    return path.resolve()
