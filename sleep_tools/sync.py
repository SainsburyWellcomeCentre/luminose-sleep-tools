"""TTL event extraction and simple behavioral timestamp alignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sleep_tools.io import SleepRecording

Edge = Literal["rise", "fall", "both"]


@dataclass
class SyncAligner:
    """Extract TTL events from Luminose annotations and align them to Bpod."""

    recording: SleepRecording

    def extract_ttl_events(self, edge: Edge = "both") -> pd.DataFrame:
        """Return TTL edge rows from the paired TSV annotations.

        Parameters
        ----------
        edge:
            ``"rise"``, ``"fall"``, or ``"both"``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``time_from_start``, ``edge``, ``channel``, ``ttl``.
            Rows are sorted by time. Duplicate channel entries are preserved;
            call :meth:`deduplicate_channels` to collapse simultaneous rows.
        """
        if edge not in ("rise", "fall", "both"):
            raise ValueError("edge must be 'rise', 'fall', or 'both'.")

        ann = self.recording.annotations
        if ann is None or ann.empty:
            return pd.DataFrame(
                columns=["time_from_start", "edge", "channel", "ttl"]
            )
        required = {"Time From Start", "Channel", "Annotation"}
        if not required <= set(ann.columns):
            missing = ", ".join(sorted(required - set(ann.columns)))
            raise ValueError(f"Annotation table is missing required columns: {missing}")

        parsed = ann["Annotation"].str.extract(
            r"^\s*(TTL\s+\d+):\s+(Rise|Fall)\s*$"
        )
        mask = parsed[0].notna()
        if edge != "both":
            mask &= parsed[1].str.lower() == edge

        events = pd.DataFrame(
            {
                "time_from_start": pd.to_numeric(
                    ann.loc[mask, "Time From Start"], errors="coerce"
                ),
                "edge": parsed.loc[mask, 1].str.lower().to_numpy(),
                "channel": ann.loc[mask, "Channel"].astype(str).to_numpy(),
                "ttl": parsed.loc[mask, 0].str.strip().to_numpy(),
            }
        )
        events = events.dropna(subset=["time_from_start"])
        return events.sort_values(["time_from_start", "edge", "channel"]).reset_index(
            drop=True
        )

    def deduplicate_channels(
        self,
        events: pd.DataFrame | None = None,
        tolerance: float = 1e-6,
    ) -> pd.DataFrame:
        """Collapse simultaneous per-channel TTL rows to unique edge times."""
        if events is None:
            events = self.extract_ttl_events()
        if events.empty:
            return events.copy()

        rows: list[dict] = []
        sort_cols = ["ttl", "edge", "time_from_start", "channel"]
        grouped = events.sort_values(sort_cols).groupby(["ttl", "edge"], sort=False)
        for (ttl, edge), grp in grouped:
            cluster_times: list[float] = []
            cluster_channels: list[str] = []
            for row in grp.itertuples(index=False):
                t = float(row.time_from_start)
                if cluster_times and abs(t - cluster_times[-1]) > tolerance:
                    rows.append(
                        {
                            "time_from_start": float(np.mean(cluster_times)),
                            "edge": edge,
                            "channel": ",".join(sorted(set(cluster_channels))),
                            "ttl": ttl,
                            "n_channels": len(set(cluster_channels)),
                        }
                    )
                    cluster_times = []
                    cluster_channels = []
                cluster_times.append(t)
                cluster_channels.append(str(row.channel))
            if cluster_times:
                rows.append(
                    {
                        "time_from_start": float(np.mean(cluster_times)),
                        "edge": edge,
                        "channel": ",".join(sorted(set(cluster_channels))),
                        "ttl": ttl,
                        "n_channels": len(set(cluster_channels)),
                    }
                )

        return pd.DataFrame(rows).sort_values(["time_from_start", "edge"]).reset_index(
            drop=True
        )

    def detect_pulses(
        self,
        ttl: str | None = None,
        tolerance: float = 1e-6,
    ) -> list[tuple[float, float, float]]:
        """Pair rising and falling TTL edges into pulses.

        Returns
        -------
        list of tuple
            ``(rise_time, fall_time, duration)`` in seconds.
        """
        events = self.deduplicate_channels(tolerance=tolerance)
        if ttl is not None:
            events = events.loc[events["ttl"] == ttl]
        rises = events.loc[events["edge"] == "rise", "time_from_start"].to_numpy(
            dtype=float
        )
        falls = events.loc[events["edge"] == "fall", "time_from_start"].to_numpy(
            dtype=float
        )

        pulses: list[tuple[float, float, float]] = []
        fall_idx = 0
        for rise in rises:
            while fall_idx < len(falls) and falls[fall_idx] < rise:
                fall_idx += 1
            if fall_idx >= len(falls):
                break
            fall = float(falls[fall_idx])
            pulses.append((float(rise), fall, fall - float(rise)))
            fall_idx += 1
        return pulses

    def align_to_bpod(
        self,
        bpod_timestamps: np.ndarray | list[float],
        ttl: str | None = None,
    ) -> pd.DataFrame:
        """Estimate a constant offset from recording TTL rises to Bpod times.

        This is intentionally a minimal Stage 3 stub: it assumes both clocks
        share the same rate and computes ``bpod_time - recording_time`` from
        matched rising edges.
        """
        bpod = np.asarray(bpod_timestamps, dtype=np.float64)
        if bpod.ndim != 1:
            raise ValueError("bpod_timestamps must be a 1-D array.")

        events = self.deduplicate_channels()
        if ttl is not None:
            events = events.loc[events["ttl"] == ttl]
        rises = events.loc[events["edge"] == "rise", "time_from_start"].to_numpy(
            dtype=float
        )
        n = min(len(rises), len(bpod))
        if n == 0:
            return pd.DataFrame(
                columns=[
                    "recording_time",
                    "bpod_time",
                    "offset",
                    "aligned_recording_time",
                ]
            )

        offsets = bpod[:n] - rises[:n]
        offset = float(np.median(offsets))
        return pd.DataFrame(
            {
                "recording_time": rises[:n],
                "bpod_time": bpod[:n],
                "offset": np.full(n, offset),
                "aligned_recording_time": rises[:n] + offset,
            }
        )

    def plot_events(self, ax=None, edge: Edge = "both"):
        """Overlay deduplicated TTL events on a Matplotlib axis."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        events = self.deduplicate_channels(self.extract_ttl_events(edge=edge))
        colors = {"rise": "tab:green", "fall": "tab:red"}
        labels_seen: set[str] = set()
        for row in events.itertuples(index=False):
            label = row.edge if row.edge not in labels_seen else None
            ax.axvline(
                float(row.time_from_start),
                color=colors.get(row.edge, "tab:gray"),
                linestyle="--",
                alpha=0.8,
                label=label,
            )
            labels_seen.add(row.edge)
        ax.set_xlabel("Time from recording start (s)")
        ax.set_title("TTL events")
        if labels_seen:
            ax.legend()
        return ax
