"""
Shared zero-referencing of exported recording timestamps.

Every CSV the app exports (frame metadata, TTL states) should express time as
elapsed seconds from the first sample of the recording, never as raw camera
ticks or wall-clock epochs. This module is the single source of truth for that
conversion so the frame-metadata path, the TTL-states path, and auxiliary
camera streams all agree.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Columns that carry wall-clock (epoch) software timestamps in seconds.
SOFTWARE_TIMESTAMP_COLUMNS = (
    "timestamp_software",
    "live_detection_timestamp_software",
    "live_detection_completed_timestamp_software",
    "user_flag_event_timestamp_software",
)

# Columns that carry raw camera-clock ticks (Spinnaker/Basler chunk data).
CAMERA_TICK_COLUMNS = (
    "timestamp_ticks",
    "timestamp_camera",
)


def infer_timestamp_tick_scale(tick_series, software_series=None) -> float:
    """Infer how many camera ticks correspond to one second.

    Prefers the per-frame ratio against software timestamps when both clocks
    advance together; falls back to magnitude heuristics (ns / us / ms).
    """
    ticks = pd.to_numeric(tick_series, errors="coerce")
    tick_delta = ticks.diff()

    if software_series is not None:
        software = pd.to_numeric(software_series, errors="coerce")
        software_delta = software.diff()
        valid = (tick_delta > 0) & (software_delta > 0)
        if bool(valid.any()):
            ratios = (
                (tick_delta[valid] / software_delta[valid])
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if not ratios.empty:
                return max(float(ratios.median()), 1.0)

    finite_deltas = tick_delta.abs().replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_deltas.empty:
        # Per-frame deltas at realistic frame rates (1-500 fps):
        # nanosecond clocks step by >= ~2e6, microsecond by >= ~2e3,
        # millisecond by >= ~2. Classify by the median step.
        median_delta = float(finite_deltas.median())
        if median_delta >= 1e6:
            return 1e9
        if median_delta >= 1e3:
            return 1e6
        if median_delta >= 1.5:
            return 1e3
    return 1.0


def _first_valid(series: pd.Series) -> Optional[float]:
    """Return the first non-NaN value of a series as a float, or None if empty.

    Used to pick the time origin (first real sample) when zero-referencing a
    timestamp column.
    """
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[0])


def normalize_recording_timestamps(df, zero_camera_frame_id: bool = True):
    """Express exported timestamps in elapsed seconds from frame 0.

    - Software (epoch) columns share one origin: the first valid
      ``timestamp_software`` sample, so events keep their offsets relative to
      frame 0 while frame 0 itself reads exactly 0.
    - Camera tick columns are zeroed at their own first sample and divided by
      one shared tick scale, so ``timestamp_ticks`` and ``timestamp_camera``
      stay in the same unit (seconds) and both start at 0.
    - ``camera_frame_id`` is rebased to 0 so it can be compared directly with
      ``frame_id`` to spot dropped frames.
    """
    if df is None or df.empty:
        return df

    normalized = df.copy()

    software_numeric = None
    software_origin = None
    if "timestamp_software" in normalized.columns:
        software_numeric = pd.to_numeric(normalized["timestamp_software"], errors="coerce")
        software_origin = _first_valid(software_numeric)
        if software_origin is not None:
            normalized["timestamp_software"] = (software_numeric - software_origin).round(6)
        else:
            normalized["timestamp_software"] = software_numeric

    if software_origin is not None:
        for column in SOFTWARE_TIMESTAMP_COLUMNS:
            if column == "timestamp_software" or column not in normalized.columns:
                continue
            column_numeric = pd.to_numeric(normalized[column], errors="coerce")
            normalized[column] = (column_numeric - software_origin).round(6)

    tick_scale = None
    for column in CAMERA_TICK_COLUMNS:
        if column not in normalized.columns:
            continue
        tick_numeric = pd.to_numeric(normalized[column], errors="coerce")
        tick_origin = _first_valid(tick_numeric)
        if tick_origin is None:
            normalized[column] = tick_numeric
            continue
        if tick_scale is None:
            tick_scale = infer_timestamp_tick_scale(tick_numeric, software_numeric)
        normalized[column] = ((tick_numeric - tick_origin) / tick_scale).round(6)

    if zero_camera_frame_id and "camera_frame_id" in normalized.columns:
        frame_id_numeric = pd.to_numeric(normalized["camera_frame_id"], errors="coerce")
        frame_id_origin = _first_valid(frame_id_numeric)
        if frame_id_origin is not None:
            rebased = frame_id_numeric - frame_id_origin
            if bool(rebased.notna().all()):
                normalized["camera_frame_id"] = rebased.astype(np.int64)
            else:
                normalized["camera_frame_id"] = rebased

    return normalized


def normalize_metadata_csv_file(csv_path) -> bool:
    """Rewrite a worker-produced metadata CSV with zero-referenced timestamps.

    Used for auxiliary camera streams whose CSVs are written directly by
    ``CameraWorker._save_metadata`` without passing through the main window's
    export pipeline. Returns True when the file was rewritten.
    """
    from pathlib import Path

    path = Path(csv_path)
    if not path.is_file():
        return False
    df = pd.read_csv(path)
    if df.empty:
        return False
    df = normalize_recording_timestamps(df)
    drop_candidates = ["raw_dtype", "raw_height", "raw_width", "raw_min", "raw_max", "raw_mean"]
    removable = [column for column in drop_candidates if column in df.columns]
    if removable:
        df = df.drop(columns=removable)
    df.to_csv(path, index=False)
    return True
