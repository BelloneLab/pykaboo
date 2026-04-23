"""Helpers for projecting user-triggered flag events onto recorded frames."""

from __future__ import annotations

from typing import Iterable, Mapping, Any

import numpy as np
import pandas as pd


def project_user_flag_events(
    frame_times,
    events: Iterable[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Map user-flag events onto frame-aligned event and TTL columns."""
    times = pd.to_numeric(pd.Series(frame_times), errors="coerce")
    event_hits = np.zeros(len(times), dtype=int)
    ttl_active = np.zeros(len(times), dtype=int)
    event_timestamps = np.full(len(times), np.nan, dtype=float)
    event_labels = np.full(len(times), "", dtype=object)
    event_shortcuts = np.full(len(times), "", dtype=object)
    event_outputs = np.full(len(times), "", dtype=object)
    event_pulse_ms = np.full(len(times), np.nan, dtype=float)

    if times.empty or not bool(times.notna().any()):
        return {
            "event": event_hits,
            "ttl": ttl_active,
            "count": np.cumsum(event_hits, dtype=int),
            "event_timestamp": event_timestamps,
            "event_label": event_labels,
            "event_shortcut": event_shortcuts,
            "event_output": event_outputs,
            "event_pulse_ms": event_pulse_ms,
        }

    time_values = times.to_numpy(dtype=float, na_value=np.nan)
    valid_mask = np.isfinite(time_values)
    valid_indices = np.flatnonzero(valid_mask)
    valid_times = time_values[valid_mask]

    for raw_event in events or []:
        event = dict(raw_event or {})
        try:
            timestamp = float(event.get("timestamp_software", np.nan))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(timestamp):
            continue

        pulse_ms = max(0, int(event.get("pulse_ms", 0) or 0))
        nearest_valid_offset = int(np.argmin(np.abs(valid_times - timestamp)))
        nearest_index = int(valid_indices[nearest_valid_offset])
        event_hits[nearest_index] += 1
        event_timestamps[nearest_index] = timestamp
        event_labels[nearest_index] = str(event.get("label", "") or "")
        event_shortcuts[nearest_index] = str(event.get("shortcut", "") or "")
        event_outputs[nearest_index] = str(event.get("output_id", "") or "")
        event_pulse_ms[nearest_index] = float(pulse_ms)

        if pulse_ms > 0:
            end_timestamp = timestamp + (float(pulse_ms) / 1000.0)
            ttl_mask = valid_mask & (time_values >= timestamp) & (time_values < end_timestamp)
            ttl_active[ttl_mask] = 1
            ttl_active[nearest_index] = 1

    return {
        "event": event_hits,
        "ttl": ttl_active,
        "count": np.cumsum(event_hits, dtype=int),
        "event_timestamp": event_timestamps,
        "event_label": event_labels,
        "event_shortcut": event_shortcuts,
        "event_output": event_outputs,
        "event_pulse_ms": event_pulse_ms,
    }
