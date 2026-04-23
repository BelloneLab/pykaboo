"""Retention policy helpers for live detection overlays."""

from __future__ import annotations

import math


def overlay_retention_limits(
    *,
    preview_fps: float,
    inference_ms: float = 0.0,
) -> tuple[float, int]:
    """Return the max overlay age in ms and allowable frame gap."""
    fps = max(1.0, float(preview_fps or 0.0))
    frame_interval_ms = 1000.0 / fps
    max_age_ms = max(160.0, 4.0 * frame_interval_ms)
    if inference_ms > 0.0:
        max_age_ms = max(max_age_ms, 2.0 * float(inference_ms))
    max_age_ms = min(750.0, max_age_ms)
    max_frame_gap = max(2, int(math.ceil(max_age_ms / frame_interval_ms)))
    return float(max_age_ms), int(max_frame_gap)


def overlay_result_is_current(
    *,
    preview_frame_index: int,
    preview_timestamp_s: float,
    result_frame_index: int,
    result_timestamp_s: float,
    preview_fps: float,
    inference_ms: float = 0.0,
) -> bool:
    """True when the latest inference result is fresh enough to keep drawing."""
    max_age_ms, max_frame_gap = overlay_retention_limits(
        preview_fps=preview_fps,
        inference_ms=inference_ms,
    )
    frame_gap = int(preview_frame_index) - int(result_frame_index)
    if frame_gap > max_frame_gap:
        return False
    age_ms = max(0.0, (float(preview_timestamp_s) - float(result_timestamp_s)) * 1000.0)
    return age_ms <= max_age_ms
