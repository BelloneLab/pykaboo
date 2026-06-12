"""Helpers for presenting and reusing live inference timing metrics."""

from __future__ import annotations


def _safe_ms(value: object) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except Exception:
        return 0.0


def live_result_retention_ms(result: object) -> float:
    lag_ms = _safe_ms(getattr(result, "end_to_end_ms", 0.0))
    if lag_ms > 0.0:
        return lag_ms
    return _safe_ms(getattr(result, "inference_ms", 0.0))


def detection_fps(result: object) -> float:
    """Effective detections-per-second from the end-to-end latency."""
    total_ms = _safe_ms(getattr(result, "inference_ms", 0.0))
    lag_ms = _safe_ms(getattr(result, "end_to_end_ms", 0.0))
    basis = total_ms if total_ms > 0.0 else lag_ms
    if basis <= 0.0:
        return 0.0
    return 1000.0 / basis


def format_live_detection_status(result: object) -> str:
    tracked_mice = getattr(result, "tracked_mice", []) or []
    mouse_count = len(tracked_mice)
    total_ms = _safe_ms(getattr(result, "inference_ms", 0.0))
    lag_ms = _safe_ms(getattr(result, "end_to_end_ms", 0.0))
    fps = detection_fps(result)

    animals = "1 animal" if mouse_count == 1 else f"{mouse_count} animals"
    if total_ms > 0.0:
        # Lead with the rate (what the user watches) then the latency detail.
        return f"● {animals}  ·  {fps:.0f} fps  ·  {total_ms:.0f} ms compute  ·  {lag_ms:.0f} ms lag"
    return f"● {animals}"
