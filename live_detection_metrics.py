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


def format_live_detection_status(result: object) -> str:
    tracked_mice = getattr(result, "tracked_mice", []) or []
    mouse_count = len(tracked_mice)
    predict_ms = _safe_ms(getattr(result, "predict_ms", 0.0))
    total_ms = _safe_ms(getattr(result, "inference_ms", 0.0))
    lag_ms = _safe_ms(getattr(result, "end_to_end_ms", 0.0))

    if predict_ms > 0.0 and lag_ms > 0.0:
        return f"{mouse_count} mice, pred {predict_ms:.1f} ms, total {total_ms:.1f} ms, lag {lag_ms:.1f} ms"
    if total_ms > 0.0:
        return f"{mouse_count} mice, {total_ms:.1f} ms"
    return f"{mouse_count} mice"
