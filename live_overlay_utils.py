"""Retention policy helpers for live detection overlays."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Sequence

import cv2
import numpy as np

from live_detection_types import LiveDetectionResult, TrackedMouseState


def overlay_retention_limits(
    *,
    preview_fps: float,
    inference_ms: float = 0.0,
) -> tuple[float, int]:
    """Return the max overlay age in ms and allowable preview-frame gap.

    The preview can run much faster than inference (e.g. 60 fps preview while a
    segmentation model delivers ~20 fps). To keep the overlay from blinking in
    the gaps between inference updates we carry the most recent result forward
    for roughly two inference intervals, clamped to a sane window. The overlay
    only disappears when inference genuinely stalls (latency far exceeds the
    window), which is the behaviour the user wants: stable masks, no flicker,
    but no frozen ghost when detection actually dies.
    """
    fps = max(1.0, float(preview_fps or 0.0))
    frame_interval_ms = 1000.0 / fps
    lag_ms = max(0.0, float(inference_ms or 0.0))
    # Bridge ~2.5 inference intervals (latency is one interval at steady state),
    # within a 150–450 ms envelope.
    max_age_ms = min(450.0, max(150.0, 2.5 * lag_ms))
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


def compensate_live_overlay_motion(
    result: Optional[LiveDetectionResult],
    history: Sequence[LiveDetectionResult],
    *,
    target_frame_index: int,
    target_timestamp_s: float,
    preview_fps: float,
    max_age_ms: float = 220.0,
) -> Optional[LiveDetectionResult]:
    """Shift a fresh overlay result toward the current preview frame.

    Live inference always finishes after the source frame has already appeared
    on screen. For moving animals that makes the mask visibly trail the body.
    This predicts a short, bounded translation from the two newest detections
    with the same track id and applies it only to the preview overlay copy.
    """
    if result is None or not getattr(result, "tracked_mice", None):
        return result

    fps = max(1.0, float(preview_fps or 0.0))
    result_frame_index = _safe_int(getattr(result, "frame_index", target_frame_index), int(target_frame_index))
    frame_gap = max(0, int(target_frame_index) - int(result_frame_index))
    try:
        age_s = max(0.0, float(target_timestamp_s) - float(result.timestamp_s))
    except Exception:
        age_s = float(frame_gap) / fps if frame_gap > 0 else 0.0

    if frame_gap <= 0 and age_s <= 0.0:
        return result
    if age_s <= (0.5 / fps):
        return result
    if age_s * 1000.0 > max(0.0, float(max_age_ms)):
        return result

    max_frame_gap = max(1, int(math.ceil(float(max_age_ms) / (1000.0 / fps))))
    if frame_gap > max_frame_gap:
        return result

    previous_by_id = _previous_mice_by_id(history, result)
    if not previous_by_id:
        return result

    max_shift_px = _motion_compensation_shift_limit(result)
    compensated_mice: list[TrackedMouseState] = []
    changed = False
    for mouse in result.tracked_mice:
        previous_pair = previous_by_id.get(_safe_int(getattr(mouse, "mouse_id", 0), 0))
        if previous_pair is None:
            compensated_mice.append(mouse)
            continue
        previous_result, previous_mouse = previous_pair
        compensated_mouse, mouse_changed = _compensate_mouse_state(
            mouse,
            previous_mouse,
            result,
            previous_result,
            age_s=age_s,
            preview_fps=fps,
            max_shift_px=max_shift_px,
        )
        compensated_mice.append(compensated_mouse)
        changed = changed or mouse_changed

    if not changed:
        return result
    return replace(result, tracked_mice=compensated_mice)


def _safe_int(value: object, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _previous_mice_by_id(
    history: Sequence[LiveDetectionResult],
    result: LiveDetectionResult,
) -> dict[int, tuple[LiveDetectionResult, TrackedMouseState]]:
    target_frame = _safe_int(getattr(result, "frame_index", 0), 0)
    target_ids = {
        _safe_int(getattr(mouse, "mouse_id", 0), 0)
        for mouse in getattr(result, "tracked_mice", []) or []
    }
    found: dict[int, tuple[LiveDetectionResult, TrackedMouseState]] = {}
    for previous_result in reversed(history or []):
        if previous_result is result:
            continue
        previous_frame = _safe_int(getattr(previous_result, "frame_index", target_frame), target_frame)
        if previous_frame >= target_frame:
            continue
        for previous_mouse in getattr(previous_result, "tracked_mice", []) or []:
            mouse_id = _safe_int(getattr(previous_mouse, "mouse_id", 0), 0)
            if mouse_id in target_ids and mouse_id not in found:
                found[mouse_id] = (previous_result, previous_mouse)
        if target_ids and len(found) >= len(target_ids):
            break
    return found


def _motion_compensation_shift_limit(result: LiveDetectionResult) -> float:
    width = max(1.0, float(getattr(result, "width", 0) or 0))
    height = max(1.0, float(getattr(result, "height", 0) or 0))
    return float(max(4.0, min(90.0, max(width, height) * 0.08)))


def _compensate_mouse_state(
    mouse: TrackedMouseState,
    previous_mouse: TrackedMouseState,
    result: LiveDetectionResult,
    previous_result: LiveDetectionResult,
    *,
    age_s: float,
    preview_fps: float,
    max_shift_px: float,
) -> tuple[TrackedMouseState, bool]:
    current_center = _point_pair(getattr(mouse, "center", None))
    previous_center = _point_pair(getattr(previous_mouse, "center", None))
    if current_center is None or previous_center is None:
        return mouse, False

    try:
        sample_dt_s = float(result.timestamp_s) - float(previous_result.timestamp_s)
    except Exception:
        sample_dt_s = 0.0
    if sample_dt_s <= 1e-6:
        frame_delta = _safe_int(getattr(result, "frame_index", 0), 0) - _safe_int(
            getattr(previous_result, "frame_index", 0),
            0,
        )
        sample_dt_s = float(frame_delta) / max(1.0, float(preview_fps)) if frame_delta > 0 else 0.0
    if sample_dt_s <= 1e-6 or sample_dt_s > 0.5:
        return mouse, False

    dx_sample = float(current_center[0]) - float(previous_center[0])
    dy_sample = float(current_center[1]) - float(previous_center[1])
    sample_distance = math.hypot(dx_sample, dy_sample)
    if sample_distance <= 0.25:
        return mouse, False

    lead_s = min(max(0.0, float(age_s)), sample_dt_s * 2.0)
    dx = (dx_sample / sample_dt_s) * lead_s
    dy = (dy_sample / sample_dt_s) * lead_s
    dx, dy = _clamp_vector(dx, dy, float(max_shift_px))
    if math.hypot(dx, dy) <= 0.5:
        return mouse, False

    width = max(1, _safe_int(getattr(result, "width", 0), 0))
    height = max(1, _safe_int(getattr(result, "height", 0), 0))
    return _shift_mouse_state(mouse, dx, dy, width, height), True


def _point_pair(value: object) -> Optional[tuple[float, float]]:
    try:
        x, y = value
        x = float(x)
        y = float(y)
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _clamp_vector(dx: float, dy: float, limit: float) -> tuple[float, float]:
    limit = max(0.0, float(limit))
    magnitude = math.hypot(float(dx), float(dy))
    if limit <= 0.0 or magnitude <= limit:
        return float(dx), float(dy)
    scale = limit / max(magnitude, 1e-9)
    return float(dx) * scale, float(dy) * scale


def _shift_mouse_state(
    mouse: TrackedMouseState,
    dx: float,
    dy: float,
    width: int,
    height: int,
) -> TrackedMouseState:
    bbox = _shift_bbox(getattr(mouse, "bbox", (0.0, 0.0, 0.0, 0.0)), dx, dy, width, height)
    center = _shift_point(getattr(mouse, "center", (0.0, 0.0)), dx, dy, width, height)
    keypoints = _shift_keypoints(getattr(mouse, "keypoints", None), dx, dy, width, height)
    mask = _shift_mask(getattr(mouse, "mask", None), dx, dy)
    return replace(mouse, center=center, bbox=bbox, keypoints=keypoints, mask=mask)


def _shift_bbox(
    bbox: object,
    dx: float,
    dy: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)
    max_x = max(0.0, float(width - 1))
    max_y = max(0.0, float(height - 1))
    nx1 = min(max(0.0, x1 + dx), max_x)
    ny1 = min(max(0.0, y1 + dy), max_y)
    nx2 = min(max(0.0, x2 + dx), max_x)
    ny2 = min(max(0.0, y2 + dy), max_y)
    return (float(nx1), float(ny1), float(max(nx1, nx2)), float(max(ny1, ny2)))


def _shift_point(
    point: object,
    dx: float,
    dy: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    pair = _point_pair(point)
    if pair is None:
        return (0.0, 0.0)
    max_x = max(0.0, float(width - 1))
    max_y = max(0.0, float(height - 1))
    return (
        float(min(max(0.0, pair[0] + dx), max_x)),
        float(min(max(0.0, pair[1] + dy), max_y)),
    )


def _shift_keypoints(
    keypoints: object,
    dx: float,
    dy: float,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    if keypoints is None:
        return None
    arr = np.asarray(keypoints, dtype=float)
    if arr.size == 0:
        return arr.reshape(-1, 2).copy()
    shifted = arr.reshape(-1, 2).copy()
    shifted[:, 0] = np.clip(shifted[:, 0] + float(dx), 0.0, max(0.0, float(width - 1)))
    shifted[:, 1] = np.clip(shifted[:, 1] + float(dy), 0.0, max(0.0, float(height - 1)))
    return shifted


def _shift_mask(mask: object, dx: float, dy: float) -> Optional[np.ndarray]:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return None
    shift_x = int(round(float(dx)))
    shift_y = int(round(float(dy)))
    if shift_x == 0 and shift_y == 0:
        return arr

    height, width = arr.shape[:2]
    src_x1 = max(0, -shift_x)
    src_x2 = min(width, width - shift_x) if shift_x >= 0 else width
    dst_x1 = max(0, shift_x)
    dst_x2 = min(width, width + shift_x) if shift_x < 0 else width
    src_y1 = max(0, -shift_y)
    src_y2 = min(height, height - shift_y) if shift_y >= 0 else height
    dst_y1 = max(0, shift_y)
    dst_y2 = min(height, height + shift_y) if shift_y < 0 else height

    if src_x2 <= src_x1 or src_y2 <= src_y1 or dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return np.zeros_like(arr, dtype=bool)
    shifted = np.zeros_like(arr, dtype=bool)
    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]
    return shifted


def clamp_mask_opacity(value: object, default: float = 0.18) -> float:
    """Normalize user mask opacity to a safe 0..1 alpha."""
    try:
        opacity = float(value)
    except (TypeError, ValueError):
        opacity = float(default)
    return float(min(1.0, max(0.0, opacity)))


def scale_live_detection_result_to_shape(
    result: Optional[LiveDetectionResult],
    frame_shape: tuple[int, int] | tuple[int, int, int],
) -> Optional[LiveDetectionResult]:
    """Return a copy of *result* whose coordinates/masks fit *frame_shape*.

    Live inference can run on the full recording frame while the preview or
    sidecar overlay renderer draws on a different-sized frame. This keeps masks,
    boxes, centers, and keypoints spatially aligned with the render target.
    """
    if result is None:
        return None
    if len(frame_shape) < 2:
        return result

    target_h = max(0, int(frame_shape[0]))
    target_w = max(0, int(frame_shape[1]))
    if target_w <= 0 or target_h <= 0:
        return result

    source_w = int(getattr(result, "width", 0) or 0)
    source_h = int(getattr(result, "height", 0) or 0)
    if source_w <= 0 or source_h <= 0:
        source_w, source_h = _infer_result_size_from_masks(result, target_w, target_h)

    scale_x = target_w / float(max(source_w, 1))
    scale_y = target_h / float(max(source_h, 1))
    same_coordinate_size = math.isclose(scale_x, 1.0) and math.isclose(scale_y, 1.0)

    scaled_mice = []
    for mouse in result.tracked_mice:
        bbox = tuple(float(v) for v in mouse.bbox)
        center = tuple(float(v) for v in mouse.center)
        if not same_coordinate_size:
            bbox = (
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y,
            )
            center = (center[0] * scale_x, center[1] * scale_y)

        keypoints = getattr(mouse, "keypoints", None)
        if keypoints is not None and not same_coordinate_size:
            keypoints_arr = np.asarray(keypoints, dtype=float).copy()
            if keypoints_arr.size:
                keypoints_arr = keypoints_arr.reshape(-1, 2)
                keypoints_arr[:, 0] *= scale_x
                keypoints_arr[:, 1] *= scale_y
            keypoints = keypoints_arr

        mask = _resize_mask_to_shape(getattr(mouse, "mask", None), target_h, target_w)
        scaled_mice.append(replace(mouse, center=center, bbox=bbox, mask=mask, keypoints=keypoints))

    return replace(result, width=target_w, height=target_h, tracked_mice=scaled_mice)


def _infer_result_size_from_masks(
    result: LiveDetectionResult,
    fallback_w: int,
    fallback_h: int,
) -> tuple[int, int]:
    for mouse in result.tracked_mice:
        mask = getattr(mouse, "mask", None)
        if mask is None:
            continue
        arr = np.asarray(mask)
        if arr.ndim >= 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
            return int(arr.shape[1]), int(arr.shape[0])
    return int(fallback_w), int(fallback_h)


def _resize_mask_to_shape(mask: object, target_h: int, target_w: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return None
    if arr.shape[:2] == (target_h, target_w):
        return arr
    resized = cv2.resize(
        arr.astype(np.uint8),
        (int(target_w), int(target_h)),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)
