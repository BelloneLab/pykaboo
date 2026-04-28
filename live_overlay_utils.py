"""Retention policy helpers for live detection overlays."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np

from live_detection_types import LiveDetectionResult


def overlay_retention_limits(
    *,
    preview_fps: float,
    inference_ms: float = 0.0,
) -> tuple[float, int]:
    """Return the max overlay age in ms and allowable frame gap."""
    fps = max(1.0, float(preview_fps or 0.0))
    frame_interval_ms = 1000.0 / fps
    # Live preview should prefer dropping stale masks over showing old animal
    # positions. Inference latency is reported separately; do not extend mask
    # retention to match slow models because that creates visible mask lag.
    _ = inference_ms
    max_age_ms = min(120.0, max(60.0, 2.25 * frame_interval_ms))
    max_frame_gap = min(2, max(1, int(math.ceil(max_age_ms / frame_interval_ms))))
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
