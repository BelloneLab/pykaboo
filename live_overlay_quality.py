"""Mask and keypoint cleanup for stable, mask-limited live overlays.

These helpers make live segmentation + pose overlays look "perfect":

* ``clean_instance_mask`` keeps only the largest connected blob of a mask and
  closes pinholes, so a detection draws one solid body instead of a speckled
  cloud that flickers frame to frame.
* ``clamp_keypoints_to_mask`` drops keypoints that fall outside the (slightly
  dilated) body mask, so pose never paints joints onto the background.

All functions are pure and operate on NumPy arrays, which keeps them fast and
unit-testable without a camera or a GPU.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


# 8-keypoint mouse pose topology (matches the project's YOLO pose model).
# Index order: 0 nose, 1 left_ear, 2 right_ear, 3 neck, 4 body,
# 5 left_hip, 6 right_hip, 7 tail_tip.
MOUSE_POSE_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "left_ear", "right_ear", "neck", "body", "left_hip", "right_hip", "tail_tip",
)
# Skeleton edges as 0-indexed keypoint pairs (converted from the dataset's
# 1-indexed skeleton). Forms a head diamond + neck→hips→tail spine.
MOUSE_POSE_SKELETON: tuple[tuple[int, int], ...] = (
    (1, 0), (0, 2), (1, 3), (3, 2),   # head: ears↔nose↔neck
    (3, 4),                            # neck → body
    (3, 5), (3, 6),                    # neck → hips
    (5, 7), (6, 7),                    # hips → tail tip
)

# Vivid, well-separated identity colours (RGB). Index by mouse id.
_IDENTITY_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (54, 211, 255),   # cyan
    (255, 138, 196),  # pink
    (124, 240, 150),  # green
    (255, 196, 84),   # amber
    (178, 148, 255),  # violet
    (255, 110, 110),  # coral
    (96, 224, 222),   # teal
    (240, 230, 120),  # yellow
)


def identity_color_rgb(mouse_id: int) -> Tuple[int, int, int]:
    """Stable vivid colour for a given mouse identity (RGB)."""
    palette = _IDENTITY_PALETTE_RGB
    return palette[int(mouse_id) % len(palette)]


def skeleton_for_keypoint_count(count: int) -> List[Tuple[int, int]]:
    """Return skeleton edges valid for *count* keypoints.

    Uses the mouse topology when the count matches; otherwise falls back to a
    simple index chain so any pose model still renders connected joints.
    """
    if count == len(MOUSE_POSE_KEYPOINT_NAMES):
        return list(MOUSE_POSE_SKELETON)
    return [(i, i + 1) for i in range(max(0, int(count) - 1))]


def clean_instance_mask(
    mask: Optional[np.ndarray],
    *,
    close_kernel: int = 5,
    min_area_ratio: float = 0.02,
) -> Optional[np.ndarray]:
    """Return a single-blob, hole-closed copy of *mask*.

    The largest connected component is kept; smaller specks (often transient
    segmentation noise that flickers) are discarded. ``min_area_ratio`` removes
    the result entirely if the kept blob is a negligible fraction of the frame,
    which suppresses spurious one-frame detections.
    """
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return None
    binary = (arr > 0).astype(np.uint8)
    if not binary.any():
        return None

    if close_kernel and close_kernel >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return None
    # Label 0 is background; pick the largest foreground component.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    largest_area = int(stats[largest, cv2.CC_STAT_AREA])
    if largest_area < int(min_area_ratio * binary.size):
        return None
    return labels == largest


def _dilated_mask(mask: np.ndarray, dilation: int) -> np.ndarray:
    if dilation <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def clamp_keypoints_to_mask(
    keypoints: Optional[np.ndarray],
    scores: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    *,
    dilation: int = 8,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Mark keypoints that land outside the dilated *mask* as invalid (NaN).

    Returns copies of (keypoints, scores). Keypoints outside the body get NaN
    coordinates and zero score so the renderer skips them. If no mask is given
    the keypoints pass through unchanged.
    """
    if keypoints is None:
        return None, scores
    kp = np.asarray(keypoints, dtype=float).reshape(-1, 2).copy()
    sc = (
        np.asarray(scores, dtype=float).reshape(-1).copy()
        if scores is not None
        else None
    )
    if mask is None:
        return kp, sc
    body = np.asarray(mask, dtype=bool)
    if body.ndim != 2 or not body.any():
        return kp, sc
    region = _dilated_mask(body, dilation)
    h, w = region.shape[:2]
    for i, (x, y) in enumerate(kp):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi = int(round(x))
        yi = int(round(y))
        inside = 0 <= xi < w and 0 <= yi < h and bool(region[yi, xi])
        if not inside:
            kp[i] = (np.nan, np.nan)
            if sc is not None and i < len(sc):
                sc[i] = 0.0
    return kp, sc


def smooth_keypoints(
    previous: Optional[np.ndarray],
    current: Optional[np.ndarray],
    *,
    alpha: float = 0.6,
    max_jump: float = 80.0,
) -> Optional[np.ndarray]:
    """Exponentially smooth keypoint tracks to remove per-frame jitter.

    ``alpha`` weights the new measurement (1.0 = no smoothing). A keypoint that
    jumps more than ``max_jump`` pixels is treated as a fresh measurement (the
    animal really moved or a new joint appeared) and is not blended.
    """
    if current is None:
        return previous
    cur = np.asarray(current, dtype=float).reshape(-1, 2)
    if previous is None:
        return cur.copy()
    prev = np.asarray(previous, dtype=float).reshape(-1, 2)
    if prev.shape != cur.shape:
        return cur.copy()
    out = cur.copy()
    a = float(min(1.0, max(0.0, alpha)))
    for i in range(len(cur)):
        cx, cy = cur[i]
        px, py = prev[i]
        if not (np.isfinite(cx) and np.isfinite(cy)):
            out[i] = (np.nan, np.nan)
            continue
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        if float(np.hypot(cx - px, cy - py)) > float(max_jump):
            continue
        out[i] = (a * cx + (1.0 - a) * px, a * cy + (1.0 - a) * py)
    return out
