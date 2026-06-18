"""Geometry-only mouse keypoint extraction from binary instance masks.

The extractor is designed for live use: no torch model, no Qt dependency, and
only NumPy plus OpenCV. It converts one mouse mask into the same eight-point
pose layout used by the live overlay:

    nose, left_ear, right_ear, neck, body, left_hip, right_hip, tail_tip

The important orientation cue is the tail filament. A mask opening removes the
thin tail from the thick body core, then the filament attachment marks the rear
of the animal. When the tail cannot be recovered, the stateful wrapper falls
back to the previous frame direction before using a low-confidence width cue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


KP_ORDER: tuple[str, ...] = (
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "body",
    "left_hip",
    "right_hip",
    "tail_tip",
)

EAR_FRAC = 0.18
NECK_FRAC = 0.30
HIP_FRAC = 0.74


@dataclass
class MaskGeom:
    mask: np.ndarray
    body_core: np.ndarray
    mask_points: np.ndarray
    body_points: np.ndarray
    centroid: np.ndarray
    body_radius: float
    tail_base: Optional[np.ndarray]
    tail_tip: Optional[np.ndarray]
    tail_confidence: float


@dataclass
class Skeleton:
    keypoints: np.ndarray
    scores: np.ndarray
    orientation_confidence: float
    method: str = "pca"

    def as_dict(self) -> dict[str, tuple[float, float]]:
        return {
            name: (float(point[0]), float(point[1]))
            for name, point in zip(KP_ORDER, self.keypoints)
        }


def extract_skeleton(
    mask: np.ndarray,
    *,
    method: str = "pca",
    direction_hint: Optional[np.ndarray] = None,
) -> Optional[Skeleton]:
    """Return an eight-point skeleton for one binary mask.

    ``direction_hint`` is a tail-to-nose vector from a previous frame. It is
    only trusted when the current frame has weak tail evidence.
    """
    geom = analyze_mask(mask)
    if geom is None:
        return None
    # The public method switch is kept for compatibility with the LISBET-style
    # API. The live path uses PCA because it is the most stable on noisy masks.
    normalized_method = str(method or "pca").strip().lower()
    if normalized_method not in {"pca", "contour", "geodesic"}:
        normalized_method = "pca"
    return keypoints_pca(geom, direction_hint=direction_hint, method=normalized_method)


def analyze_mask(mask: np.ndarray) -> Optional[MaskGeom]:
    """Separate a mouse mask into a thick body core and a tail filament."""
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return None
    full_binary = arr > 0
    if not bool(full_binary.any()):
        return None

    # Crop to the mask's bounding box (plus a small margin) so every heavy
    # OpenCV pass below (connected components, distance transform, up to four
    # morphological opens) runs on a tight ROI instead of the whole frame. A
    # mouse occupies a fraction of the frame, so this is the dominant speed-up
    # for live geometry. All work stays in crop space; the point-bearing
    # outputs are offset back to full-frame coordinates before returning, so
    # downstream keypoints are identical to the full-frame computation.
    ys0, xs0 = np.nonzero(full_binary)
    pad = 8
    y0 = max(0, int(ys0.min()) - pad)
    y1 = min(arr.shape[0], int(ys0.max()) + 1 + pad)
    x0 = max(0, int(xs0.min()) - pad)
    x1 = min(arr.shape[1], int(xs0.max()) + 1 + pad)
    offset = np.array([float(x0), float(y0)], dtype=np.float64)

    binary = full_binary[y0:y1, x0:x1].astype(np.uint8)

    binary = _largest_component(binary)
    if binary is None or not bool(binary.any()):
        return None
    binary = _close_small_holes(binary)

    ys, xs = np.nonzero(binary)
    if len(xs) < 12:
        return None

    mask_points = np.column_stack([xs, ys]).astype(np.float64)
    centroid = np.mean(mask_points, axis=0)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    body_radius = float(np.max(dist)) if dist.size else 1.0
    body_radius = max(1.0, body_radius)

    body_core = _opened_body_core(binary, body_radius)
    if body_core is None or int(np.count_nonzero(body_core)) < 8:
        body_core = binary.copy()

    body_ys, body_xs = np.nonzero(body_core)
    body_points = np.column_stack([body_xs, body_ys]).astype(np.float64)
    if len(body_points) < 8:
        body_points = mask_points
        body_core = binary.copy()

    tail_base, tail_tip, tail_confidence = _find_tail_attachment(
        binary,
        body_core,
        centroid,
        body_radius,
    )

    # Lift every point-bearing result from crop space to full-frame space.
    # body_points may alias mask_points (small-mask fallback), so offset once.
    body_points_shared = body_points is mask_points
    mask_points = mask_points + offset
    body_points = mask_points if body_points_shared else (body_points + offset)
    centroid = centroid + offset
    if tail_base is not None:
        tail_base = np.asarray(tail_base, dtype=np.float64) + offset
    if tail_tip is not None:
        tail_tip = np.asarray(tail_tip, dtype=np.float64) + offset

    return MaskGeom(
        mask=binary.astype(bool),
        body_core=body_core.astype(bool),
        mask_points=mask_points,
        body_points=body_points,
        centroid=centroid.astype(np.float64),
        body_radius=float(body_radius),
        tail_base=tail_base,
        tail_tip=tail_tip,
        tail_confidence=float(tail_confidence),
    )


def keypoints_pca(
    geom: MaskGeom,
    *,
    direction_hint: Optional[np.ndarray] = None,
    method: str = "pca",
) -> Optional[Skeleton]:
    """Build pose keypoints from body-core PCA and mask cross sections."""
    points = np.asarray(geom.body_points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return None

    center = np.mean(points, axis=0)
    centered = points - center
    try:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float64)
    except Exception:
        axis = np.array([1.0, 0.0], dtype=np.float64)
        eigvals = np.array([0.0, 1.0], dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        axis = np.array([1.0, 0.0], dtype=np.float64)
    else:
        axis /= axis_norm

    body_proj = (points - center) @ axis
    low_proj = float(np.min(body_proj))
    high_proj = float(np.max(body_proj))
    body_length = max(1.0, high_proj - low_proj)

    orientation_confidence = 0.0
    if geom.tail_base is not None:
        tail_proj = float((np.asarray(geom.tail_base, dtype=np.float64) - center) @ axis)
        if abs(tail_proj - low_proj) <= abs(tail_proj - high_proj):
            rear_proj = low_proj
            nose_proj = high_proj
        else:
            rear_proj = high_proj
            nose_proj = low_proj
        end_gap = abs(abs(tail_proj - low_proj) - abs(tail_proj - high_proj))
        orientation_confidence = float(np.clip(0.35 + geom.tail_confidence * 0.55 + 0.10 * end_gap / body_length, 0.0, 1.0))
    else:
        hint_dot = _orientation_hint_dot(axis, direction_hint)
        if hint_dot is not None:
            if hint_dot >= 0.0:
                nose_proj = high_proj
                rear_proj = low_proj
            else:
                nose_proj = low_proj
                rear_proj = high_proj
            orientation_confidence = 0.42
        else:
            low_width = _section_width(geom.mask_points, center, axis, low_proj, body_length)
            high_width = _section_width(geom.mask_points, center, axis, high_proj, body_length)
            if low_width >= high_width:
                rear_proj = low_proj
                nose_proj = high_proj
            else:
                rear_proj = high_proj
                nose_proj = low_proj
            denom = max(1.0, low_width, high_width)
            orientation_confidence = float(np.clip(0.15 + 0.20 * abs(low_width - high_width) / denom, 0.0, 0.35))

    nose_to_rear_sign = 1.0 if rear_proj >= nose_proj else -1.0
    tail_to_nose_vec = axis * (-nose_to_rear_sign)
    tail_to_nose_vec = _unit(tail_to_nose_vec, fallback=np.array([1.0, 0.0], dtype=np.float64))
    lateral = np.array([-tail_to_nose_vec[1], tail_to_nose_vec[0]], dtype=np.float64)

    nose = _end_point(geom.mask_points, center, axis, nose_proj, prefer_high=nose_proj > rear_proj)
    if geom.tail_tip is not None:
        tail = np.asarray(geom.tail_tip, dtype=np.float64)
        tail_score = 0.70 + 0.25 * float(np.clip(geom.tail_confidence, 0.0, 1.0))
    else:
        tail = _end_point(geom.mask_points, center, axis, rear_proj, prefer_high=rear_proj > nose_proj)
        tail_score = 0.45

    neck_center = _section_center(geom.mask_points, center, axis, nose_proj, rear_proj, NECK_FRAC, body_length)
    ear_center = _section_center(geom.mask_points, center, axis, nose_proj, rear_proj, EAR_FRAC, body_length)
    hip_center = _section_center(geom.mask_points, center, axis, nose_proj, rear_proj, HIP_FRAC, body_length)

    left_ear, right_ear, ear_width = _section_edges(
        geom.mask_points,
        center,
        axis,
        lateral,
        _fraction_to_projection(nose_proj, rear_proj, EAR_FRAC),
        body_length,
        fallback_center=ear_center,
    )
    left_hip, right_hip, hip_width = _section_edges(
        geom.mask_points,
        center,
        axis,
        lateral,
        _fraction_to_projection(nose_proj, rear_proj, HIP_FRAC),
        body_length,
        fallback_center=hip_center,
    )

    keypoints = np.vstack(
        [
            nose,
            left_ear,
            right_ear,
            neck_center,
            geom.centroid,
            left_hip,
            right_hip,
            tail,
        ]
    ).astype(np.float64)

    axis_strength = _axis_strength(eigvals)
    edge_score = float(np.clip(min(ear_width, hip_width) / max(1.0, geom.body_radius * 1.5), 0.35, 1.0))
    base_score = float(np.clip(0.45 + 0.35 * axis_strength, 0.45, 0.90))
    scores = np.array(
        [
            float(np.clip(0.50 + 0.45 * orientation_confidence, 0.0, 1.0)),
            edge_score,
            edge_score,
            base_score,
            1.0,
            edge_score,
            edge_score,
            float(np.clip(tail_score, 0.0, 1.0)),
        ],
        dtype=np.float64,
    )

    return Skeleton(
        keypoints=keypoints,
        scores=scores,
        orientation_confidence=float(np.clip(orientation_confidence, 0.0, 1.0)),
        method=method,
    )


class MaskSkeletonExtractor:
    """Stateful live wrapper for mask-to-skeleton extraction."""

    def __init__(
        self,
        *,
        method: str = "pca",
        smooth: bool = True,
        alpha: float = 0.65,
        max_jump: float = 90.0,
    ) -> None:
        self.method = str(method or "pca")
        self.smooth = bool(smooth)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.max_jump = float(max(1.0, max_jump))
        self._states: dict[object, dict[str, np.ndarray]] = {}

    def reset(self) -> None:
        self._states.clear()

    def estimate(
        self,
        mask: np.ndarray,
        *,
        track_id: object = "default",
        offset: tuple[float, float] | np.ndarray = (0.0, 0.0),
    ) -> Optional[Skeleton]:
        previous = self._states.get(track_id)
        direction_hint = None
        if previous is not None:
            direction_hint = previous.get("direction")

        skeleton = extract_skeleton(mask, method=self.method, direction_hint=direction_hint)
        if skeleton is None:
            return None

        keypoints = np.asarray(skeleton.keypoints, dtype=np.float64).reshape(-1, 2)
        offset_arr = np.asarray(offset, dtype=np.float64).reshape(2)
        if np.any(offset_arr):
            keypoints = keypoints + offset_arr.reshape(1, 2)
            skeleton = Skeleton(
                keypoints=keypoints,
                scores=np.asarray(skeleton.scores, dtype=np.float64).reshape(-1),
                orientation_confidence=skeleton.orientation_confidence,
                method=skeleton.method,
            )
        if previous is not None:
            keypoints = self._lock_left_right(previous.get("keypoints"), keypoints)
            if self.smooth:
                keypoints = _smooth_keypoints(previous.get("keypoints"), keypoints, self.alpha, self.max_jump)
            skeleton = Skeleton(
                keypoints=keypoints,
                scores=np.asarray(skeleton.scores, dtype=np.float64).reshape(-1),
                orientation_confidence=skeleton.orientation_confidence,
                method=skeleton.method,
            )

        direction = _tail_to_nose_direction(skeleton.keypoints)
        self._states[track_id] = {
            "keypoints": np.asarray(skeleton.keypoints, dtype=np.float64).reshape(-1, 2).copy(),
            "direction": direction,
        }
        return skeleton

    @staticmethod
    def _lock_left_right(previous: Optional[np.ndarray], current: np.ndarray) -> np.ndarray:
        if previous is None:
            return current
        prev = np.asarray(previous, dtype=np.float64).reshape(-1, 2)
        cur = np.asarray(current, dtype=np.float64).reshape(-1, 2).copy()
        if prev.shape != cur.shape or len(cur) < len(KP_ORDER):
            return cur
        for left_index, right_index in ((1, 2), (5, 6)):
            same = _finite_distance(cur[left_index], prev[left_index]) + _finite_distance(cur[right_index], prev[right_index])
            swapped = _finite_distance(cur[left_index], prev[right_index]) + _finite_distance(cur[right_index], prev[left_index])
            if swapped + 1e-6 < same:
                cur[[left_index, right_index]] = cur[[right_index, left_index]]
        return cur


def _largest_component(binary: np.ndarray) -> Optional[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    return (labels == largest).astype(np.uint8)


def _close_small_holes(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def _opened_body_core(binary: np.ndarray, body_radius: float) -> Optional[np.ndarray]:
    area = int(np.count_nonzero(binary))
    if area <= 0:
        return None
    radius_candidates = [
        int(round(body_radius * 0.55)),
        int(round(body_radius * 0.42)),
        int(round(body_radius * 0.32)),
        2,
    ]
    seen: set[int] = set()
    for radius in radius_candidates:
        radius = int(max(1, min(25, radius)))
        if radius in seen:
            continue
        seen.add(radius)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        opened = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        component = _largest_component(opened)
        if component is None:
            continue
        component_area = int(np.count_nonzero(component))
        if component_area >= max(8, int(area * 0.22)):
            return component.astype(np.uint8)
    return None


def _find_tail_attachment(
    binary: np.ndarray,
    body_core: np.ndarray,
    centroid: np.ndarray,
    body_radius: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    filament = np.logical_and(binary.astype(bool), ~body_core.astype(bool)).astype(np.uint8)
    if int(np.count_nonzero(filament)) < 3:
        return None, None, 0.0

    core_dilate_radius = max(1, int(round(min(5.0, max(1.0, body_radius * 0.20)))))
    core_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (core_dilate_radius * 2 + 1, core_dilate_radius * 2 + 1),
    )
    core_neighborhood = cv2.dilate(body_core.astype(np.uint8), core_kernel).astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filament, connectivity=8)
    if num_labels <= 1:
        return None, None, 0.0

    best_score = -float("inf")
    best_base = None
    best_tip = None
    best_length = 0.0
    min_area = max(3, int(0.002 * np.count_nonzero(binary)))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        touching = np.logical_and(component, core_neighborhood)
        ys, xs = np.nonzero(component)
        if len(xs) == 0:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        if bool(touching.any()):
            tys, txs = np.nonzero(touching)
            base = np.array([float(np.mean(txs)), float(np.mean(tys))], dtype=np.float64)
            touch_bonus = 1.0
        else:
            distances = np.linalg.norm(pts - centroid.reshape(1, 2), axis=1)
            base = pts[int(np.argmin(distances))]
            touch_bonus = 0.35
        tip_distances = np.linalg.norm(pts - base.reshape(1, 2), axis=1)
        tip_index = int(np.argmax(tip_distances))
        tip = pts[tip_index]
        length = float(tip_distances[tip_index])
        score = touch_bonus * length + 0.03 * float(area)
        if score > best_score:
            best_score = score
            best_base = base
            best_tip = tip
            best_length = length

    if best_base is None or best_tip is None:
        return None, None, 0.0
    confidence = float(np.clip(best_length / max(1.0, body_radius * 1.8), 0.0, 1.0))
    return best_base, best_tip, confidence


def _orientation_hint_dot(axis: np.ndarray, direction_hint: Optional[np.ndarray]) -> Optional[float]:
    if direction_hint is None:
        return None
    hint = np.asarray(direction_hint, dtype=np.float64).reshape(-1)
    if hint.size < 2 or not np.all(np.isfinite(hint[:2])):
        return None
    hint_norm = float(np.linalg.norm(hint[:2]))
    if hint_norm <= 1e-6:
        return None
    return float(np.dot(axis, hint[:2] / hint_norm))


def _section_width(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    target_proj: float,
    body_length: float,
) -> float:
    lateral = np.array([-axis[1], axis[0]], dtype=np.float64)
    selected = _section_points(points, center, axis, target_proj, body_length)
    if len(selected) < 2:
        return 0.0
    lat = (selected - center.reshape(1, 2)) @ lateral
    return float(np.max(lat) - np.min(lat))


def _axis_strength(eigvals: np.ndarray) -> float:
    vals = np.asarray(eigvals, dtype=np.float64).reshape(-1)
    if vals.size < 2:
        return 0.0
    vals = np.sort(np.maximum(vals, 0.0))
    denom = float(vals[-1] + vals[-2])
    if denom <= 1e-9:
        return 0.0
    return float(np.clip((vals[-1] - vals[-2]) / denom, 0.0, 1.0))


def _fraction_to_projection(nose_proj: float, rear_proj: float, frac: float) -> float:
    return float(nose_proj + float(frac) * (rear_proj - nose_proj))


def _section_center(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    nose_proj: float,
    rear_proj: float,
    frac: float,
    body_length: float,
) -> np.ndarray:
    target = _fraction_to_projection(nose_proj, rear_proj, frac)
    selected = _section_points(points, center, axis, target, body_length)
    if len(selected) > 0:
        return np.mean(selected, axis=0).astype(np.float64)
    return center + axis * target


def _section_edges(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    lateral: np.ndarray,
    target_proj: float,
    body_length: float,
    *,
    fallback_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    selected = _section_points(points, center, axis, target_proj, body_length)
    if len(selected) < 2:
        offset = lateral * max(2.0, body_length * 0.035)
        return fallback_center + offset, fallback_center - offset, float(np.linalg.norm(offset) * 2.0)
    lat = (selected - center.reshape(1, 2)) @ lateral
    left = selected[int(np.argmax(lat))]
    right = selected[int(np.argmin(lat))]
    return left.astype(np.float64), right.astype(np.float64), float(np.max(lat) - np.min(lat))


def _section_points(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    target_proj: float,
    body_length: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    projs = (pts - center.reshape(1, 2)) @ axis
    base_window = max(2.0, float(body_length) * 0.045)
    for multiplier in (1.0, 1.8, 2.8, 4.0):
        mask = np.abs(projs - float(target_proj)) <= base_window * multiplier
        if int(np.count_nonzero(mask)) >= 3:
            return pts[mask]
    nearest = np.argsort(np.abs(projs - float(target_proj)))[: min(5, len(pts))]
    return pts[nearest]


def _end_point(
    points: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    target_proj: float,
    *,
    prefer_high: bool,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return center + axis * target_proj
    projs = (pts - center.reshape(1, 2)) @ axis
    index = int(np.argmax(projs) if prefer_high else np.argmin(projs))
    return pts[index].astype(np.float64)


def _unit(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return np.asarray(fallback, dtype=np.float64).reshape(2)
    return vec / norm


def _tail_to_nose_direction(keypoints: np.ndarray) -> np.ndarray:
    kp = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    if len(kp) < len(KP_ORDER):
        return np.array([0.0, 0.0], dtype=np.float64)
    nose = kp[0]
    tail = kp[7]
    if not (np.all(np.isfinite(nose)) and np.all(np.isfinite(tail))):
        return np.array([0.0, 0.0], dtype=np.float64)
    return nose - tail


def _smooth_keypoints(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    alpha: float,
    max_jump: float,
) -> np.ndarray:
    if previous is None:
        return current.copy()
    prev = np.asarray(previous, dtype=np.float64).reshape(-1, 2)
    cur = np.asarray(current, dtype=np.float64).reshape(-1, 2)
    if prev.shape != cur.shape:
        return cur.copy()
    out = cur.copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    for index in range(len(cur)):
        cx, cy = cur[index]
        px, py = prev[index]
        if not (np.isfinite(cx) and np.isfinite(cy)):
            out[index] = (np.nan, np.nan)
            continue
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        if float(np.hypot(cx - px, cy - py)) > float(max_jump):
            continue
        out[index] = (a * cx + (1.0 - a) * px, a * cy + (1.0 - a) * py)
    return out


def _finite_distance(a: np.ndarray, b: np.ndarray) -> float:
    pa = np.asarray(a, dtype=np.float64).reshape(2)
    pb = np.asarray(b, dtype=np.float64).reshape(2)
    if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
        return 1e9
    return float(np.linalg.norm(pa - pb))
