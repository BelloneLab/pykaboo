"""Deterministic mask-derived feature extraction.

Features are computed from decoded instance masks using OpenCV and NumPy so the
package does not require scikit-image. The output is a tidy feature table keyed
by ``(video_id, frame_idx, subject_id, object_id)``. Single-mouse features use an
empty ``object_id``; pairwise social features fill both ids.

Feature groups:

- geometric  : centroid, bbox, area, perimeter, solidity, eccentricity, axes...
- motion     : speed, acceleration, jerk, angular velocity, mask IoU with prev...
- social     : nearest-animal centroid/bbox distance and overlap per identity
- rolling    : mean/std/min/max/median/slope over configurable second windows
- pairwise   : centroid distance, IoU, approach/follow scores, contact flag...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .coco_masks import CocoVideo, MaskRecord, load_coco_videos
from .config import AppConfig, FeaturesConfig, load_config
from .storage import write_table

GEOMETRIC_FEATURES = [
    "center_x",
    "center_y",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "area",
    "perimeter",
    "convex_area",
    "solidity",
    "extent",
    "eccentricity",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "aspect_ratio",
    "equivalent_diameter",
    "mask_score",
]

MOTION_FEATURES = [
    "velocity_x",
    "velocity_y",
    "speed",
    "speed_body_norm",
    "forward_velocity",
    "lateral_velocity",
    "movement_bearing_sin",
    "movement_bearing_cos",
    "heading_alignment",
    "acceleration",
    "acceleration_x",
    "acceleration_y",
    "acceleration_magnitude",
    "acceleration_body_norm",
    "jerk",
    "jerk_x",
    "jerk_y",
    "jerk_magnitude",
    "jerk_body_norm",
    "angular_velocity",
    "angular_acceleration",
    "area_velocity",
    "area_acceleration",
    "orientation_sin",
    "orientation_cos",
    "delta_center_x",
    "delta_center_y",
    "mask_iou_with_previous_frame",
]

INTERANIMAL_FEATURES = [
    "nearest_neighbor_count",
    "nearest_centroid_distance",
    "nearest_centroid_dx",
    "nearest_centroid_dy",
    "nearest_centroid_abs_dx",
    "nearest_centroid_abs_dy",
    "nearest_centroid_distance_velocity",
    "nearest_centroid_distance_acceleration",
    "nearest_bbox_edge_distance_velocity",
    "nearest_bbox_edge_distance_acceleration",
    "nearest_subject_speed",
    "nearest_neighbor_speed",
    "nearest_relative_velocity_x",
    "nearest_relative_velocity_y",
    "nearest_relative_speed",
    "nearest_relative_acceleration_x",
    "nearest_relative_acceleration_y",
    "nearest_relative_acceleration_magnitude",
    "nearest_radial_velocity",
    "nearest_radial_acceleration",
    "nearest_lateral_speed",
    "nearest_approach_speed",
    "nearest_bbox_edge_distance",
    "nearest_bbox_iou",
    "nearest_bbox_contact_flag",
    "nearest_area_ratio",
    "nearest_bearing_sin",
    "nearest_bearing_cos",
    "nearest_heading_alignment",
]

PAIRWISE_DISTANCE_FEATURES = [
    "pair_centroid_distance",
    "pair_centroid_dx",
    "pair_centroid_dy",
    "pair_centroid_abs_dx",
    "pair_centroid_abs_dy",
    "pair_centroid_distance_body_norm",
    "pair_relative_speed",
    "pair_relative_acceleration",
    "pair_approaching_score",
    "pair_following_score",
    "pair_relative_orientation",
    "pair_bbox_edge_distance",
    "pair_bbox_edge_distance_body_norm",
    "pair_bbox_iou",
    "pair_contact_flag",
    "pair_bbox_contact_flag",
    "pair_area_ratio",
    "pair_area_abs_difference",
    "pair_subject_speed",
    "pair_object_speed",
    "pair_speed_ratio",
    "pair_subject_acceleration",
    "pair_object_acceleration",
    "pair_relative_velocity_x",
    "pair_relative_velocity_y",
    "pair_relative_motion_speed",
    "pair_relative_motion_speed_body_norm",
    "pair_radial_velocity",
    "pair_radial_velocity_body_norm",
    "pair_lateral_velocity",
    "pair_lateral_speed",
    "pair_lateral_speed_body_norm",
    "pair_distance_acceleration",
    "pair_approach_acceleration",
    "pair_relative_acceleration_x",
    "pair_relative_acceleration_y",
    "pair_relative_acceleration_magnitude",
    "pair_radial_acceleration",
    "pair_bbox_edge_distance_velocity",
    "pair_bbox_edge_distance_acceleration",
]

ROLLING_STATS = ["mean", "std", "min", "max", "median", "slope"]

# Columns that should not receive rolling summaries (positions, raw flags).
ROLLING_EXCLUDE = {"missing_mask_flag"}


def mask_shape_features(
    mask: np.ndarray | None, record: MaskRecord, frame_width: int, frame_height: int
) -> dict[str, float]:
    """Compute geometric shape features for one instance at one frame."""

    bx, by, bw, bh = record.bbox
    out: dict[str, float] = {name: float("nan") for name in GEOMETRIC_FEATURES}
    out["bbox_x"] = bx
    out["bbox_y"] = by
    out["bbox_width"] = bw
    out["bbox_height"] = bh
    out["mask_score"] = float(record.score)
    out["missing_mask_flag"] = 0.0
    out["aspect_ratio"] = bw / bh if bh > 0 else float("nan")

    if mask is None or mask.sum() < 2:
        out["center_x"] = bx + bw / 2.0
        out["center_y"] = by + bh / 2.0
        out["area"] = float(mask.sum()) if mask is not None else bw * bh
        out["missing_mask_flag"] = 1.0 if mask is None else 0.0
        return out

    mask_u8 = mask.astype(np.uint8)
    moments = cv2.moments(mask_u8, binaryImage=True)
    area = moments["m00"]
    if area <= 0:
        out["missing_mask_flag"] = 1.0
        out["center_x"] = bx + bw / 2.0
        out["center_y"] = by + bh / 2.0
        return out

    cx = moments["m10"] / area
    cy = moments["m01"] / area
    mu20 = moments["mu20"] / area
    mu02 = moments["mu02"] / area
    mu11 = moments["mu11"] / area

    common = np.sqrt(max((mu20 - mu02) ** 2 + 4.0 * mu11**2, 0.0))
    lambda1 = (mu20 + mu02 + common) / 2.0
    lambda2 = (mu20 + mu02 - common) / 2.0
    lambda1 = max(lambda1, 1e-9)
    lambda2 = max(lambda2, 0.0)

    out["center_x"] = cx
    out["center_y"] = cy
    out["area"] = float(area)
    out["major_axis_length"] = 4.0 * np.sqrt(lambda1)
    out["minor_axis_length"] = 4.0 * np.sqrt(lambda2)
    out["eccentricity"] = float(np.sqrt(max(1.0 - lambda2 / lambda1, 0.0)))
    out["orientation"] = 0.5 * np.arctan2(2.0 * mu11, (mu20 - mu02))
    out["equivalent_diameter"] = float(np.sqrt(4.0 * area / np.pi))

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        largest = max(contours, key=cv2.contourArea)
        out["perimeter"] = float(cv2.arcLength(largest, True))
        hull = cv2.convexHull(largest)
        convex_area = float(cv2.contourArea(hull))
        out["convex_area"] = convex_area
        out["solidity"] = float(area / convex_area) if convex_area > 0 else float("nan")
    bbox_area = bw * bh
    out["extent"] = float(area / bbox_area) if bbox_area > 0 else float("nan")
    return out


def mask_iou(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> float:
    if mask_a is None or mask_b is None:
        return float("nan")
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def safe_divide(numer: Any, denom: Any, eps: float = 1e-9) -> np.ndarray:
    numer_arr = np.asarray(numer, dtype=np.float64)
    denom_arr = np.asarray(denom, dtype=np.float64)
    return np.divide(
        numer_arr,
        denom_arr,
        out=np.zeros_like(numer_arr, dtype=np.float64),
        where=np.abs(denom_arr) > eps,
    )


def temporal_derivative(
    values: Any, frame_indices: Any, frame_rate: float
) -> np.ndarray:
    """First derivative per second, respecting gaps in frame indices."""

    arr = np.asarray(values, dtype=np.float64)
    frames = np.asarray(frame_indices, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    if arr.size < 2:
        return out
    frame_delta = np.diff(frames)
    value_delta = np.diff(arr)
    scale = np.divide(
        frame_rate,
        frame_delta,
        out=np.zeros_like(frame_delta, dtype=np.float64),
        where=frame_delta > 0,
    )
    valid = np.isfinite(value_delta) & np.isfinite(scale) & (scale > 0)
    out[1:] = np.where(valid, value_delta * scale, 0.0)
    return out


def body_scale_from_geometry(df: pd.DataFrame) -> np.ndarray:
    """Representative body length for body-normalized kinematic features."""

    n = len(df)
    scale = np.full(n, np.nan, dtype=np.float64)
    for col in ("major_axis_length", "equivalent_diameter"):
        if col in df:
            candidate = df[col].to_numpy(dtype=np.float64)
            scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, candidate)
    if {"bbox_width", "bbox_height"}.issubset(df.columns):
        bbox_diag = np.hypot(
            df["bbox_width"].to_numpy(dtype=np.float64),
            df["bbox_height"].to_numpy(dtype=np.float64),
        )
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, bbox_diag)
    return np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)


def bbox_edge_distance(
    ax: Any,
    ay: Any,
    aw: Any,
    ah: Any,
    bx: Any,
    by: Any,
    bw: Any,
    bh: Any,
) -> np.ndarray:
    """Distance between bbox edges, zero when boxes touch or overlap."""

    ax = np.asarray(ax, dtype=np.float64)
    ay = np.asarray(ay, dtype=np.float64)
    aw = np.asarray(aw, dtype=np.float64)
    ah = np.asarray(ah, dtype=np.float64)
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    bw = np.asarray(bw, dtype=np.float64)
    bh = np.asarray(bh, dtype=np.float64)

    sep_x = np.maximum(np.maximum(bx - (ax + aw), ax - (bx + bw)), 0.0)
    sep_y = np.maximum(np.maximum(by - (ay + ah), ay - (by + bh)), 0.0)
    return np.hypot(sep_x, sep_y)


def bbox_iou_from_geometry(
    ax: Any,
    ay: Any,
    aw: Any,
    ah: Any,
    bx: Any,
    by: Any,
    bw: Any,
    bh: Any,
) -> np.ndarray:
    """Intersection-over-union for bbox geometry columns."""

    ax = np.asarray(ax, dtype=np.float64)
    ay = np.asarray(ay, dtype=np.float64)
    aw = np.asarray(aw, dtype=np.float64)
    ah = np.asarray(ah, dtype=np.float64)
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    bw = np.asarray(bw, dtype=np.float64)
    bh = np.asarray(bh, dtype=np.float64)

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    inter_w = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax, bx))
    inter_h = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay, by))
    inter = inter_w * inter_h
    area_a = np.maximum(aw, 0.0) * np.maximum(ah, 0.0)
    area_b = np.maximum(bw, 0.0) * np.maximum(bh, 0.0)
    union = area_a + area_b - inter
    return safe_divide(inter, union)


def normalize_geometry(
    df: pd.DataFrame, frame_width: int, frame_height: int, config: FeaturesConfig
) -> pd.DataFrame:
    """Normalize spatial features by frame size and/or body size in place."""

    diag = float(np.hypot(frame_width, frame_height)) or 1.0
    if config.normalize_by_frame_size and frame_width > 0 and frame_height > 0:
        for col in ["center_x", "bbox_x", "bbox_width", "major_axis_length",
                    "minor_axis_length", "equivalent_diameter", "perimeter"]:
            if col in df:
                df[col] = df[col] / frame_width
        for col in ["center_y", "bbox_y", "bbox_height"]:
            if col in df:
                df[col] = df[col] / frame_height
        for col in ["area", "convex_area"]:
            if col in df:
                df[col] = df[col] / (frame_width * frame_height)
    if config.normalize_by_body_size:
        median_area = df["area"].median()
        if median_area and median_area > 0 and not np.isnan(median_area):
            df["area_norm"] = df["area"] / median_area
    return df


def interpolate_missing(
    df: pd.DataFrame, columns: list[str], max_gap: int, policy: str
) -> pd.DataFrame:
    """Fill short gaps of missing-mask frames according to policy."""

    if policy == "keep_nan":
        return df
    if policy == "zero":
        df[columns] = df[columns].fillna(0.0)
        return df
    # interpolate_short_gaps
    for col in columns:
        series = df[col]
        interpolated = series.interpolate(
            method="linear", limit=max_gap, limit_area="inside"
        )
        df[col] = interpolated
    df[columns] = df[columns].ffill(limit=max_gap).bfill(limit=max_gap)
    return df


def rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling linear-regression slope via a fixed ramp kernel."""

    if window < 3:
        return np.zeros_like(values, dtype=np.float64)
    half = (window - 1) / 2.0
    kernel = np.arange(window, dtype=np.float64) - half
    denom = float(np.sum(kernel**2)) or 1.0
    filled = np.nan_to_num(values.astype(np.float64), nan=0.0)
    slope = np.convolve(filled, kernel[::-1] / denom, mode="same")
    return slope


def add_rolling_features(
    df: pd.DataFrame, columns: list[str], windows_frames: list[int]
) -> pd.DataFrame:
    """Add rolling summaries for every column over every window size."""

    new_columns: dict[str, np.ndarray] = {}
    for win_idx, window in enumerate(windows_frames):
        window = max(int(window), 1)
        for col in columns:
            series = df[col]
            roll = series.rolling(window=window, center=True, min_periods=1)
            new_columns[f"{col}_mean_w{win_idx}"] = roll.mean().to_numpy()
            new_columns[f"{col}_std_w{win_idx}"] = roll.std().fillna(0.0).to_numpy()
            new_columns[f"{col}_min_w{win_idx}"] = roll.min().to_numpy()
            new_columns[f"{col}_max_w{win_idx}"] = roll.max().to_numpy()
            new_columns[f"{col}_median_w{win_idx}"] = roll.median().to_numpy()
            new_columns[f"{col}_slope_w{win_idx}"] = rolling_slope(
                series.to_numpy(), window
            )
    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df


def compute_motion_features(df: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    """Compute temporal derivative features from a per-identity sorted table."""

    frames = (
        df["frame_idx"].to_numpy(dtype=np.float64)
        if "frame_idx" in df
        else np.arange(len(df), dtype=np.float64)
    )
    center_x = df["center_x"].to_numpy(dtype=np.float64)
    center_y = df["center_y"].to_numpy(dtype=np.float64)
    dx = np.zeros(len(df), dtype=np.float64)
    dy = np.zeros(len(df), dtype=np.float64)
    if len(df) > 1:
        dx[1:] = np.nan_to_num(np.diff(center_x), nan=0.0)
        dy[1:] = np.nan_to_num(np.diff(center_y), nan=0.0)

    velocity_x = temporal_derivative(center_x, frames, frame_rate)
    velocity_y = temporal_derivative(center_y, frames, frame_rate)
    speed = np.hypot(velocity_x, velocity_y)
    acceleration = temporal_derivative(speed, frames, frame_rate)
    acceleration_x = temporal_derivative(velocity_x, frames, frame_rate)
    acceleration_y = temporal_derivative(velocity_y, frames, frame_rate)
    acceleration_magnitude = np.hypot(acceleration_x, acceleration_y)
    jerk = temporal_derivative(acceleration, frames, frame_rate)
    jerk_x = temporal_derivative(acceleration_x, frames, frame_rate)
    jerk_y = temporal_derivative(acceleration_y, frames, frame_rate)
    jerk_magnitude = np.hypot(jerk_x, jerk_y)
    body_scale = body_scale_from_geometry(df)

    df["delta_center_x"] = dx
    df["delta_center_y"] = dy
    df["velocity_x"] = velocity_x
    df["velocity_y"] = velocity_y
    df["speed"] = speed
    df["speed_body_norm"] = safe_divide(speed, body_scale)
    df["acceleration"] = acceleration
    df["acceleration_x"] = acceleration_x
    df["acceleration_y"] = acceleration_y
    df["acceleration_magnitude"] = acceleration_magnitude
    df["acceleration_body_norm"] = safe_divide(acceleration_magnitude, body_scale)
    df["jerk"] = jerk
    df["jerk_x"] = jerk_x
    df["jerk_y"] = jerk_y
    df["jerk_magnitude"] = jerk_magnitude
    df["jerk_body_norm"] = safe_divide(jerk_magnitude, body_scale)
    df["area_velocity"] = temporal_derivative(df["area"], frames, frame_rate)
    df["area_acceleration"] = temporal_derivative(
        df["area_velocity"], frames, frame_rate
    )

    orientation = df["orientation"].to_numpy(dtype=np.float64)
    orientation = np.nan_to_num(orientation, nan=0.0)
    unwrapped = np.unwrap(2.0 * orientation) / 2.0
    angular = temporal_derivative(unwrapped, frames, frame_rate)
    df["angular_velocity"] = angular
    df["angular_acceleration"] = temporal_derivative(angular, frames, frame_rate)
    df["orientation_sin"] = np.sin(orientation)
    df["orientation_cos"] = np.cos(orientation)

    moving = speed > 1e-9
    bearing = np.zeros_like(speed)
    bearing[moving] = np.arctan2(velocity_y[moving], velocity_x[moving])
    df["movement_bearing_sin"] = np.where(moving, np.sin(bearing), 0.0)
    df["movement_bearing_cos"] = np.where(moving, np.cos(bearing), 0.0)
    forward_velocity = velocity_x * np.cos(orientation) + velocity_y * np.sin(
        orientation
    )
    lateral_velocity = -velocity_x * np.sin(orientation) + velocity_y * np.cos(
        orientation
    )
    df["forward_velocity"] = forward_velocity
    df["lateral_velocity"] = lateral_velocity
    df["heading_alignment"] = safe_divide(forward_velocity, speed)
    return df


def add_nearest_animal_features(
    df: pd.DataFrame, frame_rate: float
) -> pd.DataFrame:
    """Add nearest-other-animal distance features to each identity row."""

    out = df.copy().reset_index(drop=True)
    for col in INTERANIMAL_FEATURES:
        out[col] = 0.0
    required = {
        "video_id",
        "frame_idx",
        "subject_id",
        "center_x",
        "center_y",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "area",
        "orientation",
    }
    if out.empty or not required.issubset(out.columns):
        return out

    values = {
        col: np.zeros(len(out), dtype=np.float64) for col in INTERANIMAL_FEATURES
    }
    for _, group in out.groupby(["video_id", "frame_idx"], sort=False):
        if len(group) < 2:
            continue
        idx = group.index.to_numpy()
        cx = group["center_x"].to_numpy(dtype=np.float64)
        cy = group["center_y"].to_numpy(dtype=np.float64)
        bx = group["bbox_x"].to_numpy(dtype=np.float64)
        by = group["bbox_y"].to_numpy(dtype=np.float64)
        bw = group["bbox_width"].to_numpy(dtype=np.float64)
        bh = group["bbox_height"].to_numpy(dtype=np.float64)
        area = group["area"].to_numpy(dtype=np.float64)
        orientation = group["orientation"].to_numpy(dtype=np.float64)
        speed = (
            group["speed"].to_numpy(dtype=np.float64)
            if "speed" in group
            else np.zeros(len(group), dtype=np.float64)
        )
        vx = (
            group["velocity_x"].to_numpy(dtype=np.float64)
            if "velocity_x" in group
            else np.zeros(len(group), dtype=np.float64)
        )
        vy = (
            group["velocity_y"].to_numpy(dtype=np.float64)
            if "velocity_y" in group
            else np.zeros(len(group), dtype=np.float64)
        )
        ax = (
            group["acceleration_x"].to_numpy(dtype=np.float64)
            if "acceleration_x" in group
            else np.zeros(len(group), dtype=np.float64)
        )
        ay = (
            group["acceleration_y"].to_numpy(dtype=np.float64)
            if "acceleration_y" in group
            else np.zeros(len(group), dtype=np.float64)
        )
        mask_score = (
            group["mask_score"].to_numpy(dtype=np.float64)
            if "mask_score" in group
            else np.ones(len(group), dtype=np.float64)
        )
        finite_centers = np.isfinite(cx) & np.isfinite(cy) & (mask_score > 0.0)

        for local_i, row_idx in enumerate(idx):
            candidates = np.where(finite_centers)[0]
            candidates = candidates[candidates != local_i]
            if candidates.size == 0 or not finite_centers[local_i]:
                continue

            dx_all = cx[candidates] - cx[local_i]
            dy_all = cy[candidates] - cy[local_i]
            dist_all = np.hypot(dx_all, dy_all)
            nearest_local = candidates[int(np.nanargmin(dist_all))]
            dx = cx[nearest_local] - cx[local_i]
            dy = cy[nearest_local] - cy[local_i]
            dist = float(np.hypot(dx, dy))
            edge_dist = float(
                bbox_edge_distance(
                    bx[local_i],
                    by[local_i],
                    bw[local_i],
                    bh[local_i],
                    bx[nearest_local],
                    by[nearest_local],
                    bw[nearest_local],
                    bh[nearest_local],
                )
            )
            bbox_iou = float(
                bbox_iou_from_geometry(
                    bx[local_i],
                    by[local_i],
                    bw[local_i],
                    bh[local_i],
                    bx[nearest_local],
                    by[nearest_local],
                    bw[nearest_local],
                    bh[nearest_local],
                )
            )
            angle = float(np.arctan2(dy, dx)) if dist > 0 else 0.0
            unit_x = dx / dist if dist > 1e-9 else 0.0
            unit_y = dy / dist if dist > 1e-9 else 0.0
            rel_vx = vx[nearest_local] - vx[local_i]
            rel_vy = vy[nearest_local] - vy[local_i]
            rel_ax = ax[nearest_local] - ax[local_i]
            rel_ay = ay[nearest_local] - ay[local_i]
            radial_velocity = rel_vx * unit_x + rel_vy * unit_y
            radial_acceleration = rel_ax * unit_x + rel_ay * unit_y
            lateral_velocity = rel_vx * (-unit_y) + rel_vy * unit_x

            values["nearest_neighbor_count"][row_idx] = float(candidates.size)
            values["nearest_centroid_distance"][row_idx] = dist
            values["nearest_centroid_dx"][row_idx] = dx
            values["nearest_centroid_dy"][row_idx] = dy
            values["nearest_centroid_abs_dx"][row_idx] = abs(dx)
            values["nearest_centroid_abs_dy"][row_idx] = abs(dy)
            values["nearest_bbox_edge_distance"][row_idx] = edge_dist
            values["nearest_subject_speed"][row_idx] = speed[local_i]
            values["nearest_neighbor_speed"][row_idx] = speed[nearest_local]
            values["nearest_relative_velocity_x"][row_idx] = rel_vx
            values["nearest_relative_velocity_y"][row_idx] = rel_vy
            values["nearest_relative_speed"][row_idx] = float(np.hypot(rel_vx, rel_vy))
            values["nearest_relative_acceleration_x"][row_idx] = rel_ax
            values["nearest_relative_acceleration_y"][row_idx] = rel_ay
            values["nearest_relative_acceleration_magnitude"][row_idx] = float(
                np.hypot(rel_ax, rel_ay)
            )
            values["nearest_radial_velocity"][row_idx] = radial_velocity
            values["nearest_radial_acceleration"][row_idx] = radial_acceleration
            values["nearest_lateral_speed"][row_idx] = abs(lateral_velocity)
            values["nearest_approach_speed"][row_idx] = -radial_velocity
            values["nearest_bbox_iou"][row_idx] = bbox_iou
            values["nearest_bbox_contact_flag"][row_idx] = float(edge_dist <= 1e-9)
            values["nearest_area_ratio"][row_idx] = float(
                safe_divide(area[local_i], area[nearest_local])
            )
            values["nearest_bearing_sin"][row_idx] = np.sin(angle)
            values["nearest_bearing_cos"][row_idx] = np.cos(angle)
            values["nearest_heading_alignment"][row_idx] = np.cos(
                angle - orientation[local_i]
            )

    for _, group in out.groupby(["video_id", "subject_id"], sort=False):
        group = group.sort_values("frame_idx")
        idx = group.index.to_numpy()
        frames = group["frame_idx"].to_numpy(dtype=np.float64)
        dist = values["nearest_centroid_distance"][idx]
        velocity = temporal_derivative(dist, frames, frame_rate)
        acceleration = temporal_derivative(velocity, frames, frame_rate)
        edge = values["nearest_bbox_edge_distance"][idx]
        edge_velocity = temporal_derivative(edge, frames, frame_rate)
        edge_acceleration = temporal_derivative(edge_velocity, frames, frame_rate)
        values["nearest_centroid_distance_velocity"][idx] = velocity
        values["nearest_centroid_distance_acceleration"][idx] = acceleration
        values["nearest_bbox_edge_distance_velocity"][idx] = edge_velocity
        values["nearest_bbox_edge_distance_acceleration"][idx] = edge_acceleration

    for col, arr in values.items():
        out[col] = arr
    return out


def extract_identity_features(
    video: CocoVideo, config: AppConfig
) -> pd.DataFrame:
    """Build the per-identity feature table for a video (object_id empty)."""

    fcfg = config.features
    frame_rate = config.data.frame_rate
    by_frame = video.records_by_frame()
    decoded_cache: dict[tuple[int, str], np.ndarray | None] = {}

    # Pick the best-scoring record per (frame, identity).
    best: dict[tuple[int, str], MaskRecord] = {}
    for frame_idx, records in by_frame.items():
        for record in records:
            key = (frame_idx, record.identity)
            if key not in best or record.score > best[key].score:
                best[key] = record

    identities = video.identities
    per_identity_frames: list[pd.DataFrame] = []

    for identity in identities:
        rows: list[dict[str, Any]] = []
        prev_mask: np.ndarray | None = None
        for frame_idx in video.frame_indices:
            key = (frame_idx, identity)
            record = best.get(key)
            if record is None:
                row = {name: float("nan") for name in GEOMETRIC_FEATURES}
                row["missing_mask_flag"] = 1.0
                mask = None
            else:
                mask = record.decode_mask()
                decoded_cache[key] = mask
                row = mask_shape_features(mask, record, video.width, video.height)
            row["video_id"] = video.video_id
            row["frame_idx"] = frame_idx
            row["subject_id"] = identity
            row["object_id"] = ""
            row["mask_iou_with_previous_frame"] = mask_iou(prev_mask, mask)
            prev_mask = mask
            rows.append(row)

        idf = pd.DataFrame(rows)
        geom_cols = [c for c in GEOMETRIC_FEATURES if c != "mask_score"]
        idf = interpolate_missing(
            idf,
            geom_cols,
            fcfg.max_interpolation_gap_frames,
            fcfg.missing_mask_policy,
        )
        idf = normalize_geometry(idf, video.width, video.height, fcfg)
        idf = compute_motion_features(idf, frame_rate)

        scalar_cols = [
            c
            for c in idf.columns
            if c not in {"video_id", "frame_idx", "subject_id", "object_id"}
            and c not in ROLLING_EXCLUDE
            and pd.api.types.is_numeric_dtype(idf[c])
        ]
        windows_frames = [
            max(int(round(sec * frame_rate)), 1)
            for sec in fcfg.rolling_windows_seconds
        ]
        idf = add_rolling_features(idf, scalar_cols, windows_frames)
        idf = idf.fillna(0.0)
        per_identity_frames.append(idf)

    if not per_identity_frames:
        return pd.DataFrame()
    result = pd.concat(per_identity_frames, ignore_index=True)
    result = add_nearest_animal_features(result, frame_rate)
    # Downcast features to float32 to keep long videos memory friendly.
    float_cols = result.select_dtypes(include=["float64"]).columns
    result[float_cols] = result[float_cols].astype(np.float32)
    return result


def extract_pairwise_features(
    video: CocoVideo, identity_features: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    """Build ordered-pair social features from already computed identity features."""

    identities = video.identities
    if len(identities) < 2:
        return pd.DataFrame()
    frame_rate = config.data.frame_rate
    indexed = identity_features.set_index(["subject_id", "frame_idx"]).sort_index()

    rows: list[dict[str, Any]] = []
    for subject in identities:
        for obj in identities:
            if subject == obj:
                continue
            try:
                subj = indexed.loc[subject]
                other = indexed.loc[obj]
            except KeyError:
                continue
            merged = subj.join(other, lsuffix="_s", rsuffix="_o", how="inner")
            if merged.empty:
                continue
            frames = merged.index.to_numpy()
            sx = merged["center_x_s"].to_numpy()
            sy = merged["center_y_s"].to_numpy()
            ox = merged["center_x_o"].to_numpy()
            oy = merged["center_y_o"].to_numpy()
            dx = ox - sx
            dy = oy - sy
            abs_dx = np.abs(dx)
            abs_dy = np.abs(dy)
            dist = np.hypot(dx, dy)
            dist_velocity = temporal_derivative(dist, frames, frame_rate)
            dist_accel = temporal_derivative(dist_velocity, frames, frame_rate)
            approaching = -dist_velocity
            svx = merged["delta_center_x_s"].to_numpy() * frame_rate
            svy = merged["delta_center_y_s"].to_numpy() * frame_rate
            if "velocity_x_s" in merged and "velocity_y_s" in merged:
                svx = merged["velocity_x_s"].to_numpy(dtype=np.float64)
                svy = merged["velocity_y_s"].to_numpy(dtype=np.float64)
            ovx = (
                merged["velocity_x_o"].to_numpy(dtype=np.float64)
                if "velocity_x_o" in merged
                else np.zeros_like(svx)
            )
            ovy = (
                merged["velocity_y_o"].to_numpy(dtype=np.float64)
                if "velocity_y_o" in merged
                else np.zeros_like(svy)
            )
            sax = (
                merged["acceleration_x_s"].to_numpy(dtype=np.float64)
                if "acceleration_x_s" in merged
                else np.zeros_like(svx)
            )
            say = (
                merged["acceleration_y_s"].to_numpy(dtype=np.float64)
                if "acceleration_y_s" in merged
                else np.zeros_like(svy)
            )
            oax = (
                merged["acceleration_x_o"].to_numpy(dtype=np.float64)
                if "acceleration_x_o" in merged
                else np.zeros_like(svx)
            )
            oay = (
                merged["acceleration_y_o"].to_numpy(dtype=np.float64)
                if "acceleration_y_o" in merged
                else np.zeros_like(svy)
            )
            subject_speed = (
                merged["speed_s"].to_numpy(dtype=np.float64)
                if "speed_s" in merged
                else np.hypot(svx, svy)
            )
            object_speed = (
                merged["speed_o"].to_numpy(dtype=np.float64)
                if "speed_o" in merged
                else np.hypot(ovx, ovy)
            )
            subject_accel = (
                merged["acceleration_magnitude_s"].to_numpy(dtype=np.float64)
                if "acceleration_magnitude_s" in merged
                else np.hypot(sax, say)
            )
            object_accel = (
                merged["acceleration_magnitude_o"].to_numpy(dtype=np.float64)
                if "acceleration_magnitude_o" in merged
                else np.hypot(oax, oay)
            )
            rel_vx = ovx - svx
            rel_vy = ovy - svy
            rel_motion_speed = np.hypot(rel_vx, rel_vy)
            rel_ax = oax - sax
            rel_ay = oay - say
            rel_accel_mag = np.hypot(rel_ax, rel_ay)
            norm = np.hypot(dx, dy) + 1e-9
            unit_x = dx / norm
            unit_y = dy / norm
            radial_velocity = rel_vx * unit_x + rel_vy * unit_y
            lateral_velocity = rel_vx * (-unit_y) + rel_vy * unit_x
            lateral_speed = np.abs(lateral_velocity)
            radial_acceleration = rel_ax * unit_x + rel_ay * unit_y
            following = (svx * dx + svy * dy) / norm
            rel_orientation = np.arctan2(dy, dx) - merged[
                "orientation_s"
            ].to_numpy()
            bbox_edge = bbox_edge_distance(
                merged["bbox_x_s"].to_numpy(),
                merged["bbox_y_s"].to_numpy(),
                merged["bbox_width_s"].to_numpy(),
                merged["bbox_height_s"].to_numpy(),
                merged["bbox_x_o"].to_numpy(),
                merged["bbox_y_o"].to_numpy(),
                merged["bbox_width_o"].to_numpy(),
                merged["bbox_height_o"].to_numpy(),
            )
            bbox_iou = bbox_iou_from_geometry(
                merged["bbox_x_s"].to_numpy(),
                merged["bbox_y_s"].to_numpy(),
                merged["bbox_width_s"].to_numpy(),
                merged["bbox_height_s"].to_numpy(),
                merged["bbox_x_o"].to_numpy(),
                merged["bbox_y_o"].to_numpy(),
                merged["bbox_width_o"].to_numpy(),
                merged["bbox_height_o"].to_numpy(),
            )
            body_scale = np.nanmean(
                np.vstack(
                    [
                        merged["major_axis_length_s"].to_numpy(),
                        merged["major_axis_length_o"].to_numpy(),
                    ]
                ),
                axis=0,
            )
            body_scale = np.where(
                np.isfinite(body_scale) & (body_scale > 1e-9), body_scale, 1.0
            )
            area_s = merged["area_s"].to_numpy()
            area_o = merged["area_o"].to_numpy()
            bbox_edge_velocity = temporal_derivative(bbox_edge, frames, frame_rate)
            bbox_edge_acceleration = temporal_derivative(
                bbox_edge_velocity, frames, frame_rate
            )
            contact_threshold = np.nanmedian(dist) * 0.5 if len(dist) else 0.0
            contact_flag = (dist < contact_threshold).astype(np.float32)
            bbox_contact_flag = (bbox_edge <= 1e-9).astype(np.float32)

            for i, frame in enumerate(frames):
                rows.append(
                    {
                        "video_id": video.video_id,
                        "frame_idx": int(frame),
                        "subject_id": subject,
                        "object_id": obj,
                        "pair_centroid_distance": dist[i],
                        "pair_centroid_dx": dx[i],
                        "pair_centroid_dy": dy[i],
                        "pair_centroid_abs_dx": abs_dx[i],
                        "pair_centroid_abs_dy": abs_dy[i],
                        "pair_centroid_distance_body_norm": dist[i] / body_scale[i],
                        "pair_relative_speed": dist_velocity[i],
                        "pair_relative_acceleration": dist_accel[i],
                        "pair_subject_speed": subject_speed[i],
                        "pair_object_speed": object_speed[i],
                        "pair_speed_ratio": float(
                            safe_divide(subject_speed[i], object_speed[i])
                        ),
                        "pair_subject_acceleration": subject_accel[i],
                        "pair_object_acceleration": object_accel[i],
                        "pair_relative_velocity_x": rel_vx[i],
                        "pair_relative_velocity_y": rel_vy[i],
                        "pair_relative_motion_speed": rel_motion_speed[i],
                        "pair_relative_motion_speed_body_norm": (
                            rel_motion_speed[i] / body_scale[i]
                        ),
                        "pair_radial_velocity": radial_velocity[i],
                        "pair_radial_velocity_body_norm": (
                            radial_velocity[i] / body_scale[i]
                        ),
                        "pair_lateral_velocity": lateral_velocity[i],
                        "pair_lateral_speed": lateral_speed[i],
                        "pair_lateral_speed_body_norm": (
                            lateral_speed[i] / body_scale[i]
                        ),
                        "pair_distance_acceleration": dist_accel[i],
                        "pair_approach_acceleration": -dist_accel[i],
                        "pair_relative_acceleration_x": rel_ax[i],
                        "pair_relative_acceleration_y": rel_ay[i],
                        "pair_relative_acceleration_magnitude": rel_accel_mag[i],
                        "pair_radial_acceleration": radial_acceleration[i],
                        "pair_approaching_score": approaching[i],
                        "pair_following_score": following[i],
                        "pair_relative_orientation": rel_orientation[i],
                        "pair_bbox_edge_distance": bbox_edge[i],
                        "pair_bbox_edge_distance_body_norm": (
                            bbox_edge[i] / body_scale[i]
                        ),
                        "pair_bbox_edge_distance_velocity": bbox_edge_velocity[i],
                        "pair_bbox_edge_distance_acceleration": (
                            bbox_edge_acceleration[i]
                        ),
                        "pair_bbox_iou": bbox_iou[i],
                        "pair_contact_flag": contact_flag[i],
                        "pair_bbox_contact_flag": bbox_contact_flag[i],
                        "pair_area_ratio": float(safe_divide(area_s[i], area_o[i])),
                        "pair_area_abs_difference": abs(area_s[i] - area_o[i]),
                    }
                )
    if not rows:
        return pd.DataFrame()
    pair_df = pd.DataFrame(rows)
    float_cols = pair_df.select_dtypes(include=["float64"]).columns
    pair_df[float_cols] = pair_df[float_cols].astype(np.float32)
    return pair_df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of numeric feature columns in a feature table."""

    meta = {"video_id", "frame_idx", "subject_id", "object_id"}
    return [
        c
        for c in df.columns
        if c not in meta and pd.api.types.is_numeric_dtype(df[c])
    ]


def extract_video_features(
    video: CocoVideo, config: AppConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract identity and (optionally) pairwise feature tables for a video."""

    identity_df = extract_identity_features(video, config)
    pair_df = pd.DataFrame()
    if config.features.include_pairwise_features:
        pair_df = extract_pairwise_features(video, identity_df, config)
    return identity_df, pair_df


def extract_features_from_coco(
    coco_path: str | Path, config: AppConfig
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Parse a COCO file and extract features for every video it contains."""

    videos = load_coco_videos(coco_path, config.data)
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for video_id, video in videos.items():
        out[video_id] = extract_video_features(video, config)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract mask-derived features.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    parser.add_argument("--coco-json", required=True, help="Path to COCO JSON.")
    parser.add_argument("--output", default=None, help="Output feature table path.")
    parser.add_argument(
        "--output-dir", default=None, help="Diagnostics output directory."
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="Write data quality diagnostics."
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    results = extract_features_from_coco(args.coco_json, config)
    all_identity = pd.concat(
        [ident for ident, _ in results.values()], ignore_index=True
    )
    all_pairs = [pair for _, pair in results.values() if not pair.empty]

    if args.output:
        out_path = write_table(all_identity, args.output)
        print(f"Wrote identity features: {out_path}  shape={all_identity.shape}")
        if all_pairs:
            pair_path = Path(args.output)
            pair_path = pair_path.with_name(pair_path.stem + "_pairs.parquet")
            written = write_table(pd.concat(all_pairs, ignore_index=True), pair_path)
            print(f"Wrote pairwise features: {written}")

    if args.diagnostics and args.output_dir:
        from .export import write_feature_diagnostics

        write_feature_diagnostics(all_identity, args.output_dir)
        print(f"Wrote diagnostics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
