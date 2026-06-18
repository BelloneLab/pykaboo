"""Pose-keypoint-graph input for the free-interaction benchmark.

Instead of the 432 engineered descriptors (effective rank ~17, highly redundant),
this builds a compact, low-redundancy *relational* input: the raw keypoints of
both animals in one animal's egocentric frame. For subject ``s`` the input is the
8 keypoints of ``s`` plus the 8 keypoints of the partner, expressed in ``s``'s
body frame (origin = body, x-axis neck->nose, unit = body length), so the
inter-animal geometry (the social signal) is encoded directly and the
representation is invariant to global translation / rotation / scale.

Channels: 2 animals x 8 keypoints x (x, y) = 32. A skeleton + dyadic graph over
those 32 channels is provided for the graph-attention backbone (``gatr``), which
is its native setting (real low-redundancy relational structure, unlike a
correlation graph over redundant descriptors).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .dataset import TrackData
from .free_social import (
    CachedVideo,
    DEFAULT_DATASET_DIR,
    build_free_label_map,
    discover_videos,
    load_scene_label_tracks,
)
from .labels import LabelMap
from .pose import KEYPOINT_ORDER, load_pose_csv
from .pose_features import egocentric_transform

# Skeleton edges (within one animal), by keypoint name.
SKELETON_EDGES = [
    ("nose", "left_ear"), ("nose", "right_ear"), ("nose", "neck"),
    ("neck", "body"), ("body", "left_hip"), ("body", "right_hip"),
    ("body", "tail"), ("left_ear", "right_ear"), ("left_hip", "right_hip"),
]
# Dyadic edges (across the two animals): contacts that define social behaviors.
DYADIC_EDGES = [
    ("nose", "nose"), ("nose", "tail"), ("tail", "nose"),
    ("nose", "body"), ("body", "nose"), ("nose", "neck"),
]


def _channel_names() -> list[str]:
    names = []
    for ai in range(2):
        for kp in KEYPOINT_ORDER:
            names.append(f"a{ai}_{kp}_x")
            names.append(f"a{ai}_{kp}_y")
    return names


def build_skeleton_graph() -> np.ndarray:
    """Boolean [32, 32] adjacency over the (animal, keypoint, coord) channels."""
    K = len(KEYPOINT_ORDER)
    kp_idx = {kp: i for i, kp in enumerate(KEYPOINT_ORDER)}
    N = 2 * K * 2

    def ch(ai, kp, coord):  # coord 0=x,1=y
        return (ai * K + kp_idx[kp]) * 2 + coord

    A = np.eye(N, dtype=bool)
    # couple the x and y of every keypoint
    for ai in range(2):
        for kp in KEYPOINT_ORDER:
            A[ch(ai, kp, 0), ch(ai, kp, 1)] = True
            A[ch(ai, kp, 1), ch(ai, kp, 0)] = True
    # skeleton edges within each animal (both coords)
    for ai in range(2):
        for u, v in SKELETON_EDGES:
            for cu in (0, 1):
                for cv in (0, 1):
                    A[ch(ai, u, cu), ch(ai, v, cv)] = True
                    A[ch(ai, v, cv), ch(ai, u, cu)] = True
    # dyadic edges across animals
    for u, v in DYADIC_EDGES:
        for cu in (0, 1):
            for cv in (0, 1):
                A[ch(0, u, cu), ch(1, v, cv)] = True
                A[ch(1, v, cv), ch(0, u, cu)] = True
    return A


def _interp_nan(x: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs along time, per channel; fill leftover with 0."""
    out = x.copy()
    n = out.shape[0]
    t = np.arange(n)
    for c in range(out.shape[1]):
        col = out[:, c]
        good = np.isfinite(col)
        if good.sum() >= 2:
            out[:, c] = np.interp(t, t[good], col[good])
        elif good.sum() == 1:
            out[:, c] = col[good][0]
        else:
            out[:, c] = 0.0
    return out


def _pose_features_for_subject(pose, subject_index: int) -> np.ndarray:
    """[F, 32] egocentric keypoints (self + partner) in subject's body frame."""
    coords, _, _ = egocentric_transform(pose, subject_index)  # [F, A, K, 2]
    F, A, K, _ = coords.shape
    # order: self animal first, then partner, each 8 kp x (x,y)
    order = [subject_index] + [a for a in range(A) if a != subject_index]
    flat = coords[:, order, :, :].reshape(F, A * K * 2)
    flat = _interp_nan(flat)
    # clip extreme values (units are body lengths; contacts are within a few)
    return np.clip(flat, -8.0, 8.0).astype(np.float32)


def build_pose_cache(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, CachedVideo], list[str], LabelMap]:
    """Build per-identity pose-coordinate tracks for every video (in memory)."""

    def emit(m: str) -> None:
        if log:
            log(m)

    label_map = build_free_label_map()
    feature_names = _channel_names()
    by_video: dict[str, CachedVideo] = {}

    for entry in discover_videos(dataset_dir):
        if entry.pose_csv is None:
            emit(f"skip {entry.video_id} (no pose csv)")
            continue
        pose = load_pose_csv(entry.pose_csv, video_id=entry.video_id, min_likelihood=0.3)
        identities = list(pose.identities)  # ["1", "2"]
        frame_indices = list(pose.frame_indices)
        label_tracks = load_scene_label_tracks(
            entry.label_csv, entry.video_id, frame_indices, identities, label_map)
        partner = {i: [o for o in identities if o != i] for i in identities}
        tracks = []
        for si, ident in enumerate(identities):
            feats = _pose_features_for_subject(pose, si)
            obj = partner[ident][0] if partner[ident] else ""
            key = (entry.video_id, ident, obj)
            labels = label_tracks.get(key)
            if labels is None:
                continue
            n = min(feats.shape[0], labels.shape[1])
            tracks.append(TrackData(
                video_id=entry.video_id, subject_id=ident, object_id=obj,
                frame_indices=np.asarray(frame_indices[:n]),
                features=feats[:n], feature_names=feature_names,
                labels=labels[:, :n], valid_mask=np.ones(n, bool)))
        by_video[entry.video_id] = CachedVideo(
            video_id=entry.video_id, tracks=tracks, frame_rate=30.0)
        emit(f"pose {entry.video_id}: {len(tracks)} tracks x {len(feature_names)} ch")

    return by_video, feature_names, label_map
