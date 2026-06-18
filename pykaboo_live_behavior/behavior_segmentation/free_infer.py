"""Build the inference input for free-interaction checkpoints (incl. fused pose).

A free-interaction checkpoint's ``feature_names`` are namespaced by modality:
``features:<col>`` for the 432 engineered descriptors and ``pose:<col>`` for the
32 egocentric keypoint coordinates. This module reconstructs the exact fused
per-identity track matrices for a new ``(coco, pose)`` pair so the GUI can run the
fused model with no other changes. A plain (non-fused) free model whose names have
no ``pose:`` entries is handled too (it just builds the feature tracks).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import AppConfig
from .dataset import build_feature_column_list, build_inference_tracks
from .free_pose import _pose_features_for_subject
from .pose import find_pose_csv_for_coco, load_pose_csv
from .social_pipeline import build_social_features

FEAT_PREFIX = "features:"
POSE_PREFIX = "pose:"


def is_fused_feature_names(feature_names: list[str]) -> bool:
    return any(n.startswith(POSE_PREFIX) for n in feature_names)


def build_free_inference_tracks(
    coco_path: str | Path,
    config: AppConfig,
    feature_names: list[str],
    pose_path: str | Path | None = None,
    log=None,
):
    """Per-identity tracks whose channels match ``feature_names`` exactly.

    Supports both the namespaced fused names (``features:`` / ``pose:``) and the
    plain 432-feature names of the non-fused free model.
    """

    namespaced = any(n.startswith(FEAT_PREFIX) or n.startswith(POSE_PREFIX) for n in feature_names)
    feat_cols = ([n[len(FEAT_PREFIX):] for n in feature_names if n.startswith(FEAT_PREFIX)]
                 if namespaced else list(feature_names))
    pose_cols = [n[len(POSE_PREFIX):] for n in feature_names if n.startswith(POSE_PREFIX)]

    feats = build_social_features(
        coco_path, config, pose_path=pose_path, use_pose=True, use_wavelets=True, log=log)
    # Use the SAME ordered feature-column list training used (subject + partner +
    # pair), so assemble_track_matrix produces matching columns; we then pick the
    # checkpoint's (lean) columns by name below.
    full_names = build_feature_column_list(feats.identity_df, feats.pair_df, is_pair=True)
    tracks = build_inference_tracks(feats.identity_df, feats.pair_df, full_names, is_pair=True)

    pose = None
    if pose_cols:
        pose_path = pose_path or find_pose_csv_for_coco(coco_path)
        if pose_path and Path(pose_path).exists():
            pose = load_pose_csv(pose_path, video_id=feats.video_id, min_likelihood=0.3)

    out = []
    for t in tracks:
        feat_index = {n: i for i, n in enumerate(t.feature_names)}
        n_frames = t.features.shape[0]
        cols = []
        # feature columns (by stripped name; missing -> zeros)
        for c in feat_cols:
            j = feat_index.get(c)
            cols.append(t.features[:, j] if j is not None else np.zeros(n_frames, np.float32))
        # pose columns for THIS subject's egocentric frame
        if pose_cols and pose is not None:
            si = pose.identity_index(t.subject_id)
            if si is not None:
                pf = _pose_features_for_subject(pose, si)        # [F, 32]
                # pose channel order is fixed by _channel_names()
                from .free_pose import _channel_names
                pnames = _channel_names()
                pidx = {n: i for i, n in enumerate(pnames)}
                m = min(pf.shape[0], n_frames)
                for c in pose_cols:
                    col = np.zeros(n_frames, np.float32)
                    k = pidx.get(c)
                    if k is not None:
                        col[:m] = pf[:m, k]
                    cols.append(col)
            else:
                for _ in pose_cols:
                    cols.append(np.zeros(n_frames, np.float32))
        elif pose_cols:
            for _ in pose_cols:
                cols.append(np.zeros(n_frames, np.float32))

        t.features = np.stack(cols, axis=1).astype(np.float32)
        t.feature_names = list(feature_names)
        out.append(t)
    return feats, out
