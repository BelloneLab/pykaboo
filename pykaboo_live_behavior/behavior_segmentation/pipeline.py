"""Shared data-preparation and inference helpers used by train, infer, and embed.

Keeping this logic in one place guarantees that the feature schema produced for
training is identical to the one produced at inference, and that sliding-window
prediction merging is consistent everywhere.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .config import AppConfig
from .coco_masks import load_coco_videos
from .dataset import TrackData
from .features import (
    INTERANIMAL_FEATURES,
    MOTION_FEATURES,
    PAIRWISE_DISTANCE_FEATURES,
    extract_video_features,
)
from .labels import (
    AlignedLabels,
    LabelMap,
    align_framewise_labels,
    align_interval_labels,
    build_label_map,
    load_labels_dataframe,
)
from .models.temporal_tcn import TemporalTcnModel
from .mask_video import render_mask_clip
from .postprocess import merge_window_logits
from .storage import read_table, write_table
from .windows import build_windows, seconds_to_frames

LogFn = Callable[[str], None]


def _forward_model(model, features: torch.Tensor, mask_clip: torch.Tensor | None = None):
    """Forward helper that passes mask clips only to two-stream models."""

    if mask_clip is not None and getattr(model, "requires_mask_clip", False):
        return model(features, mask_clip=mask_clip)
    return model(features)


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _merge_on_meta(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    if left is None or left.empty:
        return right
    cols = keys + [c for c in right.columns if c not in left.columns]
    return left.merge(right[cols], on=keys, how="left")


def _merge_pair_context_into_identity(
    identity_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    prefixes: tuple[str, ...] = ("dy_", "mask_"),
) -> pd.DataFrame:
    """Attach directed pair context to subject-level rows for single-track labels."""

    if identity_df is None or identity_df.empty or pair_df is None or pair_df.empty:
        return identity_df
    keys = ["video_id", "frame_idx", "subject_id"]
    context_cols = [
        c
        for c in pair_df.columns
        if c not in {"video_id", "frame_idx", "subject_id", "object_id"}
        and any(str(c).startswith(prefix) for prefix in prefixes)
    ]
    if not context_cols:
        return identity_df
    context = (
        pair_df[keys + context_cols]
        .groupby(keys, as_index=False, sort=False)
        .mean(numeric_only=True)
    )
    context = context.rename(columns={c: f"ctx_{c}" for c in context_cols})
    return identity_df.merge(context, on=keys, how="left")


def _augment_pose_features(video, coco_path, identity_df, pair_df, config):
    """Merge pose, dyadic, and mask-overlap features.

    Best-effort: if pose is unavailable the mask-only features still work.
    """
    feats = config.features
    want_pose = bool(getattr(feats, "pose", False))
    want_dyadic = bool(getattr(feats, "dyadic", False))
    want_overlap = bool(getattr(feats, "mask_overlap", False))
    if not (want_pose or want_dyadic or want_overlap):
        return identity_df, pair_df
    if pair_df is None:
        pair_df = pd.DataFrame()
    keys = ["video_id", "frame_idx", "subject_id", "object_id"]
    try:
        pose = None
        if want_pose or want_dyadic:
            from .pose import find_pose_csv_for_coco, load_pose_csv

            pose_csv = find_pose_csv_for_coco(coco_path)
            if pose_csv is not None:
                pose = load_pose_csv(pose_csv, video_id=video.video_id, min_likelihood=0.0)
        if want_pose and pose is not None:
            from .pose_features import extract_pose_feature_tables

            pose_id, pose_pair = extract_pose_feature_tables(
                pose,
                video.width,
                video.height,
                config.data.frame_rate,
                egocentric=bool(getattr(feats, "egocentric", True)),
            )
            identity_df = _merge_on_meta(identity_df, pose_id, keys)
            if not pose_pair.empty:
                pair_df = _merge_on_meta(pair_df, pose_pair, keys)
        if want_dyadic and pose is not None:
            from .dyadic_features import extract_dyadic_features

            if pose is not None:
                dy = extract_dyadic_features(pose, config.data.frame_rate)
                if not dy.empty:
                    dy["subject_id"] = dy["subject_id"].astype(str)
                    dy["object_id"] = dy["object_id"].astype(str)
                    if pair_df.empty:
                        pair_df = dy
                    else:
                        pair_df["subject_id"] = pair_df["subject_id"].astype(str)
                        pair_df["object_id"] = pair_df["object_id"].astype(str)
                        cols = keys + [c for c in dy.columns if c not in pair_df.columns]
                        pair_df = pair_df.merge(dy[cols], on=keys, how="left")
        if want_overlap and not pair_df.empty:
            from .mask_video import mask_overlap_features

            ov = mask_overlap_features(video, video.identities)
            ovdf = pd.DataFrame({"frame_idx": list(video.frame_indices)})
            for name, arr in ov.items():
                ovdf[name] = arr
            cols = ["frame_idx"] + [c for c in ovdf.columns if c not in pair_df.columns]
            pair_df = pair_df.merge(ovdf[cols], on="frame_idx", how="left")
    except Exception:
        pass  # mask-only features remain usable
    if want_dyadic or want_overlap:
        identity_df = _merge_pair_context_into_identity(identity_df, pair_df)
    if identity_df is not None and not identity_df.empty:
        identity_df = identity_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if pair_df is not None and not pair_df.empty:
        pair_df = pair_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return identity_df, pair_df


def _extract_one_coco(coco_path, config, cache_dir, log=None):
    out_identity = []
    out_pair = []
    frames = {}
    videos = load_coco_videos(coco_path, config.data)
    for video_id, video in videos.items():
        frames[video_id] = video.frame_indices
        cached = Path(cache_dir) / f"{video_id}_features.parquet" if cache_dir else None
        identity_df = None
        pair_df = None
        if cached is not None and (cached.exists() or cached.with_suffix(".csv").exists()):
            identity_df = read_table(cached)
            pair_path = cached.with_name(cached.stem + "_pairs.parquet")
            pair_df = (
                read_table(pair_path)
                if pair_path.exists() or pair_path.with_suffix(".csv").exists()
                else pd.DataFrame()
            )
            stale = any(
                c not in identity_df.columns
                for c in (*MOTION_FEATURES, *INTERANIMAL_FEATURES)
            )
            needs_pair = config.features.include_pairwise_features and len(video.identities) > 1
            if needs_pair and (pair_df is None or pair_df.empty):
                stale = True
            if needs_pair and getattr(config.features, "dyadic", False) and (
                pair_df is None or "dy_min_contact" not in getattr(pair_df, "columns", [])
            ):
                stale = True
            if getattr(config.features, "pose", False) and "pose_body_length" not in identity_df.columns:
                stale = True
            if getattr(config.features, "dyadic", False) and "ctx_dy_min_contact" not in identity_df.columns:
                stale = True
            if getattr(config.features, "mask_overlap", False) and "ctx_mask_iou" not in identity_df.columns:
                stale = True
            if stale:
                if log:
                    log(f"Cached features for {video_id} are stale; regenerating.")
                identity_df = None
                pair_df = None
        if identity_df is None or pair_df is None:
            if log:
                log(f"Extracting features for {video_id} ...")
            identity_df, pair_df = extract_video_features(video, config)
            identity_df, pair_df = _augment_pose_features(
                video, coco_path, identity_df, pair_df, config
            )
            if cache_dir is not None:
                write_table(identity_df, cached)
                if pair_df is not None and not pair_df.empty:
                    write_table(pair_df, cached.with_name(cached.stem + "_pairs.parquet"))
        out_identity.append(identity_df)
        if pair_df is not None and not pair_df.empty:
            out_pair.append(pair_df)
    return out_identity, out_pair, frames



_LEAN_DROP_RE = re.compile(r"_(mean|std|min|max|median|slope)_w[0-9]+$")
_MOTION_ROLLING_TOKENS = (
    "acceleration",
    "angular",
    "approach",
    "delta_",
    "distance_velocity",
    "distance_acceleration",
    "forward_velocity",
    "jerk",
    "lateral",
    "mask_iou_with_previous_frame",
    "motion",
    "radial",
    "relative",
    "speed",
    "velocity",
)
_CORE_ROLLING_STATS = {"mean", "std", "slope"}
_CORE_ROLLING_TOKENS = (
    "acceleration",
    "angular_velocity",
    "approach",
    "area_acceleration",
    "area_velocity",
    "contact",
    "distance",
    "forward_velocity",
    "heading_alignment",
    "iou",
    "jerk",
    "lateral_velocity",
    "motion",
    "radial",
    "relative",
    "speed",
    "velocity",
)


def _rolling_base_name(column: str) -> str | None:
    match = _LEAN_DROP_RE.search(str(column))
    return str(column)[: match.start()] if match else None


def _rolling_stat_name(column: str) -> str | None:
    match = _LEAN_DROP_RE.search(str(column))
    return match.group(1) if match else None


def _is_motion_rolling_base(base: str) -> bool:
    return any(token in base for token in _MOTION_ROLLING_TOKENS)


def _is_core_rolling_column(column: str, base: str) -> bool:
    stat = _rolling_stat_name(column)
    if stat not in _CORE_ROLLING_STATS:
        return False
    return any(token in base for token in _CORE_ROLLING_TOKENS)


def rolling_stat_columns(df: pd.DataFrame, policy: str = "none") -> list[str]:
    """Return generated rolling-stat feature columns removed by a pruning policy."""

    if df is None or df.empty:
        return []
    meta = {"video_id", "frame_idx", "subject_id", "object_id"}
    drop: list[str] = []
    for column in df.columns:
        if column in meta:
            continue
        base = _rolling_base_name(str(column))
        if base is None:
            continue
        if policy == "none":
            drop.append(column)
        elif policy == "motion_only" and not _is_motion_rolling_base(base):
            drop.append(column)
        elif policy == "core" and not _is_core_rolling_column(str(column), base):
            drop.append(column)
    return drop


def apply_rolling_stat_policy(
    df: pd.DataFrame,
    policy: str = "none",
) -> pd.DataFrame:
    """Strip generated rolling statistics according to a feature pruning policy."""

    if df is None or df.empty:
        return df
    if policy == "full":
        return df
    drop = rolling_stat_columns(df, policy)
    return df.drop(columns=drop) if drop else df


def apply_lean_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Strip all generated rolling statistics while keeping instantaneous cues."""

    return apply_rolling_stat_policy(df, "none")


def _apply_lean(df):
    return apply_lean_feature_set(df)

def extract_features_for_videos(
    config: AppConfig,
    cache_dir: str | Path | None = None,
    log: LogFn | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]]]:
    """Extract (and optionally cache) identity + pairwise features for all videos.

    Per-video extraction runs in parallel across CPU cores when enabled. Pose-derived
    egocentric/relational (dyadic) and inter-animal mask-overlap features are merged
    into the pairwise table when ``features.dyadic`` / ``features.mask_overlap`` are set.
    """

    cache_dir = Path(cache_dir) if cache_dir else None
    cocos = list(config.data.coco_jsons)
    results = []
    parallel = bool(getattr(config.features, "parallel_extraction", False)) and len(cocos) > 1
    if parallel:
        import os
        from concurrent.futures import ProcessPoolExecutor

        nw = config.features.num_workers or min(len(cocos), os.cpu_count() or 1)
        if log:
            log(f"Extracting features for {len(cocos)} videos with {nw} CPU workers ...")
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futures = [ex.submit(_extract_one_coco, c, config, cache_dir, None) for c in cocos]
            for fut in futures:
                results.append(fut.result())
    else:
        for c in cocos:
            results.append(_extract_one_coco(c, config, cache_dir, log))

    identity_frames = []
    pair_frames = []
    frame_indices: dict[str, list[int]] = {}
    for oi, op, fr in results:
        identity_frames.extend(oi)
        pair_frames.extend(op)
        frame_indices.update(fr)
    identity_all = (
        pd.concat(identity_frames, ignore_index=True) if identity_frames else pd.DataFrame()
    )
    pair_all = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    rolling_policy = (
        "none"
        if getattr(config.features, "lean", False)
        else getattr(config.features, "rolling_stat_policy", "full")
    )
    if rolling_policy != "full":
        before_identity = len(identity_all.columns)
        before_pair = len(pair_all.columns)
        identity_all = apply_rolling_stat_policy(identity_all, rolling_policy)
        pair_all = apply_rolling_stat_policy(pair_all, rolling_policy)
        if log:
            log(
                f"Feature pruning ({rolling_policy} rolling stats): "
                f"identity {before_identity}->{len(identity_all.columns)}, "
                f"pair {before_pair}->{len(pair_all.columns)} columns."
            )
    return identity_all, pair_all, frame_indices


def align_all_labels(
    config: AppConfig, frame_indices: dict[str, list[int]]
) -> AlignedLabels:
    """Align manual labels for every video into per-track framewise targets."""

    if not config.data.labels_csv:
        raise ValueError("No labels CSV configured; cannot align labels.")
    labels_df = load_labels_dataframe(
        config.data.labels_csv,
        config.data.label_format,
        frame_rate=config.data.frame_rate,
    )
    label_map = build_label_map(labels_df, config.labels)
    aligned = AlignedLabels(label_map=label_map)
    for video_id, frames in frame_indices.items():
        if config.data.label_format == "interval":
            tracks = align_interval_labels(
                labels_df,
                video_id,
                frames,
                label_map,
                config.labels,
                config.data.end_frame_inclusive,
            )
        else:
            tracks = align_framewise_labels(
                labels_df, video_id, frames, label_map, config.labels
            )
        aligned.tracks.update(tracks)
    return aligned


def attach_mask_clips_to_tracks(
    tracks: list[TrackData],
    coco_jsons: list[str | Path],
    data_config,
    size: int = 48,
    log: LogFn | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Attach subject-first clean mask clips to tracks in-place."""

    if not tracks:
        return
    wanted = {str(track.video_id) for track in tracks}
    videos = {}
    for coco_json in coco_jsons:
        videos.update(load_coco_videos(coco_json, data_config))
    rendered: dict[str, tuple[np.ndarray, list[int], list[str]]] = {}
    for video_id, video in videos.items():
        if should_stop and should_stop():
            raise InterruptedError("Inference stopped by user.")
        if str(video_id) not in wanted:
            continue
        ids = list(video.identities)[:2]
        clip, frames = render_mask_clip(
            video,
            identities=ids,
            size=size,
            with_overlap=True,
            smooth_box=True,
        )
        rendered[str(video_id)] = (
            (clip * 255.0).clip(0, 255).astype(np.uint8),
            list(frames),
            [str(i) for i in ids],
        )
        if log:
            log(
                f"Cached mask clip for {video_id}: "
                f"{len(frames)} frames, {size}x{size}."
            )
    for track in tracks:
        if should_stop and should_stop():
            raise InterruptedError("Inference stopped by user.")
        payload = rendered.get(str(track.video_id))
        if payload is None:
            continue
        clip, frames, ids = payload
        index = {int(frame): pos for pos, frame in enumerate(frames)}
        positions = [index.get(int(frame), -1) for frame in track.frame_indices]
        out = np.zeros((len(positions), *clip.shape[1:]), dtype=np.uint8)
        valid = np.asarray([pos >= 0 for pos in positions], dtype=bool)
        if valid.any():
            out[valid] = clip[[pos for pos in positions if pos >= 0]]
        if len(ids) > 1 and str(track.subject_id) == ids[1]:
            out = out.copy()
            out[:, [0, 1]] = out[:, [1, 0]]
        track.mask_clip = out


@torch.no_grad()
def sliding_window_inference(
    model: TemporalTcnModel,
    track: TrackData,
    normalizer,
    window_seconds: float,
    stride_seconds: float,
    frame_rate: float,
    device: str = "cpu",
    should_stop: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run overlapping-window inference for one track.

    Returns merged logits ``[num_classes, T]`` and embeddings ``[embedding_dim, T]``.
    """

    num_frames = track.num_frames
    window_length = seconds_to_frames(window_seconds, frame_rate)
    stride = seconds_to_frames(stride_seconds, frame_rate)
    spans = build_windows(num_frames, window_length, stride)

    features = normalizer.transform(track.features, track.video_id)  # [T, C]
    num_classes = model.num_classes
    embedding_dim = model.embedding_dim

    logit_windows: list[tuple[int, int, np.ndarray]] = []
    embed_accum = np.zeros((embedding_dim, num_frames), dtype=np.float64)
    embed_counts = np.zeros(num_frames, dtype=np.float64)

    for span in spans:
        if should_stop and should_stop():
            raise InterruptedError("Inference stopped by user.")
        segment = features[span.start : span.end]  # [t, C]
        tensor = torch.from_numpy(segment.T.copy()).unsqueeze(0).to(device)
        mask_clip = None
        if track.mask_clip is not None and getattr(model, "requires_mask_clip", False):
            clip = track.mask_clip[span.start : span.end].astype(np.float32)
            if clip.size and float(clip.max()) > 1.0:
                clip = clip / 255.0
            mask_clip = torch.from_numpy(clip.copy()).unsqueeze(0).to(device)
        output = _forward_model(model, tensor, mask_clip)
        logits = output.logits.squeeze(0).cpu().numpy()  # [K, t]
        embeddings = output.embeddings.squeeze(0).cpu().numpy()  # [E, t]
        logit_windows.append((span.start, span.end, logits))
        width = span.end - span.start
        embed_accum[:, span.start : span.end] += embeddings[:, :width]
        embed_counts[span.start : span.end] += 1.0

    merged_logits = merge_window_logits(num_frames, num_classes, logit_windows)
    embed_counts = np.where(embed_counts < 1.0, 1.0, embed_counts)
    merged_embeddings = embed_accum / embed_counts
    return merged_logits, merged_embeddings


def logits_to_probabilities(logits: np.ndarray, multilabel: bool) -> np.ndarray:
    if multilabel:
        return 1.0 / (1.0 + np.exp(-logits))
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=0, keepdims=True)
