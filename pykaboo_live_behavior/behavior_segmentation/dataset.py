"""Assemble per-track feature/label tensors and a windowed PyTorch dataset.

A *track* is one ``(video_id, subject_id, object_id)`` series. For single-mouse
tasks ``object_id`` is empty and the feature vector is just the subject's
identity features. For pairwise social tasks the feature vector concatenates the
subject features, the object features (prefixed ``obj_``), and the pair features.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import feature_columns
from .config import BehaviorRolesConfig
from .labels import LabelMap
from .normalization import FeatureNormalizer
from .roles import (
    constrain_label_array_for_subject,
    should_expand_global_labels_to_subject,
)
from .windows import WindowSpan, build_windows, seconds_to_frames

META_COLUMNS = ["video_id", "frame_idx", "subject_id", "object_id"]


@dataclass
class TrackData:
    video_id: str
    subject_id: str
    object_id: str
    frame_indices: np.ndarray
    features: np.ndarray  # [T, C] float32
    feature_names: list[str]
    labels: np.ndarray | None = None  # [T] multiclass or [K, T] multilabel
    valid_mask: np.ndarray | None = None  # [T] bool, False where mask missing
    mask_clip: np.ndarray | None = None  # [T, C, H, W] uint8/float, subject first

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)


def build_feature_column_list(
    identity_df: pd.DataFrame, pair_df: pd.DataFrame, is_pair: bool
) -> list[str]:
    """Return the canonical, ordered feature column names for a task type."""

    identity_cols = feature_columns(identity_df)
    if not is_pair:
        return identity_cols
    pair_cols = feature_columns(pair_df) if not pair_df.empty else []
    return identity_cols + [f"obj_{c}" for c in identity_cols] + pair_cols


def subjects_for_video(identity_df: pd.DataFrame, video_id: str) -> list[str]:
    rows = identity_df[identity_df["video_id"].astype(str) == str(video_id)]
    return sorted(rows["subject_id"].astype(str).unique())


def expand_global_label_tracks(
    identity_df: pd.DataFrame,
    label_tracks: dict[tuple[str, str, str], np.ndarray],
    label_map: LabelMap | None = None,
    behavior_roles: BehaviorRolesConfig | None = None,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Apply video-level labels with blank subject/object to every identity.

    Some annotation tools export one behavior stream for the whole dyad/video
    rather than one stream per tracked identity. Expanding these global labels
    lets the trainer build usable tracks from COCO identities without requiring
    users to manually duplicate the label CSV.
    """

    expanded: dict[tuple[str, str, str], np.ndarray] = {}
    for (video_id, subject_id, object_id), labels in label_tracks.items():
        if subject_id or object_id:
            expanded[(video_id, subject_id, object_id)] = labels
            continue
        subjects = subjects_for_video(identity_df, video_id)
        if not subjects:
            expanded[(video_id, subject_id, object_id)] = labels
            continue
        for subject in subjects:
            if behavior_roles and not should_expand_global_labels_to_subject(
                subject, behavior_roles
            ):
                continue
            subject_labels = labels.copy()
            if label_map is not None and behavior_roles is not None:
                subject_labels = constrain_label_array_for_subject(
                    subject_labels, label_map, behavior_roles, subject
                )
            expanded[(video_id, subject, "")] = subject_labels
    return expanded


def assemble_track_matrix(
    identity_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    video_id: str,
    subject_id: str,
    object_id: str,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``[T, C]`` feature matrix for one track on its frame grid."""

    subj = identity_df[
        (identity_df["video_id"].astype(str) == str(video_id))
        & (identity_df["subject_id"].astype(str) == str(subject_id))
    ].sort_values("frame_idx")
    if subj.empty:
        raise ValueError(
            f"No identity features for video={video_id} subject={subject_id}"
        )
    frame_indices = subj["frame_idx"].to_numpy()
    identity_cols = feature_columns(identity_df)
    frame_table = subj.set_index("frame_idx")

    valid = (1.0 - subj.get("missing_mask_flag", pd.Series(0.0, index=subj.index))
             .to_numpy()).astype(bool)

    if not object_id:
        matrix = frame_table[identity_cols].to_numpy(dtype=np.float32)
        return frame_indices, matrix, valid

    obj = identity_df[
        (identity_df["video_id"].astype(str) == str(video_id))
        & (identity_df["subject_id"].astype(str) == str(object_id))
    ].sort_values("frame_idx").set_index("frame_idx")
    obj = obj.reindex(frame_indices).ffill().bfill()

    pieces = [frame_table[identity_cols].to_numpy(dtype=np.float32)]
    pieces.append(obj[identity_cols].to_numpy(dtype=np.float32))
    if not pair_df.empty:
        pair = pair_df[
            (pair_df["video_id"].astype(str) == str(video_id))
            & (pair_df["subject_id"].astype(str) == str(subject_id))
            & (pair_df["object_id"].astype(str) == str(object_id))
        ].set_index("frame_idx")
        pair = pair.reindex(frame_indices).ffill().bfill().fillna(0.0)
        pair_cols = feature_columns(pair_df)
        pieces.append(pair[pair_cols].to_numpy(dtype=np.float32))
    matrix = np.concatenate(pieces, axis=1)
    matrix = np.nan_to_num(matrix, nan=0.0)
    return frame_indices, matrix, valid


def build_tracks(
    identity_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    label_tracks: dict[tuple[str, str, str], np.ndarray],
    feature_names: list[str] | None = None,
    label_map: LabelMap | None = None,
    behavior_roles: BehaviorRolesConfig | None = None,
) -> tuple[list[TrackData], list[str]]:
    """Build labeled tracks aligned to features. Returns (tracks, feature_names)."""

    label_tracks = expand_global_label_tracks(
        identity_df, label_tracks, label_map, behavior_roles
    )
    is_pair = any(object_id for (_, _, object_id) in label_tracks.keys())
    if feature_names is None:
        feature_names = build_feature_column_list(identity_df, pair_df, is_pair)

    tracks: list[TrackData] = []
    for (video_id, subject_id, object_id), labels in label_tracks.items():
        if label_map is not None and behavior_roles is not None:
            labels = constrain_label_array_for_subject(
                labels, label_map, behavior_roles, subject_id
            )
        try:
            frame_indices, matrix, valid = assemble_track_matrix(
                identity_df, pair_df, video_id, subject_id, object_id, feature_names
            )
        except ValueError:
            continue
        if matrix.shape[1] != len(feature_names):
            # Pad/trim defensively so all tracks share channel count.
            matrix = _fit_channels(matrix, len(feature_names))
        n = min(matrix.shape[0], labels.shape[-1])
        matrix = matrix[:n]
        valid = valid[:n]
        if labels.ndim == 2:
            track_labels = labels[:, :n]
        else:
            track_labels = labels[:n]
        tracks.append(
            TrackData(
                video_id=str(video_id),
                subject_id=str(subject_id),
                object_id=str(object_id),
                frame_indices=frame_indices[:n],
                features=matrix,
                feature_names=feature_names,
                labels=track_labels,
                valid_mask=valid,
            )
        )
    return tracks, feature_names


def build_inference_tracks(
    identity_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    feature_names: list[str],
    is_pair: bool,
) -> list[TrackData]:
    """Build unlabeled tracks for every identity (or ordered pair) in a video."""

    tracks: list[TrackData] = []
    videos = identity_df["video_id"].astype(str).unique()
    for video_id in videos:
        subjects = identity_df[identity_df["video_id"].astype(str) == video_id][
            "subject_id"
        ].astype(str).unique()
        if not is_pair:
            for subject in subjects:
                tracks.append(
                    _make_inference_track(
                        identity_df, pair_df, video_id, subject, "", feature_names
                    )
                )
        else:
            for subject in subjects:
                for obj in subjects:
                    if subject == obj:
                        continue
                    tracks.append(
                        _make_inference_track(
                            identity_df, pair_df, video_id, subject, obj, feature_names
                        )
                    )
    return [t for t in tracks if t is not None and t.num_frames > 0]


def _make_inference_track(identity_df, pair_df, video_id, subject, obj, feature_names):
    try:
        frame_indices, matrix, valid = assemble_track_matrix(
            identity_df, pair_df, video_id, subject, obj, feature_names
        )
    except ValueError:
        return None
    if matrix.shape[1] != len(feature_names):
        matrix = _fit_channels(matrix, len(feature_names))
    return TrackData(
        video_id=str(video_id),
        subject_id=str(subject),
        object_id=str(obj),
        frame_indices=frame_indices,
        features=matrix,
        feature_names=feature_names,
        labels=None,
        valid_mask=valid,
    )


def _fit_channels(matrix: np.ndarray, target: int) -> np.ndarray:
    current = matrix.shape[1]
    if current == target:
        return matrix
    if current > target:
        return matrix[:, :target]
    pad = np.zeros((matrix.shape[0], target - current), dtype=matrix.dtype)
    return np.concatenate([matrix, pad], axis=1)


def split_tracks_by_group(
    tracks: list[TrackData],
    train_fraction: float,
    validation_fraction: float,
    seed: int = 42,
) -> dict[str, list[TrackData]]:
    """Split tracks into train/val/test by their ``video_id`` group, deterministically."""

    groups = sorted({t.video_id for t in tracks})
    if len(groups) == 1:
        # Single video: fall back to splitting tracks themselves.
        return _split_list(tracks, train_fraction, validation_fraction, seed)

    def group_hash(group: str) -> float:
        digest = hashlib.md5(f"{seed}:{group}".encode()).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    train, val, test = [], [], []
    for group in groups:
        score = group_hash(group)
        if score < train_fraction:
            target = train
        elif score < train_fraction + validation_fraction:
            target = val
        else:
            target = test
        target.extend(t for t in tracks if t.video_id == group)
    if not train:
        train = test or val
    if not val and train:
        # Few-video case: the whole-video split produced no validation set, which
        # silently breaks early stopping (the model keeps its epoch-0 weights).
        # Carve a temporal tail from each train track so early stopping has signal.
        train, val = _temporal_tail_split(train, max(validation_fraction, 0.15))
        if not test:
            test = val
    return {"train": train, "val": val, "test": test}


def _temporal_tail_split(
    tracks: list[TrackData], val_fraction: float
) -> tuple[list[TrackData], list[TrackData]]:
    """Split each track into an earlier train part and a later validation tail."""

    import copy

    train, val = [], []
    for t in tracks:
        n = t.num_frames
        cut = int(round(n * (1.0 - val_fraction)))
        if cut <= 0 or cut >= n:
            train.append(t)
            continue
        tr, va = copy.copy(t), copy.copy(t)
        tr.frame_indices, va.frame_indices = t.frame_indices[:cut], t.frame_indices[cut:]
        tr.features, va.features = t.features[:cut], t.features[cut:]
        if t.labels is not None:
            if t.labels.ndim == 2:
                tr.labels, va.labels = t.labels[:, :cut], t.labels[:, cut:]
            else:
                tr.labels, va.labels = t.labels[:cut], t.labels[cut:]
        if t.valid_mask is not None:
            tr.valid_mask, va.valid_mask = t.valid_mask[:cut], t.valid_mask[cut:]
        if t.mask_clip is not None:
            tr.mask_clip, va.mask_clip = t.mask_clip[:cut], t.mask_clip[cut:]
        train.append(tr)
        val.append(va)
    return train, val


def _split_list(tracks, train_fraction, validation_fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tracks))
    n_train = max(int(round(len(tracks) * train_fraction)), 1)
    n_val = int(round(len(tracks) * validation_fraction))
    train = [tracks[i] for i in order[:n_train]]
    val = [tracks[i] for i in order[n_train : n_train + n_val]]
    test = [tracks[i] for i in order[n_train + n_val :]] or val or train
    return {"train": train, "val": val or test, "test": test}


class WindowDataset(Dataset):
    """Windowed view over a list of tracks for framewise segmentation."""

    def __init__(
        self,
        tracks: list[TrackData],
        window_seconds: float,
        stride_seconds: float,
        frame_rate: float,
        normalizer: FeatureNormalizer,
        multilabel: bool = False,
        min_valid_fraction: float = 0.0,
    ) -> None:
        self.tracks = tracks
        self.normalizer = normalizer
        self.multilabel = multilabel
        window_length = seconds_to_frames(window_seconds, frame_rate)
        stride = seconds_to_frames(stride_seconds, frame_rate)
        self.spans: list[WindowSpan] = []
        for idx, track in enumerate(tracks):
            spans = build_windows(track.num_frames, window_length, stride, idx)
            for span in spans:
                if min_valid_fraction > 0 and track.valid_mask is not None:
                    frac = track.valid_mask[span.start : span.end].mean()
                    if frac < min_valid_fraction:
                        continue
                self.spans.append(span)

    def __len__(self) -> int:
        return len(self.spans)

    def labels_per_track(self) -> list[np.ndarray]:
        return [t.labels for t in self.tracks if t.labels is not None]

    def __getitem__(self, index: int) -> dict[str, Any]:
        span = self.spans[index]
        track = self.tracks[span.track_index]
        features = track.features[span.start : span.end]
        features = self.normalizer.transform(features, track.video_id)
        feat_tensor = torch.from_numpy(features.T.copy())  # [C, T]
        item: dict[str, Any] = {
            "features": feat_tensor,
            "video_id": track.video_id,
            "subject_id": track.subject_id,
            "object_id": track.object_id,
            "start": span.start,
            "end": span.end,
        }
        if track.valid_mask is not None:
            valid = track.valid_mask[span.start : span.end]
            item["valid_mask"] = torch.from_numpy(valid.astype(bool))
        if track.mask_clip is not None:
            clip = track.mask_clip[span.start : span.end]
            clip_tensor = torch.from_numpy(clip.astype(np.float32, copy=False))
            if clip_tensor.numel() > 0 and float(clip_tensor.max()) > 1.0:
                clip_tensor = clip_tensor / 255.0
            item["mask_clip"] = clip_tensor
        if track.labels is not None:
            if track.labels.ndim == 2:
                labels = track.labels[:, span.start : span.end]
                item["labels"] = torch.from_numpy(labels.astype(np.float32))
            else:
                labels = track.labels[span.start : span.end]
                item["labels"] = torch.from_numpy(labels.astype(np.int64))
        return item


def collate_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad a batch of variable-length windows along time to the max length."""

    max_len = max(item["features"].shape[1] for item in batch)
    channels = batch[0]["features"].shape[0]
    has_labels = "labels" in batch[0]
    multilabel = has_labels and batch[0]["labels"].dim() == 2
    has_mask_clip = "mask_clip" in batch[0]

    features = torch.zeros(len(batch), channels, max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    if has_labels:
        if multilabel:
            num_classes = batch[0]["labels"].shape[0]
            labels = torch.zeros(len(batch), num_classes, max_len)
        else:
            labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    if has_mask_clip:
        clip_shape = batch[0]["mask_clip"].shape[1:]
        mask_clips = torch.zeros(len(batch), max_len, *clip_shape)
    meta: list[dict[str, Any]] = []
    for i, item in enumerate(batch):
        length = item["features"].shape[1]
        features[i, :, :length] = item["features"]
        mask[i, :length] = True
        if "valid_mask" in item:
            mask[i, :length] &= item["valid_mask"].bool()
        if has_mask_clip:
            mask_clips[i, :length] = item["mask_clip"]
        if has_labels:
            if multilabel:
                labels[i, :, :length] = item["labels"]
            else:
                labels[i, :length] = item["labels"]
        meta.append(
            {
                "video_id": item["video_id"],
                "subject_id": item["subject_id"],
                "object_id": item["object_id"],
                "start": item["start"],
                "end": item["end"],
            }
        )
    out = {"features": features, "mask": mask, "meta": meta}
    if has_mask_clip:
        out["mask_clip"] = mask_clips
    if has_labels:
        out["labels"] = labels
    return out
