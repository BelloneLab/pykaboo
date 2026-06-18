"""Corrected social-behavior pipeline: per-identity multi-label attribution.

This module fixes the three structural defects of the original single-stream
pipeline:

1. *Identity attribution.* The wide annotation CSV is one behavior stream for the
   whole cage. Instead of duplicating it onto both mice (which trains the model to
   predict ``CD1 approach`` from the BL6 mouse), each behavior is routed to the
   animal that performs it: ``BL6 *`` -> mouse1, ``CD1/attack* `` -> mouse2,
   mutual behaviors (``reciprocal``, ``attack_wrestling``) -> both.
2. *Multi-label.* Targets are multi-hot per frame, so co-occurring and
   hierarchical behaviors (``attack`` + ``attack_biting``) are preserved.
3. *Rich features.* Mask-shape features are fused with pose-keypoint posture and
   dyadic geometry, plus optional wavelet rhythm channels.

Each per-identity track still carries the partner's features and the pairwise
geometry, so the model sees the social context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import AppConfig
from .coco_masks import load_coco_videos
from .features import extract_video_features, feature_columns
from .labels import LabelMap, infer_video_id_from_label_path
from .pose import find_pose_csv_for_coco, load_pose_csv
from .pose_features import extract_pose_feature_tables
from .wavelet_features import DEFAULT_WAVELET_SIGNALS, add_wavelet_features

META_COLUMNS = ["video_id", "frame_idx", "subject_id", "object_id"]

# Default identity vocabulary: strain name -> COCO track_id identity string.
DEFAULT_IDENTITY_MAP = {"BL6": "1", "CD1": "2"}


def attribute_behavior(
    name: str,
    identity_map: dict[str, str] | None = None,
    overrides: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return the identity strings that perform behavior ``name``.

    Rules (applied in order): explicit override; ``reciprocal``/``*wrestling`` ->
    both; ``BL6 *`` -> BL6 identity; ``CD1/Cd1/attack* `` -> CD1 identity; else
    both animals.
    """

    identity_map = identity_map or DEFAULT_IDENTITY_MAP
    overrides = overrides or {}
    if name in overrides:
        return list(overrides[name])
    bl6 = identity_map.get("BL6", "1")
    cd1 = identity_map.get("CD1", "2")
    lower = name.lower()
    if "reciprocal" in lower or "wrestl" in lower:
        return [bl6, cd1]
    if lower.startswith("bl6"):
        return [bl6]
    if lower.startswith("cd1") or lower.startswith("attack"):
        return [cd1]
    return [bl6, cd1]


def _times_to_frames(
    times: np.ndarray,
    frame_rate: float | None,
    frame_times: np.ndarray | None,
    frame_indices: np.ndarray | None,
) -> np.ndarray:
    """Map annotation wall-clock ``times`` (s) to COCO frame indices.

    Prefers nearest match against the COCO per-frame ``frame_times`` (robust to a
    non-zero start time and to a frame-rate mismatch between the annotation clock
    and the video). Falls back to ``round(time * frame_rate)``.
    """

    times = np.asarray(times, dtype=np.float64)
    if frame_times is not None and frame_indices is not None and len(frame_times):
        ft = np.asarray(frame_times, dtype=np.float64)
        fi = np.asarray(frame_indices)
        order = np.argsort(ft)
        ft_s, fi_s = ft[order], fi[order]
        pos = np.clip(np.searchsorted(ft_s, times), 0, len(ft_s) - 1)
        left = np.clip(pos - 1, 0, len(ft_s) - 1)
        choose_left = np.abs(times - ft_s[left]) <= np.abs(times - ft_s[pos])
        nearest = np.where(choose_left, left, pos)
        return fi_s[nearest].astype(np.int64)
    if frame_rate:
        return np.round(times * float(frame_rate)).astype(np.int64)
    return np.round(times).astype(np.int64)


def read_wide_csv(
    path: str | Path,
    frame_rate: float | None = None,
    frame_times: np.ndarray | None = None,
    frame_indices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Read a wide framewise behavior CSV -> (frame, behavior_cols, frame_idx).

    If the CSV carries an explicit ``frame``/``frame_idx`` column it is trusted.
    Otherwise, when a ``time`` column is present, frames are derived from time so
    annotations that do not start at t=0 (or use a slightly different fps than the
    video) line up with the masks. Row-index is only a last resort.
    """

    from .labels import read_label_csv, wide_behavior_columns

    df = read_label_csv(path)
    beh_cols = wide_behavior_columns(df)
    if "frame_idx" in df.columns:
        frame_idx = pd.to_numeric(df["frame_idx"], errors="coerce").round().astype("Int64").to_numpy()
    elif "frame" in df.columns:
        frame_idx = pd.to_numeric(df["frame"], errors="coerce").round().astype("Int64").to_numpy()
    elif "time" in df.columns:
        times = pd.to_numeric(df["time"], errors="coerce").to_numpy()
        frame_idx = _times_to_frames(times, frame_rate, frame_times, frame_indices)
    else:
        frame_idx = np.arange(len(df), dtype=np.int64)
    return df, beh_cols, frame_idx


def build_label_map_from_csvs(
    csv_paths: list[str],
    background_label: str = "background",
) -> LabelMap:
    """Stable label map: background first, then all behaviors sorted by name."""

    behaviors: set[str] = set()
    for path in csv_paths:
        _, beh_cols, _ = read_wide_csv(path)
        behaviors.update(beh_cols)
    names = [background_label] + sorted(behaviors)
    return LabelMap(names=names, background_label=background_label)


def align_wide_labels_per_identity(
    csv_path: str | Path,
    video_id: str,
    frame_indices: list[int],
    identities: list[str],
    label_map: LabelMap,
    identity_map: dict[str, str] | None = None,
    overrides: dict[str, list[str]] | None = None,
    frame_rate: float | None = None,
    frame_times: np.ndarray | None = None,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Build per-identity multi-hot targets ``[K, T]`` from a wide CSV.

    The returned keys are ``(video_id, subject_id, partner_id)`` with the partner
    being the other tracked animal, so each track carries social context.

    Label rows are aligned to mask frames by **time** (using the CSV ``time``
    column against the video frame rate / per-frame timestamps) so a non-zero
    annotation start or fps mismatch does not shift the targets.
    """

    df, beh_cols, frame_idx = read_wide_csv(
        csv_path, frame_rate=frame_rate, frame_times=frame_times,
        frame_indices=np.asarray(frame_indices),
    )
    frame_to_pos = {int(f): p for p, f in enumerate(frame_indices)}
    num_frames = len(frame_indices)
    name_to_id = label_map.name_to_id
    bg = label_map.background_id

    # partner lookup for the (typically two) tracked identities
    partner = {}
    for ident in identities:
        others = [o for o in identities if o != ident]
        partner[ident] = others[0] if others else ""

    tracks: dict[tuple[str, str, str], np.ndarray] = {}

    def get(key: tuple[str, str, str]) -> np.ndarray:
        if key not in tracks:
            arr = np.zeros((label_map.num_classes, num_frames), dtype=np.int8)
            arr[bg, :] = 1
            tracks[key] = arr
        return tracks[key]

    # ensure every identity has a track even if it never acts
    for ident in identities:
        get((str(video_id), str(ident), str(partner.get(ident, ""))))

    for behavior in beh_cols:
        if behavior not in name_to_id:
            continue
        class_id = name_to_id[behavior]
        active = pd.to_numeric(df[behavior], errors="coerce").fillna(0).to_numpy() > 0
        owners = attribute_behavior(behavior, identity_map, overrides)
        owners = [o for o in owners if o in identities]
        for pos_in_df in np.flatnonzero(active):
            frame = int(frame_idx[pos_in_df]) if frame_idx[pos_in_df] is not None else -1
            if frame not in frame_to_pos:
                continue
            t = frame_to_pos[frame]
            for owner in owners:
                key = (str(video_id), str(owner), str(partner.get(owner, "")))
                arr = get(key)
                arr[class_id, t] = 1
                arr[bg, t] = 0
    return tracks


# --------------------------------------------------------------------------- #
# Feature fusion
# --------------------------------------------------------------------------- #

def _merge_on_meta(
    base: pd.DataFrame, extra: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    if extra is None or extra.empty:
        return base
    extra_cols = [c for c in extra.columns if c not in keys]
    dedup = [c for c in extra_cols if c in base.columns]
    extra = extra.drop(columns=dedup) if dedup else extra
    return base.merge(extra, on=keys, how="left")


@dataclass
class SocialFeatures:
    video_id: str
    identity_df: pd.DataFrame
    pair_df: pd.DataFrame
    frame_indices: list[int]
    identities: list[str]
    frame_rate: float = 25.0
    frame_times: list[float] = field(default_factory=list)
    pose_quality: dict[str, Any] = field(default_factory=dict)


def build_social_features(
    coco_path: str | Path,
    config: AppConfig,
    pose_path: str | Path | None = None,
    use_pose: bool = True,
    use_wavelets: bool = True,
    wavelet_signals: list[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> SocialFeatures:
    """Fuse mask + pose (+ wavelet) features for one COCO video."""

    def emit(msg: str) -> None:
        if log:
            log(msg)

    videos = load_coco_videos(coco_path, config.data)
    video = max(videos.values(), key=lambda v: v.num_frames)
    # Use the video's true frame rate (from COCO timestamps) for all kinematics,
    # overriding any stale config default so features match across train/infer.
    detected_fps = video.fps
    if abs(detected_fps - config.data.frame_rate) > 0.2:
        emit(f"Using detected fps {detected_fps:.4f} (config had {config.data.frame_rate})")
        config = config.model_copy(deep=True)
        config.data.frame_rate = detected_fps
    emit(f"Mask features: {video.video_id} ({video.num_frames} frames @ {detected_fps:.4f} fps)")
    identity_df, pair_df = extract_video_features(video, config)

    if use_pose:
        pose_path = pose_path or find_pose_csv_for_coco(coco_path)
        if pose_path and Path(pose_path).exists():
            pose = load_pose_csv(
                pose_path, video_id=video.video_id, min_likelihood=0.3
            )
            pose_id, pose_pair = extract_pose_feature_tables(
                pose,
                video.width,
                video.height,
                config.data.frame_rate,
                egocentric=config.features.egocentric,
            )
            identity_df = _merge_on_meta(
                identity_df, pose_id, ["video_id", "frame_idx", "subject_id", "object_id"]
            )
            pair_df = _merge_on_meta(
                pair_df, pose_pair, ["video_id", "frame_idx", "subject_id", "object_id"]
            )
            emit(f"Fused pose features ({pose_id.shape[1]} id, {pose_pair.shape[1]} pair cols)")

            # Mask contact-location-on-body-axis + occlusion features (the proven
            # free-interaction lever; reusable here for both models). Directed,
            # per ordered pair, merged into pair_df. Needs the loaded pose.
            if getattr(config.features, "contact_geometry", False):
                from .free_mask_contact import contact_pair_dataframe

                contact_df = contact_pair_dataframe(video, pose, video.video_id)
                if not contact_df.empty:
                    pair_df = _merge_on_meta(
                        pair_df, contact_df,
                        ["video_id", "frame_idx", "subject_id", "object_id"],
                    )
                    emit(f"Added contact-geometry features ({contact_df.shape[1] - 4} cols)")
        else:
            emit("WARNING: no pose CSV found; continuing with mask features only.")

    if use_wavelets:
        signals = wavelet_signals or DEFAULT_WAVELET_SIGNALS
        present = [c for c in signals if c in identity_df.columns]
        # merge a couple of pair signals into identity series per subject for CWT
        pair_signal_map = {
            "pp_body_body": "pp_body_body",
            "pp_nose_nose": "pp_nose_nose",
            "pp_approach_speed": "pp_approach_speed",
        }
        parts: list[pd.DataFrame] = []
        for subject, grp in identity_df.groupby("subject_id", sort=False):
            grp = grp.sort_values("frame_idx").copy()
            # attach pair signals (subject vs its partner) if available
            if not pair_df.empty:
                pj = pair_df[pair_df["subject_id"].astype(str) == str(subject)]
                if not pj.empty:
                    pj = pj.sort_values("frame_idx")
                    merged = grp.merge(
                        pj[["frame_idx", *[c for c in pair_signal_map if c in pj.columns]]],
                        on="frame_idx",
                        how="left",
                    )
                    grp = merged
            sigs = present + [c for c in pair_signal_map if c in grp.columns]
            grp = add_wavelet_features(grp, sigs, config.data.frame_rate)
            # drop the temporarily attached pair signals (keep their cwt cols)
            drop = [c for c in pair_signal_map if c in grp.columns]
            grp = grp.drop(columns=drop, errors="ignore")
            parts.append(grp)
        identity_df = pd.concat(parts, ignore_index=True)
        emit(f"Added wavelet channels (signals: {len(present)})")

    identity_df = identity_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if not pair_df.empty:
        pair_df = pair_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return SocialFeatures(
        video_id=video.video_id,
        identity_df=identity_df,
        pair_df=pair_df,
        frame_indices=list(video.frame_indices),
        identities=sorted(identity_df["subject_id"].astype(str).unique()),
        frame_rate=detected_fps,
        frame_times=list(video.timestamps),
    )
