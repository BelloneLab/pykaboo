"""Free social-interaction pipeline (two identical BL6 mice, scene-level labels).

This is the second model in the project. Unlike the aggression dataset, here the
two animals are the *same* strain (``mouse1``/``mouse2``) and the behavior
annotations describe the *dyad* (``nose-to-nose``, ``fighting``, ``mounting`` ...),
not one animal. There are therefore no roles to attribute: every behavior is a
property of the interaction, so each per-identity track receives the SAME
scene-level multi-hot targets (the partner's geometry differs between the two
tracks, which doubles the supervised signal from one symmetric viewpoint each).

Key facts about this dataset (``free_interaction_model/dataset``):
- 21 videos (~9000 frames each, ~30 fps), perfectly frame-aligned to the masks.
- Label CSV header ``Frames`` is 1-based; ``Frames=1`` <-> COCO ``frame_index=0``.
- Behaviors: nose-to-nose, nose-to-body, anogenital, passive, rearing, fighting,
  mounting. ``fighting`` occurs in only one video (18 frames) and is essentially
  unlearnable cross-video; it is kept in the label map but reported honestly.

The "unsupervised network" requested is the EmbTCN-Attention-Transformer
(``models/embtcn_attention.py``): it is self-supervised pretrained (masked
reconstruction) on ALL 21 videos with no labels, then a multi-label head is
trained on the labeled tracks. An MS-TCN baseline shares the exact same feature
surface and split for an honest comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import AppConfig, load_config
from .dataset import TrackData, build_feature_column_list, build_tracks
from .labels import LabelMap
from .social_pipeline import build_social_features
from .social_train import select_feature_indices, subset_tracks

# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #

DEFAULT_DATASET_DIR = Path(
    "/home/andry/tracking_project/free_interaction_model/dataset"
)

# NOTE: ``fighting`` is intentionally excluded. It appears in only one video
# (760_juv, 18 frames) and is unlearnable cross-video; keeping it only added a
# permanent F1=0 row. To re-include it, add "fighting" back here and rerun.
BEHAVIORS = [
    "nose-to-nose",
    "nose-to-body",
    "anogenital",
    "passive",
    "rearing",
    "mounting",
]

# The lean feature surface that won the aggression ablation (mask-core + pose +
# wavelet). The full ~1300-col set overfits; see TCN.md section 16.
LEAN_GROUPS = ["maskcore", "pose", "wavelet"]

# Default 3-video validation split: one adult / female / juvenile partner from
# three different cage IDs, each with strong support on the six common behaviors.
DEFAULT_VAL_VIDEOS = ["745_juv", "760_adu", "812_fem"]


@dataclass
class VideoEntry:
    video_id: str          # e.g. "225_adu"
    coco: Path
    label_csv: Path
    pose_csv: Path | None


def discover_videos(dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> list[VideoEntry]:
    """Find every ``<id>_masks_coco.json`` with a sibling label CSV + tracking CSV."""

    dataset_dir = Path(dataset_dir)
    out: list[VideoEntry] = []
    for coco in sorted(dataset_dir.glob("*_masks_coco.json")):
        vid = coco.name[: -len("_masks_coco.json")]
        label_csv = dataset_dir / f"{vid}.csv"
        pose_csv = dataset_dir / f"{vid}_tracking.csv"
        if not label_csv.exists():
            continue
        out.append(
            VideoEntry(
                video_id=vid,
                coco=coco,
                label_csv=label_csv,
                pose_csv=pose_csv if pose_csv.exists() else None,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Labels: scene-level multi-hot, attributed to BOTH identities
# --------------------------------------------------------------------------- #

def build_free_label_map(background_label: str = "background") -> LabelMap:
    names = [background_label] + list(BEHAVIORS)
    return LabelMap(names=names, background_label=background_label)


def load_scene_label_tracks(
    label_csv: str | Path,
    video_id: str,
    frame_indices: list[int],
    identities: list[str],
    label_map: LabelMap,
) -> dict[tuple[str, str, str], np.ndarray]:
    """Per-identity multi-hot targets ``[K, T]`` from a wide scene CSV.

    The CSV ``Frames`` column is 1-based and frame-aligned to the masks, so
    ``frame_idx = Frames - 1``. Every behavior is attributed to BOTH identities
    (the annotation is a property of the dyad). Keys are
    ``(video_id, subject, partner)``.
    """

    df = pd.read_csv(label_csv)
    if "Frames" in df.columns:
        frame_idx = pd.to_numeric(df["Frames"], errors="coerce").round().astype("Int64").to_numpy() - 1
    else:
        frame_idx = np.arange(len(df), dtype=np.int64)

    frame_to_pos = {int(f): p for p, f in enumerate(frame_indices)}
    num_frames = len(frame_indices)
    name_to_id = label_map.name_to_id
    bg = label_map.background_id

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

    for ident in identities:
        get((str(video_id), str(ident), str(partner.get(ident, ""))))

    for behavior in BEHAVIORS:
        if behavior not in df.columns or behavior not in name_to_id:
            continue
        class_id = name_to_id[behavior]
        active = pd.to_numeric(df[behavior], errors="coerce").fillna(0).to_numpy() > 0
        for pos_in_df in np.flatnonzero(active):
            f = frame_idx[pos_in_df]
            if f is None:
                continue
            f = int(f)
            if f not in frame_to_pos:
                continue
            t = frame_to_pos[f]
            for ident in identities:
                key = (str(video_id), str(ident), str(partner.get(ident, "")))
                arr = get(key)
                arr[class_id, t] = 1
                arr[bg, t] = 0
    return tracks


# --------------------------------------------------------------------------- #
# Feature + track caching
# --------------------------------------------------------------------------- #

@dataclass
class CachedVideo:
    video_id: str
    tracks: list[TrackData]
    frame_rate: float


def _track_to_npz_dict(t: TrackData, prefix: str) -> dict[str, np.ndarray]:
    out = {
        f"{prefix}_features": t.features.astype(np.float32),
        f"{prefix}_frame_indices": np.asarray(t.frame_indices),
        f"{prefix}_labels": (t.labels if t.labels is not None else np.zeros((0,))),
        f"{prefix}_valid": (t.valid_mask if t.valid_mask is not None else np.ones(t.num_frames, bool)),
        f"{prefix}_meta": np.array([t.video_id, t.subject_id, t.object_id]),
    }
    return out


def build_and_cache_video(
    entry: VideoEntry,
    config: AppConfig,
    label_map: LabelMap,
    feature_names: list[str] | None,
    cache_dir: Path,
    lean: bool = True,
    log: Callable[[str], None] | None = None,
) -> tuple[CachedVideo, list[str]]:
    """Build fused features + scene-labeled tracks for one video, cache to npz."""

    def emit(m: str) -> None:
        if log:
            log(m)

    cache_dir.mkdir(parents=True, exist_ok=True)
    feats = build_social_features(
        entry.coco, config, pose_path=entry.pose_csv,
        use_pose=True, use_wavelets=True, log=emit,
    )
    # Use a STABLE video_id (the dataset stem, not the coco-derived one).
    vid = entry.video_id
    feats.identity_df["video_id"] = vid
    if not feats.pair_df.empty:
        feats.pair_df["video_id"] = vid

    # IMPORTANT: always assemble the FULL ordered column list for this video and
    # subset to lean afterwards. Passing a previously-returned LEAN list straight
    # into build_tracks makes assemble_track_matrix/_fit_channels trim the wide
    # matrix to len(lean) columns by position, scrambling features. The full
    # column list is deterministic across videos, so the lean result is identical.
    full_names = build_feature_column_list(
        feats.identity_df, feats.pair_df, is_pair=True
    )

    label_tracks = load_scene_label_tracks(
        entry.label_csv, vid, feats.frame_indices, feats.identities, label_map
    )
    tracks, full_names = build_tracks(
        feats.identity_df, feats.pair_df, label_tracks,
        feature_names=full_names, label_map=None, behavior_roles=None,
    )
    if lean:
        keep = select_feature_indices(full_names, LEAN_GROUPS)
        tracks, feature_names = subset_tracks(tracks, keep, full_names)
    else:
        feature_names = full_names

    cached = CachedVideo(video_id=vid, tracks=tracks, frame_rate=feats.frame_rate)
    payload: dict[str, np.ndarray] = {}
    for i, t in enumerate(tracks):
        payload.update(_track_to_npz_dict(t, f"t{i}"))
    np.savez_compressed(
        cache_dir / f"{vid}.npz",
        n_tracks=np.array([len(tracks)]),
        frame_rate=np.array([feats.frame_rate]),
        feature_names=np.array(feature_names, dtype=object),
        **payload,
    )
    return cached, feature_names


def load_cached_video(path: Path) -> tuple[CachedVideo, list[str]]:
    z = np.load(path, allow_pickle=True)
    n = int(z["n_tracks"][0])
    fr = float(z["frame_rate"][0])
    feature_names = list(z["feature_names"])
    tracks = []
    for i in range(n):
        p = f"t{i}"
        meta = z[f"{p}_meta"]
        labels = z[f"{p}_labels"]
        tracks.append(
            TrackData(
                video_id=str(meta[0]),
                subject_id=str(meta[1]),
                object_id=str(meta[2]),
                frame_indices=z[f"{p}_frame_indices"],
                features=z[f"{p}_features"].astype(np.float32),
                feature_names=feature_names,
                labels=(labels if labels.ndim == 2 else None),
                valid_mask=z[f"{p}_valid"],
            )
        )
    return CachedVideo(video_id=path.stem, tracks=tracks, frame_rate=fr), feature_names


def _relabel_cached_video(cached: CachedVideo, entry: VideoEntry, label_map: LabelMap) -> None:
    """Recompute each cached track's labels from the CSV under the current map.

    Features and frame grid are left untouched; only the ``[K, T]`` target rows are
    rebuilt, so a change to ``BEHAVIORS`` does not require re-extracting features.
    """

    if not cached.tracks:
        return
    frame_indices = list(cached.tracks[0].frame_indices)
    identities = sorted({t.subject_id for t in cached.tracks})
    label_tracks = load_scene_label_tracks(
        entry.label_csv, cached.video_id, frame_indices, identities, label_map
    )
    for t in cached.tracks:
        key = (str(cached.video_id), str(t.subject_id), str(t.object_id))
        new = label_tracks.get(key)
        if new is not None:
            n = min(new.shape[1], t.features.shape[0])
            t.labels = new[:, :n]


def _resave_cached_video(npz_path: Path, cached: CachedVideo, feature_names: list[str]) -> None:
    payload: dict[str, np.ndarray] = {}
    for i, t in enumerate(cached.tracks):
        payload.update(_track_to_npz_dict(t, f"t{i}"))
    np.savez_compressed(
        npz_path,
        n_tracks=np.array([len(cached.tracks)]),
        frame_rate=np.array([cached.frame_rate]),
        feature_names=np.array(feature_names, dtype=object),
        **payload,
    )


def build_or_load_cache(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    cache_dir: str | Path = "outputs/free_social/cache",
    config_path: str = "configs/default.yaml",
    lean: bool = True,
    rebuild: bool = False,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, CachedVideo], list[str], LabelMap]:
    """Build (or load) cached tracks for every video. Returns (by_video, names, lm)."""

    def emit(m: str) -> None:
        if log:
            log(m)

    cache_dir = Path(cache_dir)
    config = load_config(config_path)
    label_map = build_free_label_map()
    entries = discover_videos(dataset_dir)
    by_video: dict[str, CachedVideo] = {}
    feature_names: list[str] | None = None

    for entry in entries:
        npz = cache_dir / f"{entry.video_id}.npz"
        if npz.exists() and not rebuild:
            cached, names = load_cached_video(npz)
            if feature_names is None:
                feature_names = names
            # Self-heal: if the behavior set changed (e.g. fighting dropped), the
            # cached label rows no longer match the current label map. Recompute
            # labels from the CSV using the current map; features are unaffected.
            stale = any(
                t.labels is not None and t.labels.shape[0] != label_map.num_classes
                for t in cached.tracks
            )
            if stale:
                emit(f"relabeling {entry.video_id} (behavior set changed)")
                _relabel_cached_video(cached, entry, label_map)
                _resave_cached_video(npz, cached, names)
            by_video[entry.video_id] = cached
            emit(f"loaded cache {entry.video_id} ({len(cached.tracks)} tracks)")
            continue
        emit(f"building {entry.video_id} ...")
        cached, feature_names = build_and_cache_video(
            entry, config, label_map, feature_names, cache_dir, lean=lean, log=emit
        )
        by_video[entry.video_id] = cached
        emit(f"cached {entry.video_id} ({len(cached.tracks)} tracks, {len(feature_names)} feats)")

    assert feature_names is not None, "no videos found"
    return by_video, feature_names, label_map


def split_videos(
    by_video: dict[str, CachedVideo], val_videos: list[str] | None = None
) -> tuple[list[str], list[str]]:
    val = [v for v in (val_videos or DEFAULT_VAL_VIDEOS) if v in by_video]
    train = [v for v in by_video if v not in val]
    return train, val


def tracks_for(by_video: dict[str, CachedVideo], video_ids: list[str]) -> list[TrackData]:
    out: list[TrackData] = []
    for v in video_ids:
        out.extend(by_video[v].tracks)
    return out
