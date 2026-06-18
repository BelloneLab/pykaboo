"""Fused-input caches for the free-interaction benchmark.

Combines complementary input modalities along the feature axis, aligned per
``(video, subject, object)`` track and per frame:

- ``features`` : the 432 engineered descriptors (mask shape + pose geometry +
  wavelets). Strong on posture/shape behaviors (rearing).
- ``pose``     : 32 egocentric keypoint coordinates. Strong on contact behaviors
  (nose-to-nose, passive) because they encode inter-animal geometry directly.
- ``maskcnn``  : 64-d per-frame embeddings from the trained 2D mask-CNN encoder,
  broadcast from the scene clip to both identity tracks.

The hypothesis: posture and contact information are complementary, so a fused
input should beat any single modality. ``build_fused_cache`` returns the same
``(by_video, feature_names, label_map)`` triple as the single-modality caches, so
the existing benchmark runs on it unchanged.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np

from .dataset import TrackData
from .free_pose import build_pose_cache
from .free_social import CachedVideo, build_or_load_cache
from .labels import LabelMap


def _index_tracks(by_video: dict[str, CachedVideo]) -> dict[tuple, TrackData]:
    out = {}
    for vid, cv in by_video.items():
        for t in cv.tracks:
            out[(vid, t.subject_id, t.object_id)] = t
    return out


def build_fused_cache(
    sources: list[str],
    maskcnn_emb_dir: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, CachedVideo], list[str], LabelMap]:
    """Concatenate the requested modality features per track. ``sources`` is an
    ordered subset of {"features", "pose", "maskcnn"}."""

    def emit(m: str) -> None:
        if log:
            log(m)

    caches: dict[str, dict[str, CachedVideo]] = {}
    names: dict[str, list[str]] = {}
    label_map: LabelMap | None = None
    frame_rate = 30.0

    if "features" in sources:
        bv, nm, lm = build_or_load_cache(log=lambda m: None)
        caches["features"], names["features"], label_map = bv, nm, lm
        frame_rate = next(iter(bv.values())).frame_rate
    if "pose" in sources:
        bv, nm, lm = build_pose_cache(log=lambda m: None)
        caches["pose"], names["pose"] = bv, nm
        label_map = label_map or lm
    if "maskcnn" in sources:
        bv, nm = _load_maskcnn_as_tracks(maskcnn_emb_dir, label_map)
        caches["maskcnn"], names["maskcnn"] = bv, nm

    assert label_map is not None
    indexed = {s: _index_tracks(caches[s]) for s in sources}
    keys = set.intersection(*[set(ix.keys()) for ix in indexed.values()])

    feature_names: list[str] = []
    for s in sources:
        feature_names += [f"{s}:{n}" for n in names[s]]

    by_video: dict[str, CachedVideo] = {}
    for (vid, subj, obj) in sorted(keys):
        mats, labels_ref, frames_ref, valid_ref = [], None, None, None
        n = min(indexed[s][(vid, subj, obj)].features.shape[0] for s in sources)
        for s in sources:
            t = indexed[s][(vid, subj, obj)]
            mats.append(t.features[:n])
            if labels_ref is None:
                labels_ref = t.labels[:, :n] if t.labels is not None else None
                frames_ref = np.asarray(t.frame_indices)[:n]
                valid_ref = (t.valid_mask[:n] if t.valid_mask is not None else np.ones(n, bool))
        fused = np.concatenate(mats, axis=1).astype(np.float32)
        track = TrackData(video_id=vid, subject_id=subj, object_id=obj,
                          frame_indices=frames_ref, features=fused,
                          feature_names=feature_names, labels=labels_ref, valid_mask=valid_ref)
        by_video.setdefault(vid, CachedVideo(video_id=vid, tracks=[], frame_rate=frame_rate)).tracks.append(track)

    emit(f"fused {sources}: {len(by_video)} videos, {len(feature_names)} channels")
    return by_video, feature_names, label_map


def _load_maskcnn_as_tracks(emb_dir, label_map):
    """Turn cached scene mask-CNN embeddings into per-identity tracks.

    The mask-CNN produces one scene embedding stream per video; it is broadcast to
    both identity tracks (subject 1 and 2) so it aligns with the other modalities.
    """
    from .free_social import build_or_load_cache  # for the (video,subject,object) keys
    by_video_feat, _, _ = build_or_load_cache(log=lambda m: None)
    emb_dir = Path(emb_dir)
    names = [f"mc_{i}" for i in range(64)]
    out: dict[str, CachedVideo] = {}
    for vid, cv in by_video_feat.items():
        npz = emb_dir / f"{vid}_maskcnn.npz"
        if not npz.exists():
            continue
        Z = np.load(npz)["Z"]  # [F, E]
        tracks = []
        for t in cv.tracks:
            n = min(Z.shape[0], t.features.shape[0])
            tracks.append(TrackData(video_id=vid, subject_id=t.subject_id, object_id=t.object_id,
                                    frame_indices=np.asarray(t.frame_indices)[:n],
                                    features=Z[:n].astype(np.float32), feature_names=names,
                                    labels=(t.labels[:, :n] if t.labels is not None else None),
                                    valid_mask=None))
        out[vid] = CachedVideo(video_id=vid, tracks=tracks, frame_rate=cv.frame_rate)
    return out, names
