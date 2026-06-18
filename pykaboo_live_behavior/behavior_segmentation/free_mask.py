"""Raw mask-image clip cache for the free-interaction 2D-CNN backbone.

This is the honest "spatial field over time" input: per frame, the two animals'
segmentation masks are rendered into a small cropped image (background removed,
one channel per animal plus their overlap), giving a clip ``[F, C, H, W]``. The
project's :class:`MaskVideoEncoder` (Conv2d per frame + temporal TCN) consumes
these directly, unlike the NEMBA sequence backbones which only take ``[C, T]``.

One scene-level clip stream per video (both animals share the crop), with the
scene-level multi-label targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .coco_masks import load_coco_videos
from .config import load_config
from .free_social import (
    DEFAULT_DATASET_DIR,
    build_free_label_map,
    discover_videos,
    load_scene_label_tracks,
)
from .labels import LabelMap
from .mask_video import render_mask_clip


@dataclass
class MaskClipVideo:
    video_id: str
    clip: np.ndarray          # [F, C, H, W] uint8 (0..255)
    labels: np.ndarray        # [K, F] int8 scene-level multi-hot
    frame_rate: float = 30.0


def build_mask_clip_cache(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    cache_dir: str | Path = "outputs/free_mask_cache",
    clip_size: int = 48,
    config_path: str = "configs/default.yaml",
    rebuild: bool = False,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, MaskClipVideo], LabelMap]:
    """Render + cache one scene clip stream per video. Returns (by_video, label_map)."""

    def emit(m: str) -> None:
        if log:
            log(m)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    label_map = build_free_label_map()
    by_video: dict[str, MaskClipVideo] = {}

    for entry in discover_videos(dataset_dir):
        npz = cache_dir / f"{entry.video_id}_s{clip_size}.npz"
        if npz.exists() and not rebuild:
            z = np.load(npz)
            by_video[entry.video_id] = MaskClipVideo(
                entry.video_id, z["clip"], z["labels"], float(z["frame_rate"][0]))
            emit(f"loaded clip cache {entry.video_id} {z['clip'].shape}")
            continue
        videos = load_coco_videos(entry.coco, config.data)
        video = max(videos.values(), key=lambda v: v.num_frames)
        frames = list(video.frame_indices)
        clip, used = render_mask_clip(video, frame_indices=frames, size=clip_size,
                                      with_overlap=True, smooth_box=True)
        clip_u8 = np.clip(clip * 255.0, 0, 255).astype(np.uint8)  # [F, C, H, W]
        identities = list(video.identities)
        label_tracks = load_scene_label_tracks(
            entry.label_csv, entry.video_id, frames, identities, label_map)
        # scene labels are identical across identities; take the first track
        labels = next(iter(label_tracks.values()))  # [K, F]
        n = min(clip_u8.shape[0], labels.shape[1])
        clip_u8, labels = clip_u8[:n], labels[:, :n]
        np.savez_compressed(npz, clip=clip_u8, labels=labels.astype(np.int8),
                            frame_rate=np.array([video.fps]))
        by_video[entry.video_id] = MaskClipVideo(entry.video_id, clip_u8, labels, float(video.fps))
        emit(f"rendered {entry.video_id} clip {clip_u8.shape} @ {video.fps:.2f} fps")

    return by_video, label_map
