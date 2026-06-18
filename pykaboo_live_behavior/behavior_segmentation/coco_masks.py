"""Parse RF-DETR COCO JSON into per-frame, per-identity mask records.

The parser is deliberately permissive about where identity and frame index live,
because different export tools place them in different fields. Identity and frame
resolution strategies are configurable through :class:`~behavior_segmentation.config.DataConfig`.

Public API:

- :class:`MaskRecord`        one decoded mask for one identity at one frame
- :class:`CocoVideo`         all records for one ``video_id`` plus frame metadata
- :func:`load_coco_video`    parse a single COCO JSON file into a :class:`CocoVideo`
- :func:`decode_segmentation` turn polygon/RLE segmentation into a binary mask
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import DataConfig
from .storage import read_json


class CocoParseError(RuntimeError):
    """Raised when a COCO JSON cannot be parsed into usable mask records."""


@dataclass
class MaskRecord:
    """One decoded instance mask for one identity at one frame."""

    frame_idx: int
    identity: str
    bbox: tuple[float, float, float, float]  # x, y, w, h
    score: float
    height: int
    width: int
    segmentation: Any = None  # raw COCO segmentation, decoded lazily
    category_id: int | None = None

    def decode_mask(self) -> np.ndarray | None:
        """Decode this record's segmentation into a boolean mask, or ``None``."""

        return decode_segmentation(self.segmentation, self.height, self.width)


@dataclass
class CocoVideo:
    """All mask records for a single video, indexed by frame and identity."""

    video_id: str
    width: int
    height: int
    frame_indices: list[int]
    records: list[MaskRecord] = field(default_factory=list)
    source_path: str | None = None
    frame_timestamps: dict[int, float] = field(default_factory=dict)

    @property
    def timestamps(self) -> list[float]:
        """Per-frame wall-clock timestamps aligned to ``frame_indices``.

        Falls back to ``frame_index / fps`` when explicit timestamps are absent.
        """

        if self.frame_timestamps:
            return [
                self.frame_timestamps.get(fi, fi / self.fps)
                for fi in self.frame_indices
            ]
        fps = self.fps
        return [fi / fps for fi in self.frame_indices]

    @property
    def fps(self) -> float:
        """Frame rate inferred from timestamps (median 1/dt), else 25.0."""

        if len(self.frame_timestamps) >= 2:
            items = sorted(self.frame_timestamps.items())
            ts = np.asarray([t for _, t in items], dtype=np.float64)
            dt = np.diff(ts)
            dt = dt[dt > 1e-9]
            if dt.size:
                return float(1.0 / np.median(dt))
        return 25.0

    @property
    def identities(self) -> list[str]:
        seen: dict[str, None] = {}
        for record in self.records:
            seen.setdefault(record.identity, None)
        return list(seen.keys())

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    def records_by_frame(self) -> dict[int, list[MaskRecord]]:
        out: dict[int, list[MaskRecord]] = {idx: [] for idx in self.frame_indices}
        for record in self.records:
            out.setdefault(record.frame_idx, []).append(record)
        return out


def get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    """Resolve a possibly dotted key such as ``attributes.identity``."""

    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def resolve_identity(
    annotation: dict[str, Any],
    priority: list[str],
    category_names: dict[int, str],
) -> str | None:
    """Resolve the identity label for an annotation using the priority order."""

    for field_name in priority:
        if field_name == "category_id":
            cat_id = annotation.get("category_id")
            if cat_id is not None:
                return category_names.get(int(cat_id), f"category_{cat_id}")
            continue
        value = get_nested(annotation, field_name)
        if value is not None and value != "":
            return str(value)
    return None


def resolve_frame_index(
    image: dict[str, Any], regex: str, fallback_order: int
) -> int:
    """Resolve a frame index from explicit fields, file name regex, or order."""

    for key in ("frame_idx", "frame_index", "frame"):
        if key in image and image[key] is not None:
            return int(image[key])
    file_name = image.get("file_name", "")
    if regex:
        match = re.search(regex, str(file_name))
        if match:
            return int(match.group(1))
    return fallback_order


def resolve_video_id(
    image: dict[str, Any], video_id_field: str, default_video_id: str
) -> str:
    value = image.get(video_id_field)
    if value is None:
        value = image.get("video_id")
    return str(value) if value is not None else default_video_id


def decode_segmentation(
    segmentation: Any, height: int, width: int
) -> np.ndarray | None:
    """Decode polygon, RLE, or compressed-RLE COCO segmentation to a bool mask."""

    if segmentation is None:
        return None
    try:
        if isinstance(segmentation, list):
            if len(segmentation) == 0:
                return None
            return decode_polygon(segmentation, height, width)
        if isinstance(segmentation, dict) and "counts" in segmentation:
            return decode_rle(segmentation, height, width)
    except Exception:
        return None
    return None


def decode_polygon(polygons: list, height: int, width: int) -> np.ndarray | None:
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    drew = False
    for polygon in polygons:
        if polygon is None or len(polygon) < 6:
            continue
        pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        drew = True
    if not drew:
        return None
    return mask.astype(bool)


def decode_rle(segmentation: dict, height: int, width: int) -> np.ndarray | None:
    from pycocotools import mask as mask_utils

    rle = dict(segmentation)
    counts = rle.get("counts")
    if isinstance(counts, list):
        rle = mask_utils.frPyObjects(rle, height, width)
    elif isinstance(counts, str):
        rle = dict(rle)
        rle["counts"] = counts.encode("ascii")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return decoded.astype(bool)


def load_coco_video(path: str | Path, data_config: DataConfig) -> CocoVideo:
    """Parse a single COCO JSON file into a :class:`CocoVideo`.

    All annotations are grouped by their resolved ``video_id``; if more than one
    video is present the one with the most frames is returned and the others are
    ignored. Use :func:`load_coco_videos` to retain every video.
    """

    videos = load_coco_videos(path, data_config)
    if not videos:
        raise CocoParseError(f"No usable mask records found in {path}")
    return max(videos.values(), key=lambda video: video.num_frames)


def load_coco_videos(
    path: str | Path, data_config: DataConfig
) -> dict[str, CocoVideo]:
    """Parse a COCO JSON file into one :class:`CocoVideo` per video id."""

    path = Path(path)
    payload = read_json(path)
    if not isinstance(payload, dict) or "images" not in payload:
        raise CocoParseError(
            f"{path} does not look like a COCO file (missing 'images')."
        )

    default_video_id = path.stem.replace(".coco", "")
    category_names: dict[int, str] = {
        int(cat["id"]): str(cat.get("name", cat["id"]))
        for cat in payload.get("categories", [])
    }

    images_sorted = sorted(
        payload["images"], key=lambda im: str(im.get("file_name", im.get("id", "")))
    )
    image_meta: dict[int, dict[str, Any]] = {}
    for order, image in enumerate(images_sorted):
        image_id = image["id"]
        frame_idx = resolve_frame_index(
            image, data_config.frame_index_regex, order
        )
        video_id = resolve_video_id(
            image, data_config.video_id_field, default_video_id
        )
        timestamp = image.get("timestamp")
        image_meta[image_id] = {
            "frame_idx": frame_idx,
            "video_id": video_id,
            "width": int(image.get("width", 0)),
            "height": int(image.get("height", 0)),
            "timestamp": float(timestamp) if timestamp is not None else None,
        }

    videos: dict[str, CocoVideo] = {}
    frames_per_video: dict[str, set[int]] = {}

    for annotation in payload.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id not in image_meta:
            continue
        meta = image_meta[image_id]
        identity = resolve_identity(
            annotation, data_config.identity_field_priority, category_names
        )
        if identity is None:
            identity = f"id_{annotation.get('id', 0)}"
        bbox = annotation.get("bbox", [0.0, 0.0, 0.0, 0.0])
        record = MaskRecord(
            frame_idx=meta["frame_idx"],
            identity=identity,
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            score=float(annotation.get("score", 1.0)),
            height=meta["height"],
            width=meta["width"],
            segmentation=annotation.get("segmentation"),
            category_id=annotation.get("category_id"),
        )
        video_id = meta["video_id"]
        if video_id not in videos:
            videos[video_id] = CocoVideo(
                video_id=video_id,
                width=meta["width"],
                height=meta["height"],
                frame_indices=[],
                records=[],
                source_path=str(path),
            )
            frames_per_video[video_id] = set()
        videos[video_id].records.append(record)
        frames_per_video[video_id].add(meta["frame_idx"])

    # Ensure every image frame is represented even if it has no annotations.
    for meta in image_meta.values():
        video_id = meta["video_id"]
        if video_id not in videos:
            videos[video_id] = CocoVideo(
                video_id=video_id,
                width=meta["width"],
                height=meta["height"],
                frame_indices=[],
                records=[],
                source_path=str(path),
            )
            frames_per_video[video_id] = set()
        frames_per_video[video_id].add(meta["frame_idx"])

    # Collect explicit per-frame timestamps (if the COCO provides them).
    for meta in image_meta.values():
        if meta.get("timestamp") is None:
            continue
        video = videos.get(meta["video_id"])
        if video is not None:
            video.frame_timestamps[meta["frame_idx"]] = meta["timestamp"]

    for video_id, video in videos.items():
        video.frame_indices = sorted(frames_per_video[video_id])
        if video.width == 0 or video.height == 0:
            for record in video.records:
                if record.width and record.height:
                    video.width = record.width
                    video.height = record.height
                    break

    return videos
