"""Clean mask-video rendering: background-removed, per-animal-colored, interaction-cropped.

Why this exists: a frozen RGB foundation model (VideoPrism) failed badly here
(weighted-F1 0.106, attack 0.151) because raw top-down grayscale cage video is
dominated by nuisance (background, cage, lighting, strain appearance) that does not
transfer across sessions, and a frozen probe cannot adapt. The remedy is to strip the
nuisance and keep only the behaviorally relevant signal: render each animal's mask
into its own channel (identity preserved), zero the background, and crop to the
two-animal interaction region (translation invariant). A small 3D-CNN / attention
encoder on THIS representation captures the fast shape deformation of attacks while
generalizing cross-video, and fuses with the EmbTCN-AT pose/mask-feature stream.

Output clip tensor: ``[T, C, H, W]`` float32 in [0, 1], background = 0.
Channels (default ``per_animal``): ch0 = animal-1 mask, ch1 = animal-2 mask,
ch2 = overlap (animal-1 AND animal-2). This keeps shape, identity, and contact.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .coco_masks import CocoVideo, MaskRecord


def _best_records_by_frame(video: CocoVideo) -> dict:
    """Best-scoring MaskRecord per (frame, identity)."""

    best: dict[tuple[int, str], MaskRecord] = {}
    for frame_idx, records in video.records_by_frame().items():
        for rec in records:
            key = (frame_idx, rec.identity)
            if key not in best or rec.score > best[key].score:
                best[key] = rec
    return best


def _union_bbox(masks: list[np.ndarray | None], width: int, height: int, margin: float):
    ys, xs = [], []
    for m in masks:
        if m is None or m.sum() == 0:
            continue
        r = np.where(m.any(axis=1))[0]
        c = np.where(m.any(axis=0))[0]
        if r.size and c.size:
            ys += [r[0], r[-1]]
            xs += [c[0], c[-1]]
    if not xs:
        return 0, 0, width, height
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    side = int(max(w, h) * (1.0 + 2 * margin))
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    half = side // 2
    return cx - half, cy - half, side, side


def render_mask_clip(
    video: CocoVideo,
    frame_indices: list[int] | None = None,
    identities: list[str] | None = None,
    size: int = 96,
    margin: float = 0.3,
    with_overlap: bool = True,
    smooth_box: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Render a clean mask clip ``[T, C, H, W]`` for the (up to 2) tracked animals.

    The crop follows the union bounding box of both animals (the interaction
    region), optionally smoothed across time to reduce jitter. Background is 0.
    """

    frames = list(frame_indices if frame_indices is not None else video.frame_indices)
    ids = list(identities if identities is not None else video.identities)[:2]
    best = _best_records_by_frame(video)
    W, H = video.width, video.height
    C = 3 if with_overlap else 2
    clip = np.zeros((len(frames), C, size, size), dtype=np.float32)

    # precompute per-frame union bbox, optionally smoothed
    boxes = []
    for f in frames:
        masks = [
            (best[(f, i)].decode_mask() if (f, i) in best else None) for i in ids
        ]
        boxes.append(_union_bbox(masks, W, H, margin))
    boxes = np.asarray(boxes, dtype=np.float32)
    if smooth_box and len(boxes) > 5:
        k = np.ones(5) / 5.0
        for j in range(4):
            boxes[:, j] = np.convolve(boxes[:, j], k, mode="same")

    for ti, f in enumerate(frames):
        x0, y0, side, _ = boxes[ti].astype(int)
        side = max(side, 8)
        decoded = []
        for ci, i in enumerate(ids):
            rec = best.get((f, i))
            m = rec.decode_mask() if rec is not None else None
            if m is None:
                decoded.append(None)
                continue
            crop = _safe_crop(m.astype(np.float32), x0, y0, side, side, H, W)
            decoded.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA))
            clip[ti, ci] = decoded[-1]
        if with_overlap and decoded[0] is not None and len(decoded) > 1 and decoded[1] is not None:
            clip[ti, 2] = np.minimum(decoded[0], decoded[1])
    return clip, frames


def _safe_crop(img: np.ndarray, x0: int, y0: int, w: int, h: int, H: int, W: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=img.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + w, W), min(y0 + h, H)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def mask_overlap_features(video: CocoVideo, identities: list[str] | None = None):
    """Inter-animal mask overlap features: IoU, intersection/min-area, contact, velocity.

    Strong attack/wrestling signal (bodies collide). Returned as a per-frame dict of
    arrays aligned to ``video.frame_indices``.
    """

    ids = list(identities if identities is not None else video.identities)[:2]
    best = _best_records_by_frame(video)
    frames = list(video.frame_indices)
    iou = np.zeros(len(frames), np.float32)
    inter_over_min = np.zeros(len(frames), np.float32)
    inter_area = np.zeros(len(frames), np.float32)
    for ti, f in enumerate(frames):
        if len(ids) < 2:
            break
        a = best.get((f, ids[0]))
        b = best.get((f, ids[1]))
        ma = a.decode_mask() if a is not None else None
        mb = b.decode_mask() if b is not None else None
        if ma is None or mb is None:
            continue
        inter = float(np.logical_and(ma, mb).sum())
        union = float(np.logical_or(ma, mb).sum())
        amin = float(min(ma.sum(), mb.sum())) or 1.0
        iou[ti] = inter / union if union > 0 else 0.0
        inter_over_min[ti] = inter / amin
        inter_area[ti] = inter / (video.width * video.height)
    vel = np.zeros_like(inter_area)
    vel[1:] = np.diff(inter_area)
    return {
        "mask_iou": iou,
        "mask_inter_over_min": inter_over_min,
        "mask_inter_area": inter_area,
        "mask_inter_velocity": vel,
        "mask_contact_flag": (iou > 0.02).astype(np.float32),
    }
