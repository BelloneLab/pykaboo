"""Render manual labels onto the video for training-data quality control.

Burns the human-annotated behaviors (per identity, color-coded) onto each frame so
you can SEE whether the labels actually match what the animals are doing. This is the
direct tool for the label-quality question (e.g. is 819's freezing annotation real?).

Time alignment is the whole point and it reuses the EXACT same path as training:
labels are mapped from the CSV ``time`` column onto the COCO frame grid via the COCO
per-frame timestamps (``align_wide_labels_per_identity``), so a label shown on video
frame N is the label the model is trained on for frame N. Mask contours are drawn per
identity so the annotated identity can be verified too.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .coco_masks import load_coco_videos
from .config import DataConfig
from .social_pipeline import align_wide_labels_per_identity, build_label_map_from_csvs

# identity -> (display name, BGR contour color). "1" = BL6 (blue), "2" = CD1 (red).
IDENTITY_STYLE = {
    "1": ("BL6", (235, 180, 40)),
    "2": ("CD1", (60, 60, 235)),
}


def _behavior_color(name: str) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR color per behavior name."""
    h = (hash(name) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def render_labeled_video(
    video_path: str | Path,
    coco_json: str | Path,
    label_csv: str | Path,
    output_path: str | Path,
    *,
    data_config: DataConfig | None = None,
    identity_map: dict[str, str] | None = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    stride: int = 1,
    draw_masks: bool = True,
    progress: Callable[[float, str], None] | None = None,
) -> dict:
    """Write ``output_path`` = the video with manual labels + mask contours burned in.

    Returns a small QC report dict (frames written, per-behavior frame counts, the
    fraction of frames that carried any label, and the alignment offset used).
    """

    video_path, coco_json, label_csv = str(video_path), str(coco_json), str(label_csv)
    output_path = str(output_path)
    data_config = data_config or DataConfig()

    videos = load_coco_videos(coco_json, data_config)
    video = max(videos.values(), key=lambda v: v.num_frames)
    frame_indices = list(video.frame_indices)
    identities = sorted({r.identity for r in video.records})
    label_map = build_label_map_from_csvs([label_csv])

    gt = align_wide_labels_per_identity(
        label_csv, video.video_id, frame_indices, identities, label_map,
        identity_map=identity_map, frame_rate=video.fps,
        frame_times=np.asarray(video.timestamps),
    )  # keyed by (video_id, subject_id, partner_id)
    gt_by_identity = {subj: arr for (_vid, subj, _partner), arr in gt.items()}
    identities = [i for i in identities if i in gt_by_identity] or list(gt_by_identity)
    bg = label_map.background_id
    names = label_map.names
    # frame_idx -> column in the aligned label arrays
    col_of = {fi: j for j, fi in enumerate(frame_indices)}
    records_by_frame = video.records_by_frame()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or video.fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base = frame_indices[0] if frame_indices else 0  # 0- vs 1-based offset

    out_fps = max(fps / max(stride, 1), 1.0)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height)
    )

    start_f = int(start_sec * fps)
    end_f = int(end_sec * fps) if end_sec is not None else n_video
    if start_f > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    per_behavior = {n: 0 for n in names if names.index(n) != bg}
    written = 0
    labeled_frames = 0
    c = start_f
    while c < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (c - start_f) % stride == 0:
            fi = base + c  # cv2 frame c (0-based) -> COCO frame_idx (handles 0/1-based)
            col = col_of.get(fi)
            _draw_overlay(frame, fi, c / fps, gt_by_identity, col, identities, names, bg,
                          records_by_frame.get(fi, []) if draw_masks else [],
                          per_behavior, width, height)
            if col is not None and any(
                col < gt_by_identity[i].shape[1]
                and gt_by_identity[i][:, col].sum() > gt_by_identity[i][bg, col]
                for i in identities
            ):
                labeled_frames += 1
            writer.write(frame)
            written += 1
            if progress and written % 200 == 0:
                progress((c - start_f) / max(end_f - start_f, 1), f"frame {c}")
        c += 1

    cap.release()
    writer.release()
    return {
        "output_path": output_path,
        "frames_written": written,
        "fraction_labeled": labeled_frames / max(written, 1),
        "per_behavior_frames": {k: v for k, v in per_behavior.items() if v > 0},
        "identities": identities,
        "fps": out_fps,
    }


def _draw_overlay(frame, frame_idx, t_sec, gt, col, identities, names, bg,
                  records, per_behavior, width, height) -> None:
    # mask contours per identity
    for rec in records:
        style = IDENTITY_STYLE.get(str(rec.identity), (str(rec.identity), (200, 200, 200)))
        m = rec.decode_mask()
        if m is None:
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, cnts, -1, style[1], 2)

    # top header: frame + time
    cv2.rectangle(frame, (0, 0), (width, 28), (20, 20, 20), -1)
    cv2.putText(frame, f"frame {frame_idx}   t={t_sec:6.2f}s", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    # per-identity active behaviors
    y = 48
    for ident in identities:
        disp, idcol = IDENTITY_STYLE.get(str(ident), (str(ident), (200, 200, 200)))
        active = []
        if col is not None and col < gt[ident].shape[1]:
            for k, nm in enumerate(names):
                if k == bg:
                    continue
                if gt[ident][k, col]:
                    active.append(nm)
                    per_behavior[nm] = per_behavior.get(nm, 0) + 1
        cv2.putText(frame, f"{disp}:", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, idcol, 2, cv2.LINE_AA)
        x = 70
        if not active:
            cv2.putText(frame, "-", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1, cv2.LINE_AA)
        for nm in active:
            (tw, _), _ = cv2.getTextSize(nm, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, nm, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _behavior_color(nm), 2, cv2.LINE_AA)
            x += tw + 16
        y += 26
