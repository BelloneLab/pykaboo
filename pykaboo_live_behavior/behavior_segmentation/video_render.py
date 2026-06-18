"""Render an annotated MP4 with mask overlays, identity, behavior, and confidence.

Reused by both the inference CLI and the GUI export panel. Mask contours are
drawn per identity with a stable color, the predicted behavior and confidence are
written near each mouse, and the frame number plus timestamp are burned in.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .coco_masks import load_coco_videos
from .config import AppConfig, load_config
from .storage import read_table


def identity_color(identity: str) -> tuple[int, int, int]:
    """Deterministic bright BGR color for an identity string."""

    palette = [
        (255, 84, 84),
        (53, 194, 255),
        (47, 224, 162),
        (255, 195, 77),
        (197, 125, 255),
        (255, 112, 194),
        (124, 231, 255),
        (166, 255, 96),
    ]
    digest = hashlib.md5(str(identity).encode("utf-8")).digest()
    if str(identity).strip().isdigit():
        idx = max(int(str(identity).strip()) - 1, 0) % len(palette)
    else:
        idx = digest[0] % len(palette)
    r, g, b = palette[idx]
    if len(str(identity)) > 2:
        hue = (int.from_bytes(digest[:2], "big") % 360) / 360.0
        r, g, b = (int(value * 255) for value in colorsys.hsv_to_rgb(hue, 0.72, 1.0))
    return int(b), int(g), int(r)


def render_annotated_video(
    video_path: str | Path,
    coco_json: str | Path,
    framewise: pd.DataFrame,
    output_path: str | Path,
    config: AppConfig,
    progress=None,
) -> Path:
    """Write an annotated MP4 from a video, its masks, and framewise predictions."""

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or config.data.frame_rate
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    videos = load_coco_videos(coco_json, config.data)
    video = next(iter(videos.values()))
    by_frame = video.records_by_frame()

    pred_lookup: dict[tuple[int, str], dict] = {}
    for row in framewise.itertuples(index=False):
        pred_lookup[(int(row.frame_idx), str(row.subject_id))] = {
            "behavior": row.predicted_behavior,
            "confidence": float(row.confidence),
        }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    opacity = config.gui.overlay_opacity
    thickness = config.gui.contour_thickness
    font_scale = config.gui.font_scale

    frame_pos = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_idx = video.frame_indices[frame_pos] if frame_pos < video.num_frames else frame_pos
        overlay = frame.copy()
        for record in by_frame.get(frame_idx, []):
            color = identity_color(record.identity)
            mask = record.decode_mask()
            if mask is not None and mask.shape[:2] == frame.shape[:2]:
                colored = np.zeros_like(frame)
                colored[mask] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored, opacity, 0)
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay, contours, -1, color, thickness)
            bx, by, bw, bh = (int(v) for v in record.bbox)
            pred = pred_lookup.get((frame_idx, record.identity))
            label = record.identity
            if config.gui.show_behavior_labels and pred:
                label = f"{record.identity}: {pred['behavior']}"
                if config.gui.show_confidence:
                    label += f" ({pred['confidence'] * 100:.0f}%)"
            cv2.putText(
                overlay,
                label,
                (max(bx, 2), max(by - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
        frame = cv2.addWeighted(frame, 1.0 - opacity, overlay, opacity, 0)
        timestamp = frame_idx / fps if fps else 0.0
        cv2.putText(
            frame,
            f"frame {frame_idx}  t={timestamp:.2f}s",
            (8, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_pos += 1
        if progress and total > 0:
            progress(frame_pos / total, f"Rendering frame {frame_pos}/{total}")

    capture.release()
    writer.release()
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render annotated behavior video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    framewise = read_table(args.predictions)
    out = render_annotated_video(
        args.video, args.coco_json, framewise, args.output, config,
        progress=lambda f, m: print(f"{f * 100:.0f}% {m}", flush=True),
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
