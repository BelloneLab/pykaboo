"""Background rendering and writing for sidecar overlay videos."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Optional

import cv2
import numpy as np

from live_detection_logic import occupied_roi_names
from live_detection_types import BehaviorROI, LiveDetectionResult
from live_overlay_quality import (
    draw_behavior_subtitle_bgr,
    draw_pose_skeleton_bgr,
    identity_color_rgb,
)
from live_overlay_utils import clamp_mask_opacity, scale_live_detection_result_to_shape

LIVE_ROI_OCCUPIED_COLOR = (34, 197, 94)


@dataclass(slots=True)
class OverlayVideoFrameTask:
    frame_rgb: np.ndarray
    timestamp_s: float
    overlay_result: Optional[LiveDetectionResult]
    rois: tuple[BehaviorROI, ...]
    show_masks: bool
    show_boxes: bool
    show_keypoints: bool
    mask_opacity: float = 0.18
    repeat_count: int = 1
    show_behavior: bool = False
    # {subject_id: {"probs": {name: float}, "binary": {...}}} carried from the latest
    # behavior decision; drives the per-mouse behavior subtitle chips.
    behavior_per_track: Optional[dict] = None


def render_overlay_video_frame_bgr(task: OverlayVideoFrameTask) -> np.ndarray:
    frame_rgb = np.asarray(task.frame_rgb)
    if frame_rgb.ndim == 2:
        display_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
    else:
        display_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    overlay = display_bgr.copy()
    overlay_result = scale_live_detection_result_to_shape(task.overlay_result, display_bgr.shape)
    mask_opacity = clamp_mask_opacity(task.mask_opacity)
    roi_map = {roi.name: roi for roi in task.rois}
    occupied_names = occupied_roi_names(roi_map, overlay_result) if roi_map else set()

    for roi in task.rois:
        color_rgb = LIVE_ROI_OCCUPIED_COLOR if str(roi.name) in occupied_names else roi.color
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        line_width = 3 if str(roi.name) in occupied_names else 2
        if roi.roi_type == "rectangle" and roi.data:
            x1, y1, x2, y2 = [int(round(value)) for value in roi.data[0]]
            cv2.rectangle(display_bgr, (x1, y1), (x2, y2), color_bgr, line_width)
            cv2.putText(
                display_bgr,
                roi.name,
                (x1 + 6, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_bgr,
                2,
                cv2.LINE_AA,
            )
        elif roi.roi_type == "circle" and roi.data:
            cx, cy, radius = roi.data[0]
            cv2.circle(display_bgr, (int(round(cx)), int(round(cy))), int(round(radius)), color_bgr, line_width)
            cv2.putText(
                display_bgr,
                roi.name,
                (int(cx) + 6, max(20, int(cy) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color_bgr,
                2,
                cv2.LINE_AA,
            )
        elif roi.roi_type == "polygon" and roi.data:
            pts = np.array([(int(round(px)), int(round(py))) for px, py in roi.data], dtype=np.int32)
            if len(pts) >= 3:
                cv2.polylines(display_bgr, [pts], True, color_bgr, line_width, cv2.LINE_AA)
                cv2.fillPoly(overlay, [pts], color_bgr)
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                cv2.putText(
                    display_bgr,
                    roi.name,
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color_bgr,
                    2,
                    cv2.LINE_AA,
                )

    if task.rois:
        cv2.addWeighted(overlay, 0.12, display_bgr, 0.88, 0, display_bgr)

    if overlay_result is not None:
        for mouse in overlay_result.tracked_mice:
            # Skip degenerate detections (NaN/inf bbox) instead of crashing the whole
            # render thread -- a single bad frame must not truncate the overlay video.
            bbox = getattr(mouse, "bbox", None)
            try:
                if bbox is None or not all(np.isfinite(float(v)) for v in bbox):
                    continue
            except (TypeError, ValueError):
                continue
            # Same per-identity colour as the live preview so the recorded overlay
            # matches what the user sees on screen.
            color_rgb = identity_color_rgb(int(mouse.mouse_id))
            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
            mask = getattr(mouse, "mask", None)
            if (
                task.show_masks
                and mask is not None
                and mask.size > 0
            ):
                mask_bool = np.asarray(mask, dtype=bool)
                _blend_mask_roi(display_bgr, mask_bool, color_bgr, mask_opacity, getattr(mouse, "bbox", None))
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            if task.show_boxes:
                cv2.rectangle(display_bgr, (x1, y1), (x2, y2), color_bgr, 2)
            center = getattr(mouse, "center", None)
            if center is not None and all(np.isfinite(float(v)) for v in center):
                cx, cy = int(round(center[0])), int(round(center[1]))
            else:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(display_bgr, (cx, cy), 4, color_bgr, -1)
            label = f"{mouse.label}  C{mouse.class_id}  {mouse.confidence:.2f}"
            label_x = x1 + 4 if task.show_boxes else min(max(8, cx + 8), max(8, display_bgr.shape[1] - 220))
            label_y = max(20, y1 - 8) if task.show_boxes else max(20, cy - 8)
            cv2.putText(
                display_bgr,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_bgr,
                2,
                cv2.LINE_AA,
            )

            if task.show_keypoints and getattr(mouse, "keypoints", None) is not None:
                # Identity-coloured joints + skeleton, identical to the live preview.
                draw_pose_skeleton_bgr(
                    display_bgr,
                    getattr(mouse, "keypoints", None),
                    getattr(mouse, "keypoint_scores", None),
                    color_bgr,
                )

            if task.show_behavior and task.behavior_per_track:
                # Same per-mouse behavior chip as the live preview.
                draw_behavior_subtitle_bgr(
                    display_bgr,
                    str(int(mouse.mouse_id)),
                    [int(round(v)) for v in mouse.bbox],
                    color_bgr,
                    task.behavior_per_track,
                )
    return display_bgr


def _blend_mask_roi(
    display_bgr: np.ndarray,
    mask_bool: np.ndarray,
    color_bgr: tuple[int, int, int],
    mask_opacity: float,
    bbox,
) -> None:
    """Blend one mask only inside its bbox ROI, avoiding full-frame layers."""
    if mask_bool.ndim != 2 or mask_bool.shape[:2] != display_bgr.shape[:2] or not bool(mask_bool.any()):
        return
    h, w = mask_bool.shape[:2]
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    except Exception:
        ys, xs = np.nonzero(mask_bool)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1 = max(0, min(w, x1 - 2))
    x2 = max(0, min(w, x2 + 3))
    y1 = max(0, min(h, y1 - 2))
    y2 = max(0, min(h, y2 + 3))
    if x2 <= x1 or y2 <= y1:
        return
    local_mask = mask_bool[y1:y2, x1:x2]
    if not bool(local_mask.any()):
        return
    roi = display_bgr[y1:y2, x1:x2]
    base = roi[local_mask].astype(np.float32)
    tint = np.asarray(color_bgr, dtype=np.float32).reshape(1, 3)
    roi[local_mask] = np.clip(
        base * (1.0 - float(mask_opacity)) + tint * float(mask_opacity),
        0.0,
        255.0,
    ).astype(np.uint8)


class OverlayVideoRecorder:
    """Render overlay frames in a background thread and write them to disk."""

    def __init__(self, path: str, fps: float, max_pending_frames: int = 4) -> None:
        self.path = str(path or "")
        self.fps = max(1.0, float(fps or 25.0))
        self.max_pending_frames = max(1, int(max_pending_frames))
        self._queue: Queue[OverlayVideoFrameTask] = Queue(maxsize=self.max_pending_frames)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._writer = None
        self._writer_size: Optional[tuple[int, int]] = None
        self.dropped_frames = 0
        self.error_message = ""
        self.frames_written = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="PyKabooOverlayVideo", daemon=True)
        self._thread.start()

    def enqueue(self, task: OverlayVideoFrameTask) -> None:
        if not self._running or not self.path:
            return
        try:
            self._queue.put_nowait(task)
            return
        except Full:
            pass
        try:
            dropped_task = self._queue.get_nowait()
            self.dropped_frames += 1
            task.repeat_count = max(1, int(task.repeat_count or 1)) + max(
                1,
                int(getattr(dropped_task, "repeat_count", 1) or 1),
            )
        except Empty:
            pass
        try:
            self._queue.put_nowait(task)
        except Full:
            self.dropped_frames += 1

    def stop(self, timeout_s: float = 30.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, float(timeout_s)))
            if self._thread.is_alive():
                self.error_message = (
                    self.error_message
                    or f"Overlay video writer did not finish within {float(timeout_s):.1f}s."
                )
                return
            self._thread = None
        self._release_writer()

    def _run(self) -> None:
        try:
            while self._running or not self._queue.empty():
                try:
                    task = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    overlay_bgr = render_overlay_video_frame_bgr(task)
                    if not self._ensure_writer(overlay_bgr):
                        continue
                    overlay_bgr = self._coerce_writer_frame_size(overlay_bgr)
                    repeat_count = max(1, int(getattr(task, "repeat_count", 1) or 1))
                    for _ in range(repeat_count):
                        self._writer.write(overlay_bgr)
                        self.frames_written += 1
                except Exception as exc:
                    # Skip a single bad frame rather than truncating the whole video.
                    self.error_message = f"Overlay video render error: {exc}"
                    self.dropped_frames += 1
                    continue
        finally:
            self._release_writer()

    def _ensure_writer(self, overlay_bgr: np.ndarray) -> bool:
        if self._writer is not None:
            return True
        if not self.path:
            return False
        height, width = overlay_bgr.shape[:2]
        if width <= 0 or height <= 0:
            return False
        writer = cv2.VideoWriter(
            self.path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (int(width), int(height)),
            True,
        )
        if not writer.isOpened():
            self.error_message = f"Could not open overlay video writer for {Path(self.path).name}."
            self._running = False
            return False
        self._writer = writer
        self._writer_size = (int(width), int(height))
        return True

    def _coerce_writer_frame_size(self, overlay_bgr: np.ndarray) -> np.ndarray:
        """Return a contiguous frame matching the already-open writer size."""
        if self._writer_size is None:
            return np.ascontiguousarray(overlay_bgr)
        writer_width, writer_height = self._writer_size
        height, width = overlay_bgr.shape[:2]
        if int(width) != int(writer_width) or int(height) != int(writer_height):
            overlay_bgr = cv2.resize(
                overlay_bgr,
                (int(writer_width), int(writer_height)),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(overlay_bgr)

    def _release_writer(self) -> None:
        writer = self._writer
        self._writer = None
        self._writer_size = None
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
