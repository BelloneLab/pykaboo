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


def render_overlay_video_frame_bgr(task: OverlayVideoFrameTask) -> np.ndarray:
    frame_rgb = np.asarray(task.frame_rgb)
    if frame_rgb.ndim == 2:
        display_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
    else:
        display_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    overlay = display_bgr.copy()
    roi_map = {roi.name: roi for roi in task.rois}
    occupied_names = occupied_roi_names(roi_map, task.overlay_result) if roi_map else set()

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

    if task.overlay_result is not None:
        if task.show_masks:
            blended_layer = np.zeros_like(display_bgr)
            blended_mask = np.zeros(display_bgr.shape[:2], dtype=bool)
        for mouse in task.overlay_result.tracked_mice:
            color_bgr = (
                90 + (mouse.mouse_id * 40) % 140,
                220 - (mouse.mouse_id * 35) % 120,
                120 + (mouse.mouse_id * 55) % 110,
            )
            mask = getattr(mouse, "mask", None)
            if (
                task.show_masks
                and mask is not None
                and mask.size > 0
                and mask.shape[:2] == display_bgr.shape[:2]
            ):
                mask_bool = np.asarray(mask, dtype=bool)
                blended_layer[mask_bool] = color_bgr
                blended_mask |= mask_bool
            x1, y1, x2, y2 = [int(round(value)) for value in mouse.bbox]
            if task.show_boxes:
                cv2.rectangle(display_bgr, (x1, y1), (x2, y2), color_bgr, 2)
            cx, cy = int(round(mouse.center[0])), int(round(mouse.center[1]))
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
                keypoints = np.asarray(mouse.keypoints, dtype=float).reshape(-1, 2)
                scores = getattr(mouse, "keypoint_scores", None)
                score_arr = np.asarray(scores, dtype=float).reshape(-1) if scores is not None else None
                for kp_index, (kx, ky) in enumerate(keypoints):
                    if not (np.isfinite(kx) and np.isfinite(ky)):
                        continue
                    if kx <= 0 and ky <= 0:
                        continue
                    kp_score = (
                        float(score_arr[kp_index])
                        if score_arr is not None and kp_index < len(score_arr)
                        else 1.0
                    )
                    if kp_score < 0.1:
                        continue
                    cv2.circle(
                        display_bgr,
                        (int(round(float(kx))), int(round(float(ky)))),
                        3,
                        (0, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )
        if task.show_masks and blended_mask.any():
            base = display_bgr[blended_mask].astype(np.float32)
            tint = blended_layer[blended_mask].astype(np.float32)
            display_bgr[blended_mask] = np.clip((base * 0.82) + (tint * 0.18), 0.0, 255.0).astype(np.uint8)

    return display_bgr


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
            self._queue.get_nowait()
            self.dropped_frames += 1
        except Empty:
            pass
        try:
            self._queue.put_nowait(task)
        except Full:
            self.dropped_frames += 1

    def stop(self, timeout_s: float = 5.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, float(timeout_s)))
            self._thread = None
        self._release_writer()

    def _run(self) -> None:
        try:
            while self._running or not self._queue.empty():
                try:
                    task = self._queue.get(timeout=0.1)
                except Empty:
                    continue
                overlay_bgr = render_overlay_video_frame_bgr(task)
                if not self._ensure_writer(overlay_bgr):
                    continue
                self._writer.write(overlay_bgr)
                self.frames_written += 1
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

    def _release_writer(self) -> None:
        writer = self._writer
        self._writer = None
        self._writer_size = None
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
