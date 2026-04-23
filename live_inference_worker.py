"""Latest-frame live segmentation worker for RF-DETR Seg and YOLO Seg models."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from live_detection_types import LiveDetectionResult, PreviewFramePacket
from live_tracking import LiveIdentityTracker, compute_body_center
from torch_runtime import import_torch


@dataclass
class LiveInferenceConfig:
    model_key: str = "rfdetr-seg-medium"
    checkpoint_path: str = ""
    threshold: float = 0.35
    selected_class_ids: list[int] | None = None
    identity_mode: str = "tracker"
    expected_mouse_count: int = 1
    inference_max_width: int = 960
    # Optional YOLO pose model run on each segmentation bbox crop to attach
    # keypoints to detections without breaking identity tracking.
    pose_checkpoint_path: str = ""
    pose_threshold: float = 0.25
    min_pose_keypoints: int = 0
    acceleration_mode: str = "balanced"

    def normalized(self) -> "LiveInferenceConfig":
        acceleration_mode = str(self.acceleration_mode or "balanced").strip().lower().replace("-", "_").replace(" ", "_")
        if acceleration_mode not in {"balanced", "max_gpu", "compatibility"}:
            acceleration_mode = "balanced"
        return LiveInferenceConfig(
            model_key=str(self.model_key or "rfdetr-seg-medium").strip(),
            checkpoint_path=str(self.checkpoint_path or "").strip(),
            threshold=float(self.threshold),
            selected_class_ids=list(self.selected_class_ids or []),
            identity_mode=str(self.identity_mode or "tracker").strip().lower(),
            expected_mouse_count=max(1, int(self.expected_mouse_count or 1)),
            inference_max_width=max(0, int(self.inference_max_width or 0)),
            pose_checkpoint_path=str(self.pose_checkpoint_path or "").strip(),
            pose_threshold=float(self.pose_threshold or 0.25),
            min_pose_keypoints=max(0, int(self.min_pose_keypoints or 0)),
            acceleration_mode=acceleration_mode,
        )

    def signature(self) -> tuple:
        normalized = self.normalized()
        return (
            normalized.model_key,
            normalized.checkpoint_path,
            round(normalized.threshold, 4),
            tuple(normalized.selected_class_ids),
            normalized.identity_mode,
            normalized.expected_mouse_count,
            normalized.inference_max_width,
            normalized.pose_checkpoint_path,
            round(normalized.pose_threshold, 4),
            normalized.min_pose_keypoints,
            normalized.acceleration_mode,
        )


class LiveInferenceWorker(QThread):
    """Run live segmentation on the newest preview frame only."""

    result_ready = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._condition = threading.Condition()
        self._running = True
        self._active = False
        self._latest_packet: Optional[PreviewFramePacket] = None
        self._config = LiveInferenceConfig()
        self._model = None
        self._model_signature: Optional[tuple] = None
        self._pose_model = None
        self._pose_model_signature: Optional[str] = None
        self._tracker = LiveIdentityTracker(expected_mice=1)
        self._rfdetr_direct_predict_enabled = True
        self._pose_batch_predict_enabled = True

    def start_inference(self, config: LiveInferenceConfig) -> None:
        normalized = config.normalized()
        with self._condition:
            self._config = normalized
            self._active = True
            self._tracker.reset(expected_mice=normalized.expected_mouse_count)
            self._latest_packet = None
            self._condition.notify_all()
        if not self.isRunning():
            self.start()
        self.status_changed.emit("Live inference armed")

    def stop_inference(self) -> None:
        with self._condition:
            self._active = False
            self._latest_packet = None
            self._condition.notify_all()
        self.status_changed.emit("Live inference stopped")

    def submit_preview(self, packet: object) -> None:
        if not isinstance(packet, PreviewFramePacket):
            return
        with self._condition:
            if not self._active or not self._running:
                return
            self._latest_packet = packet
            self._condition.notify_all()

    def shutdown(self) -> None:
        with self._condition:
            self._running = False
            self._active = False
            self._latest_packet = None
            self._condition.notify_all()
        self.wait(5000)
        self._release_accelerator_memory(self._model)
        self._model = None
        self._release_accelerator_memory(self._pose_model)
        self._pose_model = None
        self._pose_model_signature = None

    def run(self) -> None:
        while True:
            with self._condition:
                while self._running and (not self._active or self._latest_packet is None):
                    self._condition.wait(timeout=0.2)
                if not self._running:
                    break
                packet = self._latest_packet
                self._latest_packet = None
                config = self._config.normalized()

            try:
                signature = config.signature()
                if signature != self._model_signature or self._model is None:
                    self.status_changed.emit(f"Loading {config.model_key} model")
                    self._release_accelerator_memory(self._model)
                    self._model = self._load_model(
                        config.model_key,
                        config.checkpoint_path,
                        acceleration_mode=config.acceleration_mode,
                    )
                    self._model_signature = signature
                    self._rfdetr_direct_predict_enabled = True
                    self._tracker.reset(expected_mice=config.expected_mouse_count)
                    self.status_changed.emit(
                        f"Model ready: {self._describe_loaded_model(self._model, config.model_key, config.acceleration_mode)}"
                    )

                if config.pose_checkpoint_path != (self._pose_model_signature or ""):
                    self._release_accelerator_memory(self._pose_model)
                    self._pose_model = None
                    self._pose_model_signature = None
                    self._pose_batch_predict_enabled = True
                    if config.pose_checkpoint_path:
                        self.status_changed.emit("Loading pose checkpoint")
                        try:
                            self._pose_model = self._load_pose_model(
                                config.pose_checkpoint_path,
                                acceleration_mode=config.acceleration_mode,
                            )
                            self._pose_model_signature = config.pose_checkpoint_path
                            self.status_changed.emit("Pose model ready")
                        except Exception as exc:
                            self._pose_model = None
                            self._pose_model_signature = None
                            self.error_occurred.emit(f"Pose model load failed: {exc}")

                start_perf = time.perf_counter()
                start_wall = time.time()
                frame_rgb = self._ensure_rgb(packet.frame)
                inference_frame, scale_x, scale_y = self._prepare_inference_frame(
                    frame_rgb,
                    config.inference_max_width,
                )
                preprocess_ms = (time.perf_counter() - start_perf) * 1000.0

                predict_start_perf = time.perf_counter()
                detections = self._predict(
                    self._model,
                    config.model_key,
                    inference_frame,
                    config.threshold,
                )
                predict_ms = (time.perf_counter() - predict_start_perf) * 1000.0

                postprocess_start_perf = time.perf_counter()
                normalized = self._normalize_detections(detections)
                if scale_x != 1.0 or scale_y != 1.0:
                    normalized = self._rescale_detections(
                        normalized,
                        frame_shape=frame_rgb.shape[:2],
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )
                records = self._build_detection_records(normalized, config)

                if self._pose_model is not None and records:
                    self._attach_pose_keypoints_in_bboxes(
                        frame_rgb,
                        records,
                        pose_threshold=config.pose_threshold,
                        min_confident_kp=config.min_pose_keypoints,
                    )

                if config.identity_mode == "model_class":
                    tracked = self._tracker.assign_by_model_class(records, config.selected_class_ids or [])
                else:
                    tracked = self._tracker.update(records)
                postprocess_ms = (time.perf_counter() - postprocess_start_perf) * 1000.0

                completed_timestamp_s = time.time()
                inference_ms = (time.perf_counter() - start_perf) * 1000.0
                queue_wait_ms = max(0.0, (start_wall - float(packet.timestamp_s)) * 1000.0)
                end_to_end_ms = max(0.0, (completed_timestamp_s - float(packet.timestamp_s)) * 1000.0)
                inference_height, inference_width = inference_frame.shape[:2]
                self.result_ready.emit(
                    LiveDetectionResult(
                        frame_index=packet.frame_index,
                        timestamp_s=packet.timestamp_s,
                        width=packet.width,
                        height=packet.height,
                        inference_ms=float(inference_ms),
                        tracked_mice=tracked,
                        model_key=config.model_key,
                        status="ok",
                        predict_ms=float(predict_ms),
                        preprocess_ms=float(preprocess_ms),
                        postprocess_ms=float(postprocess_ms),
                        queue_wait_ms=float(queue_wait_ms),
                        end_to_end_ms=float(end_to_end_ms),
                        completed_timestamp_s=float(completed_timestamp_s),
                        inference_width=int(inference_width),
                        inference_height=int(inference_height),
                    )
                )
            except Exception as exc:
                self.error_occurred.emit(f"Live inference error: {str(exc)}")

    def _ensure_rgb(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return np.asarray(frame)

    def _prepare_inference_frame(
        self,
        frame_rgb: np.ndarray,
        max_width: int,
    ) -> tuple[np.ndarray, float, float]:
        target_width = max(0, int(max_width or 0))
        height, width = frame_rgb.shape[:2]
        if target_width <= 0 or width <= target_width:
            return frame_rgb, 1.0, 1.0

        scale = target_width / float(width)
        target_height = max(1, int(round(height * scale)))
        resized = cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return resized, width / float(target_width), height / float(target_height)

    def _load_model(self, model_key: str, checkpoint: str, *, acceleration_mode: str = "balanced"):
        torch = import_torch()
        self._configure_torch_acceleration(torch, acceleration_mode)

        if str(model_key).startswith("rfdetr"):
            import rfdetr as rfdetr_module

            class_name_map = {
                "rfdetr-seg-nano": "RFDETRSegNano",
                "rfdetr-seg-small": "RFDETRSegSmall",
                "rfdetr-seg-medium": "RFDETRSegMedium",
                "rfdetr-seg-large": "RFDETRSegLarge",
            }
            default_class_name = class_name_map["rfdetr-seg-medium"]
            class_name = class_name_map.get(model_key, default_class_name)
            model_cls = getattr(rfdetr_module, class_name, None)
            if model_cls is None:
                raise RuntimeError(
                    f"The installed rfdetr package does not expose {class_name}. "
                    "Upgrade rfdetr to use this live inference model."
                )
            if not checkpoint:
                model = model_cls()
                if hasattr(model, "eval"):
                    model.eval()
                model = self._align_rfdetr_postprocess_num_select(model)
                return self._optimize_loaded_model(model, model_key, acceleration_mode=acceleration_mode)
            for key in ("pretrain_weights", "checkpoint_path", "weights"):
                try:
                    model = model_cls(**{key: checkpoint})
                    if hasattr(model, "eval"):
                        model.eval()
                    model = self._align_rfdetr_postprocess_num_select(model)
                    return self._optimize_loaded_model(model, model_key, acceleration_mode=acceleration_mode)
                except TypeError:
                    continue
            model = model_cls()
            if hasattr(model, "eval"):
                model.eval()
            model = self._align_rfdetr_postprocess_num_select(model)
            return self._optimize_loaded_model(model, model_key, acceleration_mode=acceleration_mode)

        from ultralytics import YOLO

        weight_path = checkpoint or "yolo11n-seg.pt"
        model = YOLO(weight_path)
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.eval()
        return model

    def _load_pose_model(self, checkpoint: str, *, acceleration_mode: str = "balanced"):
        """Load a YOLO pose checkpoint for paired keypoint inference."""
        torch = import_torch()
        self._configure_torch_acceleration(torch, acceleration_mode)
        from ultralytics import YOLO

        path = str(checkpoint or "").strip()
        if not path:
            raise ValueError("Pose checkpoint path is empty")
        model = YOLO(path)
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.eval()
        return model

    def _attach_pose_keypoints_in_bboxes(
        self,
        frame_rgb: np.ndarray,
        records: list[dict],
        *,
        pose_threshold: float,
        min_confident_kp: int,
    ) -> None:
        """Run the YOLO pose model on each detection's bbox crop, in place.

        Each record gains ``keypoints`` (Kx2, image-space pixels) and
        ``keypoint_scores`` (K,) when the pose model produces a detection
        inside the crop. Records that fail the ``min_confident_kp`` gate keep
        their seg-only state untouched (the seg detection is still emitted —
        we never drop identity for missing keypoints).
        """
        if self._pose_model is None or not records:
            return

        torch = import_torch(required=False)

        height, width = frame_rgb.shape[:2]
        crop_jobs: list[dict] = []
        for record in records:
            try:
                bbox = record.get("bbox")
                if bbox is None:
                    continue
                x1, y1, x2, y2 = (float(value) for value in bbox)
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                pad_x = max(12.0, bw * 0.12)
                pad_y = max(12.0, bh * 0.12)
                mask = record.get("mask")
                if mask is not None and np.size(mask):
                    mask_bool = np.asarray(mask, dtype=bool)
                    if mask_bool.shape[:2] == frame_rgb.shape[:2] and bool(mask_bool.any()):
                        ys, xs = np.nonzero(mask_bool)
                        x1 = min(x1, float(np.min(xs)))
                        y1 = min(y1, float(np.min(ys)))
                        x2 = max(x2, float(np.max(xs) + 1))
                        y2 = max(y2, float(np.max(ys) + 1))
                        bw = max(1.0, x2 - x1)
                        bh = max(1.0, y2 - y1)
                        pad_x = max(pad_x, bw * 0.08)
                        pad_y = max(pad_y, bh * 0.08)
                cx1 = max(0, int(round(x1 - pad_x)))
                cy1 = max(0, int(round(y1 - pad_y)))
                cx2 = min(width, int(round(x2 + pad_x)))
                cy2 = min(height, int(round(y2 + pad_y)))
                if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                    continue
                crop_jobs.append(
                    {
                        "record": record,
                        "crop": frame_rgb[cy1:cy2, cx1:cx2],
                        "crop_origin": (float(cx1), float(cy1)),
                        "target_bbox_crop": np.asarray(
                            [x1 - float(cx1), y1 - float(cy1), x2 - float(cx1), y2 - float(cy1)],
                            dtype=float,
                        ),
                        "target_center_crop": np.asarray(
                            [
                                float(record.get("center", (0.0, 0.0))[0]) - float(cx1),
                                float(record.get("center", (0.0, 0.0))[1]) - float(cy1),
                            ],
                            dtype=float,
                        ),
                    }
                )
            except Exception:
                continue

        if not crop_jobs:
            return

        pose_results = self._predict_pose_batch(
            [job["crop"] for job in crop_jobs],
            pose_threshold=pose_threshold,
            torch_module=torch,
        )
        if not pose_results:
            return

        for job, pose_result in zip(crop_jobs, pose_results):
            try:
                if pose_result is None:
                    continue
                kp_obj = getattr(pose_result, "keypoints", None)
                if kp_obj is None:
                    continue

                kp_xy = self._to_numpy(getattr(kp_obj, "xy", None), dtype=float, shape=None)
                kp_conf = self._to_numpy(getattr(kp_obj, "conf", None), dtype=float, shape=None)
                if kp_xy is None or kp_xy.size == 0 or kp_xy.ndim < 2:
                    continue

                # Pick the highest-confidence pose detection inside the crop.
                pose_boxes = getattr(pose_result, "boxes", None)
                pose_scores = self._to_numpy(
                    getattr(pose_boxes, "conf", None) if pose_boxes is not None else None,
                    dtype=float,
                    shape=(-1,),
                )
                pose_boxes_xyxy = self._to_numpy(
                    getattr(pose_boxes, "xyxy", None) if pose_boxes is not None else None,
                    dtype=float,
                    shape=(-1, 4),
                )
                if kp_xy.ndim == 2:
                    selected_xy = kp_xy
                    selected_conf = kp_conf if (kp_conf is not None and kp_conf.ndim == 1) else None
                else:
                    best_index = self._select_best_pose_candidate(
                        kp_xy,
                        pose_scores,
                        pose_boxes_xyxy,
                        job["target_bbox_crop"],
                        job["target_center_crop"],
                    )
                    if best_index < 0 or best_index >= len(kp_xy):
                        continue
                    selected_xy = kp_xy[best_index]
                    if kp_conf is not None and kp_conf.ndim == 2 and best_index < len(kp_conf):
                        selected_conf = kp_conf[best_index]
                    else:
                        selected_conf = None

                if selected_xy.shape[-1] < 2:
                    continue
                keypoints_image = selected_xy.astype(float, copy=True)
                origin_x, origin_y = job["crop_origin"]
                keypoints_image[:, 0] = keypoints_image[:, 0] + float(origin_x)
                keypoints_image[:, 1] = keypoints_image[:, 1] + float(origin_y)
                if selected_conf is not None:
                    selected_conf = selected_conf.astype(float, copy=False).reshape(-1)

                if min_confident_kp > 0:
                    if selected_conf is None:
                        confident_count = int(np.sum(np.all(np.isfinite(keypoints_image), axis=1)))
                    else:
                        confident_count = int(
                            np.sum(np.asarray(selected_conf) >= float(pose_threshold))
                        )
                    if confident_count < int(min_confident_kp):
                        continue

                job["record"]["keypoints"] = keypoints_image
                job["record"]["keypoint_scores"] = selected_conf
            except Exception:
                # Per-detection pose failures must not abort live inference.
                continue

    def _predict_pose_batch(
        self,
        crops: list[np.ndarray],
        *,
        pose_threshold: float,
        torch_module,
    ) -> list:
        """Run YOLO pose on a batch of crops, falling back to per-crop mode if needed."""
        if self._pose_model is None or not crops:
            return []

        max_dim = max(max(int(crop.shape[0]), int(crop.shape[1])) for crop in crops)
        pose_imgsz = int(min(640, max(256, int(np.ceil(max_dim / 32.0) * 32))))

        def _run_predict(input_value):
            return self._pose_model.predict(
                input_value,
                conf=float(pose_threshold),
                imgsz=pose_imgsz,
                verbose=False,
            )

        if self._pose_batch_predict_enabled:
            try:
                if torch_module is not None:
                    with torch_module.inference_mode():
                        results = _run_predict(crops)
                else:
                    results = _run_predict(crops)
                if isinstance(results, (list, tuple)) and len(results) == len(crops):
                    return list(results)
            except Exception as exc:
                self._pose_batch_predict_enabled = False
                self.status_changed.emit(f"Pose batch path disabled: {exc}")

        results: list = []
        for crop in crops:
            try:
                if torch_module is not None:
                    with torch_module.inference_mode():
                        per_crop = _run_predict(crop)
                else:
                    per_crop = _run_predict(crop)
                results.append(per_crop[0] if per_crop else None)
            except Exception:
                results.append(None)
        return results

    def _select_best_pose_candidate(
        self,
        kp_xy: np.ndarray,
        pose_scores: np.ndarray,
        pose_boxes_xyxy: np.ndarray,
        target_bbox_crop: np.ndarray,
        target_center_crop: np.ndarray,
    ) -> int:
        """Choose the pose candidate that best matches the segmented animal geometry."""
        if kp_xy.ndim < 3 or len(kp_xy) == 0:
            return 0

        target_box = np.asarray(target_bbox_crop, dtype=float).reshape(4)
        target_center = np.asarray(target_center_crop, dtype=float).reshape(2)
        target_diag = max(
            1.0,
            float(np.hypot(max(1.0, target_box[2] - target_box[0]), max(1.0, target_box[3] - target_box[1]))),
        )
        best_index = -1
        best_score = -float("inf")

        for index, candidate_xy in enumerate(kp_xy):
            candidate_xy = np.asarray(candidate_xy, dtype=float).reshape(-1, candidate_xy.shape[-1])
            finite_mask = np.all(np.isfinite(candidate_xy[:, :2]), axis=1)
            candidate_points = candidate_xy[finite_mask, :2]
            if candidate_points.size == 0:
                continue

            if index < len(pose_boxes_xyxy):
                candidate_box = np.asarray(pose_boxes_xyxy[index], dtype=float).reshape(4)
            else:
                candidate_box = np.asarray(
                    [
                        float(np.min(candidate_points[:, 0])),
                        float(np.min(candidate_points[:, 1])),
                        float(np.max(candidate_points[:, 0])),
                        float(np.max(candidate_points[:, 1])),
                    ],
                    dtype=float,
                )

            candidate_center = np.asarray(
                [
                    float((candidate_box[0] + candidate_box[2]) * 0.5),
                    float((candidate_box[1] + candidate_box[3]) * 0.5),
                ],
                dtype=float,
            )
            center_distance = float(np.linalg.norm(candidate_center - target_center)) / target_diag
            iou = self._bbox_iou(candidate_box, target_box)

            expanded_target = np.asarray(
                [
                    target_box[0] - 0.1 * (target_box[2] - target_box[0]),
                    target_box[1] - 0.1 * (target_box[3] - target_box[1]),
                    target_box[2] + 0.1 * (target_box[2] - target_box[0]),
                    target_box[3] + 0.1 * (target_box[3] - target_box[1]),
                ],
                dtype=float,
            )
            inside_fraction = float(
                np.mean(
                    (candidate_points[:, 0] >= expanded_target[0])
                    & (candidate_points[:, 0] <= expanded_target[2])
                    & (candidate_points[:, 1] >= expanded_target[1])
                    & (candidate_points[:, 1] <= expanded_target[3])
                )
            )
            pose_conf = float(pose_scores[index]) if index < len(pose_scores) else 0.0
            score = (2.5 * iou) + (1.5 * inside_fraction) + (0.35 * pose_conf) - (1.25 * center_distance)
            if score > best_score:
                best_score = score
                best_index = index

        return best_index if best_index >= 0 else 0

    def _bbox_iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU for two xyxy boxes."""
        a = np.asarray(box_a, dtype=float).reshape(4)
        b = np.asarray(box_b, dtype=float).reshape(4)
        inter_x1 = max(float(a[0]), float(b[0]))
        inter_y1 = max(float(a[1]), float(b[1]))
        inter_x2 = min(float(a[2]), float(b[2]))
        inter_y2 = min(float(a[3]), float(b[3]))
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
        area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        union_area = area_a + area_b - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def _predict(self, model, model_key: str, frame_rgb: np.ndarray, threshold: float):
        torch = import_torch(required=False)

        def _run_prediction():
            if str(model_key).startswith("rfdetr"):
                if torch is not None and self._rfdetr_direct_predict_enabled:
                    try:
                        return self._predict_rfdetr_direct(model, frame_rgb, threshold, torch)
                    except Exception as exc:
                        self._rfdetr_direct_predict_enabled = False
                        self.status_changed.emit(f"RF-DETR direct path disabled: {exc}")
                return model.predict(frame_rgb, threshold=float(threshold))

            results = model.predict(frame_rgb, conf=float(threshold), verbose=False)
            return results[0] if results else None

        if torch is not None:
            with torch.inference_mode():
                return _run_prediction()
        return _run_prediction()

    def _predict_rfdetr_direct(self, model, frame_rgb: np.ndarray, threshold: float, torch):
        model_context = getattr(model, "model", None)
        if model_context is None:
            raise RuntimeError("RF-DETR model context is unavailable")

        device = getattr(model_context, "device", None)
        if device is None:
            raise RuntimeError("RF-DETR device is unavailable")

        inference_module = getattr(model_context, "inference_model", None)
        use_optimized = bool(getattr(model, "_is_optimized_for_inference", False) and inference_module is not None)
        model_module = getattr(model_context, "model", None)
        if not use_optimized and model_module is None:
            raise RuntimeError("RF-DETR base model is unavailable")
        input_dtype = getattr(model, "_optimized_dtype", None) if use_optimized else None
        if not isinstance(input_dtype, torch.dtype):
            input_dtype = torch.float32

        target_resolution = int(
            getattr(model, "_optimized_resolution", 0) if use_optimized else getattr(model_context, "resolution", 0)
        )
        if target_resolution <= 0:
            target_resolution = int(getattr(model_context, "resolution", 0) or 0)
        if target_resolution <= 0:
            raise RuntimeError("RF-DETR inference resolution is unavailable")

        source_frame = np.ascontiguousarray(frame_rgb)
        if source_frame.ndim != 3 or source_frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame with shape HxWx3, got {source_frame.shape!r}")
        source_height, source_width = source_frame.shape[:2]

        batch_tensor = torch.from_numpy(source_frame).permute(2, 0, 1).unsqueeze(0).contiguous()
        batch_tensor = batch_tensor.to(device=device)
        batch_tensor = batch_tensor.to(dtype=input_dtype).div_(255.0)
        if batch_tensor.shape[-2] != target_resolution or batch_tensor.shape[-1] != target_resolution:
            batch_tensor = torch.nn.functional.interpolate(
                batch_tensor,
                size=(target_resolution, target_resolution),
                mode="bilinear",
                align_corners=False,
            )

        mean_tensor = torch.tensor(getattr(model, "means", [0.485, 0.456, 0.406]), device=device, dtype=input_dtype)
        std_tensor = torch.tensor(getattr(model, "stds", [0.229, 0.224, 0.225]), device=device, dtype=input_dtype)
        batch_tensor = (batch_tensor - mean_tensor.view(1, 3, 1, 1)) / std_tensor.view(1, 3, 1, 1)

        if use_optimized:
            predictions = inference_module(batch_tensor.to(dtype=input_dtype))
        else:
            predictions = model_module(batch_tensor)
        if isinstance(predictions, tuple):
            normalized_predictions = {
                "pred_logits": predictions[1],
                "pred_boxes": predictions[0],
            }
            if len(predictions) >= 3:
                normalized_predictions["pred_masks"] = predictions[2]
            predictions = normalized_predictions

        postprocess = getattr(model_context, "postprocess", None)
        if postprocess is None:
            raise RuntimeError("RF-DETR postprocess module is unavailable")

        target_sizes = torch.tensor([[source_height, source_width]], device=device, dtype=torch.int64)
        results = postprocess(predictions, target_sizes=target_sizes)
        if not results:
            return self._empty_detections()

        result = results[0]
        scores = result.get("scores")
        labels = result.get("labels")
        boxes = result.get("boxes")
        if scores is None or labels is None or boxes is None:
            return self._empty_detections()

        keep = scores > float(threshold)
        if int(torch.count_nonzero(keep).item()) <= 0:
            return self._empty_detections()

        output = {
            "xyxy": boxes[keep].float().cpu().numpy(),
            "confidence": scores[keep].float().cpu().numpy(),
            "class_id": labels[keep].to(dtype=torch.int64).cpu().numpy(),
        }
        masks = result.get("masks")
        if masks is not None:
            output["mask"] = masks[keep].squeeze(1).to(dtype=torch.bool).cpu().numpy()
        return output

    def _build_detection_records(self, detections: dict[str, np.ndarray], config: LiveInferenceConfig) -> list[dict]:
        boxes = np.asarray(detections.get("xyxy", np.empty((0, 4), dtype=float)), dtype=float).reshape(-1, 4)
        confidence = np.asarray(detections.get("confidence", np.empty((0,), dtype=float)), dtype=float).reshape(-1)
        class_id = np.asarray(detections.get("class_id", np.empty((0,), dtype=int)), dtype=int).reshape(-1)
        masks = self._to_masks(detections.get("mask"))
        selected_classes = set(int(value) for value in (config.selected_class_ids or []))
        records: list[dict] = []
        for index, bbox in enumerate(boxes):
            cls = int(class_id[index]) if index < len(class_id) else 0
            if selected_classes and cls not in selected_classes:
                continue
            mask = None
            if masks is not None and index < len(masks):
                mask = masks[index]
            bbox_tuple = tuple(float(value) for value in bbox)
            center = compute_body_center(mask, bbox_tuple)
            records.append(
                {
                    "bbox": bbox_tuple,
                    "confidence": float(confidence[index]) if index < len(confidence) else 0.0,
                    "class_id": cls,
                    "mask": mask,
                    "center": center,
                }
            )
        return records

    def _normalize_detections(self, detections) -> dict[str, np.ndarray]:
        if detections is None:
            return self._empty_detections()

        if isinstance(detections, dict) and "xyxy" in detections:
            xyxy = self._to_numpy(detections.get("xyxy"), dtype=float, shape=(-1, 4))
            confidence = self._to_numpy(detections.get("confidence"), dtype=float, shape=(-1,))
            class_id = self._to_numpy(detections.get("class_id"), dtype=int, shape=(-1,))
            masks = self._to_masks(detections.get("mask", detections.get("masks")))
            return self._coerce_lengths(xyxy, confidence, class_id, masks)

        if hasattr(detections, "boxes"):
            boxes = detections.boxes
            xyxy = self._to_numpy(getattr(boxes, "xyxy", None), dtype=float, shape=(-1, 4))
            confidence = self._to_numpy(getattr(boxes, "conf", None), dtype=float, shape=(-1,))
            class_id = self._to_numpy(getattr(boxes, "cls", None), dtype=int, shape=(-1,))
            masks_obj = getattr(detections, "masks", None)
            masks = None
            if masks_obj is not None:
                masks = self._to_masks(getattr(masks_obj, "data", masks_obj))
            return self._coerce_lengths(xyxy, confidence, class_id, masks)

        xyxy = self._to_numpy(getattr(detections, "xyxy", None), dtype=float, shape=(-1, 4))
        confidence = self._to_numpy(
            getattr(detections, "confidence", getattr(detections, "conf", None)),
            dtype=float,
            shape=(-1,),
        )
        class_id = self._to_numpy(getattr(detections, "class_id", None), dtype=int, shape=(-1,))
        masks = self._to_masks(getattr(detections, "mask", getattr(detections, "masks", None)))
        return self._coerce_lengths(xyxy, confidence, class_id, masks)

    def _coerce_lengths(
        self,
        xyxy: np.ndarray,
        confidence: np.ndarray,
        class_id: np.ndarray,
        masks: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        count = len(xyxy)
        if count == 0:
            return self._empty_detections()
        if len(confidence) != count:
            confidence = np.ones(count, dtype=float)
        if len(class_id) != count:
            class_id = np.zeros(count, dtype=int)
        result = {
            "xyxy": xyxy.astype(float, copy=False),
            "confidence": confidence.astype(float, copy=False),
            "class_id": class_id.astype(int, copy=False),
        }
        if masks is not None:
            masks = np.asarray(masks)
            if masks.ndim == 2:
                masks = masks[None, ...]
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0, :, :]
            if masks.ndim == 3 and len(masks) == count:
                result["mask"] = masks.astype(bool, copy=False)
        return result

    def _empty_detections(self) -> dict[str, np.ndarray]:
        return {
            "xyxy": np.empty((0, 4), dtype=float),
            "confidence": np.empty((0,), dtype=float),
            "class_id": np.empty((0,), dtype=int),
            "mask": np.empty((0, 0, 0), dtype=bool),
        }

    def _to_numpy(self, value, dtype=float, shape=None) -> np.ndarray:
        if value is None:
            arr = np.empty((0,), dtype=dtype)
        elif hasattr(value, "detach"):
            arr = value.detach().cpu().numpy()
        elif hasattr(value, "cpu") and hasattr(value, "numpy"):
            arr = value.cpu().numpy()
        else:
            arr = np.asarray(value)
        arr = np.asarray(arr, dtype=dtype)
        if shape is None:
            return arr
        if arr.size == 0:
            if shape == (-1, 4):
                return np.empty((0, 4), dtype=dtype)
            if shape == (-1,):
                return np.empty((0,), dtype=dtype)
        if shape == (-1, 4):
            return arr.reshape(-1, 4)
        if shape == (-1,):
            return arr.reshape(-1)
        return arr

    def _to_masks(self, value) -> Optional[np.ndarray]:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.cpu().numpy()
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, :, :]
        if arr.ndim != 3:
            return None
        return arr.astype(bool, copy=False)

    def _align_rfdetr_postprocess_num_select(self, model):
        """Work around rfdetr building postprocess with an overly large num_select."""
        model_context = getattr(model, "model", None)
        postprocess = getattr(model_context, "postprocess", None)
        if postprocess is None:
            return model

        model_config = getattr(model, "model_config", None)
        desired_num_select = getattr(model_config, "num_select", None)
        query_count = getattr(getattr(model_context, "args", None), "num_queries", None)

        if not isinstance(desired_num_select, int) or desired_num_select <= 0:
            desired_num_select = query_count
        if not isinstance(desired_num_select, int) or desired_num_select <= 0:
            return model
        if isinstance(query_count, int) and query_count > 0:
            desired_num_select = min(desired_num_select, query_count)

        setattr(postprocess, "num_select", desired_num_select)
        args = getattr(model_context, "args", None)
        if args is not None:
            try:
                args.num_select = desired_num_select
            except Exception:
                pass
        return model

    def _optimize_loaded_model(self, model, model_key: str, *, acceleration_mode: str = "balanced"):
        if not str(model_key).startswith("rfdetr"):
            return model
        if not hasattr(model, "optimize_for_inference"):
            return model

        torch = import_torch(required=False)
        if torch is None:
            return model

        self._configure_torch_acceleration(torch, acceleration_mode)
        optimize_kwargs = {"batch_size": 1}
        try:
            model_device = str(getattr(getattr(model, "model", None), "device", "") or "").lower()
        except Exception:
            model_device = ""
        use_cuda = torch.cuda.is_available() and "cuda" in model_device
        compatibility_mode = str(acceleration_mode or "balanced") == "compatibility"
        optimize_kwargs["dtype"] = torch.float16 if (use_cuda and not compatibility_mode) else torch.float32
        optimize_kwargs["compile"] = bool(use_cuda and not compatibility_mode)

        try:
            self.status_changed.emit("Optimizing RF-DETR for live inference")
            model.optimize_for_inference(**optimize_kwargs)
            self.status_changed.emit(
                f"RF-DETR optimized ({'fp16' if optimize_kwargs['dtype'] == torch.float16 else 'fp32'})"
            )
        except Exception as exc:
            self.status_changed.emit(f"RF-DETR optimization skipped: {exc}")
        return model

    def _configure_torch_acceleration(self, torch, acceleration_mode: str) -> None:
        mode = str(acceleration_mode or "balanced").strip().lower().replace("-", "_").replace(" ", "_")
        max_gpu = mode == "max_gpu"
        compatibility = mode == "compatibility"

        benchmark_enabled = bool(max_gpu)
        tf32_enabled = bool(max_gpu and torch.cuda.is_available())
        matmul_precision = "highest" if compatibility else ("high" if max_gpu else "highest")

        try:
            torch.backends.cudnn.benchmark = benchmark_enabled
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = tf32_enabled
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception:
            pass

    def _describe_loaded_model(self, model, model_key: str, acceleration_mode: str = "balanced") -> str:
        try:
            model_config = getattr(model, "model_config", None)
            resolution = int(getattr(model_config, "resolution", 0) or 0)
            num_queries = int(getattr(model_config, "num_queries", 0) or 0)
            details: list[str] = []
            if resolution > 0:
                details.append(f"{resolution}px")
            if num_queries > 0:
                details.append(f"{num_queries} queries")
            mode = str(acceleration_mode or "balanced").strip().lower().replace("-", "_").replace(" ", "_")
            if mode == "max_gpu":
                details.append("max GPU")
            elif mode == "compatibility":
                details.append("compatibility")
            if details:
                return f"{model_key} ({', '.join(details)})"
        except Exception:
            pass
        return str(model_key or "")

    def _rescale_detections(
        self,
        detections: dict[str, np.ndarray],
        *,
        frame_shape: tuple[int, int],
        scale_x: float,
        scale_y: float,
    ) -> dict[str, np.ndarray]:
        if not detections:
            return self._empty_detections()

        height, width = int(frame_shape[0]), int(frame_shape[1])
        result = dict(detections)
        boxes = np.asarray(detections.get("xyxy", np.empty((0, 4), dtype=float)), dtype=float).reshape(-1, 4).copy()
        if len(boxes):
            boxes[:, [0, 2]] *= float(scale_x)
            boxes[:, [1, 3]] *= float(scale_y)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, max(0.0, width - 1.0))
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, max(0.0, height - 1.0))
        result["xyxy"] = boxes

        masks = detections.get("mask")
        if masks is not None:
            resized_masks: list[np.ndarray] = []
            for mask in np.asarray(masks, dtype=bool):
                resized = cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                resized_masks.append(resized.astype(bool))
            result["mask"] = (
                np.asarray(resized_masks, dtype=bool)
                if resized_masks
                else np.empty((0, height, width), dtype=bool)
            )
        return result

    def _release_accelerator_memory(self, model) -> None:
        del model
        torch = import_torch(required=False)
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
