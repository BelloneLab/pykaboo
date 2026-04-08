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

    def normalized(self) -> "LiveInferenceConfig":
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
                    self._model = self._load_model(config.model_key, config.checkpoint_path)
                    self._model_signature = signature
                    self._tracker.reset(expected_mice=config.expected_mouse_count)
                    self.status_changed.emit(f"Model ready: {config.model_key}")

                if config.pose_checkpoint_path != (self._pose_model_signature or ""):
                    self._release_accelerator_memory(self._pose_model)
                    self._pose_model = None
                    self._pose_model_signature = None
                    if config.pose_checkpoint_path:
                        self.status_changed.emit("Loading pose checkpoint")
                        try:
                            self._pose_model = self._load_pose_model(config.pose_checkpoint_path)
                            self._pose_model_signature = config.pose_checkpoint_path
                            self.status_changed.emit("Pose model ready")
                        except Exception as exc:
                            self._pose_model = None
                            self._pose_model_signature = None
                            self.error_occurred.emit(f"Pose model load failed: {exc}")

                start_time = time.perf_counter()
                frame_rgb = self._ensure_rgb(packet.frame)
                inference_frame, scale_x, scale_y = self._prepare_inference_frame(
                    frame_rgb,
                    config.inference_max_width,
                )
                detections = self._predict(
                    self._model,
                    config.model_key,
                    inference_frame,
                    config.threshold,
                )
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

                inference_ms = (time.perf_counter() - start_time) * 1000.0
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

    def _load_model(self, model_key: str, checkpoint: str):
        if str(model_key).startswith("rfdetr"):
            from rfdetr import RFDETRSegLarge, RFDETRSegMedium

            model_map = {
                "rfdetr-seg-medium": RFDETRSegMedium,
                "rfdetr-seg-large": RFDETRSegLarge,
            }
            model_cls = model_map.get(model_key, RFDETRSegMedium)
            if not checkpoint:
                model = model_cls()
                if hasattr(model, "eval"):
                    model.eval()
                return self._optimize_loaded_model(model, model_key)
            for key in ("pretrain_weights", "checkpoint_path", "weights"):
                try:
                    model = model_cls(**{key: checkpoint})
                    if hasattr(model, "eval"):
                        model.eval()
                    return self._optimize_loaded_model(model, model_key)
                except TypeError:
                    continue
            model = model_cls()
            if hasattr(model, "eval"):
                model.eval()
            return self._optimize_loaded_model(model, model_key)

        from ultralytics import YOLO

        weight_path = checkpoint or "yolo11n-seg.pt"
        model = YOLO(weight_path)
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.eval()
        return model

    def _load_pose_model(self, checkpoint: str):
        """Load a YOLO pose checkpoint for paired keypoint inference."""
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

        try:
            import torch
        except Exception:
            torch = None

        height, width = frame_rgb.shape[:2]
        pad_frac = 0.05

        for record in records:
            try:
                bbox = record.get("bbox")
                if bbox is None:
                    continue
                x1, y1, x2, y2 = (float(value) for value in bbox)
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                pad_x = bw * pad_frac
                pad_y = bh * pad_frac
                cx1 = max(0, int(round(x1 - pad_x)))
                cy1 = max(0, int(round(y1 - pad_y)))
                cx2 = min(width, int(round(x2 + pad_x)))
                cy2 = min(height, int(round(y2 + pad_y)))
                if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                    continue
                crop = frame_rgb[cy1:cy2, cx1:cx2]

                def _run_pose():
                    return self._pose_model.predict(
                        crop,
                        conf=float(pose_threshold),
                        verbose=False,
                    )

                if torch is not None:
                    with torch.inference_mode():
                        results = _run_pose()
                else:
                    results = _run_pose()

                if not results:
                    continue
                pose_result = results[0]
                kp_obj = getattr(pose_result, "keypoints", None)
                if kp_obj is None:
                    continue

                kp_xy = self._to_numpy(getattr(kp_obj, "xy", None), dtype=float, shape=None)
                kp_conf = self._to_numpy(getattr(kp_obj, "conf", None), dtype=float, shape=None)
                if kp_xy is None or kp_xy.size == 0 or kp_xy.ndim < 2:
                    continue

                # Pick the highest-confidence pose detection inside the crop.
                pose_boxes = getattr(pose_result, "boxes", None)
                pose_scores = None
                if pose_boxes is not None:
                    pose_scores = self._to_numpy(getattr(pose_boxes, "conf", None), dtype=float, shape=(-1,))
                if pose_scores is not None and len(pose_scores):
                    best_index = int(np.argmax(pose_scores))
                else:
                    best_index = 0
                if kp_xy.ndim == 2:
                    selected_xy = kp_xy
                    selected_conf = kp_conf if (kp_conf is not None and kp_conf.ndim == 1) else None
                else:
                    if best_index >= len(kp_xy):
                        continue
                    selected_xy = kp_xy[best_index]
                    if kp_conf is not None and kp_conf.ndim == 2 and best_index < len(kp_conf):
                        selected_conf = kp_conf[best_index]
                    else:
                        selected_conf = None

                if selected_xy.shape[-1] < 2:
                    continue
                keypoints_image = selected_xy.astype(float, copy=True)
                keypoints_image[:, 0] = keypoints_image[:, 0] + float(cx1)
                keypoints_image[:, 1] = keypoints_image[:, 1] + float(cy1)
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

                record["keypoints"] = keypoints_image
                record["keypoint_scores"] = selected_conf
            except Exception:
                # Per-detection pose failures must not abort live inference.
                continue

    def _predict(self, model, model_key: str, frame_rgb: np.ndarray, threshold: float):
        try:
            import torch
        except Exception:
            torch = None

        def _run_prediction():
            if str(model_key).startswith("rfdetr"):
                return model.predict(frame_rgb, threshold=float(threshold))

            results = model.predict(frame_rgb, conf=float(threshold), verbose=False)
            return results[0] if results else None

        if torch is not None:
            with torch.inference_mode():
                return _run_prediction()
        return _run_prediction()

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

    def _optimize_loaded_model(self, model, model_key: str):
        if not str(model_key).startswith("rfdetr"):
            return model
        if not hasattr(model, "optimize_for_inference"):
            return model

        try:
            import torch
        except Exception:
            return model

        optimize_kwargs = {"batch_size": 1}
        try:
            model_device = str(getattr(getattr(model, "model", None), "device", "") or "").lower()
        except Exception:
            model_device = ""
        use_cuda = torch.cuda.is_available() and "cuda" in model_device
        optimize_kwargs["dtype"] = torch.float16 if use_cuda else torch.float32
        optimize_kwargs["compile"] = bool(use_cuda)

        try:
            self.status_changed.emit("Optimizing RF-DETR for live inference")
            model.optimize_for_inference(**optimize_kwargs)
            self.status_changed.emit(
                f"RF-DETR optimized ({'fp16' if optimize_kwargs['dtype'] == torch.float16 else 'fp32'})"
            )
        except Exception as exc:
            self.status_changed.emit(f"RF-DETR optimization skipped: {exc}")
        return model

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
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
