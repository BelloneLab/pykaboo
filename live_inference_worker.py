"""Latest-frame live segmentation worker for RF-DETR Seg and YOLO Seg models."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

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

    def normalized(self) -> "LiveInferenceConfig":
        return LiveInferenceConfig(
            model_key=str(self.model_key or "rfdetr-seg-medium").strip(),
            checkpoint_path=str(self.checkpoint_path or "").strip(),
            threshold=float(self.threshold),
            selected_class_ids=list(self.selected_class_ids or []),
            identity_mode=str(self.identity_mode or "tracker").strip().lower(),
            expected_mouse_count=max(1, int(self.expected_mouse_count or 1)),
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

                start_time = time.perf_counter()
                frame_rgb = self._ensure_rgb(packet.frame)
                detections = self._predict(self._model, config.model_key, frame_rgb, config.threshold)
                normalized = self._normalize_detections(detections)
                records = self._build_detection_records(normalized, config)

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
            import cv2

            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return np.asarray(frame)

    def _load_model(self, model_key: str, checkpoint: str):
        if str(model_key).startswith("rfdetr"):
            from rfdetr import RFDETRSegLarge, RFDETRSegMedium

            model_map = {
                "rfdetr-seg-medium": RFDETRSegMedium,
                "rfdetr-seg-large": RFDETRSegLarge,
            }
            model_cls = model_map.get(model_key, RFDETRSegMedium)
            if not checkpoint:
                return model_cls()
            for key in ("pretrain_weights", "checkpoint_path", "weights"):
                try:
                    return model_cls(**{key: checkpoint})
                except TypeError:
                    continue
            return model_cls()

        from ultralytics import YOLO

        weight_path = checkpoint or "yolo11n-seg.pt"
        return YOLO(weight_path)

    def _predict(self, model, model_key: str, frame_rgb: np.ndarray, threshold: float):
        if str(model_key).startswith("rfdetr"):
            from PIL import Image

            return model.predict(Image.fromarray(frame_rgb), threshold=float(threshold))

        results = model.predict(frame_rgb, conf=float(threshold), verbose=False)
        return results[0] if results else None

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

    def _release_accelerator_memory(self, model) -> None:
        del model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
