"""
Camera Worker Thread - Clean Implementation
Handles camera acquisition, GPU-accelerated recording, and metadata logging.
"""
import time
import subprocess
import os
import re
import threading
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import cv2
from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker
from camera_backends import (
    Boson,
    Lepton,
    PYSPIN_AVAILABLE,
    PYPYLON_AVAILABLE,
    PySpin,
    TeaxGrabber,
    _read_pyspin_string_node,
    pylon,
)
from config import CAMERA_CONFIG
from live_detection_types import PreviewFramePacket


@dataclass
class FramePacket:
    """Frame data handed from acquisition to the processing pipeline."""

    backend: str
    frame: np.ndarray
    metadata: Dict[str, object] = field(default_factory=dict)
    requested_format: str = "Mono8"
    pixel_format: str = ""
    color_filter: str = ""


def frames_for_duration(fps: Optional[float], seconds: Optional[float]) -> Optional[int]:
    """Exact frame count for a recording of ``seconds`` at ``fps``.

    Returns ``round(fps * seconds)`` (at least 1) so a file written at ``fps``
    plays back for exactly ``seconds`` (frame-accurate, error < one frame), or
    None when either input is missing/non-positive (i.e. unlimited recording).
    This is the single source of truth for converting a requested duration into
    the acquisition-thread frame cap that enforces it.
    """
    if not fps or not seconds:
        return None
    if float(fps) <= 0.0 or float(seconds) <= 0.0:
        return None
    return max(1, int(round(float(fps) * float(seconds))))


class CameraWorker(QThread):
    """
    Worker thread for camera operations.
    Manages frame acquisition, FFmpeg encoding, and metadata collection.
    """

    # Signals
    frame_ready = Signal(np.ndarray)
    preview_packet_ready = Signal(object)
    live_inference_packet_ready = Signal(object)
    record_frame_packet_ready = Signal(object)
    status_update = Signal(str)
    fps_update = Signal(float)
    buffer_update = Signal(int)
    error_occurred = Signal(str)
    recording_stopped = Signal()
    frame_recorded = Signal(dict)  # Signal for each recorded frame with metadata
    frame_metadata_ready = Signal(dict)
    frame_drop_stats_updated = Signal(dict)
    resolution_changed = Signal(int, int)  # actual (width, height) after a capture reconfigure

    def __init__(self):
        super().__init__()

        # Camera
        self.camera: Optional[Any] = None
        self.flir_camera: Optional[Any] = None
        self.usb_capture: Optional[cv2.VideoCapture] = None
        # Capture-backend reconfiguration (resolution) must run on the worker
        # thread: calling cv2.VideoCapture.set()/get() from the GUI thread while
        # this thread is in cap.read() hangs the camera driver (app freeze).
        self._reconfig_lock = threading.Lock()
        self._pending_capture_reconfig: Optional[Tuple[int, int]] = None
        self.usb_index: Optional[int] = None
        self.usb_backend = ""
        self.usb_auto_white_balance_enabled = True
        self.usb_white_balance_gains_bgr: Optional[np.ndarray] = None
        self.pyspin_system: Optional[Any] = None
        self.pyspin_cam_list: Optional[Any] = None
        self.pyspin_image_processor: Optional[Any] = None
        self.camera_type: Optional[str] = None
        self.basler_device_class = ""
        self.flir_backend: Optional[str] = None
        self.flir_video_index: Optional[int] = None
        self.flir_serial_port: Optional[str] = None
        self.flir_status_cache: Dict[str, object] = {}
        self.flir_status_cache_time = 0.0
        self.camera_settings_cache: Dict[str, object] = {}
        self.camera_settings_cache_time = 0.0
        self.spinnaker_native_pixel_format = ""
        self.spinnaker_color_filter = ""
        self.spinnaker_is_color = False
        self.spinnaker_pause_requested = False
        self.spinnaker_paused = False
        self._line_debug_frame_counter: int = 0
        self._spinnaker_cached_line_selectors: List[str] = []
        self._cached_line_capabilities: List[Dict] = []
        self.basler_pause_requested = False
        self.basler_paused = False
        self.converter = pylon.ImageFormatConverter() if PYPYLON_AVAILABLE else None
        if self.converter is not None:
            self.converter.OutputPixelFormat = pylon.PixelType_Mono8
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # Thread control
        self.running = False
        self.mutex = QMutex()
        self.processing_thread: Optional[threading.Thread] = None
        self.processing_condition = threading.Condition()
        self.processing_queue: deque[FramePacket] = deque()
        self.processing_queue_max_frames = 128
        self.processing_queue_drop_count = 0
        self.processing_queue_high_water = 0
        self.processing_queue_last_drop_notice = 0.0
        self.recording_lock = threading.RLock()
        self.metadata_stats_counter = 0

        # Recording
        self.is_recording = False
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.ffmpeg_stderr_thread: Optional[threading.Thread] = None
        self.metadata_buffer: List[Dict] = []
        self.recording_filename = ""
        self.frame_counter = 0
        self.max_record_frames: Optional[int] = None
        self.recording_duration_seconds: Optional[float] = None
        self.camera_reported_fps: Optional[float] = None
        self.recording_output_fps: Optional[float] = None

        # FFmpeg encoder settings
        self.encoder = "h264_nvenc"  # Default to NVIDIA GPU
        self.encoder_preset = "p4"
        self.bitrate = "5M"

        # Camera config
        self.trigger_mode = "FreeRun"
        self.width = 0
        self.height = 0
        self.fps_target = 30.0
        self.image_format = "Mono8"
        self.roi = None
        self.preview_enabled = True
        self.preview_target_fps = 25.0
        self.preview_max_width = 1280
        self.preview_last_emit_time = 0.0
        self.live_inference_packets_enabled = False
        # Inference packets are throttled and downscaled independently of the
        # display preview: the live inference worker only ever consumes the
        # newest frame, so emitting full-rate full-resolution RGB frames just
        # burns CPU and starves the inference thread. ~30 emit fps at <=960 px
        # keeps the GPU fed without saturating a core.
        self.live_inference_emit_fps = 30.0
        self.live_inference_emit_max_width = 960
        self.live_inference_last_emit_time = 0.0
        self.record_frame_packets_enabled = False
        self.metadata_stats_interval_frames = 25
        self.stream_buffer_target = 128
        self.active_encoder = self.encoder

        # FPS calculation
        self.fps_frame_count = 0
        self.fps_last_time = time.time()
        self.line_label_map: Dict[str, str] = {}
        self.frame_timing_reference_fps = 0.0
        self.frame_timing_expected_interval = 0.0
        self.frame_timing_last_timestamp: Optional[float] = None
        self.frame_timing_interval_sum = 0.0
        self.frame_timing_interval_count = 0
        self.frame_timing_max_gap = 0.0
        self.frame_timing_last_interval = 0.0
        self.frame_drop_estimate = 0
        self.recording_started_at: Optional[float] = None
        self.last_recording_output_fps: Optional[float] = None
        self.last_recording_stats: Dict[str, object] = {}
        self._reset_frame_drop_stats()
    def set_encoder(self, encoder: str, preset: str = "p4", bitrate: str = "5M"):
        """Set FFmpeg encoder and parameters."""
        self.encoder = encoder
        self.encoder_preset = preset
        self.bitrate = bitrate

    def set_preview_enabled(self, enabled: bool):
        """Enable or disable live preview emission."""
        self.preview_enabled = bool(enabled)
        if not self.preview_enabled:
            self.preview_last_emit_time = 0.0

    def set_preview_fps(self, fps: float):
        """Set the preview emission rate independent of acquisition FPS."""
        try:
            fps_value = float(fps)
        except (TypeError, ValueError):
            fps_value = 25.0
        self.preview_target_fps = max(1.0, fps_value)

    def set_preview_max_width(self, width: int):
        """Limit preview frame width before shipping frames to the UI."""
        try:
            width_value = int(width)
        except (TypeError, ValueError):
            width_value = 1280
        self.preview_max_width = max(0, width_value)

    def set_live_inference_packets_enabled(self, enabled: bool):
        """Emit processed frames for live inference independent of preview FPS."""
        self.live_inference_packets_enabled = bool(enabled)
        self.live_inference_last_emit_time = 0.0

    def set_live_inference_emit_fps(self, fps: float):
        """Cap how often inference frames are shipped to the inference worker."""
        try:
            value = float(fps)
        except (TypeError, ValueError):
            value = 30.0
        self.live_inference_emit_fps = max(1.0, value)

    def set_live_inference_emit_max_width(self, width: int):
        """Downscale inference frames to this width before emitting (0 = native)."""
        try:
            value = int(width)
        except (TypeError, ValueError):
            value = 960
        self.live_inference_emit_max_width = max(0, value)

    def _should_emit_inference_packet(self) -> bool:
        """Throttle inference frame emission to the configured rate."""
        fps = max(1.0, float(self.live_inference_emit_fps or 30.0))
        interval = 1.0 / fps
        now = time.monotonic()
        if (now - self.live_inference_last_emit_time) < interval:
            return False
        self.live_inference_last_emit_time = now
        return True

    def _downscale_for_inference(self, frame: np.ndarray) -> np.ndarray:
        """Resize a record frame down to the inference emit width to cut cost."""
        max_width = int(self.live_inference_emit_max_width or 0)
        if max_width <= 0 or frame.ndim < 2 or frame.shape[1] <= max_width:
            return frame
        scale = max_width / float(frame.shape[1])
        target_height = max(1, int(round(frame.shape[0] * scale)))
        return cv2.resize(frame, (max_width, target_height), interpolation=cv2.INTER_LINEAR)

    def set_record_frame_packets_enabled(self, enabled: bool):
        """Emit full-resolution recorded frames for sidecar overlay video export."""
        self.record_frame_packets_enabled = bool(enabled)

    def set_metadata_stats_interval(self, frames: int):
        """Control how often expensive raw frame statistics are computed."""
        try:
            frame_value = int(frames)
        except (TypeError, ValueError):
            frame_value = 25
        self.metadata_stats_interval_frames = max(0, frame_value)

    def set_frame_buffer_size(self, frames: int):
        """Resize the app-level processing queue and camera stream buffers."""
        try:
            frame_value = int(frames)
        except (TypeError, ValueError):
            frame_value = 128
        frame_value = max(8, frame_value)
        self.processing_queue_max_frames = frame_value
        self.stream_buffer_target = max(32, frame_value)

        effective_capacity = self._get_effective_processing_queue_capacity()
        with self.processing_condition:
            while len(self.processing_queue) > effective_capacity:
                self.processing_queue.popleft()
                self.processing_queue_drop_count += 1
            self.processing_condition.notify_all()

        if self.is_spinnaker_camera():
            self._configure_spinnaker_stream()
        elif self.camera_type == "basler" and self.camera and self.camera.IsOpen():
            self._configure_basler_stream_buffers()
        self._emit_processing_buffer_usage()

    def _is_basler_gige_camera(self) -> bool:
        """True when the connected Basler camera uses the GigE transport."""
        return self.camera_type == "basler" and "gige" in str(self.basler_device_class or "").lower()

    def _get_effective_processing_queue_capacity(self) -> int:
        """Clamp live frame buffering to avoid runaway memory on high-rate cameras."""
        requested = max(8, int(self.processing_queue_max_frames))
        if self._is_basler_gige_camera():
            return min(requested, 32)
        return requested

    def _get_basler_camera_buffer_target(self) -> int:
        """Size Basler SDK buffers conservatively for real-time recording stability."""
        target = self._get_effective_processing_queue_capacity()
        if self._is_basler_gige_camera():
            return min(target, 32)
        return min(target, 64)

    def _configure_basler_stream_buffers(self):
        """Apply a conservative Basler camera-side buffer depth."""
        if self.camera_type != "basler" or not self.camera or not self.camera.IsOpen():
            return
        try:
            target = int(self._get_basler_camera_buffer_target())
            if hasattr(self.camera, "MaxNumBuffer"):
                try:
                    self.camera.MaxNumBuffer.SetValue(target)
                except Exception:
                    self.camera.MaxNumBuffer = target
        except Exception:
            pass

    def _get_basler_grab_strategy(self):
        """Pick a grab strategy that favors live stability over unbounded backlog."""
        if self._is_basler_gige_camera():
            return pylon.GrabStrategy_LatestImageOnly
        strategy_name = str(CAMERA_CONFIG.get("grab_strategy", "LatestImageOnly") or "").strip().lower()
        if strategy_name == "onebyone":
            return pylon.GrabStrategy_OneByOne
        return pylon.GrabStrategy_LatestImageOnly

    def _reset_frame_drop_stats(self):
        """Reset frame-timing counters for a new recording session."""
        reference_fps = float(self.camera_reported_fps or self.fps_target or 0.0)
        if reference_fps <= 0:
            reference_fps = 30.0

        self.frame_timing_reference_fps = reference_fps
        self.frame_timing_expected_interval = 1.0 / reference_fps if reference_fps > 0 else 0.0
        self.frame_timing_last_timestamp = None
        self.frame_timing_interval_sum = 0.0
        self.frame_timing_interval_count = 0
        self.frame_timing_max_gap = 0.0
        self.frame_timing_last_interval = 0.0
        self.frame_drop_estimate = 0
        self.recording_started_at = None
        self.last_recording_stats = self._build_frame_drop_stats(active=False)

    def _build_frame_drop_stats(self, active: Optional[bool] = None) -> Dict[str, object]:
        """Build a snapshot of current frame-drop estimates for UI and file export."""
        recorded_frames = int(self.frame_counter)
        estimated_total_frames = recorded_frames + int(self.frame_drop_estimate)
        drop_percent = 0.0
        if estimated_total_frames > 0:
            drop_percent = (float(self.frame_drop_estimate) / float(estimated_total_frames)) * 100.0

        average_interval_ms = 0.0
        if self.frame_timing_interval_count > 0:
            average_interval_ms = (self.frame_timing_interval_sum / self.frame_timing_interval_count) * 1000.0

        elapsed_seconds = 0.0
        if self.recording_started_at is not None:
            elapsed_seconds = max(0.0, time.time() - self.recording_started_at)

        return {
            "active": self.is_recording if active is None else bool(active),
            "recorded_frames": recorded_frames,
            "estimated_dropped_frames": int(self.frame_drop_estimate),
            "estimated_total_frames": int(estimated_total_frames),
            "drop_percent": float(drop_percent),
            "reference_fps": float(self.frame_timing_reference_fps),
            "expected_interval_ms": float(self.frame_timing_expected_interval * 1000.0),
            "average_interval_ms": float(average_interval_ms),
            "max_gap_ms": float(self.frame_timing_max_gap * 1000.0),
            "last_interval_ms": float(self.frame_timing_last_interval * 1000.0),
            "elapsed_seconds": float(elapsed_seconds),
            "timestamp_source": "software",
            "camera_type": self.camera_type or "",
        }

    def _emit_frame_drop_stats(self, active: Optional[bool] = None):
        """Publish current frame-drop statistics."""
        self.last_recording_stats = self._build_frame_drop_stats(active=active)
        self.frame_drop_stats_updated.emit(dict(self.last_recording_stats))

    def _track_recorded_frame_timing(self, metadata: Dict):
        """Estimate dropped frames from gaps between recorded-frame timestamps."""
        timestamp_raw = metadata.get("timestamp_software")
        try:
            timestamp_value = float(timestamp_raw)
        except (TypeError, ValueError):
            timestamp_value = time.time()

        if self.recording_started_at is None:
            self.recording_started_at = timestamp_value

        if self.frame_timing_last_timestamp is not None:
            interval = max(0.0, timestamp_value - self.frame_timing_last_timestamp)
            self.frame_timing_last_interval = interval
            self.frame_timing_interval_sum += interval
            self.frame_timing_interval_count += 1
            self.frame_timing_max_gap = max(self.frame_timing_max_gap, interval)

            if self.frame_timing_expected_interval > 0:
                interval_ratio = interval / self.frame_timing_expected_interval
                if interval_ratio >= 1.5:
                    missing_frames = max(int(interval_ratio + 0.5) - 1, 0)
                    self.frame_drop_estimate += missing_frames

        self.frame_timing_last_timestamp = timestamp_value

        if self.frame_counter <= 3 or self.frame_counter % 30 == 0:
            self._emit_frame_drop_stats(active=True)

    def set_image_format(self, image_format: str):
        """Set image format for recording/display."""
        self.image_format = image_format
        if self.camera_type == "basler" and self.converter is not None:
            if image_format == "BGR8":
                self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            else:
                self.converter.OutputPixelFormat = pylon.PixelType_Mono8
        elif self.is_spinnaker_camera():
            paused_for_reconfigure = False
            if self.isRunning():
                paused_for_reconfigure = self._pause_spinnaker_acquisition_for_reconfigure()
            try:
                current_pixel_format = self._read_enum_node_symbolic("PixelFormat")
                current_is_color = self._is_color_pixel_format(current_pixel_format)
                if image_format == "BGR8" and self.spinnaker_is_color and not current_is_color:
                    # Keep color cameras in their native Bayer/RGB format and
                    # debayer on the host when needed. This is more reliable than
                    # assuming the camera can switch directly to RGB8.
                    candidates = self._get_preferred_spinnaker_color_pixel_formats()
                    if not candidates:
                        candidates = ["RGB8", "BGR8"]
                    for candidate in candidates:
                        if self._set_enum_node_by_name("PixelFormat", candidate):
                            if "Bayer" in candidate:
                                self.spinnaker_native_pixel_format = candidate
                                self.spinnaker_color_filter = self._read_enum_node_symbolic("PixelColorFilter") or self.spinnaker_color_filter
                            break
                self._refresh_camera_settings_cache(force=True)
            finally:
                if paused_for_reconfigure:
                    self._resume_spinnaker_acquisition_after_reconfigure()

    def set_target_fps(self, fps: float):
        """Set target FPS for recording output."""
        self.fps_target = float(fps)

    def _should_emit_preview(self) -> bool:
        """Return True when the next processed frame should be sent to the UI."""
        if not self.preview_enabled:
            return False

        preview_fps = max(1.0, float(self.preview_target_fps or 25.0))
        interval = 1.0 / preview_fps
        now = time.monotonic()
        if (now - self.preview_last_emit_time) < interval:
            return False
        self.preview_last_emit_time = now
        return True

    def _resize_preview_frame(self, frame: np.ndarray) -> np.ndarray:
        """Downscale preview frames before they reach the GUI."""
        preview = frame
        max_width = int(self.preview_max_width or 0)
        if max_width <= 0 or preview.ndim < 2 or preview.shape[1] <= max_width:
            return preview

        scale = max_width / float(preview.shape[1])
        target_height = max(1, int(round(preview.shape[0] * scale)))
        return cv2.resize(preview, (max_width, target_height), interpolation=cv2.INTER_AREA)

    def _emit_processing_buffer_usage(self):
        """Publish internal processing queue occupancy as a percentage."""
        with self.processing_condition:
            queue_len = len(self.processing_queue)
            capacity = max(1, int(self._get_effective_processing_queue_capacity()))
        usage = int(min(100.0, (100.0 * queue_len) / float(capacity)))
        self.buffer_update.emit(usage)

    def _enqueue_frame_packet(self, packet: FramePacket) -> bool:
        """Push a captured frame into the processing queue."""
        dropped = False
        with self.processing_condition:
            capacity = max(1, int(self._get_effective_processing_queue_capacity()))
            while self.running and len(self.processing_queue) >= capacity:
                if not self.is_recording:
                    self.processing_queue.popleft()
                    self.processing_queue_drop_count += 1
                    dropped = True
                    break
                if self._is_basler_gige_camera():
                    self.processing_queue.popleft()
                    self.processing_queue_drop_count += 1
                    dropped = True
                    break
                self.processing_condition.wait(timeout=0.01)

            if not self.running:
                return False

            self.processing_queue.append(packet)
            self.processing_queue_high_water = max(self.processing_queue_high_water, len(self.processing_queue))
            self.processing_condition.notify()

        if dropped and (time.time() - self.processing_queue_last_drop_notice) > 1.0:
            self.processing_queue_last_drop_notice = time.time()
            self.status_update.emit("Processing queue overflowed; dropping oldest buffered frames")
        self._emit_processing_buffer_usage()
        return True

    def _dequeue_frame_packet(self, timeout: float = 0.1) -> Optional[FramePacket]:
        """Read the next frame waiting for processing."""
        with self.processing_condition:
            if not self.processing_queue and self.running:
                self.processing_condition.wait(timeout=max(0.0, float(timeout)))
            if not self.processing_queue:
                return None
            packet = self.processing_queue.popleft()
            self.processing_condition.notify_all()
        self._emit_processing_buffer_usage()
        return packet

    def _start_processing_thread(self):
        """Launch the pipeline stage that converts, previews, and records frames."""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            return
        self.preview_last_emit_time = 0.0
        self.processing_queue_drop_count = 0
        self.processing_queue_high_water = 0
        self.processing_queue_last_drop_notice = 0.0
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="CameraProcessingWorker",
            daemon=True,
        )
        self.processing_thread.start()

    def _stop_processing_thread(self, timeout: float = 5.0):
        """Wait for the processing stage to finish draining queued frames."""
        with self.processing_condition:
            self.processing_condition.notify_all()
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=max(0.1, float(timeout)))
        self.processing_thread = None
        self._emit_processing_buffer_usage()

    def _processing_loop(self):
        """Convert, preview, and record frames independently from acquisition."""
        try:
            while self.running or self.processing_queue:
                packet = self._dequeue_frame_packet(timeout=0.1)
                if packet is None:
                    continue
                try:
                    self._process_frame_packet(packet)
                except Exception as exc:
                    self.error_occurred.emit(f"Frame processing error: {str(exc)}")
        finally:
            self._emit_processing_buffer_usage()

    def set_recording_duration_limit(self, seconds: Optional[float]):
        """Request the next recording to last an exact wall-clock duration.

        The duration is converted to an exact frame count at record start (see
        ``start_recording``) using the encode FPS, so the saved file is exactly
        ``seconds`` long and the stop is enforced deterministically on the
        acquisition thread. Pass None or <= 0 for unlimited.
        """
        if seconds is None or float(seconds) <= 0.0:
            self.recording_duration_seconds = None
        else:
            self.recording_duration_seconds = float(seconds)

    def set_recording_frame_limit(self, max_frames: Optional[int]):
        """Set an optional hard cap for the number of frames written per recording."""
        stop_now = False
        with QMutexLocker(self.mutex):
            if max_frames is None:
                self.max_record_frames = None
            else:
                self.max_record_frames = max(1, int(max_frames))
                stop_now = bool(self.is_recording and self.frame_counter >= self.max_record_frames)

        if stop_now:
            self.status_update.emit(f"Reached frame target: {self.max_record_frames} frames")
            self.stop_recording()

    def update_resolution(self, width: int, height: int):
        """Update cached resolution for recording output."""
        self.width = int(width)
        self.height = int(height)

    def _request_usb_mjpg(self, capture) -> None:
        """Best-effort: ask an OpenCV USB capture for the MJPG pixel format.

        OpenCV opens UVC webcams in raw (YUY2) mode by default, which is
        USB-bandwidth-limited to only a few fps at 1080p; requesting MJPG lets
        the camera stream compressed frames so high resolutions can reach their
        full frame rate. Harmless when unsupported: the camera simply keeps its
        current mode, so this never blocks a connect or a resolution change.
        Must be set before width/height, as some drivers reset to raw on a mode
        change.
        """
        if capture is None:
            return
        try:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

    def apply_resolution(self, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Apply a capture resolution across whatever backend is active.

        Returns the actual (width, height) the device accepted, or None if the
        backend does not support a runtime resolution change. Used by the
        auxiliary-stream settings popup so each stream mirrors the primary
        camera's resolution control without duplicating backend logic.
        """
        width = int(width)
        height = int(height)
        if self.camera_type == "usb" and self.usb_capture is not None:
            try:
                self._request_usb_mjpg(self.usb_capture)
                self.usb_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.usb_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_w = int(self.usb_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
                actual_h = int(self.usb_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
                self.update_resolution(actual_w, actual_h)
                return actual_w, actual_h
            except Exception:
                return None
        if self.camera_type == "flir":
            cap = getattr(self.flir_camera, "cap", None)
            if cap is not None:
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
                    self.update_resolution(actual_w, actual_h)
                    return actual_w, actual_h
                except Exception:
                    return None
        if self.is_genicam_camera():
            applied = self.set_camera_resolution(width, height)
            if applied is not None:
                return int(applied[0]), int(applied[1])
        return None

    def request_resolution(self, width: int, height: int) -> bool:
        """Request a capture-resolution change safely.

        For OpenCV-backed cameras (USB and FLIR/cv2) the change is queued and
        applied on the acquisition thread between frame reads, then reported via
        ``resolution_changed``. Calling ``cv2.VideoCapture.set`` from another
        thread mid-read freezes the driver, which is why this indirection
        exists. Returns True if the change was queued; False means the caller
        should apply it directly (GenICam handles its own pause/resume).
        """
        uses_opencv_capture = self.camera_type == "usb" or (
            self.camera_type == "flir" and getattr(self.flir_camera, "cap", None) is not None
        )
        if not uses_opencv_capture:
            return False
        with self._reconfig_lock:
            self._pending_capture_reconfig = (int(width), int(height))
        return True

    def _apply_pending_capture_reconfig(self) -> None:
        """Apply a queued capture-resolution change on the acquisition thread."""
        with self._reconfig_lock:
            pending = self._pending_capture_reconfig
            self._pending_capture_reconfig = None
        if pending is None:
            return
        width, height = pending
        cap = self.usb_capture if self.camera_type == "usb" else getattr(self.flir_camera, "cap", None)
        if cap is None:
            return
        try:
            if self.camera_type == "usb":
                self._request_usb_mjpg(cap)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(width)
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(height)
            self.update_resolution(actual_w, actual_h)
            self.resolution_changed.emit(actual_w, actual_h)
        except Exception as exc:
            self.error_occurred.emit(f"Resolution change failed: {exc}")

    def set_roi(self, roi: Optional[dict]):
        """Set software ROI (x, y, w, h) for cropping."""
        self.roi = roi

    def set_line_label_map(self, label_map: Dict[str, str]):
        """Set optional suffixes for line status column labels."""
        self.line_label_map = label_map or {}

    def is_spinnaker_camera(self) -> bool:
        """True when the active camera is a FLIR Spinnaker / PySpin device."""
        return (
            self.camera_type == "flir"
            and self.flir_backend == "spinnaker"
            and self.camera is not None
            and PYSPIN_AVAILABLE
            and PySpin is not None
        )

    def is_genicam_camera(self) -> bool:
        """True when the active camera exposes direct GenICam-style nodes."""
        if self.camera_type == "basler" and self.camera:
            try:
                return bool(self.camera.IsOpen())
            except Exception:
                return False
        if self.is_spinnaker_camera():
            return True
        return False

    def _get_camera_node(self, node_name: str):
        """Return a direct-access camera node when available."""
        if not self.camera:
            return None
        try:
            node = getattr(self.camera, node_name)
        except Exception:
            node = None
        if node is not None:
            return node

        if self.is_spinnaker_camera() and PySpin is not None:
            try:
                node_map = self.camera.GetNodeMap()
                raw_node = node_map.GetNode(node_name) if node_map is not None else None
            except Exception:
                return None
            return self._wrap_spinnaker_nodemap_node(raw_node)

        return None

    def _wrap_spinnaker_nodemap_node(self, raw_node):
        """Cast a PySpin nodemap node to the interface pointer that exposes Get/Set helpers."""
        if raw_node is None or PySpin is None:
            return None

        try:
            if not PySpin.IsAvailable(raw_node):
                return None
        except Exception:
            pass

        try:
            interface_type = raw_node.GetPrincipalInterfaceType()
        except Exception:
            interface_type = None

        for interface_name, caster_name in (
            ("intfIEnumeration", "CEnumerationPtr"),
            ("intfIInteger", "CIntegerPtr"),
            ("intfIFloat", "CFloatPtr"),
            ("intfIBoolean", "CBooleanPtr"),
            ("intfIString", "CStringPtr"),
            ("intfICommand", "CCommandPtr"),
        ):
            expected_interface = getattr(PySpin, interface_name, None)
            caster = getattr(PySpin, caster_name, None)
            if expected_interface is None or caster is None or interface_type != expected_interface:
                continue
            try:
                return caster(raw_node)
            except Exception:
                return None

        return raw_node

    def _node_is_readable(self, node) -> bool:
        if node is None:
            return False
        try:
            if hasattr(node, "IsReadable"):
                return bool(node.IsReadable())
        except Exception:
            pass
        if self.is_spinnaker_camera() and PySpin is not None:
            try:
                return bool(PySpin.IsReadable(node))
            except Exception:
                pass
        return True

    def _node_is_writable(self, node) -> bool:
        if node is None:
            return False
        try:
            if hasattr(node, "IsWritable"):
                return bool(node.IsWritable())
        except Exception:
            pass
        if self.is_spinnaker_camera() and PySpin is not None:
            try:
                return bool(PySpin.IsWritable(node))
            except Exception:
                pass
        return True

    def _set_enum_node_by_name(self, node_name: str, entry_name: str) -> bool:
        """Set an enumeration node to a named entry if supported."""
        node = self._get_camera_node(node_name)
        if node is None or not self._node_is_writable(node):
            return False
        requested = str(entry_name or "").strip()
        if not requested:
            return False

        def _normalize_enum_value(value: str) -> str:
            return "".join(ch for ch in str(value or "").lower() if ch.isalnum())

        # Prefer native string SetValue when the SDK supports it.
        try:
            node.SetValue(requested)
            return True
        except Exception:
            pass

        try:
            entry = node.GetEntryByName(requested)
            node.SetIntValue(entry.GetValue())
            return True
        except Exception:
            pass

        requested_normalized = _normalize_enum_value(requested)
        try:
            raw_entries = node.GetEntries()
        except Exception:
            raw_entries = []

        for entry in raw_entries:
            labels = []
            try:
                labels.append(str(entry.GetSymbolic()).strip())
            except Exception:
                pass
            try:
                labels.append(str(entry.ToString()).strip())
            except Exception:
                pass

            if not any(label for label in labels):
                continue
            if not any(_normalize_enum_value(label) == requested_normalized for label in labels if label):
                continue

            try:
                node.SetIntValue(entry.GetValue())
                return True
            except Exception:
                continue

        return False

    def _read_numeric_node(self, node_name: str) -> Optional[float]:
        """Read a numeric node value when available."""
        node = self._get_camera_node(node_name)
        if node is None or not self._node_is_readable(node):
            return None
        try:
            return float(node.GetValue())
        except Exception:
            return None

    def _apply_line_status_metadata(
        self,
        metadata: Dict[str, object],
        line_status: Optional[int],
        display_line_status: Optional[int] = None,
    ):
        """Project line status bits into the UI-friendly line fields."""
        if line_status is None and display_line_status is None:
            metadata['line_status_all'] = None
            metadata['line1_status'] = None
            metadata['line2_status'] = None
            metadata['line3_status'] = None
            metadata['line4_status'] = None
            return

        raw_status = int(line_status) if line_status is not None else int(display_line_status)
        packed_status = int(display_line_status) if display_line_status is not None else raw_status
        metadata['line_status_all'] = raw_status
        metadata['line1_status'] = (packed_status >> 0) & 0x01
        metadata['line2_status'] = (packed_status >> 1) & 0x01
        metadata['line3_status'] = (packed_status >> 2) & 0x01
        metadata['line4_status'] = (packed_status >> 3) & 0x01

    def _map_spinnaker_line_status_all_to_display_bits(self, raw_status: int) -> int:
        """Map a Spinnaker LineStatusAll bitmask into the app's line1..line4 display slots."""
        packed_status = int(raw_status)
        selectors = self._list_enum_node_entries("LineSelector") or self._spinnaker_cached_line_selectors
        if not selectors:
            return packed_status

        selector_numbers = []
        for selector in selectors:
            match = re.search(r"(\d+)", str(selector))
            if match:
                selector_numbers.append(int(match.group(1)))
        if not selector_numbers:
            return packed_status

        zero_based = 0 in selector_numbers
        mapped_status = 0
        found_any = False

        for selector in selectors:
            match = re.search(r"(\d+)", str(selector))
            if not match:
                continue

            raw_index = int(match.group(1))
            line_number = raw_index + 1 if zero_based else raw_index
            if line_number < 1 or line_number > 4:
                continue

            raw_bit_index = raw_index + 1 if zero_based else raw_index - 1
            if raw_bit_index < 0:
                continue

            found_any = True
            if packed_status & (1 << raw_bit_index):
                mapped_status |= (1 << (line_number - 1))

        return mapped_status if found_any else packed_status

    def _read_spinnaker_live_line_status(self) -> Optional[int]:
        """Fallback to live node-map reads when Spinnaker chunk line data is unavailable."""
        if not self.is_spinnaker_camera() or self.camera is None:
            return None

        selectors = self._list_enum_node_entries("LineSelector") or self._spinnaker_cached_line_selectors
        if selectors:
            selector_numbers = []
            for selector in selectors:
                match = re.search(r"(\d+)", str(selector))
                if match:
                    selector_numbers.append(int(match.group(1)))
            zero_based = 0 in selector_numbers

            original_selector = self._read_enum_node_symbolic("LineSelector")
            packed_status = 0
            found_any = False

            try:
                for selector in selectors:
                    if not self._set_enum_node_by_name("LineSelector", selector):
                        continue

                    match = re.search(r"(\d+)", str(selector))
                    if not match:
                        continue

                    raw_index = int(match.group(1))
                    line_number = raw_index + 1 if zero_based else raw_index
                    if line_number < 1 or line_number > 4:
                        continue

                    line_state = self._read_numeric_node("LineStatus")
                    if line_state is None:
                        continue

                    found_any = True
                    if int(line_state):
                        packed_status |= (1 << (line_number - 1))
            finally:
                if original_selector:
                    self._set_enum_node_by_name("LineSelector", original_selector)

            if found_any:
                return packed_status

        direct_status = self._read_numeric_node("LineStatusAll")
        if direct_status is not None:
            return self._map_spinnaker_line_status_all_to_display_bits(int(direct_status))

        return None

    def _read_enum_node_symbolic(self, node_name: str) -> str:
        """Read the symbolic/current value of an enumeration node."""
        node = self._get_camera_node(node_name)
        if node is None or not self._node_is_readable(node):
            return ""

        try:
            current_entry = node.GetCurrentEntry()
            if current_entry is not None:
                try:
                    return str(current_entry.GetSymbolic()).strip()
                except Exception:
                    pass
                try:
                    return str(current_entry.ToString()).strip()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            value = node.GetValue()
            if isinstance(value, str):
                return value.strip()
        except Exception:
            pass
        return ""

    def _list_enum_node_entries(self, node_name: str) -> List[str]:
        """Return symbolic values exposed by an enumeration node."""
        node = self._get_camera_node(node_name)
        if node is None or not self._node_is_readable(node):
            return []

        entries: List[str] = []

        try:
            symbolics = node.GetSymbolics()
            if symbolics:
                entries = [str(value).strip() for value in symbolics if str(value).strip()]
        except Exception:
            entries = []

        if entries:
            return entries

        try:
            raw_entries = node.GetEntries()
        except Exception:
            raw_entries = []

        for entry in raw_entries:
            try:
                if self.is_spinnaker_camera() and PySpin is not None and not PySpin.IsReadable(entry):
                    continue
            except Exception:
                pass

            symbolic = ""
            try:
                symbolic = str(entry.GetSymbolic()).strip()
            except Exception:
                symbolic = ""
            if not symbolic:
                try:
                    symbolic = str(entry.ToString()).strip()
                except Exception:
                    symbolic = ""
            if symbolic:
                entries.append(symbolic)

        return entries

    def _select_enum_node_info(self, candidate_names: List[str]) -> Tuple[str, List[str], bool]:
        """Return the best enum node match from a list of candidate node names."""
        fallback_name = ""
        fallback_entries: List[str] = []
        fallback_writable = False

        for node_name in candidate_names:
            entries = self._list_enum_node_entries(node_name)
            if not entries:
                continue
            node = self._get_camera_node(node_name)
            writable = bool(
                node is not None and (
                    self._node_is_writable(node)
                    or (self.camera_type == "basler" and self.isRunning())
                )
            )
            if writable:
                return str(node_name), entries, True
            if not fallback_name:
                fallback_name = str(node_name)
                fallback_entries = list(entries)
                fallback_writable = writable

        return fallback_name, fallback_entries, fallback_writable

    def _is_supported_processing_pixel_format(self, pixel_format: str) -> bool:
        """Return True when the app can safely preview and record this raw camera format."""
        normalized = str(pixel_format or "").strip()
        if not normalized:
            return False
        return normalized.startswith(("Mono", "Bayer", "RGB", "BGR"))

    def _is_color_pixel_format(self, pixel_format: str) -> bool:
        """Return True for raw pixel formats that carry color information."""
        normalized = str(pixel_format or "").strip()
        return normalized.startswith(("Bayer", "RGB", "BGR"))

    def _get_camera_pixel_format_info(self) -> Dict[str, object]:
        """Describe the active camera's native pixel-format control."""
        if not self.is_genicam_camera():
            return {"node_name": "PixelFormat", "options": [], "current": "", "writable": False}

        options = [
            entry
            for entry in self._list_enum_node_entries("PixelFormat")
            if self._is_supported_processing_pixel_format(entry)
        ]
        current = self._read_enum_node_symbolic("PixelFormat")
        if current and current not in options and self._is_supported_processing_pixel_format(current):
            options.insert(0, current)

        node = self._get_camera_node("PixelFormat")
        writable = bool(
            node is not None and (
                self._node_is_writable(node)
                or (self.camera_type == "basler" and self.isRunning())
            )
        )
        return {
            "node_name": "PixelFormat",
            "options": options,
            "current": current,
            "writable": writable,
        }

    def _get_camera_bit_depth_info(self) -> Dict[str, object]:
        """Describe the active camera's bit-depth / ADC control when available."""
        if not self.is_genicam_camera():
            return {"node_name": "", "options": [], "current": "", "writable": False}

        node_name, options, writable = self._select_enum_node_info(
            ["AdcBitDepth", "TransferBitDepth", "SensorBitDepth", "BitDepth", "PixelSize"]
        )
        current = self._read_enum_node_symbolic(node_name) if node_name else ""
        if current and current not in options:
            options = [current] + list(options)
        return {
            "node_name": node_name,
            "options": options,
            "current": current,
            "writable": writable,
        }

    def get_camera_pixel_format_options(self) -> Dict[str, object]:
        """Expose camera-native pixel-format options for the advanced UI."""
        return self._get_camera_pixel_format_info()

    def get_camera_bit_depth_options(self) -> Dict[str, object]:
        """Expose camera-native bit-depth options for the advanced UI."""
        return self._get_camera_bit_depth_info()

    def set_camera_pixel_format(self, pixel_format: str) -> Optional[str]:
        """Set the active camera's native pixel format when supported."""
        pixel_format = str(pixel_format or "").strip()
        if not self.is_genicam_camera() or not pixel_format:
            return None

        info = self._get_camera_pixel_format_info()
        if pixel_format not in info.get("options", []):
            return None

        paused_for_reconfigure = False
        paused_backend = ""
        if self.is_spinnaker_camera() and self.isRunning():
            paused_for_reconfigure = self._pause_spinnaker_acquisition_for_reconfigure()
            paused_backend = "spinnaker"
        elif self.camera_type == "basler" and self.isRunning():
            paused_for_reconfigure = self._pause_basler_acquisition_for_reconfigure()
            paused_backend = "basler"
        try:
            if not self._set_enum_node_by_name("PixelFormat", pixel_format):
                return None

            current_pixel_format = self._read_enum_node_symbolic("PixelFormat")
            if current_pixel_format:
                self.spinnaker_native_pixel_format = current_pixel_format
            updated_color_filter = self._read_enum_node_symbolic("PixelColorFilter")
            if updated_color_filter:
                self.spinnaker_color_filter = updated_color_filter
            self.spinnaker_is_color = self.spinnaker_is_color or self._is_color_pixel_format(current_pixel_format)
            self._refresh_camera_settings_cache(force=True)
            return current_pixel_format or pixel_format
        finally:
            if paused_for_reconfigure and paused_backend == "spinnaker":
                self._resume_spinnaker_acquisition_after_reconfigure()
            elif paused_for_reconfigure and paused_backend == "basler":
                self._resume_basler_acquisition_after_reconfigure()

    def set_camera_bit_depth(self, bit_depth: str) -> Optional[str]:
        """Set the active camera's bit-depth / ADC mode when supported."""
        bit_depth = str(bit_depth or "").strip()
        if not self.is_genicam_camera() or not bit_depth:
            return None

        info = self._get_camera_bit_depth_info()
        node_name = str(info.get("node_name", "") or "")
        if not node_name or bit_depth not in info.get("options", []):
            return None

        paused_for_reconfigure = False
        paused_backend = ""
        if self.is_spinnaker_camera() and self.isRunning():
            paused_for_reconfigure = self._pause_spinnaker_acquisition_for_reconfigure()
            paused_backend = "spinnaker"
        elif self.camera_type == "basler" and self.isRunning():
            paused_for_reconfigure = self._pause_basler_acquisition_for_reconfigure()
            paused_backend = "basler"
        try:
            if not self._set_enum_node_by_name(node_name, bit_depth):
                return None
            self._refresh_camera_settings_cache(force=True)
            return self._read_enum_node_symbolic(node_name) or bit_depth
        finally:
            if paused_for_reconfigure and paused_backend == "spinnaker":
                self._resume_spinnaker_acquisition_after_reconfigure()
            elif paused_for_reconfigure and paused_backend == "basler":
                self._resume_basler_acquisition_after_reconfigure()

    def get_camera_line_capabilities(self) -> List[Dict[str, object]]:
        """Enumerate GenICam camera line selectors, modes, and sources."""
        if not self.is_genicam_camera():
            return []

        # For Spinnaker cameras the nodemap is often locked while connected.
        # Attempt a direct read first; otherwise use the cache populated at connect time.
        if self.is_spinnaker_camera():
            fresh = self._read_line_capabilities_direct()
            if fresh:
                self._cached_line_capabilities = fresh
                return fresh
            return list(self._cached_line_capabilities)

        selectors = self._list_enum_node_entries("LineSelector")
        if not selectors:
            return []

        original_selector = self._read_enum_node_symbolic("LineSelector")
        capabilities: List[Dict[str, object]] = []

        try:
            for selector in selectors:
                if not self._set_enum_node_by_name("LineSelector", selector):
                    continue

                modes = self._list_enum_node_entries("LineMode")
                sources = self._list_enum_node_entries("LineSource")
                capabilities.append({
                    "selector": str(selector),
                    "mode": self._read_enum_node_symbolic("LineMode"),
                    "mode_options": modes,
                    "source": self._read_enum_node_symbolic("LineSource"),
                    "source_options": sources,
                })
        finally:
            if original_selector:
                self._set_enum_node_by_name("LineSelector", original_selector)

        return capabilities

    def apply_camera_line_configuration(self, line_configs: List[Dict[str, object]]) -> List[str]:
        """Apply per-line GenICam mode/source settings when supported by the camera."""
        if not self.is_genicam_camera():
            return []

        applied: List[str] = []
        original_selector = self._read_enum_node_symbolic("LineSelector")
        paused_for_reconfigure = False
        if self.is_spinnaker_camera() and self.isRunning():
            paused_for_reconfigure = self._pause_spinnaker_acquisition_for_reconfigure()
        try:
            for config in line_configs or []:
                selector = str(config.get("selector", "")).strip()
                if not selector or not self._set_enum_node_by_name("LineSelector", selector):
                    continue

                mode = str(config.get("mode", "")).strip()
                source = str(config.get("source", "")).strip()

                if mode:
                    self._set_enum_node_by_name("LineMode", mode)
                if source:
                    self._set_enum_node_by_name("LineSource", source)
                applied.append(selector)

            # Refresh capabilities cache while the nodemap is still writable.
            if self.is_spinnaker_camera():
                refreshed = self._read_line_capabilities_direct()
                if refreshed:
                    self._cached_line_capabilities = refreshed
                    self._spinnaker_cached_line_selectors = [c["selector"] for c in refreshed]
        finally:
            if original_selector:
                self._set_enum_node_by_name("LineSelector", original_selector)
            if paused_for_reconfigure:
                self._resume_spinnaker_acquisition_after_reconfigure()

        return applied

    def _clamp_numeric_node_value(self, node_name: str, value: float, integer: bool = False):
        """Clamp a value to the min/max/inc limits exposed by a camera node."""
        node = self._get_camera_node(node_name)
        if node is None or not self._node_is_writable(node):
            return None
        try:
            min_val = float(node.GetMin())
            max_val = float(node.GetMax())
        except Exception:
            return None

        clamped = max(min_val, min(max_val, float(value)))
        try:
            inc = float(node.GetInc())
        except Exception:
            inc = 1.0 if integer else 0.0

        if integer:
            if inc > 1:
                clamped = min_val + (int((clamped - min_val) // inc) * inc)
            return int(clamped)

        if inc and inc > 0:
            steps = round((clamped - min_val) / inc)
            clamped = min_val + (steps * inc)
            clamped = max(min_val, min(max_val, clamped))
        return float(clamped)

    def _write_numeric_node(self, node_name: str, value: float, integer: bool = False):
        """Write a numeric node after clamping to the allowed range."""
        node = self._get_camera_node(node_name)
        if node is None or not self._node_is_writable(node):
            return None
        clamped = self._clamp_numeric_node_value(node_name, value, integer=integer)
        if clamped is None:
            return None
        try:
            node.SetValue(int(clamped) if integer else float(clamped))
        except Exception:
            return None
        return clamped

    def _set_bool_node(self, node_name: str, value: bool) -> bool:
        """Set a boolean node when available."""
        node = self._get_camera_node(node_name)
        if node is None or not self._node_is_writable(node):
            return False
        try:
            node.SetValue(bool(value))
            return True
        except Exception:
            return False

    def sync_camera_fps(self):
        """Read and cache the camera's reported FPS if available."""
        if self.camera_type == "basler":
            if not self.camera or not self.camera.IsOpen():
                self.camera_reported_fps = None
                return None
            fps = self._read_camera_fps()
            if fps:
                self.camera_reported_fps = fps
                return fps
        elif self.is_spinnaker_camera():
            fps = self._read_camera_fps()
            if fps:
                self.camera_reported_fps = fps
                return fps
        elif self.camera_type == "usb" and self.usb_capture:
            usb_fps = self.usb_capture.get(cv2.CAP_PROP_FPS)
            if usb_fps and usb_fps > 0:
                self.camera_reported_fps = float(usb_fps)
                return self.camera_reported_fps
        elif self.camera_type == "flir":
            flir_fps = self._read_flir_camera_fps()
            if flir_fps and flir_fps > 0:
                self.camera_reported_fps = float(flir_fps)
                return self.camera_reported_fps
        self.camera_reported_fps = None
        return None

    def set_camera_frame_rate(self, fps: float) -> Optional[float]:
        """Set camera frame rate through the active backend when supported."""
        if self.camera_type == "usb" and self.usb_capture:
            self.usb_capture.set(cv2.CAP_PROP_FPS, float(fps))
            self.set_target_fps(fps)
            return self.sync_camera_fps()

        if not self.is_genicam_camera():
            self.set_target_fps(fps)
            return None

        if self.is_spinnaker_camera():
            self._set_enum_node_by_name("AcquisitionFrameRateAuto", "Off")

        self._fit_exposure_to_requested_fps(fps)
        self._set_bool_node("AcquisitionFrameRateEnable", True)
        applied = self._write_numeric_node("AcquisitionFrameRate", fps, integer=False)
        if applied is None:
            applied = self._write_numeric_node("AcquisitionFrameRateAbs", fps, integer=False)
        self.set_target_fps(float(applied if applied is not None else fps))
        return self.sync_camera_fps()

    def set_camera_exposure_ms(self, exposure_ms: float) -> Optional[float]:
        """Set manual exposure in milliseconds when the backend supports it."""
        if not self.is_genicam_camera():
            return None

        exposure_us = max(1.0, float(exposure_ms) * 1000.0)
        self._set_enum_node_by_name("ExposureAuto", "Off")
        self._set_enum_node_by_name("ExposureMode", "Timed")
        applied = self._write_numeric_node("ExposureTime", exposure_us, integer=False)
        if applied is None:
            applied = self._write_numeric_node("ExposureTimeAbs", exposure_us, integer=False)

        if applied is not None:
            self.camera_settings_cache["exposure_time_us"] = float(applied)
            self.camera_settings_cache_time = time.time()
            return float(applied) / 1000.0
        return None

    def set_camera_gain(self, gain_db: float) -> Optional[float]:
        """Set manual gain in dB when the backend supports it."""
        if not self.is_genicam_camera():
            return None
        self._set_enum_node_by_name("GainAuto", "Off")
        applied = self._write_numeric_node("Gain", gain_db, integer=False)
        if applied is not None:
            self.camera_settings_cache["gain_db"] = float(applied)
            self.camera_settings_cache_time = time.time()
        return float(applied) if applied is not None else None

    def get_camera_exposure_ms(self) -> Optional[float]:
        """Read the current manual exposure time in milliseconds."""
        exposure_us = self._read_numeric_node("ExposureTime")
        if exposure_us is None:
            exposure_us = self._read_numeric_node("ExposureTimeAbs")
        if exposure_us is None:
            return None
        return float(exposure_us) / 1000.0

    def _fit_exposure_to_requested_fps(self, fps: float) -> Optional[float]:
        """Clamp exposure so the requested frame rate remains physically achievable."""
        if not self.is_genicam_camera():
            return None
        fps = float(fps)
        if fps <= 0:
            return None

        current_exposure_ms = self.get_camera_exposure_ms()
        if current_exposure_ms is None:
            return None

        safe_exposure_us = max(1.0, (1_000_000.0 / fps) * 0.92)
        current_exposure_us = current_exposure_ms * 1000.0
        if current_exposure_us <= safe_exposure_us:
            return current_exposure_ms

        self._set_enum_node_by_name("ExposureAuto", "Off")
        self._set_enum_node_by_name("ExposureMode", "Timed")
        applied = self._write_numeric_node("ExposureTime", safe_exposure_us, integer=False)
        if applied is None:
            applied = self._write_numeric_node("ExposureTimeAbs", safe_exposure_us, integer=False)
        if applied is None:
            return current_exposure_ms

        applied_ms = float(applied) / 1000.0
        self.camera_settings_cache["exposure_time_us"] = float(applied)
        self.camera_settings_cache_time = time.time()
        return applied_ms

    def set_camera_white_balance_auto(self, mode: str) -> bool:
        """Set white-balance auto mode when supported."""
        if not self.is_genicam_camera():
            return False
        return self._set_enum_node_by_name("BalanceWhiteAuto", str(mode).strip())

    def get_camera_white_balance_ratio(self, selector: str) -> Optional[float]:
        """Read one white-balance ratio channel."""
        if not self.is_genicam_camera():
            return None
        if not self._set_enum_node_by_name("BalanceRatioSelector", str(selector).strip()):
            return None
        ratio = self._read_numeric_node("BalanceRatio")
        return float(ratio) if ratio is not None else None

    def set_camera_white_balance_ratio(self, selector: str, ratio: float) -> Optional[float]:
        """Set one manual white-balance ratio channel."""
        if not self.is_genicam_camera():
            return None
        self._set_enum_node_by_name("BalanceWhiteAuto", "Off")
        if not self._set_enum_node_by_name("BalanceRatioSelector", str(selector).strip()):
            return None
        applied = self._write_numeric_node("BalanceRatio", ratio, integer=False)
        return float(applied) if applied is not None else None

    def _pause_spinnaker_acquisition_for_reconfigure(self, timeout_s: float = 3.0) -> bool:
        """Ask the acquisition loop to pause streaming so config changes can be applied safely."""
        if not self.is_spinnaker_camera() or not self.isRunning():
            return True
        self.spinnaker_pause_requested = True
        deadline = time.time() + max(0.1, float(timeout_s))
        while time.time() < deadline:
            if self.spinnaker_paused:
                return True
            time.sleep(0.01)
        return False

    def _resume_spinnaker_acquisition_after_reconfigure(self, timeout_s: float = 3.0) -> bool:
        """Resume a paused Spinnaker acquisition loop."""
        if not self.is_spinnaker_camera() or not self.isRunning():
            self.spinnaker_pause_requested = False
            self.spinnaker_paused = False
            return True
        self.spinnaker_pause_requested = False
        deadline = time.time() + max(0.1, float(timeout_s))
        while time.time() < deadline:
            if not self.spinnaker_paused:
                return True
            time.sleep(0.01)
        return False

    def _pause_basler_acquisition_for_reconfigure(self, timeout_s: float = 3.0) -> bool:
        """Ask the Basler acquisition loop to pause grabbing for safe reconfiguration."""
        if self.camera_type != "basler" or not self.isRunning():
            return True
        self.basler_pause_requested = True
        deadline = time.time() + max(0.1, float(timeout_s))
        while time.time() < deadline:
            if self.basler_paused:
                return True
            time.sleep(0.01)
        return False

    def _resume_basler_acquisition_after_reconfigure(self, timeout_s: float = 3.0) -> bool:
        """Resume a paused Basler acquisition loop."""
        if self.camera_type != "basler" or not self.isRunning():
            self.basler_pause_requested = False
            self.basler_paused = False
            return True
        self.basler_pause_requested = False
        deadline = time.time() + max(0.1, float(timeout_s))
        while time.time() < deadline:
            if not self.basler_paused:
                return True
            time.sleep(0.01)
        return False

    def set_camera_resolution(self, width: int, height: int) -> Optional[Tuple[int, int]]:
        """Set Width/Height on GenICam-style cameras, resetting offsets first."""
        if not self.is_genicam_camera():
            return None

        paused_for_reconfigure = False
        if self.is_spinnaker_camera() and self.isRunning():
            paused_for_reconfigure = self._pause_spinnaker_acquisition_for_reconfigure()
            if not paused_for_reconfigure:
                raise RuntimeError("Timed out while pausing FLIR acquisition for resolution change")

        if self.is_spinnaker_camera() and self.spinnaker_is_color:
            width_node = self._get_camera_node("Width")
            height_node = self._get_camera_node("Height")
            current_width_max = None
            current_height_max = None
            try:
                if width_node is not None:
                    current_width_max = int(width_node.GetMax())
                if height_node is not None:
                    current_height_max = int(height_node.GetMax())
            except Exception:
                current_width_max = None
                current_height_max = None

            if (
                (current_width_max is not None and int(width) > current_width_max)
                or (current_height_max is not None and int(height) > current_height_max)
            ):
                self._ensure_spinnaker_full_resolution_color_mode()

        was_streaming = False
        try:
            if self.is_spinnaker_camera() and not paused_for_reconfigure:
                try:
                    was_streaming = bool(self.camera.IsStreaming())
                except Exception:
                    was_streaming = False
                if was_streaming:
                    self.camera.EndAcquisition()
            else:
                try:
                    was_streaming = bool(self.camera.IsGrabbing())
                except Exception:
                    was_streaming = False
                if was_streaming:
                    self.camera.StopGrabbing()

            for node_name in ("OffsetX", "OffsetY"):
                node = self._get_camera_node(node_name)
                if node is not None and self._node_is_writable(node):
                    try:
                        node.SetValue(int(node.GetMin()))
                    except Exception:
                        pass

            applied_width = self._write_numeric_node("Width", width, integer=True)
            applied_height = self._write_numeric_node("Height", height, integer=True)
            if applied_width is None or applied_height is None:
                raise RuntimeError("Width/Height not supported by camera")

            # Centre a sub-sensor ROI so a Full HD crop frames the middle of the
            # arena instead of the top-left corner.
            self._center_camera_roi(int(applied_width), int(applied_height))

            self.update_resolution(int(applied_width), int(applied_height))
            return int(applied_width), int(applied_height)
        finally:
            try:
                if self.is_spinnaker_camera() and paused_for_reconfigure:
                    self._resume_spinnaker_acquisition_after_reconfigure()
                elif was_streaming:
                    if self.is_spinnaker_camera():
                        self.camera.BeginAcquisition()
                    else:
                        self.camera.StartGrabbing(self._get_basler_grab_strategy())
            except Exception:
                pass

    def _center_camera_roi(self, width: int, height: int) -> None:
        """Centre the active ROI on the sensor by adjusting OffsetX/OffsetY.

        Called right after Width/Height are applied (acquisition already paused
        by the caller), so it writes the offset nodes directly without touching
        the streaming state.
        """
        if not self.is_genicam_camera():
            return
        for node_name, span in (("OffsetX", int(width)), ("OffsetY", int(height))):
            try:
                node = self._get_camera_node(node_name)
                if node is None or not self._node_is_writable(node):
                    continue
                try:
                    sensor_max = int(node.GetMax()) + int(node.GetValue())
                except Exception:
                    # GetMax already reflects (sensor - current size); centre on it.
                    sensor_max = int(node.GetMax())
                inc = 1
                try:
                    inc = max(1, int(node.GetInc()))
                except Exception:
                    inc = 1
                offset = max(0, (sensor_max - span) // 2)
                offset = (offset // inc) * inc
                try:
                    offset = min(offset, int(node.GetMax()))
                except Exception:
                    pass
                self._write_numeric_node(node_name, int(offset), integer=True)
            except Exception:
                continue

    def set_camera_offset(self, node_name: str, value: int) -> Optional[int]:
        """Set OffsetX/OffsetY safely on GenICam cameras."""
        node_name = str(node_name or "").strip()
        if node_name not in {"OffsetX", "OffsetY"} or not self.is_genicam_camera():
            return None

        paused_for_spinnaker = False
        paused_for_basler = False
        if self.is_spinnaker_camera() and self.isRunning():
            paused_for_spinnaker = self._pause_spinnaker_acquisition_for_reconfigure()
            if not paused_for_spinnaker:
                raise RuntimeError(f"Timed out while pausing FLIR acquisition for {node_name}")
        elif self.camera_type == "basler" and self.isRunning():
            paused_for_basler = self._pause_basler_acquisition_for_reconfigure()
            if not paused_for_basler:
                raise RuntimeError(f"Timed out while pausing Basler acquisition for {node_name}")

        try:
            applied = self._write_numeric_node(node_name, int(value), integer=True)
            if applied is None:
                return None
            self.camera_settings_cache[node_name] = int(applied)
            self.camera_settings_cache_time = time.time()
            return int(applied)
        finally:
            if paused_for_spinnaker:
                self._resume_spinnaker_acquisition_after_reconfigure()
            if paused_for_basler:
                self._resume_basler_acquisition_after_reconfigure()

    def connect_camera(self, camera_info: Optional[dict] = None) -> bool:
        """Connect to a Basler, FLIR, or generic USB camera."""
        try:
            camera_info = camera_info or {"type": "basler", "index": 0}
            camera_type = camera_info.get("type")

            if camera_type == "usb":
                index = int(camera_info.get("index", 0))
                preferred_backend = str(camera_info.get("cv2_backend", "") or "")
                capture, backend_name = self._open_usb_capture(index, preferred_backend)
                if capture is None:
                    self.error_occurred.emit("No USB camera found!")
                    return False
                self.usb_capture = capture
                self.usb_backend = backend_name

                self.camera_type = "usb"
                self.usb_index = index
                self._request_usb_mjpg(self.usb_capture)
                self.usb_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width or 1080)
                self.usb_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height or 1080)
                self.usb_capture.set(cv2.CAP_PROP_FPS, self.fps_target)
                self._configure_usb_color_controls(self.usb_capture)
                try:
                    self.usb_capture.set(cv2.CAP_PROP_BUFFERSIZE, self.processing_queue_max_frames)
                except Exception:
                    pass

                self.width = int(self.usb_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.usb_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                usb_fps = self.usb_capture.get(cv2.CAP_PROP_FPS)
                if usb_fps and usb_fps > 0:
                    self.camera_reported_fps = float(usb_fps)

                self.status_update.emit(f"USB camera connected: {self.width}x{self.height}")
                return True

            if camera_type == "flir":
                return self._connect_flir_camera(camera_info)

            if not PYPYLON_AVAILABLE or pylon is None:
                self.error_occurred.emit("Basler support is unavailable: pypylon / Pylon SDK is not installed.")
                return False

            # Get Basler camera
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()

            if len(devices) == 0:
                self.error_occurred.emit("No Basler camera found!")
                return False

            index = int(camera_info.get("index", 0))
            index = max(0, min(index, len(devices) - 1))
            selected_device = devices[index]
            try:
                self.basler_device_class = str(selected_device.GetDeviceClass() or "")
            except Exception:
                self.basler_device_class = ""

            try:
                self.camera = pylon.InstantCamera(tlFactory.CreateDevice(selected_device))
                self.camera.Open()
            except Exception as exc:
                self.camera = None
                self.error_occurred.emit(self._format_basler_open_error(selected_device, exc))
                return False
            self.camera_type = "basler"

            self._configure_basler_stream_buffers()

            # Apply image format
            self.set_image_format(self.image_format)

            # Get camera resolution
            self.width = self.camera.Width.GetValue()
            self.height = self.camera.Height.GetValue()

            # Enable chunk mode for metadata
            self.camera.ChunkModeActive.SetValue(True)
            self.camera.ChunkSelector.SetValue('Timestamp')
            self.camera.ChunkEnable.SetValue(True)
            self.camera.ChunkSelector.SetValue('ExposureTime')
            self.camera.ChunkEnable.SetValue(True)

            # Try to enable line status
            try:
                self.camera.ChunkSelector.SetValue('LineStatusAll')
                self.camera.ChunkEnable.SetValue(True)
            except:
                pass  # Not all cameras support this

            if self._is_basler_gige_camera():
                try:
                    packet_size = int(CAMERA_CONFIG.get("gige_packet_size", 1500))
                    if hasattr(self.camera, "GevSCPSPacketSize"):
                        packet_size = max(
                            int(self.camera.GevSCPSPacketSize.GetMin()),
                            min(int(self.camera.GevSCPSPacketSize.GetMax()), packet_size),
                        )
                        self.camera.GevSCPSPacketSize.SetValue(packet_size)
                except Exception:
                    pass
                effective_capacity = self._get_effective_processing_queue_capacity()
                self.status_update.emit(
                    f"Basler GigE camera connected: {self.width}x{self.height} "
                    f"(live queue capped at {effective_capacity} frames)"
                )
                return True

            self.status_update.emit(f"Basler camera connected: {self.width}x{self.height}")
            return True

        except Exception as e:
            self.error_occurred.emit(f"Camera connection error: {str(e)}")
            return False

    def _format_basler_open_error(self, selected_device: Any, exc: Exception) -> str:
        """Translate common Basler open failures into actionable user-facing text."""
        model = ""
        serial = ""
        try:
            model = str(selected_device.GetModelName() or "").strip()
        except Exception:
            pass
        try:
            serial = str(selected_device.GetSerialNumber() or "").strip()
        except Exception:
            pass

        label = model or "camera"
        if serial:
            label = f"{label} ({serial})"

        error_text = str(exc).strip()
        lowered_error = error_text.lower()
        if (
            "exclusively opened by another client" in lowered_error
            or "access is denied" in lowered_error
            or "device is already open" in lowered_error
        ):
            return (
                f"Basler {label} is detected but currently in use by another application. "
                "Close Pylon Viewer or any other camera app, then retry."
            )
        return f"Camera connection error: {error_text}"

    def _connect_flir_camera(self, camera_info: Dict) -> bool:
        """Connect to a FLIR camera through the selected backend."""
        backend = str(camera_info.get("backend", "")).strip().lower()
        video_index_raw = camera_info.get("video_index", camera_info.get("index", None))
        serial_port = camera_info.get("serial_port", None)

        try:
            if backend == "spinnaker":
                return self._connect_flir_spinnaker_camera(camera_info)
            if backend == "boson":
                if Boson is None:
                    self.error_occurred.emit("FLIR Boson support is unavailable: flirpy is not installed.")
                    return False
                self.flir_camera = Boson(port=serial_port)
                self.flir_camera.setup_video(device_id=video_index_raw)
            elif backend == "lepton":
                if Lepton is None:
                    self.error_occurred.emit("FLIR Lepton support is unavailable: flirpy is not installed.")
                    return False
                self.flir_camera = Lepton()
                self.flir_camera.setup_video(device_id=video_index_raw)
            elif backend == "teax":
                if TeaxGrabber is None:
                    self.error_occurred.emit("FLIR Tau / TeAx support is unavailable: flirpy Teax dependencies are not installed.")
                    return False
                self.flir_camera = TeaxGrabber()
            else:
                self.error_occurred.emit(f"Unsupported FLIR backend: {backend or 'unknown'}")
                return False

            self.camera_type = "flir"
            self.flir_backend = backend
            self.flir_video_index = int(video_index_raw) if video_index_raw is not None else None
            self.flir_serial_port = str(serial_port) if serial_port else None
            self.flir_status_cache = {}
            self.flir_status_cache_time = 0.0
            self.camera_settings_cache = {}
            self.camera_settings_cache_time = 0.0
            self.width, self.height = self._read_flir_frame_dimensions()
            self.camera_reported_fps = self._read_flir_camera_fps()
            self.status_update.emit(
                f"FLIR {backend.upper()} connected: {self.width}x{self.height}"
            )
            return True
        except Exception as e:
            self._close_flir_camera()
            self.camera_type = None
            self.flir_backend = None
            self.error_occurred.emit(f"FLIR connection error: {str(e)}")
            return False

    def _connect_flir_spinnaker_camera(self, camera_info: Dict) -> bool:
        """Connect to a FLIR machine-vision camera through Spinnaker/PySpin."""
        if not PYSPIN_AVAILABLE or PySpin is None:
            self.error_occurred.emit("FLIR Spinnaker support is unavailable: install the Spinnaker SDK and PySpin.")
            return False

        index = int(camera_info.get("index", 0))
        try:
            self.pyspin_system = PySpin.System.GetInstance()
            self.pyspin_cam_list = self._get_spinnaker_cameras_with_retry()
            if self.pyspin_cam_list.GetSize() == 0:
                raise RuntimeError("No FLIR Spinnaker camera found")

            index = self._select_spinnaker_camera_index(camera_info, index)
            self.camera = self.pyspin_cam_list.GetByIndex(index)
            self.camera.Init()
            self.camera_type = "flir"
            self.flir_backend = "spinnaker"
            self.pyspin_image_processor = PySpin.ImageProcessor()
            self.pyspin_image_processor.SetColorProcessing(
                PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR
            )
            self.flir_camera = None
            self.flir_video_index = None
            self.flir_serial_port = None
            self.flir_status_cache = {}
            self.flir_status_cache_time = 0.0
            self.camera_settings_cache = {}
            self.camera_settings_cache_time = 0.0

            self._configure_spinnaker_stream()
            self._configure_spinnaker_chunks()
            self.spinnaker_native_pixel_format = self._read_enum_node_symbolic("PixelFormat")
            self.spinnaker_color_filter = self._read_enum_node_symbolic("PixelColorFilter")
            self.spinnaker_is_color = self._infer_spinnaker_is_color_camera(
                self.spinnaker_native_pixel_format,
                self.spinnaker_color_filter,
            )
            self.set_image_format(self.image_format)

            width = self._read_numeric_node("Width")
            height = self._read_numeric_node("Height")
            self.width = int(width or 0)
            self.height = int(height or 0)
            self.camera_reported_fps = self._read_camera_fps()
            self._refresh_camera_settings_cache(force=True)

            self.status_update.emit(f"FLIR SPINNAKER connected: {self.width}x{self.height}")
            return True
        except Exception as e:
            self._release_spinnaker_camera()
            self.camera_type = None
            self.flir_backend = None
            self.error_occurred.emit(f"FLIR Spinnaker connection error: {str(e)}")
            return False

    def _get_spinnaker_cameras_with_retry(self):
        """Return a PySpin camera list, retrying transient empty enumerations."""
        cam_list = None
        for attempt in range(3):
            if cam_list is not None:
                try:
                    cam_list.Clear()
                except Exception:
                    pass
                cam_list = None

            try:
                self.pyspin_system.UpdateCameras(True)
            except TypeError:
                try:
                    self.pyspin_system.UpdateCameras()
                except Exception:
                    pass
            except Exception:
                pass

            try:
                cam_list = self.pyspin_system.GetCameras(True, True)
            except TypeError:
                cam_list = self.pyspin_system.GetCameras()

            if int(cam_list.GetSize()) > 0 or attempt == 2:
                return cam_list
            time.sleep(0.25)
        return cam_list

    def _select_spinnaker_camera_index(self, camera_info: Dict, fallback_index: int) -> int:
        """Prefer serial matching over the scan-time index, which can change."""
        camera_count = int(self.pyspin_cam_list.GetSize())
        selected_index = max(0, min(int(fallback_index), camera_count - 1))
        target_serial = str(camera_info.get("serial", "") or "").strip()
        if not target_serial:
            return selected_index

        for candidate_index in range(camera_count):
            candidate = self.pyspin_cam_list.GetByIndex(candidate_index)
            try:
                candidate_serial = _read_pyspin_string_node(
                    candidate.GetTLDeviceNodeMap(),
                    "DeviceSerialNumber",
                )
            finally:
                del candidate
            if candidate_serial == target_serial:
                return candidate_index
        return selected_index

    def _configure_spinnaker_stream(self):
        """Tune the Spinnaker stream for stable continuous acquisition."""
        if not self.is_spinnaker_camera() or PySpin is None:
            return
        try:
            stream_map = self.camera.GetTLStreamNodeMap()
        except Exception:
            return

        try:
            mode_node = PySpin.CEnumerationPtr(stream_map.GetNode("StreamBufferHandlingMode"))
            if PySpin.IsAvailable(mode_node) and PySpin.IsWritable(mode_node):
                mode_entry = mode_node.GetEntryByName("OldestFirst")
                if not (PySpin.IsAvailable(mode_entry) and PySpin.IsReadable(mode_entry)):
                    mode_entry = mode_node.GetEntryByName("NewestOnly")
                if PySpin.IsAvailable(mode_entry) and PySpin.IsReadable(mode_entry):
                    mode_node.SetIntValue(mode_entry.GetValue())
        except Exception:
            pass

        try:
            count_mode = PySpin.CEnumerationPtr(stream_map.GetNode("StreamBufferCountMode"))
            if PySpin.IsAvailable(count_mode) and PySpin.IsWritable(count_mode):
                manual = count_mode.GetEntryByName("Manual")
                if PySpin.IsAvailable(manual) and PySpin.IsReadable(manual):
                    count_mode.SetIntValue(manual.GetValue())
        except Exception:
            pass

        try:
            buffer_count = PySpin.CIntegerPtr(stream_map.GetNode("StreamBufferCountManual"))
            if PySpin.IsAvailable(buffer_count) and PySpin.IsWritable(buffer_count):
                target = max(
                    int(buffer_count.GetMin()),
                    min(int(buffer_count.GetMax()), int(self.stream_buffer_target)),
                )
                buffer_count.SetValue(target)
        except Exception:
            pass

    def _configure_spinnaker_chunks(self):
        """Enable useful chunk data on Spinnaker cameras when available."""
        if not self.is_spinnaker_camera():
            return
        try:
            chunk_mode = self._get_camera_node("ChunkModeActive")
            if chunk_mode is not None and self._node_is_writable(chunk_mode):
                try:
                    chunk_mode.SetValue(True)
                except Exception:
                    pass
            for selector in ("Timestamp", "ExposureTime", "FrameID", "LineStatusAll"):
                if not self._set_enum_node_by_name("ChunkSelector", selector):
                    continue
                enable_node = self._get_camera_node("ChunkEnable")
                if enable_node is not None and self._node_is_writable(enable_node):
                    try:
                        enable_node.SetValue(True)
                    except Exception:
                        pass
        except Exception:
            pass

        # Cache line selector entries now, before streaming locks the nodemap.
        self._spinnaker_cached_line_selectors = self._list_enum_node_entries("LineSelector") or []
        # Cache full capabilities (mode/source per line) before BeginAcquisition locks the nodemap.
        self._cached_line_capabilities = self._read_line_capabilities_direct() or []

    def _read_line_capabilities_direct(self) -> List[Dict[str, object]]:
        """Read GenICam line capabilities directly from the nodemap (call before acquisition starts)."""
        selectors = self._list_enum_node_entries("LineSelector")
        if not selectors:
            return []
        original_selector = self._read_enum_node_symbolic("LineSelector")
        capabilities: List[Dict[str, object]] = []
        try:
            for selector in selectors:
                if not self._set_enum_node_by_name("LineSelector", selector):
                    continue
                modes = self._list_enum_node_entries("LineMode")
                sources = self._list_enum_node_entries("LineSource")
                capabilities.append({
                    "selector": str(selector),
                    "mode": self._read_enum_node_symbolic("LineMode"),
                    "mode_options": modes,
                    "source": self._read_enum_node_symbolic("LineSource"),
                    "source_options": sources,
                })
        finally:
            if original_selector:
                self._set_enum_node_by_name("LineSelector", original_selector)
        return capabilities

    def _infer_spinnaker_is_color_camera(self, pixel_format: str, color_filter: str) -> bool:
        """Infer whether the current Spinnaker camera is color-capable."""
        pixel_format = str(pixel_format or "")
        color_filter = str(color_filter or "")
        if "Bayer" in pixel_format or "RGB" in pixel_format or "BGR" in pixel_format:
            return True
        if "Bayer" in color_filter:
            return True
        return False

    def _get_preferred_spinnaker_color_pixel_formats(self) -> List[str]:
        """Ordered list of color pixel formats that preserve full sensor resolution."""
        formats: List[str] = []
        native = str(self.spinnaker_native_pixel_format or "")
        current = self._read_enum_node_symbolic("PixelFormat")

        # Prefer native Bayer formats first so the app can debayer on the host
        # and keep access to the full-resolution sensor geometry.
        for candidate in (
            native,
            current,
            "BayerRG8",
            "BayerBG8",
            "BayerGB8",
            "BayerGR8",
            "RGB8",
            "BGR8",
        ):
            candidate = str(candidate or "").strip()
            if not candidate or candidate in formats:
                continue
            if self.spinnaker_is_color and ("Bayer" not in candidate) and candidate in {native, current, "RGB8", "BGR8"}:
                # Skip processed RGB/BGR as the top priority if we know the
                # camera is color-capable; Bayer paths preserve full resolution.
                continue
            formats.append(candidate)

        if not formats and self.spinnaker_is_color:
            formats.extend(["BayerRG8", "BayerBG8", "BayerGB8", "BayerGR8", "RGB8", "BGR8"])
        return formats

    def _ensure_spinnaker_full_resolution_color_mode(self) -> bool:
        """Switch a color Spinnaker camera into raw full-sensor mode when possible."""
        if not self.is_spinnaker_camera() or not self.spinnaker_is_color:
            return False

        changed = False
        current = self._read_enum_node_symbolic("PixelFormat")

        isp_enabled = self._read_numeric_node("IspEnable")
        if isp_enabled is not None and bool(isp_enabled):
            changed = self._set_bool_node("IspEnable", False) or changed

        for node_name in ("BinningHorizontal", "BinningVertical", "DecimationHorizontal", "DecimationVertical"):
            current_value = self._read_numeric_node(node_name)
            if current_value is not None and float(current_value) > 1.0:
                applied = self._write_numeric_node(node_name, 1, integer=True)
                changed = (applied is not None and int(applied) == 1) or changed

        for candidate in self._get_preferred_spinnaker_color_pixel_formats():
            if "Bayer" not in candidate:
                continue
            if "Bayer" in current and candidate == current:
                break
            if self._set_enum_node_by_name("PixelFormat", candidate):
                self.spinnaker_native_pixel_format = candidate
                self.spinnaker_color_filter = self._read_enum_node_symbolic("PixelColorFilter") or self.spinnaker_color_filter
                changed = True
                self._refresh_camera_settings_cache(force=True)
                break

        if changed:
            width_max = self._read_numeric_node("Width")
            height_max = self._read_numeric_node("Height")
            self.status_update.emit(
                "FLIR Spinnaker raw full-sensor mode enabled"
                + (
                    f": max {int(width_max)}x{int(height_max)}"
                    if width_max is not None and height_max is not None
                    else ""
                )
            )
        return changed

    def _close_flir_camera(self):
        """Release any active flirpy camera object."""
        if not self.flir_camera:
            return

        close_methods = ("close", "release", "disconnect")
        for method_name in close_methods:
            method = getattr(self.flir_camera, method_name, None)
            if not callable(method):
                continue
            try:
                method()
            except Exception:
                pass
        self.flir_camera = None

    def _release_spinnaker_camera(self):
        """Release Spinnaker camera, list, and system resources."""
        if self.camera is not None:
            try:
                if hasattr(self.camera, "IsStreaming") and self.camera.IsStreaming():
                    self.camera.EndAcquisition()
            except Exception:
                pass
            try:
                self.camera.DeInit()
            except Exception:
                pass
            self.camera = None
        self.pyspin_image_processor = None

        if self.pyspin_cam_list is not None:
            try:
                self.pyspin_cam_list.Clear()
            except Exception:
                pass
            self.pyspin_cam_list = None

        if self.pyspin_system is not None:
            try:
                self.pyspin_system.ReleaseInstance()
            except Exception:
                pass
            self.pyspin_system = None

    def _read_flir_camera_fps(self) -> Optional[float]:
        """Read FPS from a FLIR video capture when available."""
        if not self.flir_camera:
            return None
        cap = getattr(self.flir_camera, "cap", None)
        if cap is None:
            return None
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        except Exception:
            return None
        return fps if fps > 0 else None

    def _read_flir_frame_dimensions(self) -> Tuple[int, int]:
        """Read FLIR frame dimensions from the capture handle or a warmup frame."""
        if not self.flir_camera:
            return self.width, self.height

        cap = getattr(self.flir_camera, "cap", None)
        if cap is not None:
            try:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width > 0 and height > 0:
                    return width, height
            except Exception:
                pass

        try:
            sample = self.flir_camera.grab()
        except Exception:
            sample = None
        if isinstance(sample, np.ndarray) and sample.ndim >= 2:
            return int(sample.shape[1]), int(sample.shape[0])
        return self.width, self.height

    def _refresh_flir_status_cache(self, force: bool = False) -> Dict[str, object]:
        """Refresh low-rate FLIR status fields without touching the frame path."""
        if not self.flir_camera:
            return {}

        now = time.time()
        if not force and self.flir_status_cache and (now - self.flir_status_cache_time) < 1.0:
            return dict(self.flir_status_cache)

        status: Dict[str, object] = {}
        if self.flir_backend == "boson":
            status["flir_serial_port"] = self.flir_serial_port
            if hasattr(self.flir_camera, "get_external_sync_mode"):
                try:
                    status["external_sync_mode"] = self.flir_camera.get_external_sync_mode()
                except Exception:
                    status["external_sync_mode"] = None
            if hasattr(self.flir_camera, "get_fpa_temperature"):
                try:
                    status["fpa_temperature_c"] = self.flir_camera.get_fpa_temperature()
                except Exception:
                    status["fpa_temperature_c"] = None

        self.flir_status_cache = status
        self.flir_status_cache_time = now
        return dict(status)

    def _refresh_camera_settings_cache(self, force: bool = False) -> Dict[str, object]:
        """Refresh low-rate camera settings for metadata export."""
        now = time.time()
        if not force and self.camera_settings_cache and (now - self.camera_settings_cache_time) < 1.0:
            return dict(self.camera_settings_cache)

        cache: Dict[str, object] = {}
        exposure = self._read_numeric_node("ExposureTime")
        if exposure is None:
            exposure = self._read_numeric_node("ExposureTimeAbs")
        if exposure is not None:
            cache["exposure_time_us"] = float(exposure)

        gain = self._read_numeric_node("Gain")
        if gain is not None:
            cache["gain_db"] = float(gain)

        fps = self._read_camera_fps()
        if fps is not None:
            cache["camera_reported_fps"] = float(fps)

        width = self._read_numeric_node("Width")
        height = self._read_numeric_node("Height")
        if width is not None:
            cache["sensor_width"] = int(width)
        if height is not None:
            cache["sensor_height"] = int(height)

        pixel_format = self._read_enum_node_symbolic("PixelFormat")
        if pixel_format:
            cache["pixel_format"] = pixel_format
        color_filter = self._read_enum_node_symbolic("PixelColorFilter")
        if color_filter:
            cache["pixel_color_filter"] = color_filter
        bit_depth_info = self._get_camera_bit_depth_info()
        bit_depth = str(bit_depth_info.get("current", "") or "").strip()
        if bit_depth:
            cache["bit_depth"] = bit_depth
        bit_depth_node = str(bit_depth_info.get("node_name", "") or "").strip()
        if bit_depth_node:
            cache["bit_depth_node"] = bit_depth_node

        self.camera_settings_cache = cache
        self.camera_settings_cache_time = now
        return dict(cache)

    def disconnect_camera(self):
        """Disconnect camera."""
        with QMutexLocker(self.mutex):
            if self.is_spinnaker_camera():
                self._release_spinnaker_camera()
            elif self.camera:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera.IsOpen():
                    self.camera.Close()
                self.camera = None
            if self.usb_capture:
                self.usb_capture.release()
                self.usb_capture = None
            self._close_flir_camera()
            self.camera_type = None
            self.basler_device_class = ""
            self.flir_backend = None
            self.flir_video_index = None
            self.flir_serial_port = None
            self.flir_status_cache = {}
            self.flir_status_cache_time = 0.0
            self.camera_settings_cache = {}
            self.camera_settings_cache_time = 0.0
            self.camera_reported_fps = None
            with self.processing_condition:
                self.processing_queue.clear()
                self.processing_condition.notify_all()
            self.processing_queue_drop_count = 0
            self.processing_queue_high_water = 0
            self._emit_processing_buffer_usage()

    def set_trigger_mode(self, mode: str):
        """Set trigger mode: FreeRun or ExternalTrigger."""
        self.trigger_mode = mode
        if self.camera_type == "basler" and self.camera and self.camera.IsOpen():
            try:
                if mode == "ExternalTrigger":
                    self.camera.TriggerMode.SetValue("On")
                    self.camera.TriggerSource.SetValue("Line1")
                else:
                    self.camera.TriggerMode.SetValue("Off")
                self.status_update.emit(f"Trigger mode: {mode}")
            except Exception as e:
                self.error_occurred.emit(f"Trigger mode error: {str(e)}")
        elif self.camera_type == "flir" and self.flir_backend == "boson" and self.flir_camera:
            if not hasattr(self.flir_camera, "set_external_sync_mode"):
                return
            try:
                sync_mode = 2 if mode == "ExternalTrigger" else 0
                self.flir_camera.set_external_sync_mode(sync_mode)
                self.status_update.emit(f"Trigger mode: {mode}")
            except Exception as e:
                self.error_occurred.emit(f"Trigger mode error: {str(e)}")

    def start_recording(self, filename: str) -> bool:
        """Start recording video and metadata."""
        with self.recording_lock:
            if self.is_recording:
                return False

            try:
                if self.camera_type == "basler" and self.camera and self.camera.IsOpen():
                    fps = self._read_camera_fps()
                    if fps:
                        self.camera_reported_fps = fps
                    try:
                        self.width = int(self.camera.Width.GetValue())
                        self.height = int(self.camera.Height.GetValue())
                    except Exception:
                        pass
                elif self.camera_type == "usb" and self.usb_capture:
                    usb_fps = self.usb_capture.get(cv2.CAP_PROP_FPS)
                    if usb_fps and usb_fps > 0:
                        self.camera_reported_fps = float(usb_fps)
                    self.width = int(self.usb_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(self.usb_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                elif self.camera_type == "flir":
                    if self.is_spinnaker_camera():
                        pyspin_fps = self._read_camera_fps()
                        if pyspin_fps and pyspin_fps > 0:
                            self.camera_reported_fps = float(pyspin_fps)
                        width = self._read_numeric_node("Width")
                        height = self._read_numeric_node("Height")
                        if width:
                            self.width = int(width)
                        if height:
                            self.height = int(height)
                    else:
                        flir_fps = self._read_flir_camera_fps()
                        if flir_fps and flir_fps > 0:
                            self.camera_reported_fps = float(flir_fps)
                        if self.width <= 0 or self.height <= 0:
                            self.width, self.height = self._read_flir_frame_dimensions()

                self.recording_filename = filename
                self.metadata_buffer = []
                self.frame_counter = 0
                self.metadata_stats_counter = 0
                self.recording_output_fps = float(self.camera_reported_fps or self.fps_target or 30.0)
                self.last_recording_output_fps = None
                # Convert any requested wall-clock duration into an exact frame
                # count at the encode FPS. Because the file is written at this
                # same FPS, the saved clip is exactly `duration` seconds long,
                # and the stop is enforced on the acquisition thread (immune to
                # GUI-thread lag, which previously let recordings overrun).
                self.max_record_frames = frames_for_duration(
                    self.recording_output_fps, self.recording_duration_seconds
                )
                self._reset_frame_drop_stats()
            except Exception as e:
                self.error_occurred.emit(f"Recording start error: {str(e)}")
                return False

        # Start FFmpeg outside the lock so the processing thread can
        # continue emitting preview frames during encoder initialisation.
        try:
            self._start_ffmpeg()
            with self.recording_lock:
                self.is_recording = True
                self.recording_started_at = time.time()
                self._emit_frame_drop_stats(active=True)
            self.status_update.emit(f"Recording: {Path(filename).name}.mp4")
            return True
        except Exception as e:
            with self.recording_lock:
                self.recording_filename = ""
                self.metadata_buffer = []
                self.frame_counter = 0
                self.metadata_stats_counter = 0
                self.recording_output_fps = None
                self.last_recording_output_fps = None
                self.ffmpeg_process = None
                self.ffmpeg_stderr_thread = None
            self.error_occurred.emit(f"Recording start error: {str(e)}")
            return False

    def _recommended_bitrate_bps(self, width: int, height: int, fps: float) -> int:
        """Quality-targeted H.264 bitrate (bits/s) scaled to resolution and rate.

        Uses ~0.10 bits/pixel which keeps 1080p60 visually lossless for the
        low-motion arena footage while staying well within NVENC throughput.
        """
        pixels = max(1, int(width) * int(height))
        rate = max(1.0, float(fps))
        bps = pixels * rate * 0.10
        # Clamp to a sane window so tiny webcams stay sharp and huge sensors
        # do not blow up file size.
        return int(max(6_000_000, min(80_000_000, bps)))

    def _build_codec_args(self, encoder: str, width: int, height: int, fps: float) -> list:
        """Return FFmpeg codec arguments for one encoder, tuned for live capture."""
        bitrate_bps = self._recommended_bitrate_bps(width, height, fps)
        maxrate = int(bitrate_bps * 1.5)
        bufsize = int(bitrate_bps * 2)
        if encoder == "h264_nvenc":
            preset = self.encoder_preset if str(self.encoder_preset).startswith("p") else "p4"
            return [
                '-c:v', 'h264_nvenc',
                '-preset', preset,
                '-tune', 'hq',
                '-rc', 'vbr',
                '-cq', '21',
                '-b:v', str(bitrate_bps),
                '-maxrate', str(maxrate),
                '-bufsize', str(bufsize),
                '-bf', '0',            # no B-frames: lowest latency, simplest decode
                '-g', str(max(1, int(round(fps * 2)))),
            ]
        if encoder == "h264_qsv":
            return [
                '-c:v', 'h264_qsv',
                '-preset', 'fast',
                '-b:v', str(bitrate_bps),
                '-maxrate', str(maxrate),
            ]
        # libx264 (software) — ultrafast keeps the CPU encoder real-time even at
        # 1080p60; crf 20 is visually clean for arena footage.
        return [
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '20',
        ]

    def _start_ffmpeg(self):
        """Start FFmpeg for video encoding, falling back to libx264 on failure."""
        output_file = f"{self.recording_filename}.mp4"
        effective_width, effective_height = self._get_effective_dimensions()
        output_fps = float(self.recording_output_fps or self.fps_target or 30.0)
        if output_fps <= 0:
            output_fps = 30.0

        pixel_format = 'gray'
        if self.image_format == "BGR8":
            pixel_format = 'bgr24'

        # Try the configured encoder first; if a hardware encoder cannot start
        # (driver/session limits), automatically fall back to software libx264
        # so a recording is never silently lost.
        encoder_chain = [self.encoder]
        if self.encoder != "libx264":
            encoder_chain.append("libx264")

        last_error = None
        for attempt_index, encoder in enumerate(encoder_chain):
            codec_args = self._build_codec_args(encoder, effective_width, effective_height, output_fps)
            ffmpeg_cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{effective_width}x{effective_height}',
                '-pix_fmt', pixel_format,
                '-r', f'{output_fps:.3f}',
                '-i', '-',
            ] + codec_args + [
                '-pix_fmt', 'yuv420p',  # Ensure compatible output format
                '-movflags', '+faststart',
                '-an',  # No audio
                output_file,
            ]

            try:
                creationflags = 0
                if os.name == "nt":
                    creationflags = subprocess.CREATE_NO_WINDOW

                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=10**8,
                    creationflags=creationflags,
                )
                self._start_ffmpeg_stderr_thread()
                # Check if process died immediately.
                time.sleep(0.1)
                if self.ffmpeg_process.poll() is not None:
                    stderr = b""
                    try:
                        _, stderr = self.ffmpeg_process.communicate(timeout=1)
                    except Exception:
                        pass
                    self.ffmpeg_process = None
                    raise Exception(stderr.decode(errors='replace').strip() or "process exited at startup")

                self.active_encoder = encoder
                if attempt_index > 0:
                    self.status_update.emit(
                        f"Hardware encoder unavailable; recording with {encoder} instead."
                    )
                else:
                    self.status_update.emit(f"FFmpeg started: {encoder}")
                return
            except FileNotFoundError:
                raise Exception("FFmpeg not found! Install FFmpeg and add to PATH")
            except Exception as e:
                last_error = e
                continue

        raise Exception(f"FFmpeg error: {last_error}")

    def stop_recording(self):
        """Stop recording, save metadata, and correct the clip length if needed.

        The MP4 is muxed at the camera's nominal fps; a camera that cannot keep
        up delivers fewer frames, which would otherwise play back
        time-compressed (e.g. a 10 s recording becoming a 2 s file). After the
        encoder closes we measure the true rate and, when it falls materially
        short, remux the file so its playback length equals the real recording
        time and stays in sync with the audio and the other cameras.
        """
        remux_plan = None
        with self.recording_lock:
            if not self.is_recording:
                return

            try:
                self.is_recording = False
                self._emit_frame_drop_stats(active=False)
                self.last_recording_output_fps = self.recording_output_fps
                muxed_fps = self.recording_output_fps
                frame_count = int(self.frame_counter)
                started_at = self.recording_started_at
                filename = self.recording_filename

                # Close FFmpeg
                if self.ffmpeg_process:
                    try:
                        self.ffmpeg_process.stdin.close()
                        self.ffmpeg_process.wait(timeout=10)
                    except:
                        self.ffmpeg_process.kill()
                    finally:
                        try:
                            if self.ffmpeg_process.stderr:
                                self.ffmpeg_process.stderr.close()
                        except Exception:
                            pass
                        self.ffmpeg_process = None
                        self.ffmpeg_stderr_thread = None

                # Save metadata
                self._save_metadata()
                self.recording_output_fps = None

                remux_plan = self._plan_duration_correction(
                    filename, frame_count, muxed_fps, started_at
                )
                self.status_update.emit("Recording stopped")

            except Exception as e:
                self.error_occurred.emit(f"Stop recording error: {str(e)}")
                self.last_recording_output_fps = self.recording_output_fps
                self.recording_output_fps = None
                remux_plan = None

        # The stream-copy remux can block briefly; run it outside the recording
        # lock so it never stalls the acquisition thread, then signal completion
        # once (covering both the success and error paths above).
        if remux_plan is not None:
            try:
                self._correct_video_duration(*remux_plan)
            except Exception as exc:
                self.error_occurred.emit(f"Video duration correction failed: {exc}")
        self.recording_stopped.emit()

    def _plan_duration_correction(
        self,
        filename: str,
        frame_count: int,
        muxed_fps: Optional[float],
        started_at: Optional[float],
    ) -> Optional[Tuple[str, float, float]]:
        """Decide whether a finished recording needs its playback rate corrected.

        Returns ``(video_path, muxed_fps, measured_fps)`` when the camera's real
        delivered rate (frames over the elapsed recording time) is materially
        below the rate the file was muxed at, otherwise ``None``. Healthy cameras
        whose true rate matches the mux rate are left completely untouched.
        """
        try:
            if not filename or frame_count < 2:
                return None
            if not muxed_fps or float(muxed_fps) <= 0 or started_at is None:
                return None
            elapsed = time.time() - float(started_at)
            if elapsed <= 0:
                return None
            measured_fps = frame_count / elapsed
            # Only correct a meaningful shortfall (>5%): this is the under-
            # delivering-camera bug, not normal frame jitter.
            if measured_fps >= float(muxed_fps) * 0.95:
                return None
            video_path = f"{filename}.mp4"
            if not os.path.exists(video_path):
                return None
            return video_path, float(muxed_fps), float(measured_fps)
        except Exception:
            return None

    def _correct_video_duration(self, video_path: str, muxed_fps: float, measured_fps: float) -> None:
        """Stretch a finished MP4 to its true duration without re-encoding.

        Uses FFmpeg ``-itsscale`` to scale the stored timestamps by
        ``muxed_fps / measured_fps`` with ``-c copy``, so the same H.264 frames
        now span the real recording window. On any failure the original file is
        left intact.
        """
        scale = float(muxed_fps) / max(0.1, float(measured_fps))
        tmp_path = f"{video_path}.fixfps.mp4"
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-itsscale', f'{scale:.6f}',
            '-i', video_path,
            '-c', 'copy',
            '-movflags', '+faststart',
            tmp_path,
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            timeout=60,
        )
        if proc.returncode != 0 or not os.path.exists(tmp_path):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise Exception((proc.stderr or b"").decode(errors="replace").strip() or "remux failed")
        os.replace(tmp_path, video_path)
        self.status_update.emit(
            f"Adjusted {Path(video_path).name} to {measured_fps:.2f} fps so its "
            "length matches the real recording time"
        )

    def _save_metadata(self):
        """Save metadata buffer to CSV."""
        if not self.metadata_buffer:
            return

        try:
            df = pd.DataFrame(self.metadata_buffer)
            if self.line_label_map:
                rename_map = {
                    key: f"{key}_{suffix}"
                    for key, suffix in self.line_label_map.items()
                    if key in df.columns and suffix
                }
                if rename_map:
                    df = df.rename(columns=rename_map)
            drop_columns = ["line_status_all"]
            for line_number in range(1, 5):
                column = f"line{line_number}_status"
                if column in self.line_label_map:
                    continue
                if column in df.columns:
                    drop_columns.append(column)
            removable = [column for column in drop_columns if column in df.columns]
            if removable:
                df = df.drop(columns=removable)
            csv_file = f"{self.recording_filename}_metadata.csv"
            df.to_csv(csv_file, index=False)
            self.status_update.emit(f"Metadata saved: {len(df)} frames")
        except Exception as e:
            self.error_occurred.emit(f"Metadata save error: {str(e)}")

    def run(self):
        """Main acquisition loop."""
        if self.camera_type == "basler":
            if not self.camera or not self.camera.IsOpen():
                self.error_occurred.emit("Camera not connected!")
                return
        elif self.is_spinnaker_camera():
            if not self.camera:
                self.error_occurred.emit("FLIR Spinnaker camera not connected!")
                return
        elif self.camera_type == "usb":
            if not self.usb_capture or not self.usb_capture.isOpened():
                self.error_occurred.emit("USB camera not connected!")
                return
        elif self.camera_type == "flir":
            if not self.flir_camera:
                self.error_occurred.emit("FLIR camera not connected!")
                return
        else:
            self.error_occurred.emit("Camera not connected!")
            return

        try:
            self.running = True
            self._start_processing_thread()
            if self.camera_type == "basler":
                self.basler_pause_requested = False
                self.basler_paused = False
                self._configure_basler_stream_buffers()
                self.camera.StartGrabbing(self._get_basler_grab_strategy())

                while self.running:
                    if self.basler_pause_requested:
                        try:
                            if self.camera.IsGrabbing():
                                self.camera.StopGrabbing()
                        except Exception:
                            pass
                        self.basler_paused = True

                        while self.running and self.basler_pause_requested:
                            time.sleep(0.01)

                        if not self.running:
                            break

                        try:
                            self.camera.StartGrabbing(self._get_basler_grab_strategy())
                        except Exception as e:
                            self.error_occurred.emit(f"Basler acquisition restart failed: {str(e)}")
                            break
                        self.basler_paused = False
                        continue

                    try:
                        grab_result = self.camera.RetrieveResult(
                            5000,
                            pylon.TimeoutHandling_ThrowException
                        )

                        if grab_result.GrabSucceeded():
                            packet = self._capture_basler_frame_packet(grab_result)
                            if packet is not None:
                                self._enqueue_frame_packet(packet)
                            self._update_fps()

                        grab_result.Release()

                    except pylon.TimeoutException:
                        self.status_update.emit("Frame timeout...")
                        continue

                self.basler_paused = False
                self.camera.StopGrabbing()
            elif self.is_spinnaker_camera():
                self.spinnaker_pause_requested = False
                self.spinnaker_paused = False
                self.camera.BeginAcquisition()

                while self.running:
                    image_result = None
                    if self.spinnaker_pause_requested:
                        try:
                            if self.camera.IsStreaming():
                                self.camera.EndAcquisition()
                        except Exception:
                            pass
                        self.spinnaker_paused = True

                        while self.running and self.spinnaker_pause_requested:
                            time.sleep(0.01)

                        if not self.running:
                            break

                        try:
                            self.camera.BeginAcquisition()
                        except Exception as e:
                            self.error_occurred.emit(f"FLIR acquisition restart failed: {str(e)}")
                            break
                        self.spinnaker_paused = False
                        continue

                    try:
                        image_result = self.camera.GetNextImage(200)
                        if image_result is None:
                            self.status_update.emit("FLIR frame timeout...")
                            continue
                        if image_result.IsIncomplete():
                            self.status_update.emit(f"FLIR incomplete frame: {image_result.GetImageStatus()}")
                            continue

                        packet = self._capture_spinnaker_frame_packet(image_result)
                        if packet is not None:
                            self._enqueue_frame_packet(packet)
                        self._update_fps()
                    except Exception as e:
                        if self.spinnaker_pause_requested:
                            continue
                        self.status_update.emit(f"FLIR frame timeout... {str(e)}")
                        time.sleep(0.01)
                    finally:
                        if image_result is not None:
                            try:
                                image_result.Release()
                            except Exception:
                                pass

                try:
                    if self.camera.IsStreaming():
                        self.camera.EndAcquisition()
                except Exception:
                    pass
                self.spinnaker_paused = False
                self.spinnaker_pause_requested = False
            elif self.camera_type == "usb":
                consecutive_failures = 0
                reopen_attempts = 0
                while self.running:
                    self._apply_pending_capture_reconfig()
                    ok, frame = self.usb_capture.read()
                    if not ok:
                        consecutive_failures += 1
                        if consecutive_failures == 30:
                            self.status_update.emit("USB frame timeout...")
                        # ~3 s without frames: the MSMF stream has stalled
                        # (commonly USB bandwidth saturation). Try a reopen.
                        if consecutive_failures >= 300 and reopen_attempts < 3:
                            reopen_attempts += 1
                            consecutive_failures = 0
                            self.error_occurred.emit(
                                "USB camera stopped delivering frames — reopening "
                                f"(attempt {reopen_attempts}/3). If this repeats, the USB bus "
                                "may be saturated: try another port or lower the resolution."
                            )
                            self._reopen_usb_capture()
                        time.sleep(0.01)
                        continue

                    if consecutive_failures >= 30:
                        self.status_update.emit("USB camera recovered.")
                    consecutive_failures = 0
                    packet = self._capture_usb_frame_packet(frame)
                    if packet is not None:
                        self._enqueue_frame_packet(packet)
                    self._update_fps()
            else:
                while self.running:
                    self._apply_pending_capture_reconfig()
                    frame = self._grab_flir_frame()
                    if frame is None:
                        self.status_update.emit("FLIR frame timeout...")
                        time.sleep(0.01)
                        continue

                    packet = self._capture_flir_frame_packet(frame)
                    if packet is not None:
                        self._enqueue_frame_packet(packet)
                    self._update_fps()

        except Exception as e:
            self.error_occurred.emit(f"Acquisition error: {str(e)}")
        finally:
            self.running = False
            self._stop_processing_thread()

    def _capture_basler_frame_packet(self, grab_result) -> Optional[FramePacket]:
        """Snapshot a Basler frame for downstream processing."""
        if self.converter is None:
            raise RuntimeError("Basler image converter is unavailable.")

        image = self.converter.Convert(grab_result)
        frame = np.array(image.GetArray(), copy=True)
        metadata = {"timestamp_software": time.time()}
        metadata.update(self._extract_chunk_data(grab_result))
        return FramePacket(
            backend="basler",
            frame=frame,
            metadata=metadata,
            requested_format=self.image_format,
        )

    def _capture_usb_frame_packet(self, frame: np.ndarray) -> Optional[FramePacket]:
        """Snapshot an OpenCV USB frame for downstream processing."""
        balanced_frame = self._apply_usb_auto_white_balance_bgr(frame)
        return FramePacket(
            backend="usb",
            frame=np.ascontiguousarray(balanced_frame),
            metadata={"timestamp_software": time.time()},
            requested_format=self.image_format,
        )

    def _configure_usb_color_controls(self, capture: cv2.VideoCapture) -> None:
        """Ask UVC/OpenCV for camera-side auto white balance where available."""
        if capture is None:
            return
        for prop_name, value in (
            ("CAP_PROP_CONVERT_RGB", 1),
            ("CAP_PROP_AUTO_WB", 1),
        ):
            prop_id = getattr(cv2, prop_name, None)
            if prop_id is None:
                continue
            try:
                capture.set(prop_id, value)
            except Exception:
                pass
        self.usb_white_balance_gains_bgr = None

    def _apply_usb_auto_white_balance_bgr(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply conservative bright-white white balance to USB BGR color frames.

        Many UVC cameras report RGB-converted frames while their hardware auto
        white balance remains strongly biased by cage illumination. OpenCV gives
        us BGR pixels, so we estimate B/G/R gains from bright non-saturated
        samples and smooth them over time to avoid flicker.
        """
        if not self.usb_auto_white_balance_enabled:
            return np.ascontiguousarray(frame)

        working = np.asarray(frame)
        if working.ndim != 3 or working.shape[2] != 3:
            return np.ascontiguousarray(working)

        if working.dtype != np.uint8:
            working = self._normalize_array_to_uint8(working)

        sample = working[::8, ::8].astype(np.float32)
        if sample.size == 0:
            return np.ascontiguousarray(working)

        brightness = sample.mean(axis=2)
        bright_threshold = float(np.percentile(brightness, 75))
        mask = (
            (brightness >= bright_threshold)
            & (brightness > 35.0)
            & (brightness < 245.0)
            & np.all(sample < 250.0, axis=2)
        )
        using_bright_sample = int(mask.sum()) >= 64
        if int(mask.sum()) < 64:
            mask = (brightness > 25.0) & (brightness < 245.0)
        if int(mask.sum()) < 64:
            return np.ascontiguousarray(working)

        pixels = sample[mask]
        if not using_bright_sample:
            low = np.percentile(pixels, 5, axis=0)
            high = np.percentile(pixels, 95, axis=0)
            robust_mask = np.all((pixels >= low) & (pixels <= high), axis=1)
            if int(robust_mask.sum()) >= 64:
                pixels = pixels[robust_mask]

        means = pixels.mean(axis=0)
        gray_level = float(means.mean())
        if gray_level <= 1.0 or np.any(means <= 1.0):
            return np.ascontiguousarray(working)

        # Bedding and cage walls should look slightly warm, not clinically gray.
        target_ratios_bgr = np.array([0.94, 1.0, 1.04], dtype=np.float32)
        target = gray_level * target_ratios_bgr / float(target_ratios_bgr.mean())
        gains = target / means
        gains = np.clip(gains, np.array([0.75, 0.65, 0.55]), np.array([1.7, 1.55, 1.55]))

        if self.usb_white_balance_gains_bgr is None:
            self.usb_white_balance_gains_bgr = gains.astype(np.float32)
        else:
            alpha = 0.12
            self.usb_white_balance_gains_bgr = (
                ((1.0 - alpha) * self.usb_white_balance_gains_bgr) + (alpha * gains)
            ).astype(np.float32)

        balanced = working.astype(np.float32) * self.usb_white_balance_gains_bgr.reshape(1, 1, 3)
        return np.ascontiguousarray(np.clip(balanced, 0, 255).astype(np.uint8))

    def _capture_spinnaker_frame_packet(self, image_result) -> Optional[FramePacket]:
        """Snapshot a Spinnaker frame before the SDK buffer is released."""
        img_array = np.array(image_result.GetNDArray(), copy=True)
        metadata: Dict[str, object] = {
            "timestamp_software": time.time(),
            "flir_backend": "spinnaker",
            "line_status_all": None,
            "line1_status": None,
            "line2_status": None,
            "line3_status": None,
            "line4_status": None,
        }
        metadata.update(self._extract_chunk_data(image_result))
        if self.is_recording:
            try:
                metadata["timestamp_ticks"] = image_result.GetTimeStamp()
            except Exception:
                metadata["timestamp_ticks"] = None
            try:
                metadata["camera_frame_id"] = image_result.GetFrameID()
            except Exception:
                metadata["camera_frame_id"] = None
            metadata.update(self._refresh_camera_settings_cache(force=False))

        cached_settings = self.camera_settings_cache or {}
        return FramePacket(
            backend="spinnaker",
            frame=img_array,
            metadata=metadata,
            requested_format=self.image_format,
            pixel_format=str(cached_settings.get("pixel_format") or self.spinnaker_native_pixel_format or ""),
            color_filter=str(cached_settings.get("pixel_color_filter") or self.spinnaker_color_filter or ""),
        )

    def _capture_flir_frame_packet(self, frame: np.ndarray) -> Optional[FramePacket]:
        """Snapshot a flirpy frame and low-rate status metadata."""
        raw_frame = np.ascontiguousarray(frame)
        metadata: Dict[str, object] = {
            "timestamp_software": time.time(),
            "flir_backend": self.flir_backend or "",
            "line_status_all": None,
            "line1_status": None,
            "line2_status": None,
            "line3_status": None,
            "line4_status": None,
        }
        if self.is_recording:
            if self.flir_backend == "boson" and self.flir_camera:
                metadata.update(self._refresh_flir_status_cache())
            elif self.flir_backend == "lepton" and self.flir_camera:
                for attr_name in ("frame_count", "uptime_ms", "fpa_temp_k", "ffc_temp_k", "ffc_elapsed_ms"):
                    metadata[attr_name] = getattr(self.flir_camera, attr_name, None)

        return FramePacket(
            backend="flir",
            frame=raw_frame,
            metadata=metadata,
            requested_format=self.image_format,
        )

    def _process_frame_packet(self, packet: FramePacket):
        """Run conversion, preview throttling, and recording for one frame packet."""
        backend = str(packet.backend or "")
        if backend == "basler":
            self._process_standard_frame_packet(packet, source_color_space="bgr")
            return
        if backend == "usb":
            self._process_standard_frame_packet(packet, source_color_space="bgr")
            return
        if backend == "spinnaker":
            self._process_standard_frame_packet(
                packet,
                source_color_space="rgb" if str(packet.pixel_format).startswith("RGB") else "bgr",
            )
            return
        self._process_flir_frame_packet(packet)

    def _process_standard_frame_packet(self, packet: FramePacket, source_color_space: str = "bgr"):
        """Handle Basler, USB, and Spinnaker machine-vision frames."""
        need_preview = self._should_emit_preview()
        record_frame, preview_frame = self._prepare_frame_buffers(
            packet.frame,
            requested_format=packet.requested_format,
            need_preview=need_preview,
            source_color_space=source_color_space,
            pixel_format=packet.pixel_format,
            color_filter=packet.color_filter,
        )
        metadata = self._build_packet_metadata(packet)
        self._finalize_processed_frame(record_frame, preview_frame, metadata)

    def _process_flir_frame_packet(self, packet: FramePacket):
        """Normalize and process FLIR thermal frames."""
        need_preview = self._should_emit_preview()
        raw_frame = np.asarray(packet.frame)
        if raw_frame.ndim == 3 and raw_frame.shape[2] == 1:
            raw_frame = raw_frame[:, :, 0]

        metadata = self._build_packet_metadata(packet)
        normalized = self._normalize_flir_frame(raw_frame)
        if packet.requested_format == "BGR8":
            color_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
            record_frame = self._apply_roi(color_bgr)
            preview_frame = self._build_preview_frame(color_bgr, source_color_space="bgr") if need_preview else None
        else:
            record_frame = self._apply_roi(normalized)
            preview_frame = self._build_preview_frame(normalized, source_color_space="gray") if need_preview else None

        self._finalize_processed_frame(record_frame, preview_frame, metadata)

    def _build_packet_metadata(self, packet: FramePacket) -> Dict[str, object]:
        """Add low-cost metadata to a captured packet and rate-limit heavy stats."""
        metadata = dict(packet.metadata or {})
        if not self.is_recording:
            return metadata

        frame = np.asarray(packet.frame)
        metadata.setdefault("raw_dtype", str(frame.dtype))
        if frame.ndim >= 2:
            metadata.setdefault("raw_height", int(frame.shape[0]))
            metadata.setdefault("raw_width", int(frame.shape[1]))
        self._attach_raw_frame_stats(metadata, frame)
        return metadata

    def _attach_raw_frame_stats(self, metadata: Dict[str, object], frame: np.ndarray):
        """Compute expensive full-frame statistics only at a configurable interval."""
        interval = int(self.metadata_stats_interval_frames or 0)
        self.metadata_stats_counter += 1
        if interval <= 0 or (self.metadata_stats_counter % interval) != 0:
            metadata.setdefault("raw_min", None)
            metadata.setdefault("raw_max", None)
            metadata.setdefault("raw_mean", None)
            return

        working = np.asarray(frame)
        if working.size == 0:
            metadata["raw_min"] = None
            metadata["raw_max"] = None
            metadata["raw_mean"] = None
            return

        metadata["raw_min"] = float(np.min(working))
        metadata["raw_max"] = float(np.max(working))
        metadata["raw_mean"] = float(np.mean(working))

    def _prepare_frame_buffers(
        self,
        frame: np.ndarray,
        requested_format: str = "Mono8",
        need_preview: bool = False,
        source_color_space: str = "bgr",
        pixel_format: str = "",
        color_filter: str = "",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert a frame into record and preview buffers while avoiding extra work."""
        working = np.asarray(frame)
        if working.ndim == 3 and working.shape[2] == 1:
            working = working[:, :, 0]

        if requested_format == "Mono8":
            if working.ndim == 2:
                mono = working
            else:
                gray_code = cv2.COLOR_RGB2GRAY if source_color_space == "rgb" else cv2.COLOR_BGR2GRAY
                mono = cv2.cvtColor(working, gray_code)
            if mono.dtype != np.uint8:
                mono = self._normalize_array_to_uint8(mono)
            record_frame = self._apply_roi(mono)
            preview_frame = self._build_preview_frame(mono, source_color_space="gray") if need_preview else None
            return record_frame, preview_frame

        if working.ndim == 2:
            color_bgr = self._convert_single_channel_frame_to_bgr(
                working,
                pixel_format=pixel_format,
                color_filter=color_filter,
            )
        else:
            color_bgr = working
            if color_bgr.dtype != np.uint8:
                color_bgr = self._normalize_array_to_uint8(color_bgr)
            if source_color_space == "rgb":
                color_bgr = cv2.cvtColor(color_bgr, cv2.COLOR_RGB2BGR)

        record_frame = self._apply_roi(color_bgr)
        preview_frame = self._build_preview_frame(color_bgr, source_color_space="bgr") if need_preview else None
        return record_frame, preview_frame

    def _build_preview_frame(self, frame: np.ndarray, source_color_space: str = "gray") -> np.ndarray:
        """Prepare a lightweight preview frame for the UI."""
        preview = self._resize_preview_frame(frame)
        if source_color_space == "gray" or preview.ndim == 2:
            return preview
        if source_color_space == "rgb":
            return preview
        return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

    def _finalize_processed_frame(
        self,
        record_frame: np.ndarray,
        preview_frame: Optional[np.ndarray],
        metadata: Dict[str, object],
    ):
        """Emit per-frame packets, write recording data, then update the UI preview."""
        self._assign_recording_frame_id(metadata)

        if self.live_inference_packets_enabled and self._should_emit_inference_packet():
            source_height, source_width = record_frame.shape[:2]
            inference_frame = self._downscale_for_inference(record_frame)
            inference_metadata = dict(metadata)
            inference_metadata["source_frame_width"] = int(source_width)
            inference_metadata["source_frame_height"] = int(source_height)
            self.live_inference_packet_ready.emit(
                self._build_preview_frame_packet(inference_frame, inference_metadata, convert_bgr_to_rgb=True)
            )

        preview_packet = None
        if preview_frame is not None:
            preview_packet = self._build_preview_frame_packet(preview_frame, metadata, convert_bgr_to_rgb=False)
            self.preview_packet_ready.emit(preview_packet)

        stop_processing = self._handle_record_frame(record_frame, metadata)
        if self.record_frame_packets_enabled and "frame_id" in metadata:
            self.record_frame_packet_ready.emit(
                self._build_preview_frame_packet(record_frame, metadata, convert_bgr_to_rgb=True)
            )
        self.frame_metadata_ready.emit(dict(metadata))
        if stop_processing:
            return
        if preview_frame is not None:
            self.frame_ready.emit(preview_frame)

    def _assign_recording_frame_id(self, metadata: Dict[str, object]) -> None:
        """Attach the current recording frame id before inference or overlay packets are emitted."""
        if "frame_id" in metadata:
            return
        try:
            with self.recording_lock:
                if self.is_recording and self.ffmpeg_process and self.ffmpeg_process.stdin:
                    metadata["frame_id"] = int(self.frame_counter)
        except Exception:
            return

    def _build_preview_frame_packet(
        self,
        frame: np.ndarray,
        metadata: Dict[str, object],
        *,
        convert_bgr_to_rgb: bool,
    ) -> PreviewFramePacket:
        packet_frame = np.asarray(frame)
        if convert_bgr_to_rgb and packet_frame.ndim == 3 and packet_frame.shape[2] == 3:
            packet_frame = cv2.cvtColor(packet_frame, cv2.COLOR_BGR2RGB)
        height, width = packet_frame.shape[:2]
        return PreviewFramePacket(
            frame=np.asarray(packet_frame),
            frame_index=int(metadata.get("frame_id", self.frame_counter)),
            timestamp_s=float(metadata.get("timestamp_software", time.time()) or time.time()),
            width=int(width),
            height=int(height),
            metadata=dict(metadata),
        )

    def _convert_single_channel_frame_to_bgr(
        self,
        frame: np.ndarray,
        pixel_format: str = "",
        color_filter: str = "",
    ) -> np.ndarray:
        """Convert a single-channel frame to BGR, including Bayer debayering."""
        working = frame
        if working.dtype != np.uint8:
            working = self._normalize_array_to_uint8(working)

        bayer_code = self._get_spinnaker_bayer_to_bgr_code(pixel_format=pixel_format, color_filter=color_filter)
        if bayer_code is not None:
            try:
                return cv2.cvtColor(working, bayer_code)
            except Exception:
                pass
        return cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)

    def _get_spinnaker_bayer_to_bgr_code(
        self,
        pixel_format: str = "",
        color_filter: str = "",
    ) -> Optional[int]:
        """Map a Bayer source descriptor to an OpenCV demosaic code."""
        source = f"{pixel_format or self.spinnaker_native_pixel_format} {color_filter or self.spinnaker_color_filter}"

        mapping = {
            # OpenCV's Bayer naming is inverted relative to the PixelColorFilter
            # names exposed by Spinnaker for BGR output.
            "BayerRG": cv2.COLOR_BAYER_BG2BGR,
            "BayerBG": cv2.COLOR_BAYER_RG2BGR,
            "BayerGB": cv2.COLOR_BAYER_GR2BGR,
            "BayerGR": cv2.COLOR_BAYER_GB2BGR,
        }
        for token, code in mapping.items():
            if token in source:
                return code
        return None

    def _normalize_array_to_uint8(self, frame: np.ndarray) -> np.ndarray:
        """Normalize an arbitrary numeric array to 8-bit for display/recording."""
        working = frame.astype(np.float32, copy=False)
        if working.size == 0:
            return np.zeros((max(self.height, 1), max(self.width, 1)), dtype=np.uint8)
        min_val = float(np.min(working))
        max_val = float(np.max(working))
        if max_val <= min_val:
            return np.zeros(working.shape[:2], dtype=np.uint8)
        normalized = ((working - min_val) * (255.0 / (max_val - min_val))).clip(0, 255)
        return normalized.astype(np.uint8)

    def _grab_flir_frame(self) -> Optional[np.ndarray]:
        """Grab a frame from the active flirpy backend."""
        if not self.flir_camera:
            return None
        try:
            frame = self.flir_camera.grab()
        except TypeError:
            frame = self.flir_camera.grab(device_id=self.flir_video_index)
        except Exception as e:
            self.error_occurred.emit(f"FLIR capture error: {str(e)}")
            return None

        if isinstance(frame, np.ndarray):
            return frame
        return None

    def _normalize_flir_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert a raw FLIR frame into an 8-bit image for display/recording."""
        if frame.size == 0:
            return np.zeros((max(self.height, 1), max(self.width, 1)), dtype=np.uint8)

        working = frame.astype(np.float32, copy=False)
        min_val = float(np.min(working))
        max_val = float(np.max(working))
        if max_val <= min_val:
            return np.zeros(working.shape[:2], dtype=np.uint8)

        normalized = ((working - min_val) * (255.0 / (max_val - min_val))).clip(0, 255)
        return normalized.astype(np.uint8)

    def _handle_record_frame(self, record_frame: np.ndarray, metadata: Dict) -> bool:
        """
        Write the current frame to FFmpeg and metadata buffers.

        Returns True when recording hit the configured frame limit and the caller
        should stop further processing of the current frame.
        """
        stop_now = False
        try:
            with self.recording_lock:
                if not (self.is_recording and self.ffmpeg_process and self.ffmpeg_process.stdin):
                    return False

                writable_frame = record_frame if record_frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(record_frame)
                self.ffmpeg_process.stdin.write(memoryview(writable_frame).cast("B"))

                frame_id = int(metadata.get("frame_id", self.frame_counter))
                metadata["frame_id"] = frame_id
                self.frame_counter = max(int(self.frame_counter), frame_id + 1)
                self._track_recorded_frame_timing(metadata)

                self.metadata_buffer.append(metadata)
                self.frame_recorded.emit(metadata)

                if self.max_record_frames is not None and self.frame_counter >= self.max_record_frames:
                    self.status_update.emit(f"Reached frame target: {self.max_record_frames} frames")
                    stop_now = True
                elif (
                    self.recording_duration_seconds
                    and self.recording_started_at is not None
                    and (time.time() - self.recording_started_at)
                    >= float(self.recording_duration_seconds) + 0.5
                ):
                    # Wall-clock backstop: a camera delivering below its nominal
                    # rate never reaches the frame target, so cap the session at
                    # the requested duration (plus a small grace) so it cannot
                    # overrun the other streams. Healthy cameras hit the frame
                    # target first and never trigger this.
                    self.status_update.emit("Reached duration limit (camera below target fps)")
                    stop_now = True
        except Exception as e:
            self.error_occurred.emit(f"Frame write error: {str(e)}")
            self.stop_recording()
            return True

        if stop_now:
            self.stop_recording()
            return True
        return False

    def _open_usb_capture(self, index: int, preferred_backend: str = ""):
        """Open a USB capture, trying the preferred cv2 backend first.

        Returns ``(capture, backend_name)``; ``(None, "")`` when every
        backend in the chain failed to open the device.
        """
        from camera_backends import _cv2_usb_backend_chain, cv2_backend_id_from_name

        chain = list(_cv2_usb_backend_chain())
        preferred_id = cv2_backend_id_from_name(preferred_backend)
        if preferred_id is not None:
            chain.sort(key=lambda item: 0 if item[0] == preferred_id else 1)

        for backend_id, backend_name in chain:
            try:
                capture = cv2.VideoCapture(int(index), backend_id)
            except Exception:
                continue
            if capture is not None and capture.isOpened():
                return capture, backend_name
            try:
                if capture is not None:
                    capture.release()
            except Exception:
                pass
        return None, ""

    def _reopen_usb_capture(self) -> bool:
        """Release and reopen a stalled OpenCV USB capture, keeping settings."""
        if self.usb_index is None:
            return False
        try:
            if self.usb_capture is not None:
                self.usb_capture.release()
        except Exception:
            pass
        capture, backend_name = self._open_usb_capture(
            int(self.usb_index), getattr(self, "usb_backend", "")
        )
        if capture is None:
            return False
        self.usb_backend = backend_name
        self._request_usb_mjpg(capture)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width or 1080)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height or 1080)
        capture.set(cv2.CAP_PROP_FPS, self.fps_target)
        self._configure_usb_color_controls(capture)
        self.usb_capture = capture
        return True

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        """Apply software ROI cropping to a frame."""
        if not self.roi:
            return frame
        x = int(self.roi.get('x', 0))
        y = int(self.roi.get('y', 0))
        w = int(self.roi.get('w', frame.shape[1]))
        h = int(self.roi.get('h', frame.shape[0]))

        max_w = frame.shape[1]
        max_h = frame.shape[0]
        x = max(0, min(x, max_w - 1))
        y = max(0, min(y, max_h - 1))
        w = max(1, min(w, max_w - x))
        h = max(1, min(h, max_h - y))
        return frame[y:y + h, x:x + w]

    def _get_effective_dimensions(self) -> tuple[int, int]:
        """Get output dimensions after ROI."""
        if not self.roi:
            return self.width, self.height
        x = int(self.roi.get('x', 0))
        y = int(self.roi.get('y', 0))
        w = int(self.roi.get('w', self.width))
        h = int(self.roi.get('h', self.height))
        w = max(1, min(w, self.width - x))
        h = max(1, min(h, self.height - y))
        return w, h

    def _extract_chunk_data(self, grab_result) -> Dict:
        """Extract chunk metadata from frame."""
        metadata = {}

        def _read_chunk_value(attr_name: str, getter_names: Tuple[str, ...]):
            sources = [grab_result]
            try:
                chunk_data = grab_result.GetChunkData()
            except Exception:
                chunk_data = None
            if chunk_data is not None:
                sources.append(chunk_data)

            for source in sources:
                attr = getattr(source, attr_name, None)
                if attr is not None:
                    try:
                        is_readable = getattr(attr, 'IsReadable', None)
                        if callable(is_readable) and not is_readable():
                            attr = None
                    except Exception:
                        pass
                if attr is not None:
                    try:
                        return attr.GetValue()
                    except Exception:
                        pass

                for getter_name in getter_names:
                    getter = getattr(source, getter_name, None)
                    if not callable(getter):
                        continue
                    try:
                        return getter()
                    except Exception:
                        continue

            return None

        try:
            metadata['timestamp_ticks'] = _read_chunk_value('ChunkTimestamp', ('GetTimestamp', 'GetTimeStamp'))
            metadata['exposure_time_us'] = _read_chunk_value('ChunkExposureTime', ('GetExposureTime',))

            line_status = _read_chunk_value('ChunkLineStatusAll', ('GetLineStatusAll',))
            display_line_status = None
            if self.is_spinnaker_camera():
                # Chunk LineStatusAll is unreliable during streaming (returns 0 even when lines
                # are active). Always use live nodemap reads instead.
                # _read_spinnaker_live_line_status prefers per-selector reads (highest fidelity)
                # and falls back to live LineStatusAll with selector-aware bit remapping.
                # We also read the raw LineStatusAll register separately to preserve it in
                # line_status_all without polluting the display-slot mapping.
                live_raw = self._read_numeric_node("LineStatusAll")
                if live_raw is not None:
                    line_status = int(live_raw)
                display_line_status = self._read_spinnaker_live_line_status()
            elif line_status is None:
                pass  # no line data available for non-Spinnaker
            self._apply_line_status_metadata(metadata, line_status, display_line_status=display_line_status)

            # --- throttled GPIO diagnostic (every 60 frames) ---
            if self.is_spinnaker_camera():
                self._line_debug_frame_counter += 1
                if self._line_debug_frame_counter % 60 == 1:
                    self.status_update.emit(
                        f"[GPIO diag] live_raw={line_status!r}  "
                        f"display={display_line_status!r}  "
                        f"cached_sel={self._spinnaker_cached_line_selectors!r}  "
                        f"l1={metadata.get('line1_status')!r} "
                        f"l2={metadata.get('line2_status')!r} "
                        f"l3={metadata.get('line3_status')!r} "
                        f"l4={metadata.get('line4_status')!r}"
                    )

        except Exception as e:
            self.error_occurred.emit(f"Chunk data error: {str(e)}")

        return metadata

    def _update_fps(self):
        """Update FPS counter."""
        self.fps_frame_count += 1

        if self.fps_frame_count % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self.fps_last_time

            if elapsed > 0:
                fps = 30.0 / elapsed
                self.fps_update.emit(fps)
                self.fps_last_time = current_time

    def _read_camera_fps(self) -> Optional[float]:
        """Read the camera's actual or configured frame rate."""
        if not self.camera:
            return None

        candidates = [
            "ResultingFrameRate",
            "ResultingFrameRateAbs",
            "AcquisitionFrameRate",
            "AcquisitionFrameRateAbs",
        ]
        for name in candidates:
            value = self._read_numeric_node(name)
            if value is None:
                continue
            if value > 0:
                return value

        return None

    def stop(self):
        """Stop the worker thread."""
        self.running = False
        self.spinnaker_pause_requested = False
        self.basler_pause_requested = False
        with self.processing_condition:
            self.processing_condition.notify_all()
        if self.is_recording:
            self.stop_recording()

    def _start_ffmpeg_stderr_thread(self):
        """Drain FFmpeg stderr to avoid blocking on full buffers."""
        if not self.ffmpeg_process or not self.ffmpeg_process.stderr:
            return

        def _drain():
            try:
                for _ in self.ffmpeg_process.stderr:
                    pass
            except Exception:
                pass

        self.ffmpeg_stderr_thread = threading.Thread(target=_drain, daemon=True)
        self.ffmpeg_stderr_thread.start()
