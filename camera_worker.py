"""
Camera Worker Thread - Clean Implementation
Handles camera acquisition, GPU-accelerated recording, and metadata logging.
"""
import time
import subprocess
import os
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


class CameraWorker(QThread):
    """
    Worker thread for camera operations.
    Manages frame acquisition, FFmpeg encoding, and metadata collection.
    """

    # Signals
    frame_ready = Signal(np.ndarray)
    preview_packet_ready = Signal(object)
    status_update = Signal(str)
    fps_update = Signal(float)
    buffer_update = Signal(int)
    error_occurred = Signal(str)
    recording_stopped = Signal()
    frame_recorded = Signal(dict)  # Signal for each recorded frame with metadata
    frame_drop_stats_updated = Signal(dict)

    def __init__(self):
        super().__init__()

        # Camera
        self.camera: Optional[Any] = None
        self.flir_camera: Optional[Any] = None
        self.usb_capture: Optional[cv2.VideoCapture] = None
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
        self.metadata_stats_interval_frames = 25
        self.stream_buffer_target = 128

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
            return None
        return node

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

        # Prefer native string SetValue when the SDK supports it.
        try:
            node.SetValue(entry_name)
            return True
        except Exception:
            pass

        try:
            entry = node.GetEntryByName(entry_name)
            node.SetIntValue(entry.GetValue())
            return True
        except Exception:
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
            writable = bool(node is not None and self._node_is_writable(node))
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
        writable = bool(node is not None and self._node_is_writable(node))
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
        if self.is_spinnaker_camera() and self.isRunning():
            paused_for_reconfigure = self._pause_spinnaker_acquisition_for_reconfigure()
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
            if paused_for_reconfigure:
                self._resume_spinnaker_acquisition_after_reconfigure()

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
        if self.is_spinnaker_camera() and self.isRunning():
            paused_for_reconfigure = self._pause_spinnaker_acquisition_for_reconfigure()
        try:
            if not self._set_enum_node_by_name(node_name, bit_depth):
                return None
            self._refresh_camera_settings_cache(force=True)
            return self._read_enum_node_symbolic(node_name) or bit_depth
        finally:
            if paused_for_reconfigure:
                self._resume_spinnaker_acquisition_after_reconfigure()

    def get_camera_line_capabilities(self) -> List[Dict[str, object]]:
        """Enumerate GenICam camera line selectors, modes, and sources."""
        if not self.is_genicam_camera():
            return []

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

    def connect_camera(self, camera_info: Optional[dict] = None) -> bool:
        """Connect to a Basler, FLIR, or generic USB camera."""
        try:
            camera_info = camera_info or {"type": "basler", "index": 0}
            camera_type = camera_info.get("type")

            if camera_type == "usb":
                index = int(camera_info.get("index", 0))
                backend = cv2.CAP_MSMF if os.name == "nt" else cv2.CAP_V4L2
                self.usb_capture = cv2.VideoCapture(index, backend)
                if not self.usb_capture or not self.usb_capture.isOpened():
                    self.error_occurred.emit("No USB camera found!")
                    return False

                self.camera_type = "usb"
                self.usb_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width or 1080)
                self.usb_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height or 1080)
                self.usb_capture.set(cv2.CAP_PROP_FPS, self.fps_target)
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

            self.camera = pylon.InstantCamera(tlFactory.CreateDevice(selected_device))
            self.camera.Open()
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
            self.pyspin_cam_list = self.pyspin_system.GetCameras()
            if self.pyspin_cam_list.GetSize() == 0:
                raise RuntimeError("No FLIR Spinnaker camera found")

            index = max(0, min(index, self.pyspin_cam_list.GetSize() - 1))
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
                self._reset_frame_drop_stats()

                # Start FFmpeg
                self._start_ffmpeg()

                self.is_recording = True
                self.recording_started_at = time.time()
                self._emit_frame_drop_stats(active=True)
                self.status_update.emit(f"Recording: {Path(filename).name}.mp4")
                return True

            except Exception as e:
                self.error_occurred.emit(f"Recording start error: {str(e)}")
                return False

    def _start_ffmpeg(self):
        """Start FFmpeg process for video encoding."""
        output_file = f"{self.recording_filename}.mp4"
        effective_width, effective_height = self._get_effective_dimensions()
        output_fps = float(self.recording_output_fps or self.fps_target or 30.0)
        if output_fps <= 0:
            output_fps = 30.0

        # Build FFmpeg command based on encoder
        if self.encoder == "h264_nvenc":
            # NVIDIA GPU encoder
            codec_args = [
                '-c:v', 'h264_nvenc',
                '-preset', self.encoder_preset,
                '-b:v', self.bitrate,
            ]
        elif self.encoder == "libx264":
            # CPU encoder (software)
            codec_args = [
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
            ]
        elif self.encoder == "h264_qsv":
            # Intel QuickSync
            codec_args = [
                '-c:v', 'h264_qsv',
                '-preset', 'fast',
                '-b:v', self.bitrate,
            ]
        else:
            # Fallback to libx264
            codec_args = [
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
            ]

        pixel_format = 'gray'
        if self.image_format == "BGR8":
            pixel_format = 'bgr24'

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
            '-pix_fmt', 'yuv420p', # Ensure compatible output format
            '-an',  # No audio
            output_file
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
                creationflags=creationflags
            )
            self._start_ffmpeg_stderr_thread()
            # Check if process died immediately
            time.sleep(0.1)
            if self.ffmpeg_process.poll() is not None:
                stderr = b""
                try:
                    _, stderr = self.ffmpeg_process.communicate(timeout=1)
                except Exception:
                    pass
                raise Exception(f"FFmpeg failed to start: {stderr.decode(errors='replace')}")

            self.status_update.emit(f"FFmpeg started: {self.encoder}")
        except FileNotFoundError:
            raise Exception("FFmpeg not found! Install FFmpeg and add to PATH")
        except Exception as e:
            raise Exception(f"FFmpeg error: {str(e)}")

    def stop_recording(self):
        """Stop recording and save metadata."""
        with self.recording_lock:
            if not self.is_recording:
                return

            try:
                self.is_recording = False
                self._emit_frame_drop_stats(active=False)

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

                self.status_update.emit("Recording stopped")
                self.recording_stopped.emit()

            except Exception as e:
                self.error_occurred.emit(f"Stop recording error: {str(e)}")
                self.recording_output_fps = None
                self.recording_stopped.emit()

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
                self._configure_basler_stream_buffers()
                self.camera.StartGrabbing(self._get_basler_grab_strategy())

                while self.running:
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
                while self.running:
                    ok, frame = self.usb_capture.read()
                    if not ok:
                        self.status_update.emit("USB frame timeout...")
                        time.sleep(0.01)
                        continue

                    packet = self._capture_usb_frame_packet(frame)
                    if packet is not None:
                        self._enqueue_frame_packet(packet)
                    self._update_fps()
            else:
                while self.running:
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
        if self.is_recording:
            metadata.update(self._extract_chunk_data(grab_result))
        return FramePacket(
            backend="basler",
            frame=frame,
            metadata=metadata,
            requested_format=self.image_format,
        )

    def _capture_usb_frame_packet(self, frame: np.ndarray) -> Optional[FramePacket]:
        """Snapshot an OpenCV USB frame for downstream processing."""
        return FramePacket(
            backend="usb",
            frame=np.ascontiguousarray(frame),
            metadata={"timestamp_software": time.time()},
            requested_format=self.image_format,
        )

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
        """Write the frame if recording, then emit preview if enabled."""
        if self._handle_record_frame(record_frame, metadata):
            return
        if preview_frame is not None:
            self.frame_ready.emit(preview_frame)
            height, width = preview_frame.shape[:2]
            self.preview_packet_ready.emit(
                PreviewFramePacket(
                    frame=np.asarray(preview_frame),
                    frame_index=int(metadata.get("frame_id", self.frame_counter)),
                    timestamp_s=float(metadata.get("timestamp_software", time.time()) or time.time()),
                    width=int(width),
                    height=int(height),
                    metadata=dict(metadata),
                )
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

                metadata["frame_id"] = self.frame_counter
                self.frame_counter += 1
                self._track_recorded_frame_timing(metadata)

                self.metadata_buffer.append(metadata)
                self.frame_recorded.emit(metadata)

                if self.max_record_frames is not None and self.frame_counter >= self.max_record_frames:
                    self.status_update.emit(f"Reached frame target: {self.max_record_frames} frames")
                    stop_now = True
        except Exception as e:
            self.error_occurred.emit(f"Frame write error: {str(e)}")
            self.stop_recording()
            return True

        if stop_now:
            self.stop_recording()
            return True
        return False

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

        try:
            # Timestamp
            if hasattr(grab_result, 'ChunkTimestamp'):
                try:
                    if callable(getattr(grab_result.ChunkTimestamp, 'IsReadable', None)):
                        if grab_result.ChunkTimestamp.IsReadable():
                            metadata['timestamp_ticks'] = grab_result.ChunkTimestamp.GetValue()
                    else:
                        metadata['timestamp_ticks'] = grab_result.ChunkTimestamp.GetValue()
                except:
                    metadata['timestamp_ticks'] = None
            else:
                metadata['timestamp_ticks'] = None

            # Exposure time
            if hasattr(grab_result, 'ChunkExposureTime'):
                try:
                    if callable(getattr(grab_result.ChunkExposureTime, 'IsReadable', None)):
                        if grab_result.ChunkExposureTime.IsReadable():
                            metadata['exposure_time_us'] = grab_result.ChunkExposureTime.GetValue()
                    else:
                        metadata['exposure_time_us'] = grab_result.ChunkExposureTime.GetValue()
                except:
                    metadata['exposure_time_us'] = None
            else:
                metadata['exposure_time_us'] = None

            # Line status (GPIO)
            if hasattr(grab_result, 'ChunkLineStatusAll'):
                try:
                    if callable(getattr(grab_result.ChunkLineStatusAll, 'IsReadable', None)):
                        if grab_result.ChunkLineStatusAll.IsReadable():
                            line_status = grab_result.ChunkLineStatusAll.GetValue()
                        else:
                            line_status = grab_result.ChunkLineStatusAll.GetValue()
                    else:
                        line_status = grab_result.ChunkLineStatusAll.GetValue()

                    metadata['line_status_all'] = line_status
                    # Correct extraction based on standard Basler Line Status All chunk
                    # Bit 0 = Line 1, Bit 1 = Line 2, etc.
                    metadata['line1_status'] = (line_status >> 0) & 0x01
                    metadata['line2_status'] = (line_status >> 1) & 0x01
                    metadata['line3_status'] = (line_status >> 2) & 0x01
                    metadata['line4_status'] = (line_status >> 3) & 0x01
                except:
                    metadata['line_status_all'] = None
                    metadata['line1_status'] = None
                    metadata['line2_status'] = None
                    metadata['line3_status'] = None
                    metadata['line4_status'] = None
            else:
                metadata['line_status_all'] = None
                metadata['line1_status'] = None
                metadata['line2_status'] = None
                metadata['line3_status'] = None
                metadata['line4_status'] = None

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

