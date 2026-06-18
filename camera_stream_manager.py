"""
Multi-camera stream management.

The main window keeps its fully featured primary camera (TTL sync, live
detection, planner metadata). This module adds auxiliary streams: each one
wraps its own CameraWorker, previews independently, and records in lock-step
with the primary recording. CSV metadata written by auxiliary workers is
zero-referenced through metadata_normalization so every output starts at
time 0, exactly like the primary stream.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from camera_worker import CameraWorker
from metadata_normalization import normalize_metadata_csv_file


def camera_identity_key(camera_info: Optional[Dict]) -> str:
    """Stable identity for a discovered camera so two streams never share one device."""
    if not camera_info:
        return ""
    camera_type = str(camera_info.get("type", "") or "").strip().lower()
    backend = str(camera_info.get("backend", "") or "").strip().lower()
    serial = str(camera_info.get("serial", "") or "").strip()
    if serial:
        return f"{camera_type}:{backend}:serial={serial}"
    video_index = camera_info.get("video_index", None)
    if video_index is not None and str(video_index).strip() != "":
        return f"{camera_type}:{backend}:video={video_index}"
    return f"{camera_type}:{backend}:index={camera_info.get('index', '')}"


def slugify_stream_suffix(label: str, fallback: str) -> str:
    """Filesystem-safe suffix for per-stream recording outputs."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(label or "")).strip("_").lower()
    return cleaned or fallback


class AuxCameraStream(QObject):
    """One auxiliary camera: its own worker, preview, and synchronized recording."""

    state_changed = Signal()
    warning = Signal(str)

    # No frames after this long into a recording means the stream has stalled.
    RECORDING_WATCHDOG_MS = 4000

    def __init__(self, stream_id: int, parent: Optional[QObject] = None):
        """Create an idle stream owning its own CameraWorker and stall watchdog.

        ``stream_id`` is 0-based among auxiliaries (the primary camera is the
        main window's own worker); the watchdog fires once shortly after a
        recording starts to warn if no frames have arrived.
        """
        super().__init__(parent)
        self.stream_id = int(stream_id)
        self.worker = CameraWorker()
        self.camera_info: Optional[Dict] = None
        self.is_connected = False
        self.recording_path: Optional[str] = None
        self.worker.recording_stopped.connect(self._on_worker_recording_stopped)
        self._recording_watchdog = QTimer(self)
        self._recording_watchdog.setSingleShot(True)
        self._recording_watchdog.timeout.connect(self._check_recording_delivers_frames)

    @property
    def display_name(self) -> str:
        """Human label for this stream, e.g. "Camera 2" (1-based for the UI)."""
        return f"Camera {self.stream_id + 1}"

    @property
    def camera_label(self) -> str:
        """Label of the connected source, or "No source" when disconnected."""
        if not self.camera_info:
            return "No source"
        return str(self.camera_info.get("label", "Camera") or "Camera")

    @property
    def is_recording(self) -> bool:
        """True while this stream's worker is actively writing a recording."""
        return bool(self.worker.is_recording)

    def filename_suffix(self) -> str:
        """Suffix appended to the primary recording base path for this stream."""
        info = self.camera_info or {}
        backend = str(info.get("backend", "") or info.get("type", "") or "cam").strip().lower()
        serial = str(info.get("serial", "") or "").strip()
        if serial:
            return slugify_stream_suffix(f"{backend}_{serial}", f"cam{self.stream_id + 1}")
        return slugify_stream_suffix(f"{backend}_{self.stream_id + 1}", f"cam{self.stream_id + 1}")

    def connect_camera(self, camera_info: Dict) -> bool:
        """Open the given camera and start its acquisition thread.

        Idempotent (returns True if already connected); returns False if the
        worker could not open the device. Emits ``state_changed`` on success.
        """
        if self.is_connected:
            return True
        if not self.worker.connect_camera(camera_info):
            return False
        self.camera_info = dict(camera_info)
        self.is_connected = True
        self.worker.start()
        self.state_changed.emit()
        return True

    def disconnect_camera(self) -> None:
        """Stop recording (if any), halt the thread, and release the device."""
        if not self.is_connected:
            return
        if self.worker.is_recording:
            self.worker.stop_recording()
        self.worker.stop()
        self.worker.wait()
        self.worker.disconnect_camera()
        self.is_connected = False
        self.camera_info = None
        self.state_changed.emit()

    def start_recording(self, base_filepath: str, duration_seconds: Optional[float] = None) -> Optional[str]:
        """Record alongside the primary stream as ``<base>_<suffix>``.

        ``duration_seconds`` (when set) caps this stream to an exact length via
        the worker's frame-count limit, so every camera's file matches the
        requested recording duration rather than drifting with GUI-thread lag.
        """
        if not self.is_connected or self.worker.is_recording:
            return None
        stream_path = f"{base_filepath}_{self.filename_suffix()}"
        self.worker.set_recording_duration_limit(duration_seconds)
        if not self.worker.start_recording(stream_path):
            return None
        self.recording_path = stream_path
        self._recording_watchdog.start(self.RECORDING_WATCHDOG_MS)
        self.state_changed.emit()
        return stream_path

    def stop_recording(self) -> None:
        """Cancel the stall watchdog and stop this stream's recording, if active."""
        self._recording_watchdog.stop()
        if self.worker.is_recording:
            self.worker.stop_recording()

    def _check_recording_delivers_frames(self) -> None:
        """Watchdog: warn if a recording is running but no frames have arrived.

        A zero-frame recording usually means the USB bus is saturated by too
        many simultaneous streams.
        """
        if not self.worker.is_recording:
            return
        if int(self.worker.frame_counter or 0) > 0:
            return
        self.warning.emit(
            f"{self.display_name} is recording but has not captured any frames. "
            "The USB bus may be saturated — try another port, lower the resolution, "
            "or reduce the number of simultaneous streams."
        )

    def _on_worker_recording_stopped(self) -> None:
        """On worker stop: zero-reference the CSV, or discard an empty recording.

        Auxiliary CSVs are written directly by the worker, so this is where they
        get the same zero-based timestamps as the primary stream's exports.
        """
        self._recording_watchdog.stop()
        path = self.recording_path
        self.recording_path = None
        if path:
            if int(self.worker.frame_counter or 0) <= 0:
                # Nothing was captured: remove the empty container so a stub
                # video cannot be mistaken for data, and tell the user why.
                self._discard_empty_recording(path)
            else:
                try:
                    normalize_metadata_csv_file(f"{path}_metadata.csv")
                except Exception:
                    pass
        self.state_changed.emit()

    def _discard_empty_recording(self, path: str) -> None:
        """Delete a near-empty stub .mp4 (zero frames) and warn the operator."""
        from pathlib import Path

        try:
            stub = Path(f"{path}.mp4")
            if stub.is_file() and stub.stat().st_size < 4096:
                stub.unlink()
        except Exception:
            pass
        self.warning.emit(
            f"{self.display_name} recorded zero frames — its video was discarded. "
            "Check USB bandwidth or camera state before the next session."
        )

    def shutdown(self) -> None:
        """Best-effort disconnect used during teardown (never raises)."""
        try:
            self.disconnect_camera()
        except Exception:
            pass


class CameraStreamManager(QObject):
    """Owns auxiliary camera streams and keeps their recordings synchronized."""

    streams_changed = Signal()
    status_message = Signal(str)
    error_message = Signal(str)

    MAX_STREAMS = 11  # auxiliary streams; the primary camera makes twelve total

    def __init__(self, parent: Optional[QObject] = None):
        """Create an empty manager; stream ids start at 1 (0 is the primary)."""
        super().__init__(parent)
        self._streams: List[AuxCameraStream] = []
        self._next_stream_id = 1  # 0 is the primary camera
        # The main window provides the identity of the primary camera so
        # auxiliary scans can exclude devices that are already in use.
        self.primary_camera_info_provider: Optional[Callable[[], Optional[Dict]]] = None

    def streams(self) -> List[AuxCameraStream]:
        """Return a shallow copy of the managed auxiliary streams."""
        return list(self._streams)

    def can_add_stream(self) -> bool:
        """True while the auxiliary stream count is below MAX_STREAMS."""
        return len(self._streams) < self.MAX_STREAMS

    def create_stream(self) -> Optional[AuxCameraStream]:
        """Create and register a new (disconnected) stream, or None if at limit."""
        if not self.can_add_stream():
            self.error_message.emit(
                f"Stream limit reached ({self.MAX_STREAMS + 1} cameras including the primary)."
            )
            return None
        stream = AuxCameraStream(self._next_stream_id, parent=self)
        stream.warning.connect(self.error_message)
        self._next_stream_id += 1
        self._streams.append(stream)
        self.streams_changed.emit()
        return stream

    def remove_stream(self, stream: AuxCameraStream) -> None:
        """Shut down and unregister one stream, freeing its camera."""
        if stream not in self._streams:
            return
        stream.shutdown()
        self._streams.remove(stream)
        stream.deleteLater()
        self.streams_changed.emit()

    def connected_streams(self) -> List[AuxCameraStream]:
        """Return only the streams that currently have a camera connected."""
        return [stream for stream in self._streams if stream.is_connected]

    def used_camera_keys(self) -> set[str]:
        """Identity keys of every camera already claimed by a stream."""
        keys = set()
        if self.primary_camera_info_provider is not None:
            try:
                primary_info = self.primary_camera_info_provider()
            except Exception:
                primary_info = None
            primary_key = camera_identity_key(primary_info)
            if primary_key:
                keys.add(primary_key)
        for stream in self._streams:
            key = camera_identity_key(stream.camera_info)
            if key:
                keys.add(key)
        return keys

    def start_recording_all(self, base_filepath: str, duration_seconds: Optional[float] = None) -> List[str]:
        """Start every connected auxiliary stream; returns the started paths.

        ``duration_seconds`` is forwarded so each stream self-stops at the exact
        requested length on its own acquisition thread.
        """
        started: List[str] = []
        for stream in self.connected_streams():
            stream_path = stream.start_recording(base_filepath, duration_seconds)
            if stream_path:
                started.append(stream_path)
                self.status_message.emit(f"{stream.display_name} recording: {stream_path}.mp4")
            else:
                self.error_message.emit(f"{stream.display_name} failed to start recording.")
        return started

    def stop_recording_all(self) -> None:
        """Stop recording on every stream (best-effort; ignores per-stream errors)."""
        for stream in self._streams:
            try:
                stream.stop_recording()
            except Exception:
                pass

    def any_recording(self) -> bool:
        """True while any managed stream is still recording."""
        return any(stream.is_recording for stream in self._streams)

    def shutdown(self) -> None:
        """Disconnect every stream (used on application exit)."""
        for stream in self._streams:
            stream.shutdown()
