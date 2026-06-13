"""
Auxiliary camera stream tiles for the live workspace.

Each tile hosts one AuxCameraStream: source picker, connect/disconnect,
live preview, recording badge, and per-stream acquisition settings
(FPS, exposure, image format) applied directly to that stream's worker.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
)

from camera_stream_manager import AuxCameraStream, camera_identity_key

CHIP_TONES = {
    "default": ("#16283c", "#9fb4cb"),
    "accent": ("#10324f", "#7cc7ff"),
    "success": ("#11351f", "#9bf57f"),
    "danger": ("#3a1420", "#ff7d9b"),
    "warning": ("#393012", "#ffc46b"),
}


def ndarray_to_qpixmap(frame: np.ndarray) -> Optional[QPixmap]:
    """Convert a preview frame (gray 2-D or RGB 3-D) to a QPixmap."""
    if frame is None:
        return None
    array = np.ascontiguousarray(frame)
    if array.ndim == 2:
        height, width = array.shape
        image = QImage(array.data, width, height, array.strides[0], QImage.Format_Grayscale8)
        return QPixmap.fromImage(image.copy())
    if array.ndim == 3 and array.shape[2] == 3:
        height, width, _ = array.shape
        image = QImage(array.data, width, height, array.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(image.copy())
    return None


class AuxCameraTile(QFrame):
    """Self-contained live tile for one auxiliary camera stream."""

    def __init__(
        self,
        stream: AuxCameraStream,
        scan_cameras: Callable[[], List[Dict]],
        used_camera_keys: Callable[[], set],
        request_remove: Callable[["AuxCameraTile"], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.stream = stream
        self._scan_cameras = scan_cameras
        self._used_camera_keys = used_camera_keys
        self._request_remove = request_remove
        self._last_pixmap: Optional[QPixmap] = None

        self.setObjectName("WorkspaceCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Small enough that a 12-stream grid (4x3) still fits a 1080p workspace.
        self.setMinimumSize(240, 185)
        self._build_ui()
        self._connect_stream_signals()
        self.refresh_sources()
        self._sync_state()

    # ----- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(6)
        self.title_label = QLabel(self.stream.display_name)
        self.title_label.setStyleSheet("font-size: 12px; font-weight: 700; color: #edf4ff;")
        header.addWidget(self.title_label)

        self.status_chip = QLabel("Offline")
        self.status_chip.setAlignment(Qt.AlignCenter)
        self._style_chip(self.status_chip, "warning")
        header.addWidget(self.status_chip)

        self.resolution_chip = QLabel("-- x --")
        self.resolution_chip.setAlignment(Qt.AlignCenter)
        self._style_chip(self.resolution_chip, "default")
        header.addWidget(self.resolution_chip)
        header.addStretch()

        self.btn_close = QToolButton()
        self.btn_close.setText("✕")
        self.btn_close.setToolTip("Remove this camera stream")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setStyleSheet(
            "QToolButton { color: #8fa6bf; border: none; font-size: 12px; padding: 2px 6px; }"
            "QToolButton:hover { color: #ff7d9b; }"
        )
        self.btn_close.clicked.connect(lambda: self._request_remove(self))
        header.addWidget(self.btn_close)
        layout.addLayout(header)

        source_row = QHBoxLayout()
        source_row.setSpacing(6)
        self.combo_source = QComboBox()
        self.combo_source.setToolTip("Pick a camera that is not already used by another stream")
        source_row.addWidget(self.combo_source, 1)

        self.btn_refresh = QToolButton()
        self.btn_refresh.setText("↻")
        self.btn_refresh.setToolTip("Rescan for available cameras")
        self.btn_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_refresh.clicked.connect(self.refresh_sources)
        source_row.addWidget(self.btn_refresh)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setToolTip("Connect this stream to the selected camera")
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        source_row.addWidget(self.btn_connect)
        layout.addLayout(source_row)

        self.preview_label = QLabel("No camera connected")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(120)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet(
            "QLabel { background-color: #050b12; border: 1px solid #203246;"
            " border-radius: 10px; color: #56708c; font-size: 11px; }"
        )
        layout.addWidget(self.preview_label, 1)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(6)

        fps_caption = QLabel("FPS")
        fps_caption.setStyleSheet("color: #8fa6bf; font-size: 10px;")
        settings_row.addWidget(fps_caption)
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1.0, 500.0)
        self.spin_fps.setValue(30.0)
        self.spin_fps.setDecimals(1)
        self.spin_fps.setMinimumWidth(68)
        self.spin_fps.setToolTip("Target acquisition frame rate for this stream")
        self.spin_fps.valueChanged.connect(self._on_fps_changed)
        settings_row.addWidget(self.spin_fps)

        exposure_caption = QLabel("Exp ms")
        exposure_caption.setStyleSheet("color: #8fa6bf; font-size: 10px;")
        settings_row.addWidget(exposure_caption)
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(0.01, 1000.0)
        self.spin_exposure.setValue(20.0)
        self.spin_exposure.setDecimals(2)
        self.spin_exposure.setMinimumWidth(76)
        self.spin_exposure.setToolTip("Exposure time (machine-vision cameras only)")
        self.spin_exposure.valueChanged.connect(self._on_exposure_changed)
        settings_row.addWidget(self.spin_exposure)

        self.combo_format = QComboBox()
        self.combo_format.addItems(["Mono8", "BGR8"])
        self.combo_format.setMinimumWidth(78)
        self.combo_format.setToolTip("Recorded image format for this stream")
        self.combo_format.currentTextChanged.connect(self._on_format_changed)
        settings_row.addWidget(self.combo_format)
        settings_row.addStretch()
        layout.addLayout(settings_row)

    def _style_chip(self, label: QLabel, tone: str) -> None:
        background, foreground = CHIP_TONES.get(tone, CHIP_TONES["default"])
        label.setStyleSheet(
            f"QLabel {{ background-color: {background}; color: {foreground};"
            " border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: 700; }}"
        )

    # ----- Stream wiring ----------------------------------------------------

    def _connect_stream_signals(self) -> None:
        worker = self.stream.worker
        worker.frame_ready.connect(self._on_frame_ready)
        worker.error_occurred.connect(self._on_worker_error)
        self.stream.state_changed.connect(self._sync_state)

    @Slot(np.ndarray)
    def _on_frame_ready(self, frame: np.ndarray) -> None:
        pixmap = ndarray_to_qpixmap(frame)
        if pixmap is None:
            return
        self._last_pixmap = pixmap
        self._apply_scaled_pixmap()
        worker = self.stream.worker
        if worker.width and worker.height:
            self.resolution_chip.setText(f"{worker.width} x {worker.height}")

    def _apply_scaled_pixmap(self) -> None:
        if self._last_pixmap is None:
            return
        target = self.preview_label.size()
        if target.width() < 8 or target.height() < 8:
            return
        self.preview_label.setPixmap(
            self._last_pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        self.status_chip.setText("Error")
        self._style_chip(self.status_chip, "danger")
        self.status_chip.setToolTip(str(message))

    # ----- Actions ----------------------------------------------------------

    def refresh_sources(self) -> None:
        """Populate the source combo with cameras not claimed by other streams."""
        current_key = camera_identity_key(self.combo_source.currentData())
        self.combo_source.clear()
        used = self._used_camera_keys()
        own_key = camera_identity_key(self.stream.camera_info)
        available = []
        for camera_info in self._scan_cameras():
            key = camera_identity_key(camera_info)
            if key and key in used and key != own_key:
                continue
            available.append(camera_info)
        if not available:
            self.combo_source.addItem("No free cameras", None)
            return
        for camera_info in available:
            self.combo_source.addItem(str(camera_info.get("label", "Camera")), camera_info)
        if current_key:
            for index in range(self.combo_source.count()):
                if camera_identity_key(self.combo_source.itemData(index)) == current_key:
                    self.combo_source.setCurrentIndex(index)
                    break

    def _on_connect_clicked(self) -> None:
        if self.stream.is_connected:
            self.stream.disconnect_camera()
            return
        camera_info = self.combo_source.currentData()
        if not camera_info:
            self.status_chip.setText("Pick a source")
            self._style_chip(self.status_chip, "warning")
            return
        worker = self.stream.worker
        worker.set_target_fps(float(self.spin_fps.value()))
        worker.set_preview_max_width(960)
        worker.image_format = self.combo_format.currentText()
        if self.stream.connect_camera(camera_info):
            self._apply_stream_settings()
        else:
            self.status_chip.setText("Connect failed")
            self._style_chip(self.status_chip, "danger")

    def _apply_stream_settings(self) -> None:
        worker = self.stream.worker
        if not self.stream.is_connected:
            return
        try:
            if worker.is_genicam_camera():
                worker.set_camera_frame_rate(float(self.spin_fps.value()))
                worker.set_camera_exposure_ms(float(self.spin_exposure.value()))
        except Exception:
            pass

    def _on_fps_changed(self, value: float) -> None:
        self.stream.worker.set_target_fps(float(value))
        if self.stream.is_connected:
            try:
                if self.stream.worker.is_genicam_camera():
                    self.stream.worker.set_camera_frame_rate(float(value))
            except Exception:
                pass

    def _on_exposure_changed(self, value: float) -> None:
        if self.stream.is_connected:
            try:
                if self.stream.worker.is_genicam_camera():
                    self.stream.worker.set_camera_exposure_ms(float(value))
            except Exception:
                pass

    def _on_format_changed(self, image_format: str) -> None:
        try:
            self.stream.worker.set_image_format(image_format)
        except Exception:
            pass

    # ----- State sync ---------------------------------------------------------

    @Slot()
    def _sync_state(self) -> None:
        connected = self.stream.is_connected
        recording = self.stream.is_recording
        if recording:
            self.status_chip.setText("Recording")
            self._style_chip(self.status_chip, "danger")
        elif connected:
            self.status_chip.setText("Live")
            self._style_chip(self.status_chip, "success")
        else:
            self.status_chip.setText("Offline")
            self._style_chip(self.status_chip, "warning")
            self.resolution_chip.setText("-- x --")
            self._last_pixmap = None
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("No camera connected")
        if connected:
            self.preview_label.setText("")
            self.title_label.setToolTip(self.stream.camera_label)

        self.btn_connect.setText("Disconnect" if connected else "Connect")
        self.btn_connect.setEnabled(not recording)
        self.btn_connect.setToolTip(
            "Stop the recording before disconnecting this stream"
            if recording
            else "Connect this stream to the selected camera"
        )
        self.combo_source.setEnabled(not connected)
        self.btn_refresh.setEnabled(not connected)
        self.btn_close.setEnabled(not recording)
        exposure_allowed = bool(
            connected and self._worker_supports_exposure() or not connected
        )
        self.spin_exposure.setEnabled(exposure_allowed)
        self.combo_format.setEnabled(not recording)

    def _worker_supports_exposure(self) -> bool:
        try:
            return bool(self.stream.worker.is_genicam_camera())
        except Exception:
            return False
