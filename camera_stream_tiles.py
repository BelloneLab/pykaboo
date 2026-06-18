"""
Auxiliary camera stream tiles for the live workspace.

Each tile hosts one AuxCameraStream: source picker, connect/disconnect,
live preview, recording badge, and per-stream acquisition settings
(FPS, exposure, image format) applied directly to that stream's worker.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
)

from camera_stream_manager import AuxCameraStream, camera_identity_key

# Auxiliary tiles are monitor views, not the science feed (that is the primary
# preview and the recorded files). Repainting them at the full acquisition rate
# saturates the GUI thread once several streams run, so the live preview is
# capped here. The recorded video is unaffected — it is encoded in the worker.
AUX_PREVIEW_FPS = 15.0
AUX_PREVIEW_MAX_WIDTH = 960
# Burst guard on the GUI side in case a worker delivers faster than its cap.
_AUX_PAINT_MIN_INTERVAL_S = 1.0 / 18.0

CHIP_TONES = {
    "default": ("#16283c", "#9fb4cb"),
    "accent": ("#10324f", "#7cc7ff"),
    "success": ("#11351f", "#9bf57f"),
    "danger": ("#3a1420", "#ff7d9b"),
    "warning": ("#393012", "#ffc46b"),
}


def ndarray_to_qpixmap(frame: np.ndarray) -> Optional[QPixmap]:
    """Convert a preview frame (gray 2-D or RGB 3-D) to a QPixmap.

    ``array`` is kept alive until ``QPixmap.fromImage`` (which deep-copies the
    pixels) returns, so the previous explicit ``QImage.copy()`` was a redundant
    full-frame memcpy on the GUI thread and is dropped.
    """
    if frame is None:
        return None
    array = np.ascontiguousarray(frame)
    if array.ndim == 2:
        height, width = array.shape
        image = QImage(array.data, width, height, array.strides[0], QImage.Format_Grayscale8)
        return QPixmap.fromImage(image)
    if array.ndim == 3 and array.shape[2] == 3:
        height, width, _ = array.shape
        image = QImage(array.data, width, height, array.strides[0], QImage.Format_RGB888)
        return QPixmap.fromImage(image)
    return None


class AuxCameraTile(QFrame):
    """Self-contained live tile for one auxiliary camera stream."""

    def __init__(
        self,
        stream: AuxCameraStream,
        scan_cameras: Callable[[], List[Dict]],
        used_camera_keys: Callable[[], set],
        request_remove: Callable[["AuxCameraTile"], None],
        settings=None,
        parent=None,
    ) -> None:
        """Build a live tile bound to one ``AuxCameraStream``.

        The callables let the tile cooperate with the manager without importing
        it: ``scan_cameras`` lists devices, ``used_camera_keys`` excludes ones
        already claimed, and ``request_remove`` asks the owner to drop this tile.
        ``settings`` is the shared ``QSettings`` used to remember this camera
        slot's acquisition defaults across launches (the "Set As Default"
        button); when ``None`` the tile simply uses built-in fallbacks.
        """
        super().__init__(parent)
        self.stream = stream
        self._scan_cameras = scan_cameras
        self._used_camera_keys = used_camera_keys
        self._request_remove = request_remove
        self._settings = settings
        self._defaults = self._load_stream_defaults()
        # One-shot guard: reseed the resolution spin-boxes to the camera's real
        # geometry the first time a frame arrives (unless a saved default has
        # already been applied), so the dialog never misreports the live size.
        self._resolution_seeded = False
        self._last_pixmap: Optional[QPixmap] = None
        self._last_paint_s: float = 0.0

        self.setObjectName("WorkspaceCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Small enough that a 12-stream grid (4x3) still fits a 1080p workspace.
        self.setMinimumSize(240, 185)
        self._build_ui()
        self._connect_stream_signals()
        self.refresh_sources()
        self._sync_state()

    # ----- Persistence (per-camera-slot defaults) --------------------------

    def _settings_key(self, name: str) -> str:
        """QSettings key for one persisted default of this camera slot."""
        return f"aux_camera/{self.stream.stream_id}/{name}"

    def _has_saved_defaults(self) -> bool:
        """True once the operator has saved a default for this camera slot."""
        return self._settings is not None and bool(
            self._settings.contains(self._settings_key("width"))
        )

    def _load_stream_defaults(self) -> Dict[str, object]:
        """Read this slot's persisted defaults, falling back to sensible values.

        Auxiliary streams default to 1920x1080 BGR8 (most monitor webcams are
        Full-HD colour), but every field is overridden by whatever the operator
        last stored with "Set As Default".
        """
        settings = self._settings

        def read(name, fallback, cast):
            if settings is None:
                return fallback
            try:
                return cast(settings.value(self._settings_key(name), fallback))
            except (TypeError, ValueError):
                return fallback

        return {
            "fps": read("fps", 30.0, float),
            "exposure_ms": read("exposure_ms", 20.0, float),
            "image_format": str(read("image_format", "BGR8", str)),
            "width": read("width", 1920, int),
            "height": read("height", 1080, int),
            "preview_fps": read("preview_fps", AUX_PREVIEW_FPS, float),
            "preview_width": read("preview_width", AUX_PREVIEW_MAX_WIDTH, int),
            "white_balance": str(read("white_balance", "Continuous", str)),
            "wb_red": read("wb_red", 0.0, float),
            "wb_blue": read("wb_blue", 0.0, float),
            "gain": read("gain", 0.0, float),
        }

    def _save_settings_as_default(self) -> None:
        """Persist the current popup settings as this slot's next-launch default."""
        if self._settings is None:
            return
        values = {
            "fps": float(self.spin_fps.value()),
            "exposure_ms": float(self.spin_exposure.value()),
            "image_format": self.combo_format.currentText(),
            "width": int(self.spin_width.value()),
            "height": int(self.spin_height.value()),
            "preview_fps": float(self.spin_preview_fps.value()),
            "preview_width": int(self.spin_preview_width.value()),
            "white_balance": self.combo_white_balance.currentText(),
            "wb_red": float(self.spin_wb_red.value()),
            "wb_blue": float(self.spin_wb_blue.value()),
            "gain": float(self.spin_gain.value()),
        }
        for name, value in values.items():
            self._settings.setValue(self._settings_key(name), value)
        self._settings.sync()
        self._defaults = dict(values)
        # Brief inline confirmation so the operator knows it took.
        self.btn_set_default.setText("Saved ✓")
        QTimer.singleShot(1600, lambda: self.btn_set_default.setText("Set As Default"))

    # ----- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        """Lay out the tile: header chips, source picker, preview, and gear footer."""
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

        # Compact footer: an at-a-glance summary plus a gear that opens the full
        # per-stream settings popup (mirrors the primary camera controls).
        footer_row = QHBoxLayout()
        footer_row.setSpacing(6)
        self.settings_summary = QLabel("30 fps · BGR8")
        self.settings_summary.setStyleSheet("color: #8fa6bf; font-size: 10px;")
        footer_row.addWidget(self.settings_summary)
        footer_row.addStretch()

        self.btn_settings = QToolButton()
        self.btn_settings.setText("⚙  Settings")
        self.btn_settings.setCursor(Qt.PointingHandCursor)
        self.btn_settings.setToolTip("Open this camera's settings (frame rate, exposure, white balance, resolution…)")
        self.btn_settings.setStyleSheet(
            "QToolButton { color: #cfe2f5; background: rgba(157, 196, 240, 0.06);"
            " border: 1px solid #2b4364; border-radius: 8px; padding: 3px 10px; font-size: 11px; }"
            "QToolButton:hover { background: rgba(157, 196, 240, 0.13); border-color: #3f6491; }"
        )
        self.btn_settings.clicked.connect(self._open_settings_dialog)
        footer_row.addWidget(self.btn_settings)
        layout.addLayout(footer_row)

        # The control widgets live in the popup but are created now so the
        # connect path can read their values before the dialog is first opened.
        self._build_settings_dialog()

    def _style_chip(self, label: QLabel, tone: str) -> None:
        """Apply one of the named CHIP_TONES (default/success/danger/...) to a label."""
        background, foreground = CHIP_TONES.get(tone, CHIP_TONES["default"])
        label.setStyleSheet(
            f"QLabel {{ background-color: {background}; color: {foreground};"
            " border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: 700; }}"
        )

    # ----- Per-stream settings popup ---------------------------------------

    def _build_settings_dialog(self) -> None:
        """Create the per-stream settings popup, mirroring the primary camera."""
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle(f"{self.stream.display_name} Settings")
        self.settings_dialog.setModal(False)
        self.settings_dialog.resize(360, 0)
        dialog_layout = QVBoxLayout(self.settings_dialog)
        dialog_layout.setContentsMargins(14, 14, 14, 14)
        dialog_layout.setSpacing(12)

        # Acquisition ------------------------------------------------------
        acquisition_group = QGroupBox("Acquisition")
        acquisition_form = QFormLayout(acquisition_group)

        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1.0, 500.0)
        self.spin_fps.setValue(float(self._defaults["fps"]))
        self.spin_fps.setDecimals(1)
        self.spin_fps.setSuffix(" fps")
        self.spin_fps.setToolTip("Target acquisition frame rate for this stream")
        acquisition_form.addRow("Frame Rate:", self.spin_fps)

        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(0.01, 1000.0)
        self.spin_exposure.setValue(float(self._defaults["exposure_ms"]))
        self.spin_exposure.setDecimals(2)
        self.spin_exposure.setSuffix(" ms")
        self.spin_exposure.setToolTip("Exposure time (machine-vision cameras only)")
        acquisition_form.addRow("Exposure:", self.spin_exposure)

        self.combo_format = QComboBox()
        self.combo_format.addItems(["Mono8", "BGR8"])
        self.combo_format.setCurrentText(str(self._defaults["image_format"]))
        self.combo_format.setToolTip("Recorded image format for this stream")
        acquisition_form.addRow("Image Format:", self.combo_format)

        resolution_row = QHBoxLayout()
        self.spin_width = QSpinBox()
        self.spin_width.setRange(64, 8192)
        self.spin_width.setValue(int(self._defaults["width"]))
        self.spin_width.setSuffix(" px")
        resolution_row.addWidget(self.spin_width)
        resolution_row.addWidget(QLabel("×"))
        self.spin_height = QSpinBox()
        self.spin_height.setRange(64, 8192)
        self.spin_height.setValue(int(self._defaults["height"]))
        self.spin_height.setSuffix(" px")
        resolution_row.addWidget(self.spin_height)
        self.btn_apply_resolution = QPushButton("Apply")
        self.btn_apply_resolution.setToolTip("Apply this capture resolution to the camera")
        resolution_row.addWidget(self.btn_apply_resolution)
        acquisition_form.addRow("Resolution:", resolution_row)
        dialog_layout.addWidget(acquisition_group)

        # Preview ----------------------------------------------------------
        preview_group = QGroupBox("Preview")
        preview_form = QFormLayout(preview_group)

        self.spin_preview_fps = QDoubleSpinBox()
        self.spin_preview_fps.setRange(1.0, 60.0)
        self.spin_preview_fps.setDecimals(1)
        self.spin_preview_fps.setValue(float(self._defaults["preview_fps"]))
        self.spin_preview_fps.setSuffix(" fps")
        self.spin_preview_fps.setToolTip(
            "On-screen preview rate. Lower it to keep the UI responsive with many\n"
            "streams — it does not change the recorded video."
        )
        preview_form.addRow("Preview FPS:", self.spin_preview_fps)

        self.spin_preview_width = QSpinBox()
        self.spin_preview_width.setRange(0, 4096)
        self.spin_preview_width.setSingleStep(64)
        self.spin_preview_width.setSpecialValueText("Full resolution")
        self.spin_preview_width.setValue(int(self._defaults["preview_width"]))
        self.spin_preview_width.setToolTip(
            "Downscale the preview to at most this width before display.\n"
            "Smaller is lighter on the GUI; it does not change the recording."
        )
        preview_form.addRow("Preview Max Width:", self.spin_preview_width)
        dialog_layout.addWidget(preview_group)

        # Colour / advanced (GenICam cameras only) -------------------------
        self.color_group = QGroupBox("Colour && Gain (FLIR / Basler)")
        color_form = QFormLayout(self.color_group)

        self.combo_white_balance = QComboBox()
        self.combo_white_balance.addItems(["Continuous", "Off"])
        self.combo_white_balance.setCurrentText(str(self._defaults["white_balance"]))
        self.combo_white_balance.setToolTip("Automatic white balance mode")
        color_form.addRow("White Balance:", self.combo_white_balance)

        self.spin_wb_red = QDoubleSpinBox()
        self.spin_wb_red.setRange(0.0, 16.0)
        self.spin_wb_red.setDecimals(4)
        self.spin_wb_red.setSingleStep(0.01)
        self.spin_wb_red.setValue(float(self._defaults["wb_red"]))
        color_form.addRow("WB Red:", self.spin_wb_red)

        self.spin_wb_blue = QDoubleSpinBox()
        self.spin_wb_blue.setRange(0.0, 16.0)
        self.spin_wb_blue.setDecimals(4)
        self.spin_wb_blue.setSingleStep(0.01)
        self.spin_wb_blue.setValue(float(self._defaults["wb_blue"]))
        color_form.addRow("WB Blue:", self.spin_wb_blue)

        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.setRange(0.0, 48.0)
        self.spin_gain.setDecimals(2)
        self.spin_gain.setSuffix(" dB")
        self.spin_gain.setValue(float(self._defaults["gain"]))
        color_form.addRow("Gain:", self.spin_gain)
        dialog_layout.addWidget(self.color_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.btn_set_default = buttons.addButton("Set As Default", QDialogButtonBox.ActionRole)
        self.btn_set_default.setToolTip(
            "Remember these settings (resolution, frame rate, format, …) as the\n"
            f"default for {self.stream.display_name} the next time it is added."
        )
        self.btn_set_default.setEnabled(self._settings is not None)
        self.btn_set_default.clicked.connect(self._save_settings_as_default)
        buttons.rejected.connect(self.settings_dialog.close)
        dialog_layout.addWidget(buttons)

        # Wire signals only after every widget exists and is seeded, so the
        # initial setValue() calls above cannot fire handlers prematurely.
        self.spin_fps.valueChanged.connect(self._on_fps_changed)
        self.spin_exposure.valueChanged.connect(self._on_exposure_changed)
        self.combo_format.currentTextChanged.connect(self._on_format_changed)
        self.btn_apply_resolution.clicked.connect(self._on_resolution_apply)
        self.spin_preview_fps.valueChanged.connect(self._on_preview_fps_changed)
        self.spin_preview_width.valueChanged.connect(self._on_preview_width_changed)
        self.combo_white_balance.currentTextChanged.connect(self._on_white_balance_auto_changed)
        self.spin_wb_red.valueChanged.connect(self._on_white_balance_red_changed)
        self.spin_wb_blue.valueChanged.connect(self._on_white_balance_blue_changed)
        self.spin_gain.valueChanged.connect(self._on_gain_changed)

    def _open_settings_dialog(self) -> None:
        """Refresh control availability and show the per-stream settings popup."""
        self._update_settings_controls_enabled()
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _worker_is_genicam(self) -> bool:
        """True when the connected camera exposes GenICam controls (FLIR/Basler).

        Exposure, white balance, and gain only apply to GenICam devices, so this
        gates those controls (USB webcams return False).
        """
        try:
            return bool(self.stream.is_connected and self.stream.worker.is_genicam_camera())
        except Exception:
            return False

    def _update_settings_controls_enabled(self) -> None:
        """Grey out controls the active backend cannot honour."""
        connected = self.stream.is_connected
        recording = self.stream.is_recording
        genicam = self._worker_is_genicam()
        # Always-applicable controls (software or any backend).
        self.spin_fps.setEnabled(not recording)
        self.combo_format.setEnabled(not recording)
        self.spin_preview_fps.setEnabled(True)
        self.spin_preview_width.setEnabled(True)
        # Resolution: USB/FLIR-cv2 honour it via OpenCV; GenICam via nodes.
        self.spin_width.setEnabled(not recording)
        self.spin_height.setEnabled(not recording)
        self.btn_apply_resolution.setEnabled(connected and not recording)
        # Exposure / white balance / gain are GenICam-only.
        self.spin_exposure.setEnabled(genicam and not recording)
        self.color_group.setEnabled(genicam and not recording)
        if genicam and not recording:
            # Manual WB ratios only apply when auto white-balance is Off.
            manual = self.combo_white_balance.currentText().strip().lower() == "off"
            self.spin_wb_red.setEnabled(manual)
            self.spin_wb_blue.setEnabled(manual)
        self._update_settings_summary()

    def _update_settings_summary(self) -> None:
        """Refresh the one-line footer summary (e.g. "30 fps · BGR8")."""
        fps = self.spin_fps.value()
        fmt = self.combo_format.currentText()
        self.settings_summary.setText(f"{fps:g} fps · {fmt}")

    # ----- Stream wiring ----------------------------------------------------

    def _connect_stream_signals(self) -> None:
        """Wire the worker's frame/error/resolution signals and stream state to the tile."""
        worker = self.stream.worker
        worker.frame_ready.connect(self._on_frame_ready)
        worker.error_occurred.connect(self._on_worker_error)
        worker.resolution_changed.connect(self._on_resolution_changed_signal)
        self.stream.state_changed.connect(self._sync_state)

    @Slot(np.ndarray)
    def _on_frame_ready(self, frame: np.ndarray) -> None:
        """Paint the newest preview frame, throttled to protect the GUI thread.

        The QPixmap conversion and rescale run on the GUI thread, so a burst of
        frames across several tiles is what freezes the window; the time-gate
        caps repaints regardless of how fast the worker delivers.
        """
        now = time.monotonic()
        if now - self._last_paint_s < _AUX_PAINT_MIN_INTERVAL_S:
            return
        self._last_paint_s = now
        pixmap = ndarray_to_qpixmap(frame)
        if pixmap is None:
            return
        self._last_pixmap = pixmap
        self._apply_scaled_pixmap()
        worker = self.stream.worker
        if worker.width and worker.height:
            self.resolution_chip.setText(f"{worker.width} x {worker.height}")
            # Reflect the camera's real geometry in the spin-boxes once, so the
            # dialog stops showing the placeholder size. Skipped when a saved
            # default is driving the resolution (that path seeds them itself).
            if not self._resolution_seeded and not self._has_saved_defaults():
                self._resolution_seeded = True
                self._seed_resolution_spins(int(worker.width), int(worker.height))

    def _apply_scaled_pixmap(self, smooth: bool = False) -> None:
        """Scale the last frame into the preview label (fast for live, smooth on resize)."""
        if self._last_pixmap is None:
            return
        target = self.preview_label.size()
        if target.width() < 8 or target.height() < 8:
            return
        # Live frames use fast (nearest) scaling — the worker already produced a
        # downscaled preview, so smooth bilinear here is wasted GUI-thread work.
        # Smooth scaling is reserved for the occasional resize repaint.
        mode = Qt.SmoothTransformation if smooth else Qt.FastTransformation
        self.preview_label.setPixmap(
            self._last_pixmap.scaled(target, Qt.KeepAspectRatio, mode)
        )

    def resizeEvent(self, event) -> None:
        """Repaint the preview with smooth scaling when the tile is resized."""
        super().resizeEvent(event)
        self._apply_scaled_pixmap(smooth=True)

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        """Show a red Error chip with the worker's message as a tooltip."""
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
        """Toggle the stream: disconnect if connected, else connect the picked source.

        Seeds the worker's target FPS, preview rate/width, and image format from
        the settings popup before opening the device.
        """
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
        worker.set_preview_max_width(int(self.spin_preview_width.value()))
        worker.set_preview_fps(float(self.spin_preview_fps.value()))
        worker.image_format = self.combo_format.currentText()
        if self.stream.connect_camera(camera_info):
            self._apply_stream_settings()
        else:
            self.status_chip.setText("Connect failed")
            self._style_chip(self.status_chip, "danger")

    def _apply_stream_settings(self) -> None:
        """Push the acquisition settings to the camera right after it connects.

        Frame rate and exposure apply to GenICam devices only. When the operator
        has saved a default for this slot (the "Set As Default" button), the rest
        of that default is honoured too: white balance and gain on GenICam, plus
        the capture resolution on every backend. That last part is what makes a
        saved 1920x1080 default come up automatically instead of the camera's
        native geometry. With no saved default we leave the camera as-is, exactly
        as before.
        """
        worker = self.stream.worker
        if not self.stream.is_connected:
            return
        has_defaults = self._has_saved_defaults()
        try:
            if worker.is_genicam_camera():
                worker.set_camera_frame_rate(float(self.spin_fps.value()))
                worker.set_camera_exposure_ms(float(self.spin_exposure.value()))
                if has_defaults:
                    wb_mode = self.combo_white_balance.currentText()
                    worker.set_camera_white_balance_auto(wb_mode)
                    if wb_mode.strip().lower() == "off":
                        worker.set_camera_white_balance_ratio("Red", float(self.spin_wb_red.value()))
                        worker.set_camera_white_balance_ratio("Blue", float(self.spin_wb_blue.value()))
                    worker.set_camera_gain(float(self.spin_gain.value()))
        except Exception:
            pass
        # Resolution is honoured by every backend; only force it when the user
        # has chosen a default, otherwise keep the camera's native geometry.
        if has_defaults:
            self._on_resolution_apply()

    def _on_fps_changed(self, value: float) -> None:
        """Update the target capture rate (and the camera's rate on GenICam)."""
        self.stream.worker.set_target_fps(float(value))
        if self.stream.is_connected:
            try:
                if self.stream.worker.is_genicam_camera():
                    self.stream.worker.set_camera_frame_rate(float(value))
            except Exception:
                pass
        self._update_settings_summary()

    def _on_exposure_changed(self, value: float) -> None:
        """Apply exposure (ms) to the connected camera (GenICam only)."""
        if self.stream.is_connected:
            try:
                if self.stream.worker.is_genicam_camera():
                    self.stream.worker.set_camera_exposure_ms(float(value))
            except Exception:
                pass

    def _on_format_changed(self, image_format: str) -> None:
        """Switch the recorded/displayed pixel format (Mono8/BGR8) for this stream."""
        try:
            self.stream.worker.set_image_format(image_format)
        except Exception:
            pass
        self._update_settings_summary()

    def _on_preview_fps_changed(self, value: float) -> None:
        """Set the on-screen preview rate (does not affect the recording)."""
        try:
            self.stream.worker.set_preview_fps(float(value))
        except Exception:
            pass

    def _on_preview_width_changed(self, value: int) -> None:
        """Set the preview downscale width (0 = full resolution; recording unaffected)."""
        try:
            self.stream.worker.set_preview_max_width(int(value))
        except Exception:
            pass

    def _on_resolution_apply(self) -> None:
        """Apply the chosen capture resolution safely for the active backend.

        OpenCV-backed cameras are reconfigured on the acquisition thread (queued
        via ``request_resolution`` to avoid the cross-thread freeze); GenICam
        cameras, which pause their own acquisition, apply inline.
        """
        if not self.stream.is_connected:
            return
        worker = self.stream.worker
        width = int(self.spin_width.value())
        height = int(self.spin_height.value())
        # OpenCV-backed cameras must be reconfigured on the acquisition thread;
        # request_resolution queues it and reports back via resolution_changed.
        if worker.request_resolution(width, height):
            return
        # GenICam cameras (handle their own acquisition pause) apply inline.
        try:
            applied = worker.apply_resolution(width, height)
        except Exception:
            applied = None
        if not applied:
            self.status_chip.setText("Res unsupported")
            self._style_chip(self.status_chip, "warning")
            return
        self._on_resolution_changed_signal(*applied)

    def _seed_resolution_spins(self, width: int, height: int) -> None:
        """Set the width/height spin-boxes without firing their change handlers."""
        for spin, val in ((self.spin_width, int(width)), (self.spin_height, int(height))):
            spin.blockSignals(True)
            spin.setValue(int(val))
            spin.blockSignals(False)

    @Slot(int, int)
    def _on_resolution_changed_signal(self, actual_w: int, actual_h: int) -> None:
        """Reflect the actual capture resolution after a reconfigure."""
        self._resolution_seeded = True
        self._seed_resolution_spins(int(actual_w), int(actual_h))
        self.resolution_chip.setText(f"{actual_w} x {actual_h}")

    def _on_gain_changed(self, value: float) -> None:
        """Apply manual gain in dB (GenICam only; ignored on USB)."""
        if self._worker_is_genicam():
            try:
                self.stream.worker.set_camera_gain(float(value))
            except Exception:
                pass

    def _on_white_balance_auto_changed(self, mode: str) -> None:
        """Set auto white-balance mode and enable manual ratios only when Off."""
        if not self._worker_is_genicam():
            return
        try:
            self.stream.worker.set_camera_white_balance_auto(str(mode))
        except Exception:
            pass
        # Manual ratios only make sense when auto white-balance is Off.
        manual = str(mode).strip().lower() == "off"
        self.spin_wb_red.setEnabled(manual)
        self.spin_wb_blue.setEnabled(manual)

    def _on_white_balance_red_changed(self, value: float) -> None:
        """Set the manual red white-balance ratio (GenICam only)."""
        if self._worker_is_genicam():
            try:
                self.stream.worker.set_camera_white_balance_ratio("Red", float(value))
            except Exception:
                pass

    def _on_white_balance_blue_changed(self, value: float) -> None:
        """Set the manual blue white-balance ratio (GenICam only)."""
        if self._worker_is_genicam():
            try:
                self.stream.worker.set_camera_white_balance_ratio("Blue", float(value))
            except Exception:
                pass

    # ----- State sync ---------------------------------------------------------

    @Slot()
    def _sync_state(self) -> None:
        """Refresh chips, buttons, and control availability from the stream state.

        Reflects offline/live/recording status and disables connect/remove and
        capture-geometry controls while a recording is in progress.
        """
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
            self._resolution_seeded = False
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
        self._update_settings_controls_enabled()
