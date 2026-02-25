"""
Enhanced Main GUI Window
PySide6-based interface for Basler camera control with Arduino integration.
"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QLineEdit,
                               QStatusBar, QGroupBox, QSpinBox, QDoubleSpinBox,
                               QFileDialog, QScrollArea, QFormLayout, QTextEdit,
                               QSplitter, QFrame, QSlider, QGridLayout,
                               QCheckBox, QToolButton, QTabWidget, QDialog,
                               QDialogButtonBox, QStyle)
from PySide6.QtCore import Qt, Slot, QTimer, QSettings, QEvent, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import numpy as np
from datetime import datetime
import pyqtgraph as pg
from collections import deque
import json
from pathlib import Path
import os
from pypylon import pylon
import cv2
from typing import Optional, Dict, List
from camera_worker import CameraWorker
from arduino_output import ArduinoOutputWorker


class MainWindow(QMainWindow):
    """
    Enhanced main application window with camera control, Arduino integration,
    and comprehensive settings.
    """

    DISPLAY_SIGNAL_ORDER = ["gate", "sync", "barcode", "lever", "cue", "reward", "iti"]
    DISPLAY_SIGNAL_META = {
        "gate": {"state_key": "gate", "group": "ttl", "name": "Gate", "role": "Output", "default_pins": [3], "color": "#22c55e"},
        "sync": {"state_key": "sync", "group": "ttl", "name": "sync", "role": "Output", "default_pins": [9], "color": "#38bdf8"},
        "barcode": {"state_key": "barcode0", "group": "ttl", "name": "Barcode", "role": "Output", "default_pins": [18], "color": "#f97316"},
        "lever": {"state_key": "lever", "group": "behavior", "name": "Lever", "role": "Input", "default_pins": [14], "color": "#facc15"},
        "cue": {"state_key": "cue", "group": "behavior", "name": "Cue LED", "role": "Output", "default_pins": [45], "color": "#34d399"},
        "reward": {"state_key": "reward", "group": "behavior", "name": "Reward LED", "role": "Output", "default_pins": [21], "color": "#60a5fa"},
        "iti": {"state_key": "iti", "group": "behavior", "name": "ITI LED", "role": "Output", "default_pins": [46], "color": "#ef4444"},
    }
    BEHAVIOR_PIN_KEYS = ["gate", "sync", "barcode", "lever", "cue", "reward", "iti"]

    def __init__(self):
        super().__init__()

        self.worker: Optional[CameraWorker] = None
        self.arduino_worker: Optional[ArduinoOutputWorker] = None
        self.is_camera_connected = False
        self.is_arduino_connected = False
        self.is_testing_ttl = False

        # Settings
        self.settings = QSettings('BaslerCam', 'CameraApp')
        self.last_save_folder = self.settings.value('last_save_folder', '.') or '.'
        self.default_fps = float(self.settings.value('camera_fps', 30.0))
        self.default_width = int(self.settings.value('camera_width', 1080))
        self.default_height = int(self.settings.value('camera_height', 1080))
        self.default_image_format = self.settings.value('image_format', 'Mono8')
        self.signal_display_config = self._load_signal_display_config()

        # Recording state
        self.recording_start_time = None
        self.current_recording_filepath = None
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self._update_recording_time)

        # ROI state
        self.roi_rect = None
        self.roi_preview = None
        self.roi_draw_mode = False
        self.roi_dragging = False
        self.roi_start_pos = QPoint()
        self.last_frame_size = None

        # TTL plot data
        self.ttl_window_seconds = 10.0
        self.ttl_max_points = 600
        self.time_data = deque(maxlen=self.ttl_max_points)
        self.plot_start_time = datetime.now()
        self.ttl_plot_data: Dict[str, deque] = {
            key: deque(maxlen=self.ttl_max_points)
            for key in self.DISPLAY_SIGNAL_ORDER
        }
        self.ttl_output_curves: Dict[str, pg.PlotDataItem] = {}
        self.behavior_curves: Dict[str, pg.PlotDataItem] = {}
        self.ttl_output_levels: Dict[str, float] = {}
        self.behavior_levels: Dict[str, float] = {}
        self.ttl_state_labels: Dict[str, QLabel] = {}
        self.ttl_count_labels: Dict[str, QLabel] = {}
        self.pin_value_labels: Dict[str, QLabel] = {}
        self.pin_name_labels: Dict[str, QLabel] = {}
        self.behavior_pin_edits: Dict[str, QLineEdit] = {}
        self.behavior_role_boxes: Dict[str, QComboBox] = {}
        self.signal_label_edits: Dict[str, QLineEdit] = {}
        self.signal_enabled_checks: Dict[str, QCheckBox] = {}
        self.sync_param_button: Optional[QToolButton] = None
        self.barcode_param_button: Optional[QToolButton] = None

        # Metadata
        self.metadata = {}

        self.setWindowTitle("Basler Camera Control - Professional Edition with Arduino")
        self.setGeometry(50, 50, 1600, 900)

        # Professional Dark Blue Theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1b2430;
                color: #ffffff;
            }
            QWidget {
                background-color: #1b2430;
                color: #ffffff;
                font-family: "Segoe UI", "Ubuntu", "Arial", sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 5px;
                margin-top: 10px;
                background-color: #1f2937;
                font-weight: bold;
                color: #e5e7eb;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #1f2937;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
            QPushButton:disabled {
                background-color: #374151;
                color: #9ca3af;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 3px;
                color: white;
                padding: 3px;
            }
            QLabel {
                color: #e5e7eb;
            }
            QStatusBar {
                background-color: #111827;
                color: #e5e7eb;
            }
            QSplitter::handle {
                background-color: #334155;
            }
            QScrollBar:vertical {
                background-color: #111827;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #334155;
                min-height: 20px;
                border-radius: 6px;
            }
        """)

        self._init_ui()
        self._load_ui_settings()
        self._setup_worker()
        self._setup_arduino_worker()
        self._load_metadata()
        self._scan_cameras()

    def _init_ui(self):
        """Initialize the user interface components."""

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        self.main_splitter = splitter

        # === LEFT PANEL: Metadata ===
        left_panel = self._create_metadata_panel()
        self.metadata_panel = left_panel
        splitter.addWidget(left_panel)

        # === CENTER PANEL: Video and Controls ===
        center_panel = self._create_center_panel()
        splitter.addWidget(center_panel)

        # === RIGHT PANEL: Arduino and TTL Plot ===
        right_panel = self._create_arduino_panel()
        splitter.addWidget(right_panel)

        # Set initial sizes
        splitter.setSizes([250, 760, 730])

        main_layout.addWidget(splitter)

        # === Status Bar ===
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.label_fps = QLabel("FPS: 0.0")
        self.label_buffer = QLabel("Buffer: 0%")
        self.label_recording = QLabel("Not Recording")
        self.label_recording_time = QLabel("00:00:00")

        self.status_bar.addWidget(self.label_fps)
        self.status_bar.addPermanentWidget(self.label_buffer)
        self.status_bar.addPermanentWidget(self.label_recording)
        self.status_bar.addPermanentWidget(self.label_recording_time)

        self.btn_toggle_metadata = QPushButton("Hide Metadata")
        self.btn_toggle_metadata.clicked.connect(self._toggle_metadata_panel)
        self.btn_toggle_metadata.setMaximumHeight(24)
        self.status_bar.addPermanentWidget(self.btn_toggle_metadata)

    def _create_metadata_panel(self) -> QWidget:
        """Create left panel for metadata input."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Experiment Metadata")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Scroll area for metadata fields
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.metadata_layout = QFormLayout(scroll_widget)

        # Default metadata fields
        self.meta_animal_id = QLineEdit()
        self.meta_animal_id.setPlaceholderText("e.g., Mouse001")
        self.metadata_layout.addRow("Animal ID:", self.meta_animal_id)

        self.meta_experiment = QLineEdit()
        self.meta_experiment.setPlaceholderText("e.g., Behavior Test")
        self.metadata_layout.addRow("Experiment:", self.meta_experiment)

        self.meta_date = QLineEdit()
        self.meta_date.setText(datetime.now().strftime('%Y-%m-%d'))
        self.meta_date.setReadOnly(True)
        self.metadata_layout.addRow("Date:", self.meta_date)

        self.meta_notes = QTextEdit()
        self.meta_notes.setPlaceholderText("Additional notes...")
        self.meta_notes.setMaximumHeight(100)
        self.metadata_layout.addRow("Notes:", self.meta_notes)

        # Custom metadata fields
        self.custom_metadata_fields = {}

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Add metadata field button
        btn_add_field = QPushButton("+ Add Custom Field")
        btn_add_field.clicked.connect(self._add_custom_metadata_field)
        layout.addWidget(btn_add_field)

        # Save metadata button
        btn_save_meta = QPushButton("Save Metadata Template")
        btn_save_meta.clicked.connect(self._save_metadata_template)
        layout.addWidget(btn_save_meta)

        return panel

    def _create_center_panel(self) -> QWidget:
        """Create center panel with video display and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # === Video Display Area ===
        display_group = QGroupBox("Live View")
        display_layout = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        self.video_label.setText("No Camera Connected")
        self.video_label.setScaledContents(False)
        self.video_label.installEventFilter(self)

        display_layout.addWidget(self.video_label)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group, stretch=3)

        # === Camera Settings ===
        settings_group = self._create_camera_settings()
        layout.addWidget(settings_group)

        # === Control Panel ===
        control_group = self._create_control_panel()
        layout.addWidget(control_group)

        return panel

    def _create_camera_settings(self) -> QGroupBox:
        """Create camera settings group."""
        settings_group = QGroupBox("Camera Settings")
        settings_container = QVBoxLayout()
        settings_layout = QFormLayout()

        # FPS setting
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1.0, 200.0)
        self.spin_fps.setValue(self.default_fps)
        self.spin_fps.setSuffix(" fps")
        self.spin_fps.valueChanged.connect(self._on_fps_changed)
        settings_layout.addRow("Frame Rate:", self.spin_fps)

        # Resolution settings
        resolution_layout = QHBoxLayout()
        self.spin_width = QSpinBox()
        self.spin_width.setRange(64, 4096)
        self.spin_width.setValue(self.default_width)
        self.spin_width.setSuffix(" px")
        resolution_layout.addWidget(QLabel("W:"))
        resolution_layout.addWidget(self.spin_width)

        self.spin_height = QSpinBox()
        self.spin_height.setRange(64, 4096)
        self.spin_height.setValue(self.default_height)
        self.spin_height.setSuffix(" px")
        resolution_layout.addWidget(QLabel("H:"))
        resolution_layout.addWidget(self.spin_height)

        btn_apply_res = QPushButton("Apply")
        btn_apply_res.clicked.connect(self._on_resolution_changed)
        resolution_layout.addWidget(btn_apply_res)
        settings_layout.addRow("Resolution:", resolution_layout)

        # Exposure time
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(0.01, 1000.0)
        self.spin_exposure.setValue(10.0)
        self.spin_exposure.setSuffix(" ms")
        self.spin_exposure.valueChanged.connect(self._on_exposure_changed)
        settings_layout.addRow("Exposure Time:", self.spin_exposure)

        # Image format selection
        self.combo_image_format = QComboBox()
        self.combo_image_format.addItems(["Mono8", "BGR8"])
        if self.default_image_format in ("Mono8", "BGR8"):
            self.combo_image_format.setCurrentText(self.default_image_format)
        self.combo_image_format.currentTextChanged.connect(self._on_image_format_changed)
        settings_layout.addRow("Image Format:", self.combo_image_format)

        # FFmpeg Encoder selection
        self.combo_encoder = QComboBox()
        self.combo_encoder.addItems([
            "h264_nvenc (NVIDIA GPU)",
            "libx264 (CPU - Software)",
            "h264_qsv (Intel QuickSync)"
        ])
        self.combo_encoder.setCurrentIndex(0)
        settings_layout.addRow("Video Encoder:", self.combo_encoder)

        settings_container.addLayout(settings_layout)

        self.btn_advanced = QPushButton("Show Advanced")
        self.btn_advanced.clicked.connect(self._toggle_advanced_settings)
        settings_container.addWidget(self.btn_advanced)

        self.advanced_group = QGroupBox("Advanced Video")
        self.advanced_group.setVisible(False)
        advanced_layout = QFormLayout()

        self.slider_offset_x = QSlider(Qt.Horizontal)
        self.slider_offset_x.setRange(0, 0)
        self.slider_offset_x.valueChanged.connect(self._on_offset_x_changed)
        self.spin_offset_x = QSpinBox()
        self.spin_offset_x.setRange(0, 0)
        self.spin_offset_x.valueChanged.connect(self._on_offset_x_changed)
        offset_x_layout = QHBoxLayout()
        offset_x_layout.addWidget(self.slider_offset_x, stretch=3)
        offset_x_layout.addWidget(self.spin_offset_x, stretch=1)
        advanced_layout.addRow("Offset X:", offset_x_layout)

        self.slider_offset_y = QSlider(Qt.Horizontal)
        self.slider_offset_y.setRange(0, 0)
        self.slider_offset_y.valueChanged.connect(self._on_offset_y_changed)
        self.spin_offset_y = QSpinBox()
        self.spin_offset_y.setRange(0, 0)
        self.spin_offset_y.valueChanged.connect(self._on_offset_y_changed)
        offset_y_layout = QHBoxLayout()
        offset_y_layout.addWidget(self.slider_offset_y, stretch=3)
        offset_y_layout.addWidget(self.spin_offset_y, stretch=1)
        advanced_layout.addRow("Offset Y:", offset_y_layout)

        self.btn_center_offsets = QPushButton("Center ROI")
        self.btn_center_offsets.clicked.connect(self._center_offsets)
        advanced_layout.addRow("", self.btn_center_offsets)

        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.valueChanged.connect(self._on_gain_changed)
        advanced_layout.addRow("Gain:", self.spin_gain)

        self.spin_brightness = QDoubleSpinBox()
        self.spin_brightness.valueChanged.connect(self._on_brightness_changed)
        advanced_layout.addRow("Brightness:", self.spin_brightness)

        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.valueChanged.connect(self._on_contrast_changed)
        advanced_layout.addRow("Contrast:", self.spin_contrast)

        roi_button_layout = QHBoxLayout()
        self.btn_draw_roi = QPushButton("Draw ROI")
        self.btn_draw_roi.clicked.connect(self._toggle_roi_draw)
        roi_button_layout.addWidget(self.btn_draw_roi)
        self.btn_clear_roi = QPushButton("Clear ROI")
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        roi_button_layout.addWidget(self.btn_clear_roi)
        advanced_layout.addRow("ROI:", roi_button_layout)

        self.advanced_group.setLayout(advanced_layout)
        settings_container.addWidget(self.advanced_group)

        self._set_advanced_controls_enabled(False)

        settings_group.setLayout(settings_container)
        return settings_group

    def _create_control_panel(self) -> QGroupBox:
        """Create control panel."""
        control_group = QGroupBox("Camera Controls")
        control_layout = QVBoxLayout()

        # Camera connection
        camera_select_layout = QHBoxLayout()
        camera_select_layout.addWidget(QLabel("Camera:"))
        self.combo_camera = QComboBox()
        camera_select_layout.addWidget(self.combo_camera, stretch=2)
        self.btn_scan_cameras = QPushButton("Refresh")
        self.btn_scan_cameras.clicked.connect(self._scan_cameras)
        camera_select_layout.addWidget(self.btn_scan_cameras)
        control_layout.addLayout(camera_select_layout)

        connection_layout = QHBoxLayout()
        self.btn_connect = QPushButton("Connect Camera")
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        self.btn_connect.setMinimumHeight(40)
        connection_layout.addWidget(self.btn_connect)
        connection_layout.addStretch()
        control_layout.addLayout(connection_layout)

        # Trigger mode selection (Simplified)
        # trigger_layout = QHBoxLayout()
        # trigger_layout.addWidget(QLabel("Trigger Mode:"))

        # self.combo_trigger = QComboBox()
        # self.combo_trigger.addItems(["Free Run"])
        # self.combo_trigger.setEnabled(False)
        # trigger_layout.addWidget(self.combo_trigger)
        # trigger_layout.addStretch()
        # control_layout.addLayout(trigger_layout)

        # Recording controls
        recording_layout = QHBoxLayout()

        # Save folder selection
        self.edit_save_folder = QLineEdit()
        self.edit_save_folder.setText(self.last_save_folder)
        self.edit_save_folder.setReadOnly(True)
        recording_layout.addWidget(QLabel("Save to:"))
        recording_layout.addWidget(self.edit_save_folder)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_save_folder)
        recording_layout.addWidget(btn_browse)
        control_layout.addLayout(recording_layout)

        # Filename and recording length
        filename_layout = QHBoxLayout()
        filename_layout.addWidget(QLabel("Filename:"))

        self.edit_filename = QLineEdit()
        self.edit_filename.setPlaceholderText("Enter filename (without extension)")
        default_filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.edit_filename.setText(default_filename)
        filename_layout.addWidget(self.edit_filename, stretch=2)
        control_layout.addLayout(filename_layout)

        # Recording length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Max Length:"))

        self.spin_hours = QSpinBox()
        self.spin_hours.setRange(0, 24)
        self.spin_hours.setValue(0)
        self.spin_hours.setSuffix(" h")
        length_layout.addWidget(self.spin_hours)

        self.spin_minutes = QSpinBox()
        self.spin_minutes.setRange(0, 59)
        self.spin_minutes.setValue(5)
        self.spin_minutes.setSuffix(" min")
        length_layout.addWidget(self.spin_minutes)

        self.spin_seconds = QSpinBox()
        self.spin_seconds.setRange(0, 59)
        self.spin_seconds.setValue(0)
        self.spin_seconds.setSuffix(" sec")
        length_layout.addWidget(self.spin_seconds)

        self.check_unlimited = QComboBox()
        self.check_unlimited.addItems(["Limited", "Unlimited"])
        self.check_unlimited.setCurrentIndex(0)
        length_layout.addWidget(self.check_unlimited)
        length_layout.addStretch()
        control_layout.addLayout(length_layout)

        # Record button
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.clicked.connect(self._on_record_clicked)
        self.btn_record.setEnabled(False)
        self.btn_record.setMinimumHeight(50)
        self.btn_record.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; }")
        control_layout.addWidget(self.btn_record)

        control_group.setLayout(control_layout)
        return control_group

    def _create_arduino_panel(self) -> QWidget:
        """Create right panel for Arduino controls and behavior TTL monitoring."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setMinimumWidth(700)
        layout = QVBoxLayout(panel)

        title = QLabel("Behavior TTL")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        tabs = QTabWidget()
        layout.addWidget(tabs, stretch=1)

        setup_tab = QWidget()
        setup_layout = QVBoxLayout(setup_tab)
        setup_layout.setSpacing(8)

        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        monitor_layout.setSpacing(8)

        tabs.addTab(setup_tab, "Setup")
        tabs.addTab(monitor_tab, "Monitoring")

        arduino_group = QGroupBox("Arduino Connection")
        arduino_layout = QVBoxLayout()

        port_layout = QHBoxLayout()
        self.combo_arduino_port = QComboBox()
        port_layout.addWidget(QLabel("Port:"))
        port_layout.addWidget(self.combo_arduino_port)

        btn_scan = QPushButton("Scan")
        btn_scan.clicked.connect(self._scan_arduino_ports)
        port_layout.addWidget(btn_scan)
        arduino_layout.addLayout(port_layout)

        self.btn_arduino_connect = QPushButton("Connect Arduino")
        self.btn_arduino_connect.clicked.connect(self._on_arduino_connect_clicked)
        arduino_layout.addWidget(self.btn_arduino_connect)

        self.label_arduino_status = QLabel("Status: Disconnected")
        self.label_arduino_status.setStyleSheet("color: gray;")
        arduino_layout.addWidget(self.label_arduino_status)

        arduino_group.setLayout(arduino_layout)
        setup_layout.addWidget(arduino_group)

        pin_group = QGroupBox("Pin Configuration (Board Defaults)")
        pin_layout = QFormLayout()
        default_pin_map = self._default_behavior_pin_map()
        for key in self.BEHAVIOR_PIN_KEYS:
            name_label = QLabel(f"{self._signal_label(key)}:")
            value_label = QLabel(self._format_pin_list(default_pin_map.get(key, [])))
            self.pin_name_labels[key] = name_label
            self.pin_value_labels[key] = value_label
            pin_layout.addRow(name_label, value_label)
        self.label_gate_pin = self.pin_value_labels["gate"]
        self.label_sync_pin = self.pin_value_labels["sync"]
        self.label_barcode_pins = self.pin_value_labels["barcode"]
        pin_group.setLayout(pin_layout)
        setup_layout.addWidget(pin_group)

        config_group = QGroupBox("Signal Mapping / Labels")
        config_layout = QVBoxLayout()
        config_grid = QGridLayout()
        config_grid.addWidget(QLabel("Use"), 0, 0)
        config_grid.addWidget(QLabel("Label"), 0, 1)
        config_grid.addWidget(QLabel("Role"), 0, 2)
        config_grid.addWidget(QLabel("Pins"), 0, 3)
        config_grid.addWidget(QLabel("Params"), 0, 4)

        default_roles = self._default_behavior_roles()
        for row, key in enumerate(self.DISPLAY_SIGNAL_ORDER, start=1):
            cfg = self.signal_display_config.get(key, {})
            enabled_check = QCheckBox()
            enabled_check.setChecked(bool(cfg.get("enabled", True)))
            self.signal_enabled_checks[key] = enabled_check

            label_edit = QLineEdit(str(cfg.get("name", self._signal_label(key))))
            label_edit.setPlaceholderText("Signal label")
            self.signal_label_edits[key] = label_edit

            role_box = QComboBox()
            role_box.addItems(["Input", "Output"])
            role_box.setCurrentText(default_roles.get(key, "Output"))
            self.behavior_role_boxes[key] = role_box

            pin_edit = QLineEdit(self._format_pin_list(default_pin_map.get(key, [])))
            pin_edit.setPlaceholderText("e.g. 8, 9")
            self.behavior_pin_edits[key] = pin_edit

            param_cell = QLabel("-")
            if key == "sync":
                self.sync_param_button = QToolButton()
                self.sync_param_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
                self.sync_param_button.setToolTip("Edit sync parameters")
                self.sync_param_button.clicked.connect(self._edit_sync_parameters)
                param_cell = self.sync_param_button
            elif key == "barcode":
                self.barcode_param_button = QToolButton()
                self.barcode_param_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
                self.barcode_param_button.setToolTip("Edit barcode parameters")
                self.barcode_param_button.clicked.connect(self._edit_barcode_parameters)
                param_cell = self.barcode_param_button

            row_widgets = [label_edit, role_box, pin_edit]
            if isinstance(param_cell, QWidget):
                row_widgets.append(param_cell)
            for widget in row_widgets:
                widget.setEnabled(enabled_check.isChecked())
            enabled_check.toggled.connect(
                lambda checked, widgets=row_widgets: [widget.setEnabled(checked) for widget in widgets]
            )

            config_grid.addWidget(enabled_check, row, 0, alignment=Qt.AlignCenter)
            config_grid.addWidget(label_edit, row, 1)
            config_grid.addWidget(role_box, row, 2)
            config_grid.addWidget(pin_edit, row, 3)
            config_grid.addWidget(param_cell, row, 4, alignment=Qt.AlignCenter)

        config_layout.addLayout(config_grid)
        self.btn_apply_behavior_config = QPushButton("Apply Mapping and Labels")
        self.btn_apply_behavior_config.clicked.connect(lambda _: self._apply_behavior_pin_configuration(persist=True))
        config_layout.addWidget(self.btn_apply_behavior_config)
        config_group.setLayout(config_layout)
        setup_layout.addWidget(config_group)

        line_group = QGroupBox("Camera Input Labels (Optional)")
        line_layout = QFormLayout()
        line_options = ["None", "Gate", "Sync", "Barcode", "Lever", "Cue", "Reward", "ITI"]

        self.combo_line1_label = QComboBox()
        self.combo_line1_label.addItems(line_options)
        self.combo_line1_label.currentTextChanged.connect(
            lambda v: self._on_line_label_changed(1, v)
        )
        line_layout.addRow("Line 1:", self.combo_line1_label)

        self.combo_line2_label = QComboBox()
        self.combo_line2_label.addItems(line_options)
        self.combo_line2_label.currentTextChanged.connect(
            lambda v: self._on_line_label_changed(2, v)
        )
        line_layout.addRow("Line 2:", self.combo_line2_label)

        self.combo_line3_label = QComboBox()
        self.combo_line3_label.addItems(line_options)
        self.combo_line3_label.currentTextChanged.connect(
            lambda v: self._on_line_label_changed(3, v)
        )
        line_layout.addRow("Line 3:", self.combo_line3_label)

        self.combo_line4_label = QComboBox()
        self.combo_line4_label.addItems(line_options)
        self.combo_line4_label.currentTextChanged.connect(
            lambda v: self._on_line_label_changed(4, v)
        )
        line_layout.addRow("Line 4:", self.combo_line4_label)

        line_group.setLayout(line_layout)
        setup_layout.addWidget(line_group)

        self.btn_test_ttl = QPushButton("Test TTL / Behavior")
        self.btn_test_ttl.clicked.connect(self._on_test_ttl_clicked)
        self.btn_test_ttl.setEnabled(False)
        self.btn_test_ttl.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        self.btn_test_ttl.setMinimumHeight(40)
        setup_layout.addWidget(self.btn_test_ttl)

        self.label_ttl_status = QLabel("TTL: IDLE")
        self.label_ttl_status.setStyleSheet("background-color: gray; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
        self.label_ttl_status.setAlignment(Qt.AlignCenter)
        setup_layout.addWidget(self.label_ttl_status)

        self.label_behavior_status = QLabel("Behavior: IDLE")
        self.label_behavior_status.setStyleSheet("background-color: #374151; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
        self.label_behavior_status.setAlignment(Qt.AlignCenter)
        setup_layout.addWidget(self.label_behavior_status)
        setup_layout.addStretch()

        self.counts_group = QGroupBox("Behavior TTL Counts")
        self.counts_layout = QGridLayout()
        self.counts_group.setLayout(self.counts_layout)
        self.counts_group.setMinimumHeight(250)
        monitor_layout.addWidget(self.counts_group, stretch=1)

        self.ttl_plot_group = QGroupBox("TTL Generator Signals")
        ttl_plot_layout = QVBoxLayout()
        pg.setConfigOptions(antialias=True)
        self.ttl_plot = pg.PlotWidget()
        self.ttl_plot.setBackground((18, 27, 36))
        self.ttl_plot.setMouseEnabled(x=False, y=False)
        self.ttl_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ttl_plot.setLabel("bottom", "Time (s)")
        self.ttl_plot.setXRange(0, self.ttl_window_seconds)
        self.ttl_plot.setLimits(xMin=0)
        self.ttl_plot.setDownsampling(auto=True, mode="peak")
        self.ttl_plot.setMinimumHeight(220)
        ttl_plot_layout.addWidget(self.ttl_plot)
        self.ttl_plot_group.setLayout(ttl_plot_layout)
        self.ttl_plot_group.setMinimumHeight(250)
        monitor_layout.addWidget(self.ttl_plot_group, stretch=2)

        self.behavior_plot_group = QGroupBox("Behavior Signals")
        behavior_plot_layout = QVBoxLayout()
        self.behavior_plot = pg.PlotWidget()
        self.behavior_plot.setBackground((18, 27, 36))
        self.behavior_plot.setMouseEnabled(x=False, y=False)
        self.behavior_plot.showGrid(x=True, y=True, alpha=0.2)
        self.behavior_plot.setLabel("bottom", "Time (s)")
        self.behavior_plot.setXRange(0, self.ttl_window_seconds)
        self.behavior_plot.setLimits(xMin=0)
        self.behavior_plot.setDownsampling(auto=True, mode="peak")
        self.behavior_plot.setMinimumHeight(220)
        behavior_plot_layout.addWidget(self.behavior_plot)
        self.behavior_plot_group.setLayout(behavior_plot_layout)
        self.behavior_plot_group.setMinimumHeight(250)
        monitor_layout.addWidget(self.behavior_plot_group, stretch=2)

        self._rebuild_monitor_visuals(reset_plot=True)
        return panel

    def _default_behavior_pin_map(self) -> Dict[str, List[int]]:
        """Default signal-to-pin mapping for behavior + TTL board."""
        return {
            key: [int(pin) for pin in self.DISPLAY_SIGNAL_META[key]["default_pins"]]
            for key in self.BEHAVIOR_PIN_KEYS
        }

    def _load_signal_display_config(self) -> Dict[str, Dict]:
        """Load user-defined signal labels and visibility."""
        config = {}
        for key in self.DISPLAY_SIGNAL_ORDER:
            meta = self.DISPLAY_SIGNAL_META[key]
            label = str(self.settings.value(f"behavior_signal_label_{key}", meta["name"]))
            raw_enabled = self.settings.value(f"behavior_signal_enabled_{key}", True)
            enabled = str(raw_enabled).strip().lower() not in ("0", "false", "no", "off")
            config[key] = {
                "name": label if label else meta["name"],
                "enabled": bool(enabled),
            }
        return config

    def _signal_label(self, key: str) -> str:
        config = self.signal_display_config.get(key, {})
        if config.get("name"):
            return str(config["name"])
        return str(self.DISPLAY_SIGNAL_META.get(key, {}).get("name", key))

    def _state_key_for_display(self, key: str) -> str:
        return str(self.DISPLAY_SIGNAL_META.get(key, {}).get("state_key", key))

    def _active_signal_keys(self, group: Optional[str] = None) -> List[str]:
        active = []
        for key in self.DISPLAY_SIGNAL_ORDER:
            cfg = self.signal_display_config.get(key, {})
            if not bool(cfg.get("enabled", True)):
                continue
            meta_group = str(self.DISPLAY_SIGNAL_META.get(key, {}).get("group", "behavior"))
            if group is not None and meta_group != group:
                continue
            active.append(key)
        return active

    def _clear_layout(self, layout):
        """Delete all widgets/items inside a layout."""
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _rebuild_monitor_visuals(self, reset_plot: bool = False):
        """Rebuild count rows and plot axes from current signal configuration."""
        self.ttl_state_labels.clear()
        self.ttl_count_labels.clear()
        self._clear_layout(self.counts_layout)
        self.counts_layout.addWidget(QLabel("Signal"), 0, 0)
        self.counts_layout.addWidget(QLabel("State"), 0, 1)
        self.counts_layout.addWidget(QLabel("Count"), 0, 2)

        active_keys = self._active_signal_keys()
        for row, key in enumerate(active_keys, start=1):
            signal_label = QLabel(self._signal_label(key))
            state_label = QLabel("LOW")
            count_label = QLabel("0")
            state_label.setStyleSheet("color: #9ca3af;")
            count_label.setStyleSheet("font-weight: bold;")
            self.counts_layout.addWidget(signal_label, row, 0)
            self.counts_layout.addWidget(state_label, row, 1)
            self.counts_layout.addWidget(count_label, row, 2)
            self.ttl_state_labels[key] = state_label
            self.ttl_count_labels[key] = count_label

        self.ttl_output_curves.clear()
        self.behavior_curves.clear()
        self.ttl_output_levels.clear()
        self.behavior_levels.clear()

        self.ttl_plot.clear()
        self.behavior_plot.clear()
        self.ttl_plot.setXRange(0, self.ttl_window_seconds)
        self.behavior_plot.setXRange(0, self.ttl_window_seconds)

        ttl_ticks = []
        ttl_keys = self._active_signal_keys(group="ttl")
        n_ttl_rows = len(ttl_keys)
        for index, key in enumerate(ttl_keys):
            meta = self.DISPLAY_SIGNAL_META[key]
            level = float(n_ttl_rows - index)
            self.ttl_output_levels[key] = level
            ttl_ticks.append((level, self._signal_label(key)))
            self.ttl_output_curves[key] = self.ttl_plot.plot(
                pen=pg.mkPen(meta["color"], width=2),
                name=self._signal_label(key),
                stepMode=True,
            )
        self.ttl_plot.setYRange(-0.6, max(1.0, float(n_ttl_rows) + 0.6))
        ttl_axis_left = self.ttl_plot.getAxis("left")
        ttl_axis_left.setTextPen(pg.mkPen("#b9c6d3"))
        ttl_axis_left.setPen(pg.mkPen("#6c7a89"))
        ttl_axis_left.setTicks([ttl_ticks] if ttl_ticks else [[]])
        ttl_axis_bottom = self.ttl_plot.getAxis("bottom")
        ttl_axis_bottom.setTextPen(pg.mkPen("#b9c6d3"))
        ttl_axis_bottom.setPen(pg.mkPen("#6c7a89"))

        behavior_ticks = []
        behavior_keys = self._active_signal_keys(group="behavior")
        n_behavior_rows = len(behavior_keys)
        for index, key in enumerate(behavior_keys):
            meta = self.DISPLAY_SIGNAL_META[key]
            level = float(n_behavior_rows - index)
            self.behavior_levels[key] = level
            behavior_ticks.append((level, self._signal_label(key)))
            self.behavior_curves[key] = self.behavior_plot.plot(
                pen=pg.mkPen(meta["color"], width=2),
                name=self._signal_label(key),
                stepMode=True,
            )
        self.behavior_plot.setYRange(-0.6, max(1.0, float(n_behavior_rows) + 0.6))
        behavior_axis_left = self.behavior_plot.getAxis("left")
        behavior_axis_left.setTextPen(pg.mkPen("#b9c6d3"))
        behavior_axis_left.setPen(pg.mkPen("#6c7a89"))
        behavior_axis_left.setTicks([behavior_ticks] if behavior_ticks else [[]])
        behavior_axis_bottom = self.behavior_plot.getAxis("bottom")
        behavior_axis_bottom.setTextPen(pg.mkPen("#b9c6d3"))
        behavior_axis_bottom.setPen(pg.mkPen("#6c7a89"))

        if reset_plot:
            self.time_data.clear()
            self.plot_start_time = datetime.now()
            for series in self.ttl_plot_data.values():
                series.clear()

    def _get_behavior_signal_spec(self, key: str) -> Dict[str, str]:
        """Lookup rendering spec for a signal key."""
        display_key = key
        if display_key not in self.DISPLAY_SIGNAL_META:
            for candidate in self.DISPLAY_SIGNAL_ORDER:
                if self._state_key_for_display(candidate) == key:
                    display_key = candidate
                    break
        if display_key in self.DISPLAY_SIGNAL_META:
            meta = self.DISPLAY_SIGNAL_META[display_key]
            return {
                "key": display_key,
                "name": self._signal_label(display_key),
                "color": str(meta["color"]),
                "group": str(meta["group"]),
            }
        return {"key": key, "name": key, "color": "#94a3b8", "group": "behavior"}

    def _default_behavior_roles(self) -> Dict[str, str]:
        """Default input/output role mapping."""
        return {
            key: str(self.DISPLAY_SIGNAL_META[key]["role"])
            for key in self.BEHAVIOR_PIN_KEYS
        }

    def _format_pin_list(self, pins: List[int]) -> str:
        """Render pin list as compact text."""
        if not pins:
            return "-"
        return ", ".join(str(int(pin)) for pin in pins)

    def _parse_pin_text(self, raw_text: str) -> List[int]:
        """Parse comma-separated pin list from user input."""
        if raw_text is None:
            return []
        tokens = [token.strip() for token in str(raw_text).replace(";", ",").split(",")]
        pins = []
        for token in tokens:
            if not token:
                continue
            pins.append(int(token))
        return pins

    def _current_behavior_pin_map(self) -> Dict[str, List[int]]:
        """Read current behavior pin mapping from UI edits."""
        pin_map = {}
        defaults = self._default_behavior_pin_map()
        for key in self.BEHAVIOR_PIN_KEYS:
            edit = self.behavior_pin_edits.get(key)
            if edit is None:
                pin_map[key] = defaults.get(key, []).copy()
                continue
            try:
                parsed = self._parse_pin_text(edit.text())
            except Exception:
                parsed = []
            pin_map[key] = parsed if parsed else defaults.get(key, []).copy()
        return pin_map

    def _current_behavior_roles(self) -> Dict[str, str]:
        """Read current behavior input/output roles from UI."""
        roles = {}
        defaults = self._default_behavior_roles()
        for key in self.BEHAVIOR_PIN_KEYS:
            box = self.behavior_role_boxes.get(key)
            if box is None:
                roles[key] = defaults.get(key, "Output")
                continue
            roles[key] = box.currentText()
        return roles

    def _refresh_pin_display_from_map(self, pin_map: Dict[str, List[int]]):
        """Update pin display labels in the behavior panel."""
        for key, label in self.pin_value_labels.items():
            pins = pin_map.get(key, [])
            label.setText(self._format_pin_list(pins))

    def _apply_behavior_pin_configuration(self, persist: bool = True):
        """Apply current behavior pin/role/label mapping and push to worker."""
        try:
            pin_map = self._current_behavior_pin_map()
        except Exception as e:
            self._on_error_occurred(f"Invalid pin configuration: {str(e)}")
            return

        role_map = self._current_behavior_roles()
        for key in self.DISPLAY_SIGNAL_ORDER:
            label_edit = self.signal_label_edits.get(key)
            enabled_check = self.signal_enabled_checks.get(key)
            raw_label = label_edit.text().strip() if label_edit is not None else ""
            default_label = str(self.DISPLAY_SIGNAL_META[key]["name"])
            self.signal_display_config.setdefault(key, {})
            self.signal_display_config[key]["name"] = raw_label if raw_label else default_label
            self.signal_display_config[key]["enabled"] = bool(enabled_check.isChecked()) if enabled_check else True

        for key, label in self.pin_name_labels.items():
            label.setText(f"{self._signal_label(key)}:")

        self._refresh_pin_display_from_map(pin_map)
        self._rebuild_monitor_visuals(reset_plot=True)

        if persist:
            for key, pins in pin_map.items():
                self.settings.setValue(f"behavior_pin_{key}", self._format_pin_list(pins))
            for key, role in role_map.items():
                self.settings.setValue(f"behavior_role_{key}", role)
            for key in self.DISPLAY_SIGNAL_ORDER:
                self.settings.setValue(f"behavior_signal_label_{key}", self._signal_label(key))
                self.settings.setValue(f"behavior_signal_enabled_{key}", int(bool(self.signal_display_config[key]["enabled"])))

        if self.arduino_worker:
            self.arduino_worker.set_manual_pin_config(pin_map)
            self.arduino_worker.set_signal_roles(role_map)

    def _load_behavior_panel_settings(self):
        """Load saved behavior pin and role settings."""
        defaults = self._default_behavior_pin_map()
        default_roles = self._default_behavior_roles()
        for key in self.BEHAVIOR_PIN_KEYS:
            pin_text = self.settings.value(f"behavior_pin_{key}", self._format_pin_list(defaults.get(key, [])))
            role_text = self.settings.value(f"behavior_role_{key}", default_roles.get(key, "Output"))
            label_text = str(self.settings.value(
                f"behavior_signal_label_{key}",
                self.signal_display_config.get(key, {}).get("name", self.DISPLAY_SIGNAL_META[key]["name"])
            ))
            enabled_raw = self.settings.value(
                f"behavior_signal_enabled_{key}",
                int(bool(self.signal_display_config.get(key, {}).get("enabled", True)))
            )
            enabled_value = str(enabled_raw).strip().lower() not in ("0", "false", "no", "off")

            pin_edit = self.behavior_pin_edits.get(key)
            if pin_edit is not None:
                pin_edit.setText(str(pin_text))

            role_box = self.behavior_role_boxes.get(key)
            if role_box is not None:
                role_box.blockSignals(True)
                if role_text in [role_box.itemText(i) for i in range(role_box.count())]:
                    role_box.setCurrentText(role_text)
                else:
                    role_box.setCurrentText(default_roles.get(key, "Output"))
                role_box.blockSignals(False)

            label_edit = self.signal_label_edits.get(key)
            if label_edit is not None:
                label_edit.setText(label_text)
            enabled_check = self.signal_enabled_checks.get(key)
            if enabled_check is not None:
                enabled_check.setChecked(bool(enabled_value))

            self.signal_display_config.setdefault(key, {})
            self.signal_display_config[key]["name"] = label_text if label_text else str(self.DISPLAY_SIGNAL_META[key]["name"])
            self.signal_display_config[key]["enabled"] = bool(enabled_value)

        self._apply_behavior_pin_configuration(persist=False)

    def _edit_sync_parameters(self):
        """Open dialog for sync pulse timing parameters."""
        if self.arduino_worker:
            period_s, pulse_s = self.arduino_worker.get_sync_parameters()
        else:
            period_s = float(self.settings.value("sync_period_s", 1.0))
            pulse_s = float(self.settings.value("sync_pulse_s", 0.05))

        dialog = QDialog(self)
        dialog.setWindowTitle("Sync Parameters")
        form = QFormLayout(dialog)

        spin_period = QDoubleSpinBox()
        spin_period.setDecimals(3)
        spin_period.setRange(0.05, 30.0)
        spin_period.setSuffix(" s")
        spin_period.setValue(float(period_s))
        form.addRow("Period:", spin_period)

        spin_pulse = QDoubleSpinBox()
        spin_pulse.setDecimals(3)
        spin_pulse.setRange(0.001, 10.0)
        spin_pulse.setSuffix(" s")
        spin_pulse.setValue(float(pulse_s))
        form.addRow("Pulse Width:", spin_pulse)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        period_val = float(spin_period.value())
        pulse_val = float(spin_pulse.value())
        if pulse_val >= period_val:
            self._on_error_occurred("Sync pulse width must be smaller than period.")
            return

        if self.arduino_worker:
            self.arduino_worker.set_sync_parameters(period_val, pulse_val)
        else:
            self.settings.setValue("sync_period_s", period_val)
            self.settings.setValue("sync_pulse_s", pulse_val)
        self._on_status_update(f"Sync params updated: period={period_val:.3f}s, pulse={pulse_val:.3f}s")

    def _edit_barcode_parameters(self):
        """Open dialog for barcode state machine timing parameters."""
        if self.arduino_worker:
            params = self.arduino_worker.get_barcode_parameters()
        else:
            params = {
                "bits": int(self.settings.value("barcode_bits", 32)),
                "start_pulse_s": float(self.settings.value("barcode_start_pulse_s", 0.1)),
                "start_low_s": float(self.settings.value("barcode_start_low_s", 0.1)),
                "bit_s": float(self.settings.value("barcode_bit_s", 0.1)),
                "interval_s": float(self.settings.value("barcode_interval_s", 5.0)),
            }

        dialog = QDialog(self)
        dialog.setWindowTitle("Barcode Parameters")
        form = QFormLayout(dialog)

        spin_bits = QSpinBox()
        spin_bits.setRange(1, 64)
        spin_bits.setValue(int(params.get("bits", 32)))
        form.addRow("Bits:", spin_bits)

        spin_start_hi = QDoubleSpinBox()
        spin_start_hi.setDecimals(3)
        spin_start_hi.setRange(0.001, 10.0)
        spin_start_hi.setSuffix(" s")
        spin_start_hi.setValue(float(params.get("start_pulse_s", 0.1)))
        form.addRow("Start HIGH:", spin_start_hi)

        spin_start_lo = QDoubleSpinBox()
        spin_start_lo.setDecimals(3)
        spin_start_lo.setRange(0.001, 10.0)
        spin_start_lo.setSuffix(" s")
        spin_start_lo.setValue(float(params.get("start_low_s", 0.1)))
        form.addRow("Start LOW:", spin_start_lo)

        spin_bit = QDoubleSpinBox()
        spin_bit.setDecimals(3)
        spin_bit.setRange(0.001, 10.0)
        spin_bit.setSuffix(" s")
        spin_bit.setValue(float(params.get("bit_s", 0.1)))
        form.addRow("Bit Duration:", spin_bit)

        spin_interval = QDoubleSpinBox()
        spin_interval.setDecimals(3)
        spin_interval.setRange(0.010, 60.0)
        spin_interval.setSuffix(" s")
        spin_interval.setValue(float(params.get("interval_s", 5.0)))
        form.addRow("Frame Interval:", spin_interval)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        bits_val = int(spin_bits.value())
        start_hi_val = float(spin_start_hi.value())
        start_lo_val = float(spin_start_lo.value())
        bit_val = float(spin_bit.value())
        interval_val = float(spin_interval.value())

        if self.arduino_worker:
            self.arduino_worker.set_barcode_parameters(
                bits=bits_val,
                start_pulse_s=start_hi_val,
                start_low_s=start_lo_val,
                bit_s=bit_val,
                interval_s=interval_val,
            )
        else:
            self.settings.setValue("barcode_bits", bits_val)
            self.settings.setValue("barcode_start_pulse_s", start_hi_val)
            self.settings.setValue("barcode_start_low_s", start_lo_val)
            self.settings.setValue("barcode_bit_s", bit_val)
            self.settings.setValue("barcode_interval_s", interval_val)

        self._on_status_update(
            "Barcode params updated: "
            f"bits={bits_val}, start={start_hi_val:.3f}/{start_lo_val:.3f}s, "
            f"bit={bit_val:.3f}s, interval={interval_val:.3f}s"
        )

    def _setup_worker(self):
        """Initialize the camera worker thread and connect signals."""
        self.worker = CameraWorker()

        # Connect worker signals to GUI slots
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.status_update.connect(self._on_status_update)
        self.worker.fps_update.connect(self._on_fps_update)
        self.worker.buffer_update.connect(self._on_buffer_update)
        self.worker.error_occurred.connect(self._on_error_occurred)
        self.worker.recording_stopped.connect(self._on_recording_stopped)

        # Connect frame recording to TTL sampling (sync TTLs with camera frames)
        self.worker.frame_recorded.connect(self._on_frame_recorded)
        self._apply_line_label_map_to_worker()

    def _setup_arduino_worker(self):
        """
        Initialize and wire the Arduino worker that talks to the external board.

        The worker runs in its own QThread so board I/O does not block the GUI.
        """
        self.arduino_worker = ArduinoOutputWorker()

        # Connect signals
        self.arduino_worker.port_list_updated.connect(self._on_port_list_updated)
        self.arduino_worker.connection_status.connect(self._on_arduino_connection_status)
        self.arduino_worker.ttl_states_updated.connect(self._on_ttl_states_updated)
        self.arduino_worker.pin_config_received.connect(self._on_pin_config_received)
        self.arduino_worker.error_occurred.connect(self._on_error_occurred)
        self._apply_behavior_pin_configuration(persist=False)

        # Initial port scan
        self._scan_arduino_ports()

        # Try to auto-connect to last used port
        if self.arduino_worker.port_name:
            for i in range(self.combo_arduino_port.count()):
                if self.arduino_worker.port_name in self.combo_arduino_port.itemText(i):
                    self.combo_arduino_port.setCurrentIndex(i)
                    break

    def _scan_cameras(self):
        """Scan for Basler and USB cameras."""
        self.combo_camera.clear()
        cameras = []

        # Basler cameras
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            for index, dev in enumerate(devices):
                model = dev.GetModelName()
                serial = dev.GetSerialNumber()
                label = f"Basler: {model} ({serial})"
                camera_info = {"type": "basler", "index": index, "serial": serial}
                self.combo_camera.addItem(label, camera_info)
                cameras.append(camera_info)
        except Exception as e:
            print(f"Error scanning Basler cameras: {e}")
            # Optionally show in status bar if UI is ready
            if hasattr(self, 'status_bar'):
                self._on_status_update(f"Basler scan error: {str(e)}")

        # USB cameras
        backend = cv2.CAP_MSMF if os.name == 'nt' else cv2.CAP_V4L2
        for index in range(10):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                label = f"USB: Device {index}"
                camera_info = {"type": "usb", "index": index}
                self.combo_camera.addItem(label, camera_info)
                cameras.append(camera_info)
                cap.release()
            else:
                cap.release()

        if not cameras:
            self.combo_camera.addItem("No cameras detected", None)
            return

        last_type = self.settings.value('last_camera_type', '')
        last_index = self.settings.value('last_camera_index', '')
        if last_type != '' and last_index != '':
            try:
                last_index = int(last_index)
            except Exception:
                return
            for i in range(self.combo_camera.count()):
                data = self.combo_camera.itemData(i)
                if not data:
                    continue
                if data.get('type') == last_type and int(data.get('index', -1)) == last_index:
                    self.combo_camera.setCurrentIndex(i)
                    break

    # ... (continue with slot implementations)

    # ===== Metadata Methods =====

    def _add_custom_metadata_field(self):
        """Add a custom metadata field."""
        from PySide6.QtWidgets import QInputDialog

        field_name, ok = QInputDialog.getText(self, "Add Custom Field", "Field Name:")
        if ok and field_name:
            field_edit = QLineEdit()
            field_edit.setPlaceholderText(f"Enter {field_name}...")
            self.metadata_layout.addRow(f"{field_name}:", field_edit)
            self.custom_metadata_fields[field_name] = field_edit

    def _save_metadata_template(self):
        """Save current metadata as template."""
        self._collect_metadata()
        template_file = Path(self.last_save_folder) / "metadata_template.json"
        with open(template_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        self._on_status_update(f"Metadata template saved to {template_file}")

    def _load_metadata(self):
        """Load metadata template if exists."""
        template_file = Path(self.last_save_folder) / "metadata_template.json"
        if template_file.exists():
            try:
                with open(template_file, 'r') as f:
                    template = json.load(f)
                    # Load standard fields
                    if 'animal_id' in template:
                        self.meta_animal_id.setText(template['animal_id'])
                    if 'experiment' in template:
                        self.meta_experiment.setText(template['experiment'])
            except:
                pass

    def _load_ui_settings(self):
        """Load saved UI settings."""
        self.spin_fps.blockSignals(True)
        self.spin_fps.setValue(float(self.settings.value('camera_fps', 30.0)))
        self.spin_fps.blockSignals(False)

        self.spin_exposure.blockSignals(True)
        self.spin_exposure.setValue(float(self.settings.value('exposure_ms', 10.0)))
        self.spin_exposure.blockSignals(False)

        self.spin_width.blockSignals(True)
        self.spin_width.setValue(int(self.settings.value('camera_width', 1080)))
        self.spin_width.blockSignals(False)

        self.spin_height.blockSignals(True)
        self.spin_height.setValue(int(self.settings.value('camera_height', 1080)))
        self.spin_height.blockSignals(False)

        image_format = self.settings.value('image_format', 'Mono8')
        if image_format in ("Mono8", "BGR8"):
            self.combo_image_format.setCurrentText(image_format)

        encoder_index = int(self.settings.value('encoder_index', 0))
        if 0 <= encoder_index < self.combo_encoder.count():
            self.combo_encoder.setCurrentIndex(encoder_index)

        self.spin_hours.setValue(int(self.settings.value('max_hours', 0)))
        self.spin_minutes.setValue(int(self.settings.value('max_minutes', 5)))
        self.spin_seconds.setValue(int(self.settings.value('max_seconds', 0)))

        unlimited = int(self.settings.value('max_unlimited', 0))
        self.check_unlimited.setCurrentIndex(1 if unlimited else 0)

        self.edit_save_folder.setText(self.last_save_folder)

        self._load_line_label_settings()
        self._load_behavior_panel_settings()

        metadata_visible = int(self.settings.value("metadata_panel_visible", 1))
        if metadata_visible:
            self.metadata_panel.show()
            self.btn_toggle_metadata.setText("Hide Metadata")
        else:
            self.metadata_panel.hide()
            self.btn_toggle_metadata.setText("Show Metadata")

        self.spin_fps.valueChanged.connect(lambda v: self._save_ui_setting('camera_fps', v))
        self.spin_exposure.valueChanged.connect(lambda v: self._save_ui_setting('exposure_ms', v))
        self.spin_width.valueChanged.connect(lambda v: self._save_ui_setting('camera_width', v))
        self.spin_height.valueChanged.connect(lambda v: self._save_ui_setting('camera_height', v))
        self.combo_encoder.currentIndexChanged.connect(lambda v: self._save_ui_setting('encoder_index', v))
        self.combo_image_format.currentTextChanged.connect(lambda v: self._save_ui_setting('image_format', v))
        self.spin_hours.valueChanged.connect(lambda v: self._save_ui_setting('max_hours', v))
        self.spin_minutes.valueChanged.connect(lambda v: self._save_ui_setting('max_minutes', v))
        self.spin_seconds.valueChanged.connect(lambda v: self._save_ui_setting('max_seconds', v))
        self.check_unlimited.currentIndexChanged.connect(lambda v: self._save_ui_setting('max_unlimited', 1 if v == 1 else 0))

    @Slot()
    def _toggle_metadata_panel(self):
        """Show/hide metadata panel."""
        if self.metadata_panel.isVisible():
            self.metadata_panel.hide()
            self.btn_toggle_metadata.setText("Show Metadata")
        else:
            self.metadata_panel.show()
            self.btn_toggle_metadata.setText("Hide Metadata")
            if hasattr(self, "main_splitter"):
                sizes = self.main_splitter.sizes()
                if sizes and sizes[0] == 0:
                    self.main_splitter.setSizes([250, 760, 730])

        self.settings.setValue("metadata_panel_visible", 1 if self.metadata_panel.isVisible() else 0)

    def _load_line_label_settings(self):
        """Load saved camera input label selections."""
        label_defaults = {
            1: self.settings.value('line_label_1', 'None'),
            2: self.settings.value('line_label_2', 'None'),
            3: self.settings.value('line_label_3', 'None'),
            4: self.settings.value('line_label_4', 'None'),
        }
        for line, value in label_defaults.items():
            combo = getattr(self, f"combo_line{line}_label", None)
            if not combo:
                continue
            value = "Sync" if str(value) == "TTL 1Hz" else str(value)
            combo.blockSignals(True)
            if value in [combo.itemText(i) for i in range(combo.count())]:
                combo.setCurrentText(value)
            else:
                combo.setCurrentText("None")
            combo.blockSignals(False)

        self._apply_line_label_map_to_worker()

    def _save_ui_setting(self, key: str, value):
        """Persist a UI setting."""
        self.settings.setValue(key, value)

    def _on_line_label_changed(self, line_number: int, value: str):
        """Handle camera input label changes."""
        self._save_ui_setting(f'line_label_{line_number}', value)
        self._apply_line_label_map_to_worker()

    def _apply_line_label_map_to_worker(self):
        """Push line label suffix mapping to the camera worker."""
        if not self.worker:
            return
        self.worker.set_line_label_map(self._get_line_label_map())

    def _get_line_label_map(self) -> dict:
        """Build line status suffix mapping for CSV output."""
        label_map = {}
        for line in range(1, 5):
            combo = getattr(self, f"combo_line{line}_label", None)
            if not combo:
                continue
            suffix = self._line_label_suffix(combo.currentText())
            if suffix:
                label_map[f"line{line}_status"] = suffix
        return label_map

    def _line_label_suffix(self, label: str) -> str:
        if label == "Gate":
            return "gate"
        if label in ("TTL 1Hz", "Sync"):
            return "ttl_1hz"
        if label == "Barcode":
            return "barcode"
        if label == "Lever":
            return "lever"
        if label == "Cue":
            return "cue"
        if label == "Reward":
            return "reward"
        if label == "ITI":
            return "iti"
        return ""

    def _apply_line_label_suffixes(self, df):
        """Rename line status columns with selected suffixes."""
        label_map = self._get_line_label_map()
        if not label_map:
            return df
        rename_map = {
            key: f"{key}_{suffix}"
            for key, suffix in label_map.items()
            if key in df.columns and suffix
        }
        if rename_map:
            return df.rename(columns=rename_map)
        return df

    def _collect_metadata(self):
        """Collect all metadata fields."""
        self.metadata = {
            'animal_id': self.meta_animal_id.text(),
            'experiment': self.meta_experiment.text(),
            'date': self.meta_date.text(),
            'notes': self.meta_notes.toPlainText(),
            'timestamp': datetime.now().isoformat()
        }

        # Add custom fields
        for field_name, field_edit in self.custom_metadata_fields.items():
            self.metadata[field_name] = field_edit.text()

        return self.metadata

    def _save_metadata_to_file(self, folder: str):
        """Save metadata to JSON file."""
        self._collect_metadata()
        metadata_file = Path(folder) / f"{self.edit_filename.text()}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    # ===== Camera Control Slots =====

    @Slot()
    def _on_connect_clicked(self):
        """Handle connect/disconnect button click."""
        if not self.is_camera_connected:
            # Connect to camera
            camera_info = self.combo_camera.currentData()
            if not camera_info:
                self._on_error_occurred("No camera selected")
                return

            if self.worker.connect_camera(camera_info):
                self.is_camera_connected = True
                self.btn_connect.setText("Disconnect Camera")
                self.btn_connect.setStyleSheet("QPushButton { background-color: #f44336; }")
                self.btn_record.setEnabled(True)
                self._save_ui_setting('last_camera_type', camera_info.get('type', ''))
                self._save_ui_setting('last_camera_index', camera_info.get('index', ''))

                # Apply initial settings
                if camera_info.get('type') == "basler":
                    self._enable_basler_frame_rate()
                self._on_fps_changed(self.spin_fps.value())
                self._on_exposure_changed(self.spin_exposure.value())
                self._on_resolution_changed()
                self._on_image_format_changed(self.combo_image_format.currentText())

                # Start the worker thread
                self.worker.start()
                self._update_advanced_controls_state()
            else:
                self._on_error_occurred("Failed to connect to camera")
        else:
            # Disconnect camera
            self._disconnect_camera()

    def _disconnect_camera(self):
        """Disconnect camera and cleanup."""
        if self.worker.is_recording:
            self.worker.stop_recording()

        self.worker.stop()
        self.worker.wait()  # Wait for thread to finish
        self.worker.disconnect_camera()

        self.is_camera_connected = False
        self.btn_connect.setText("Connect Camera")
        self.btn_connect.setStyleSheet("")
        self.btn_record.setEnabled(False)

        self.video_label.clear()
        self.video_label.setText("Camera Disconnected")
        self.label_fps.setText("FPS: 0.0")
        self.label_buffer.setText("Buffer: 0%")
        self._update_advanced_controls_state()

    # @Slot(str)
    # def _on_trigger_mode_changed(self, mode_text: str):
    #     """Handle trigger mode change."""
    #     mode = "FreeRun"  # Default
    #
    #     if mode_text == "Free Run":
    #         mode = "FreeRun"
    #     elif mode_text == "External Trigger":
    #         mode = "ExternalTrigger"
    #
    #     if self.worker:
    #         self.worker.set_trigger_mode(mode)

    @Slot()
    def _on_record_clicked(self):
        """Handle record button click."""
        if not self.worker.is_recording:
            # Generate filename from metadata
            animal_id = self.meta_animal_id.text().strip()
            experiment = self.meta_experiment.text().strip()
            condition = ""
            if "Condition" in self.custom_metadata_fields:
                condition = self.custom_metadata_fields["Condition"].text().strip()

            # Build filename
            if animal_id and experiment:
                if condition:
                    filename = f"{animal_id}_{experiment}_{condition}"
                else:
                    filename = f"{animal_id}_{experiment}"
            else:
                filename = self.edit_filename.text().strip()

            if not filename:
                self._on_error_occurred("Please enter metadata or filename")
                return

            # Save directly to selected folder (NO subfolder creation)
            save_folder = Path(self.edit_save_folder.text())
            filepath = str(self._get_unique_recording_path(save_folder, filename))
            self.current_recording_filepath = filepath
            self.edit_filename.setText(Path(filepath).name)

            # Save metadata JSON
            self._collect_metadata()
            metadata_file = f"{filepath}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=4)

            # Set encoder from combo box
            encoder_text = self.combo_encoder.currentText()
            if "nvenc" in encoder_text:
                encoder = "h264_nvenc"
            elif "libx264" in encoder_text:
                encoder = "libx264"
            elif "qsv" in encoder_text:
                encoder = "h264_qsv"
            else:
                encoder = "h264_nvenc"

            self.worker.set_encoder(encoder)

            # Start Arduino TTLs if connected
            if self.is_arduino_connected:
                # Keep recording I/O roles in sync with current setup values.
                self._apply_behavior_pin_configuration(persist=True)
                if not self.arduino_worker.start_recording():
                    self._on_status_update("Warning: Arduino TTLs failed to start; recording will continue.")
                    self.label_ttl_status.setText("TTL: START FAILED")
                    self.label_ttl_status.setStyleSheet("background-color: #b45309; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
                    self.label_behavior_status.setText("Behavior: IDLE")
                    self.label_behavior_status.setStyleSheet("background-color: #374151; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
                else:
                    # Reset and clear plot for new recording
                    self._reset_ttl_plot()

                    self.label_ttl_status.setText("TTL: RECORDING")
                    self.label_ttl_status.setStyleSheet("background-color: green; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
                    self.label_behavior_status.setText("Behavior: ARMED")
                    self.label_behavior_status.setStyleSheet("background-color: #2563eb; color: white; padding: 10px; font-weight: bold; font-size: 16px;")

            # Start camera recording
            if not self.worker.start_recording(filepath):
                if self.is_arduino_connected:
                    self._stop_arduino_generation()
                self.current_recording_filepath = None
                return

            # Start recording timer
            self.recording_start_time = datetime.now()
            self.recording_timer.start(1000)

            self.btn_record.setText("Stop Recording")
            self.btn_record.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; font-size: 14px; }")
            self.label_recording.setText("Recording")
            self.label_recording.setStyleSheet("QLabel { color: red; font-weight: bold; }")

            # Disable controls during recording
            self.btn_connect.setEnabled(False)
            self.edit_filename.setEnabled(False)
            if self.is_arduino_connected:
                self.btn_test_ttl.setEnabled(False)
        else:
            # Stop recording
            self.worker.stop_recording()
            self.recording_timer.stop()

            # Stop Arduino TTLs if active
            if self.is_arduino_connected:
                self._stop_arduino_generation()

    @Slot()
    def _on_recording_stopped(self):
        """Handle recording stopped signal."""
        self.btn_record.setText("Start Recording")
        self.btn_record.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; }")
        self.label_recording.setText("Not Recording")
        self.label_recording.setStyleSheet("")
        self.label_recording_time.setText("00:00:00")

        # Re-enable controls
        self.btn_connect.setEnabled(True)
        self.edit_filename.setEnabled(True)
        if self.is_arduino_connected:
            self.btn_test_ttl.setEnabled(True)

        # Save Arduino TTL data if connected
        if self.is_arduino_connected:
            self._stop_arduino_generation()

            filepath = self.current_recording_filepath
            if not filepath:
                save_folder = Path(self.edit_save_folder.text())
                fallback_name = self.edit_filename.text().strip() or "recording"
                filepath = str(self._get_unique_recording_path(save_folder, fallback_name))

            # Save TTL history
            self._save_arduino_ttl_data(filepath)
            self.current_recording_filepath = None

            # Reset TTL status
            self.label_ttl_status.setText("TTL: IDLE")
            self.label_ttl_status.setStyleSheet("background-color: gray; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
            self.label_behavior_status.setText("Behavior: IDLE")
            self.label_behavior_status.setStyleSheet("background-color: #374151; color: white; padding: 10px; font-weight: bold; font-size: 16px;")

        # Generate new default filename
        default_filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.edit_filename.setText(default_filename)

    def _update_recording_time(self):
        """Update recording time display."""
        if self.recording_start_time:
            elapsed = datetime.now() - self.recording_start_time
            total_seconds = int(elapsed.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            self.label_recording_time.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

            # Check max recording length
            if self.check_unlimited.currentText() == "Limited":
                max_seconds = (self.spin_hours.value() * 3600 +
                             self.spin_minutes.value() * 60 +
                             self.spin_seconds.value())
                if elapsed.total_seconds() >= max_seconds:
                    self.worker.stop_recording()
                    self.recording_timer.stop()

    def _get_unique_recording_path(self, folder: Path, base_name: str) -> Path:
        """Return a unique base path for recording outputs."""
        base_name = base_name.strip() or "recording"
        candidate = folder / base_name
        suffix = 1
        while self._recording_files_exist(candidate):
            candidate = folder / f"{base_name}_{suffix}"
            suffix += 1
        return candidate

    def _recording_files_exist(self, base_path: Path) -> bool:
        """Check whether any output file for the given base path already exists."""
        stems = [
            f"{base_path}.mp4",
            f"{base_path}_metadata.json",
            f"{base_path}_ttl_states.csv",
            f"{base_path}_ttl_live_states.csv",
            f"{base_path}_ttl_events.csv",
            f"{base_path}_ttl_counts.csv",
            f"{base_path}_behavior_summary.csv",
        ]
        return any(Path(path).exists() for path in stems)

    def _browse_save_folder(self):
        """Browse for save folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder",
                                                  self.last_save_folder)
        if folder:
            self.last_save_folder = folder
            self.edit_save_folder.setText(folder)
            self.settings.setValue('last_save_folder', folder)

    # ===== Camera Settings Slots =====

    @Slot(float)
    def _on_fps_changed(self, value):
        """Handle FPS change."""
        if not self.worker:
            return
        try:
            if self.worker.camera_type == "usb" and self.worker.usb_capture:
                self.worker.usb_capture.set(cv2.CAP_PROP_FPS, float(value))
                self.worker.set_target_fps(value)
            elif self.worker.camera and self.worker.camera.IsOpen():
                self.worker.camera.AcquisitionFrameRate.SetValue(value)
                self.worker.set_target_fps(value)
                self.worker.sync_camera_fps()
            self._on_status_update(f"FPS set to {value:.1f}")
        except Exception as e:
            self._on_error_occurred(f"Failed to set FPS: {str(e)}")

    @Slot()
    def _on_resolution_changed(self):
        """Handle resolution change."""
        if not self.worker:
            return
        if self.worker.camera_type == "usb" and self.worker.usb_capture:
            try:
                width = int(self.spin_width.value())
                height = int(self.spin_height.value())
                self.worker.usb_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.worker.usb_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(self.worker.usb_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.worker.usb_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.worker.update_resolution(actual_width, actual_height)
                self._on_status_update(f"Resolution set to {actual_width}x{actual_height}")
                self._update_advanced_controls_state()
            except Exception as e:
                self._on_error_occurred(f"Failed to set resolution: {str(e)}")
            return

        if self.worker.camera and self.worker.camera.IsOpen():
            try:
                width = self.spin_width.value()
                height = self.spin_height.value()

                # Stop grabbing to change resolution
                was_grabbing = self.worker.camera.IsGrabbing()
                if was_grabbing:
                    self.worker.camera.StopGrabbing()

                for node_name in ("OffsetX", "OffsetY"):
                    node = self._get_camera_node(node_name)
                    if node and hasattr(node, "IsWritable") and node.IsWritable():
                        node.SetValue(int(node.GetMin()))

                width = self._clamp_int_node("Width", width)
                height = self._clamp_int_node("Height", height)
                if width is None or height is None:
                    raise RuntimeError("Width/Height not supported by camera")

                self.worker.camera.Width.SetValue(int(width))
                self.worker.camera.Height.SetValue(int(height))
                self.worker.update_resolution(width, height)

                if was_grabbing:
                    self.worker.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                self._on_status_update(f"Resolution set to {width}x{height}")
                self._update_advanced_controls_state()
            except Exception as e:
                self._on_error_occurred(f"Failed to set resolution: {str(e)}")

    @Slot(float)
    def _on_exposure_changed(self, value):
        """Handle exposure time change."""
        if self.worker and self.worker.camera_type == "basler" and self.worker.camera:
            try:
                # Convert ms to microseconds
                exposure_us = value * 1000.0
                self.worker.camera.ExposureTime.SetValue(exposure_us)
                self._on_status_update(f"Exposure set to {value:.2f} ms")
            except Exception as e:
                self._on_error_occurred(f"Failed to set exposure: {str(e)}")

    @Slot(str)
    def _on_image_format_changed(self, format_text: str):
        """Handle image format change."""
        if self.worker:
            self.worker.set_image_format(format_text)
            self._save_ui_setting('image_format', format_text)

    def _toggle_advanced_settings(self):
        """Show or hide advanced video settings."""
        is_visible = not self.advanced_group.isVisible()
        self.advanced_group.setVisible(is_visible)
        self.btn_advanced.setText("Hide Advanced" if is_visible else "Show Advanced")

    def _update_advanced_controls_state(self):
        """Update advanced controls based on camera availability."""
        if not self.worker or self.worker.camera_type != "basler" or not self.worker.camera or not self.worker.camera.IsOpen():
            self._set_advanced_controls_enabled(False)
            return

        self._set_advanced_controls_enabled(True)

        self._configure_int_node("OffsetX", self.slider_offset_x, self.spin_offset_x)
        self._configure_int_node("OffsetY", self.slider_offset_y, self.spin_offset_y)

        self._configure_float_node("Gain", self.spin_gain)
        if not self._configure_float_node("Brightness", self.spin_brightness):
            self._configure_float_node("BlackLevel", self.spin_brightness)
        if not self._configure_float_node("Contrast", self.spin_contrast):
            self._configure_float_node("Gamma", self.spin_contrast)

        if self.settings.contains('offset_x') and self.settings.contains('offset_y'):
            try:
                self._on_offset_x_changed(int(self.settings.value('offset_x')))
                self._on_offset_y_changed(int(self.settings.value('offset_y')))
            except Exception:
                self._center_offsets()
        else:
            self._center_offsets()

        if self.settings.contains('gain'):
            try:
                self._on_gain_changed(float(self.settings.value('gain')))
            except Exception:
                pass
        if self.settings.contains('brightness'):
            try:
                self._on_brightness_changed(float(self.settings.value('brightness')))
            except Exception:
                pass
        if self.settings.contains('contrast'):
            try:
                self._on_contrast_changed(float(self.settings.value('contrast')))
            except Exception:
                pass

    def _enable_basler_frame_rate(self) -> bool:
        """Ensure Basler frame rate control is enabled by default."""
        node = self._get_camera_node("AcquisitionFrameRateEnable")
        if not node:
            return False
        try:
            if hasattr(node, "IsWritable") and not node.IsWritable():
                return False
        except Exception:
            pass
        try:
            node.SetValue(True)
        except Exception as e:
            self._on_error_occurred(f"Failed to enable AcquisitionFrameRate: {str(e)}")
            return False
        return True

    def _set_advanced_controls_enabled(self, enabled: bool):
        """Enable/disable advanced controls."""
        for widget in (
            self.slider_offset_x, self.spin_offset_x,
            self.slider_offset_y, self.spin_offset_y,
            self.btn_center_offsets,
            self.spin_gain, self.spin_brightness, self.spin_contrast,
        ):
            widget.setEnabled(enabled)

    def _configure_int_node(self, node_name: str, slider: QSlider, spin: QSpinBox):
        node = self._get_camera_node(node_name)
        if not node:
            slider.setEnabled(False)
            spin.setEnabled(False)
            return False

        min_val = int(node.GetMin())
        max_val = int(node.GetMax())
        inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
        current = int(node.GetValue())

        for widget in (slider, spin):
            widget.blockSignals(True)
            widget.setRange(min_val, max_val)
            widget.setSingleStep(inc)
            widget.setValue(current)
            widget.blockSignals(False)
            widget.setEnabled(True)
        return True

    def _configure_float_node(self, node_name: str, spin: QDoubleSpinBox):
        node = self._get_camera_node(node_name)
        if not node:
            spin.setEnabled(False)
            return False
        try:
            if hasattr(node, "IsWritable") and not node.IsWritable():
                spin.setEnabled(False)
                return False
        except Exception:
            pass

        min_val = float(node.GetMin())
        max_val = float(node.GetMax())
        inc = 0.1
        if hasattr(node, "GetInc"):
            try:
                inc = float(node.GetInc())
            except Exception:
                inc = 0.1
        current = float(node.GetValue())

        spin.blockSignals(True)
        spin.setDecimals(2)
        spin.setRange(min_val, max_val)
        spin.setSingleStep(max(inc, 0.01))
        spin.setValue(current)
        spin.blockSignals(False)
        spin.setEnabled(True)
        return True

    def _get_camera_node(self, node_name: str):
        if not self.worker or not self.worker.camera:
            return None
        try:
            node = getattr(self.worker.camera, node_name)
        except Exception:
            return None
        try:
            if hasattr(node, "IsReadable") and not node.IsReadable():
                return None
        except Exception:
            return None
        return node

    def _clamp_int_node(self, node_name: str, value: int):
        node = self._get_camera_node(node_name)
        if not node:
            return None
        try:
            if hasattr(node, "IsWritable") and not node.IsWritable():
                return None
        except Exception:
            pass
        try:
            min_val = int(node.GetMin())
            max_val = int(node.GetMax())
            inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
        except Exception:
            return None
        value = max(min_val, min(max_val, int(value)))
        if inc > 1:
            value = min_val + ((value - min_val) // inc) * inc
        return value

    def _set_camera_int_node(self, node_name: str, value: int):
        node = self._get_camera_node(node_name)
        if not node:
            return
        try:
            node.SetValue(int(value))
        except Exception as e:
            self._on_error_occurred(f"Failed to set {node_name}: {str(e)}")

    def _set_camera_float_node(self, node_name: str, value: float):
        node = self._get_camera_node(node_name)
        if not node:
            return
        try:
            node.SetValue(float(value))
        except Exception as e:
            self._on_error_occurred(f"Failed to set {node_name}: {str(e)}")

    def _sync_offset_controls(self, value: int, slider: QSlider, spin: QSpinBox):
        if slider.value() != value:
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        if spin.value() != value:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _on_offset_x_changed(self, value: int):
        self._sync_offset_controls(value, self.slider_offset_x, self.spin_offset_x)
        self._set_camera_int_node("OffsetX", value)
        self._save_ui_setting('offset_x', value)

    def _on_offset_y_changed(self, value: int):
        self._sync_offset_controls(value, self.slider_offset_y, self.spin_offset_y)
        self._set_camera_int_node("OffsetY", value)
        self._save_ui_setting('offset_y', value)

    def _center_offsets(self):
        offset_x_node = self._get_camera_node("OffsetX")
        offset_y_node = self._get_camera_node("OffsetY")
        if not offset_x_node or not offset_y_node:
            return

        def _center(node):
            min_val = int(node.GetMin())
            max_val = int(node.GetMax())
            inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
            center = min_val + ((max_val - min_val) // (2 * inc)) * inc
            return center

        centered_x = _center(offset_x_node)
        centered_y = _center(offset_y_node)
        self._on_offset_x_changed(centered_x)
        self._on_offset_y_changed(centered_y)

    def _on_gain_changed(self, value: float):
        self._set_camera_float_node("Gain", value)
        self._save_ui_setting('gain', value)

    def _on_brightness_changed(self, value: float):
        if not self._get_camera_node("Brightness"):
            self._set_camera_float_node("BlackLevel", value)
        else:
            self._set_camera_float_node("Brightness", value)
        self._save_ui_setting('brightness', value)

    def _on_contrast_changed(self, value: float):
        if not self._get_camera_node("Contrast"):
            self._set_camera_float_node("Gamma", value)
        else:
            self._set_camera_float_node("Contrast", value)
        self._save_ui_setting('contrast', value)

    def _toggle_roi_draw(self):
        """Toggle ROI drawing mode."""
        self.roi_draw_mode = not self.roi_draw_mode
        self.roi_preview = None
        self.roi_dragging = False
        if self.roi_draw_mode:
            self.btn_draw_roi.setText("Drawing...")
            self.btn_draw_roi.setStyleSheet("QPushButton { background-color: #f59e0b; color: white; font-weight: bold; }")
        else:
            self.btn_draw_roi.setText("Draw ROI")
            self.btn_draw_roi.setStyleSheet("")

    def _clear_roi(self):
        """Clear ROI and reset cropping."""
        self.roi_rect = None
        self.roi_preview = None
        if self.worker:
            self.worker.set_roi(None)

    def eventFilter(self, obj, event):
        if obj is self.video_label and self.roi_draw_mode:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                pos = event.position().toPoint()
                image_pos = self._map_pos_to_image(pos)
                if image_pos:
                    self.roi_dragging = True
                    self.roi_start_pos = image_pos
                    self.roi_preview = {'x': image_pos.x(), 'y': image_pos.y(), 'w': 1, 'h': 1}
                    return True
            elif event.type() == QEvent.MouseMove and self.roi_dragging:
                pos = event.position().toPoint()
                image_pos = self._map_pos_to_image(pos)
                if image_pos:
                    self._update_roi_preview(image_pos)
                    return True
            elif event.type() == QEvent.MouseButtonRelease and self.roi_dragging:
                pos = event.position().toPoint()
                image_pos = self._map_pos_to_image(pos)
                if image_pos:
                    self._update_roi_preview(image_pos)
                    self.roi_rect = self.roi_preview
                    self.roi_preview = None
                    if self.worker:
                        self.worker.set_roi(self.roi_rect)
                self.roi_dragging = False
                self._toggle_roi_draw()
                return True
        return super().eventFilter(obj, event)

    def _update_roi_preview(self, current_pos: QPoint):
        start = self.roi_start_pos
        x1 = min(start.x(), current_pos.x())
        y1 = min(start.y(), current_pos.y())
        x2 = max(start.x(), current_pos.x())
        y2 = max(start.y(), current_pos.y())
        self.roi_preview = {'x': x1, 'y': y1, 'w': max(1, x2 - x1), 'h': max(1, y2 - y1)}

    def _map_pos_to_image(self, pos: QPoint) -> Optional[QPoint]:
        if not self.last_frame_size:
            return None
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        img_w, img_h = self.last_frame_size
        scale = min(label_w / img_w, label_h / img_h)
        disp_w = img_w * scale
        disp_h = img_h * scale
        offset_x = (label_w - disp_w) / 2
        offset_y = (label_h - disp_h) / 2
        x = pos.x() - offset_x
        y = pos.y() - offset_y
        if x < 0 or y < 0 or x > disp_w or y > disp_h:
            return None
        img_x = int(x / scale)
        img_y = int(y / scale)
        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))
        return QPoint(img_x, img_y)

    def _roi_to_display_rect(self, roi: dict, pixmap: QPixmap) -> QRect:
        if not self.last_frame_size:
            return QRect()
        img_w, img_h = self.last_frame_size
        scale_x = pixmap.width() / img_w
        scale_y = pixmap.height() / img_h
        x = int(roi.get('x', 0) * scale_x)
        y = int(roi.get('y', 0) * scale_y)
        w = int(roi.get('w', img_w) * scale_x)
        h = int(roi.get('h', img_h) * scale_y)
        return QRect(x, y, max(1, w), max(1, h))

    # ===== Arduino Slots =====

    def _scan_arduino_ports(self):
        """Scan for Arduino ports."""
        self.arduino_worker.scan_ports()

    @Slot(list)
    def _on_port_list_updated(self, ports):
        """Update port list."""
        self.combo_arduino_port.clear()
        self.combo_arduino_port.addItems(ports)

    @Slot()
    def _on_arduino_connect_clicked(self):
        """
        Connect/disconnect board from the selected serial COM port.

        On connect:
        - open serial port
        - start worker thread if needed
        - enable test button

        On disconnect:
        - stop test mode
        - stop worker thread
        - reset UI state labels/counters
        """
        if not self.is_arduino_connected:
            # Ensure the latest UI role/pin mapping is active before connecting.
            self._apply_behavior_pin_configuration(persist=True)
            port = self.combo_arduino_port.currentText()

            if self.arduino_worker.connect_to_port(port):
                self.is_arduino_connected = True
                if not self.arduino_worker.isRunning():
                    self.arduino_worker.start()
                self.btn_arduino_connect.setText("Disconnect Arduino")
                self.btn_arduino_connect.setStyleSheet("QPushButton { background-color: #f44336; }")
                self.btn_test_ttl.setEnabled(True)
        else:
            if self.is_testing_ttl:
                self.arduino_worker.stop_test()
                self.is_testing_ttl = False
                self.btn_test_ttl.setText("Test TTL / Behavior")
                self.btn_test_ttl.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")

            self.arduino_worker.stop()
            self.arduino_worker.wait()
            self.is_arduino_connected = False
            self.btn_arduino_connect.setText("Connect Arduino")
            self.btn_arduino_connect.setStyleSheet("")
            self.btn_test_ttl.setEnabled(False)

            self._apply_behavior_pin_configuration(persist=False)
            for key, label in self.ttl_state_labels.items():
                label.setText("LOW")
                label.setStyleSheet("color: #9ca3af;")
            for key, label in self.ttl_count_labels.items():
                label.setText("0")

            self.label_ttl_status.setText("TTL: IDLE")
            self.label_ttl_status.setStyleSheet("background-color: gray; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
            self.label_behavior_status.setText("Behavior: IDLE")
            self.label_behavior_status.setStyleSheet("background-color: #374151; color: white; padding: 10px; font-weight: bold; font-size: 16px;")

    @Slot(bool, str)
    def _on_arduino_connection_status(self, connected, message):
        """Handle Arduino connection status."""
        if connected:
            self.label_arduino_status.setText(f"Status: {message}")
            self.label_arduino_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.label_arduino_status.setText(f"Status: {message}")
            self.label_arduino_status.setStyleSheet("color: red;")
            if self.is_arduino_connected:
                self.is_arduino_connected = False
                self.is_testing_ttl = False
                self.btn_arduino_connect.setText("Connect Arduino")
                self.btn_arduino_connect.setStyleSheet("")
                self.btn_test_ttl.setEnabled(False)
                self.btn_test_ttl.setText("Test TTL / Behavior")
                self.btn_test_ttl.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
                self.label_ttl_status.setText("TTL: IDLE")
                self.label_ttl_status.setStyleSheet("background-color: gray; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
                self.label_behavior_status.setText("Behavior: IDLE")
                self.label_behavior_status.setStyleSheet("background-color: #374151; color: white; padding: 10px; font-weight: bold; font-size: 16px;")

    @Slot(dict)
    def _on_pin_config_received(self, config):
        """Handle pin configuration from Arduino."""
        normalized = {}
        for key, value in config.items():
            k = str(key).lower()
            if isinstance(value, list):
                normalized[k] = [int(v) for v in value]
            else:
                normalized[k] = [int(value)]
        for key, pins in normalized.items():
            if key in self.pin_value_labels:
                self.pin_value_labels[key].setText(self._format_pin_list(pins))
            pin_edit = self.behavior_pin_edits.get(key)
            if pin_edit is not None:
                pin_edit.setText(self._format_pin_list(pins))

    def _ttl_count_key_for_signal(self, key: str) -> str:
        """Map signal key to arduino count key."""
        if key in ("barcode", "barcode0", "barcode1"):
            return "barcode_count"
        return f"{key}_count"

    @Slot(dict)
    def _on_ttl_states_updated(self, states):
        """
        Receive one normalized state packet from ArduinoOutputWorker.

        This slot is the central UI update path for:
        - TTL and behavior plots
        - HIGH/LOW state labels
        - rising-edge counters
        - status banners (TESTING / MONITORING / ACTIVE)
        """
        # Update plot
        current_time = (datetime.now() - self.plot_start_time).total_seconds()
        self.time_data.append(current_time)
        amplitude = 0.35
        for key in self.DISPLAY_SIGNAL_ORDER:
            if key in self.ttl_output_levels:
                level = self.ttl_output_levels.get(key, 0.0)
            elif key in self.behavior_levels:
                level = self.behavior_levels.get(key, 0.0)
            else:
                continue
            state_key = self._state_key_for_display(key)
            state = bool(states.get(state_key, False))
            self.ttl_plot_data[key].append(level + amplitude if state else level - amplitude)

        # Update curves
        times = np.fromiter(self.time_data, dtype=float)
        if times.size == 0:
            return
        if times.size == 1:
            step = 0.03
        else:
            step = max(0.01, times[-1] - times[-2])
        times_step = np.append(times, times[-1] + step)

        for key, curve in self.ttl_output_curves.items():
            curve.setData(times_step, np.fromiter(self.ttl_plot_data[key], dtype=float))

        for key, curve in self.behavior_curves.items():
            curve.setData(times_step, np.fromiter(self.ttl_plot_data[key], dtype=float))

        end_time = times[-1]
        start_time = max(0.0, end_time - self.ttl_window_seconds)
        self.ttl_plot.setXRange(start_time, end_time)
        self.behavior_plot.setXRange(start_time, end_time)

        if states.get("passive_mode"):
            if self.label_ttl_status.text() != "TTL: MONITORING":
                self.label_ttl_status.setText("TTL: MONITORING")
                self.label_ttl_status.setStyleSheet("background-color: #0ea5e9; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
        elif self.label_ttl_status.text() == "TTL: MONITORING" and self.is_testing_ttl:
            self.label_ttl_status.setText("TTL: TESTING")
            self.label_ttl_status.setStyleSheet("background-color: blue; color: white; padding: 10px; font-weight: bold; font-size: 16px;")

        behavior_active = any(
            bool(states.get(self._state_key_for_display(key), False))
            for key in self._active_signal_keys(group="behavior")
        )
        if behavior_active:
            self.label_behavior_status.setText("Behavior: ACTIVE")
            self.label_behavior_status.setStyleSheet("background-color: #16a34a; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
        elif states.get("passive_mode"):
            self.label_behavior_status.setText("Behavior: MONITORING")
            self.label_behavior_status.setStyleSheet("background-color: #0ea5e9; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
        elif self.is_testing_ttl:
            self.label_behavior_status.setText("Behavior: ARMED")
            self.label_behavior_status.setStyleSheet("background-color: #2563eb; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
        else:
            self.label_behavior_status.setText("Behavior: IDLE")
            self.label_behavior_status.setStyleSheet("background-color: #374151; color: white; padding: 10px; font-weight: bold; font-size: 16px;")

        pulse_counts = states.get("pulse_counts", {})
        for key, state_label in self.ttl_state_labels.items():
            state_key = self._state_key_for_display(key)
            state = bool(states.get(state_key, False))
            if state_label:
                state_label.setText("HIGH" if state else "LOW")
                state_label.setStyleSheet("color: #22c55e; font-weight: bold;" if state else "color: #9ca3af;")

            count_label = self.ttl_count_labels.get(key)
            if not count_label:
                continue
            count_key = self._ttl_count_key_for_signal(key)
            if states.get("passive_mode"):
                if key == "barcode":
                    count_value = max(int(pulse_counts.get("barcode0", 0)), int(pulse_counts.get("barcode1", 0)))
                else:
                    count_value = pulse_counts.get(state_key, 0)
            elif count_key in states:
                count_value = states.get(count_key, 0)
            else:
                if key == "barcode":
                    count_value = max(int(pulse_counts.get("barcode0", 0)), int(pulse_counts.get("barcode1", 0)))
                else:
                    count_value = pulse_counts.get(state_key, 0)
            count_label.setText(str(int(count_value)))

    @Slot()
    def _on_test_ttl_clicked(self):
        """Handle test TTL button click."""
        if not self.is_testing_ttl:
            # Start test
            self._apply_behavior_pin_configuration(persist=True)
            if self.arduino_worker.start_test():
                self.is_testing_ttl = True
                self.btn_test_ttl.setText("Stop Test / Monitor")
                self.btn_test_ttl.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
                self.label_ttl_status.setText("TTL: TESTING")
                self.label_ttl_status.setStyleSheet("background-color: blue; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
                self.label_behavior_status.setText("Behavior: ARMED")
                self.label_behavior_status.setStyleSheet("background-color: #2563eb; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
                self._reset_ttl_plot()
            else:
                self._on_error_occurred("Failed to start test/monitor mode on Arduino.")
        else:
            # Stop test
            self.arduino_worker.stop_test()
            self.is_testing_ttl = False
            self.btn_test_ttl.setText("Test TTL / Behavior")
            self.btn_test_ttl.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
            self.label_ttl_status.setText("TTL: IDLE")
            self.label_ttl_status.setStyleSheet("background-color: gray; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
            self.label_behavior_status.setText("Behavior: IDLE")
            self.label_behavior_status.setStyleSheet("background-color: #374151; color: white; padding: 10px; font-weight: bold; font-size: 16px;")

    @Slot(dict)
    def _on_frame_recorded(self, frame_metadata: dict):
        """
        Called for each recorded camera frame.
        Samples TTL state synchronized with camera acquisition.
        """
        if self.is_arduino_connected and self.arduino_worker.is_generating:
            # Sample TTL state for this frame
            self.arduino_worker.sample_ttl_state(frame_metadata)

    def _augment_ttl_state_columns(self, df):
        """Add aggregate TTL/behavior state columns for CSV readability."""
        import pandas as pd

        if df is None or df.empty:
            return df

        df = df.copy()
        mapping = {
            "gate": ["gate_ttl", "gate"],
            "sync": ["sync_1hz_ttl", "sync"],
            "barcode0": ["barcode_pin0_ttl", "barcode0"],
            "barcode1": ["barcode_pin1_ttl", "barcode1"],
            "lever": ["lever_ttl", "lever"],
            "cue": ["cue_ttl", "cue"],
            "reward": ["reward_ttl", "reward"],
            "iti": ["iti_ttl", "iti"],
        }

        resolved = {}
        for key, candidates in mapping.items():
            series = None
            for col in candidates:
                if col in df.columns:
                    series = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
                    break
            if series is None:
                series = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
            resolved[key] = series
            col_name = f"{key}_state"
            if col_name not in df.columns:
                df[col_name] = series

        barcode_state = (resolved["barcode0"] | resolved["barcode1"]).astype(int)
        ttl_active = (resolved["gate"] | resolved["sync"] | barcode_state).astype(int)
        behavior_active = (resolved["lever"] | resolved["cue"] | resolved["reward"] | resolved["iti"]).astype(int)

        df["ttl_state"] = np.where(ttl_active > 0, "HIGH", "LOW")
        df["behavior_state"] = np.where(behavior_active > 0, "ACTIVE", "IDLE")
        df["ttl_state_vector"] = (
            "gate=" + resolved["gate"].astype(str)
            + "|sync=" + resolved["sync"].astype(str)
            + "|barcode=" + barcode_state.astype(str)
        )
        df["behavior_state_vector"] = (
            "lever=" + resolved["lever"].astype(str)
            + "|cue=" + resolved["cue"].astype(str)
            + "|reward=" + resolved["reward"].astype(str)
            + "|iti=" + resolved["iti"].astype(str)
        )

        return df

    def _build_behavior_summary_df(self, source_df, ttl_counts: Dict) -> "pd.DataFrame":
        """Build behavior summary (counts and cumulative HIGH durations)."""
        import pandas as pd

        signals = ["lever", "cue", "reward", "iti"]
        rows = []

        if source_df is None or source_df.empty or "timestamp_software" not in source_df.columns:
            for signal in signals:
                rows.append(
                    {
                        "signal": signal,
                        "count": int(ttl_counts.get(signal, 0)),
                        "cumulative_duration_s": 0.0,
                        "duty_cycle_pct": 0.0,
                        "last_state": "LOW",
                    }
                )
            return pd.DataFrame(rows)

        df = source_df.copy()
        t = pd.to_numeric(df["timestamp_software"], errors="coerce")
        t = t.fillna(method="ffill").fillna(method="bfill")
        if t.empty:
            t = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        if len(t) > 1:
            dt = (t.shift(-1) - t).fillna(0.0)
            dt = dt.where(dt > 0.0, 0.0)
            total_duration = float(max(t.iloc[-1] - t.iloc[0], 0.0))
        else:
            dt = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
            total_duration = 0.0

        for signal in signals:
            if signal in df.columns:
                state = pd.to_numeric(df[signal], errors="coerce").fillna(0).astype(int).clip(0, 1)
            elif f"{signal}_ttl" in df.columns:
                state = pd.to_numeric(df[f"{signal}_ttl"], errors="coerce").fillna(0).astype(int).clip(0, 1)
            else:
                state = pd.Series(np.zeros(len(df), dtype=int), index=df.index)

            rises = int(((state == 1) & (state.shift(1, fill_value=0) == 0)).sum())
            duration_high = float((dt * state).sum())
            duty_cycle = (100.0 * duration_high / total_duration) if total_duration > 0 else 0.0
            count_value = int(ttl_counts.get(signal, rises))

            rows.append(
                {
                    "signal": signal,
                    "count": count_value,
                    "cumulative_duration_s": round(duration_high, 4),
                    "duty_cycle_pct": round(duty_cycle, 2),
                    "last_state": "HIGH" if int(state.iloc[-1]) == 1 else "LOW",
                }
            )

        return pd.DataFrame(rows)

    def _save_arduino_ttl_data(self, filepath: str):
        """
        Export board/behavior telemetry to CSV files.

        Files produced (when data is available):
        - *_ttl_states.csv: frame-synced samples (during recording)
        - *_ttl_live_states.csv: live monitor samples from worker thread
        - *_ttl_events.csv: edge events detected by worker
        - *_ttl_counts.csv: final pulse counters
        - *_behavior_summary.csv: count + cumulative HIGH duration summary
        """
        if self.is_arduino_connected:
            import pandas as pd

            ttl_counts = self.arduino_worker.get_ttl_pulse_counts() or {}
            df_history = None
            df_live = None

            ttl_history = self.arduino_worker.get_ttl_history()
            if ttl_history:
                df_history = pd.DataFrame(ttl_history)
                df_history = self._apply_line_label_suffixes(df_history)
                df_history = self._augment_ttl_state_columns(df_history)
                csv_file = filepath + "_ttl_states.csv"
                df_history.to_csv(csv_file, index=False)
                self._on_status_update(f"TTL states saved: {csv_file}")

            live_history = self.arduino_worker.get_live_state_history()
            if live_history:
                df_live = pd.DataFrame(live_history)
                df_live = self._augment_ttl_state_columns(df_live)
                csv_file = filepath + "_ttl_live_states.csv"
                df_live.to_csv(csv_file, index=False)
                self._on_status_update(f"TTL live states saved: {csv_file}")

            ttl_events = self.arduino_worker.get_ttl_event_history()
            if ttl_events:
                df = pd.DataFrame(ttl_events)
                csv_file = filepath + "_ttl_events.csv"
                df.to_csv(csv_file, index=False)
                self._on_status_update(f"TTL events saved: {csv_file}")

            if ttl_counts:
                df = pd.DataFrame([ttl_counts])
                csv_file = filepath + "_ttl_counts.csv"
                df.to_csv(csv_file, index=False)
                self._on_status_update(f"TTL counts saved: {csv_file}")

            summary_source = df_live if df_live is not None else df_history
            if summary_source is not None:
                behavior_summary = self._build_behavior_summary_df(summary_source, ttl_counts)
                csv_file = filepath + "_behavior_summary.csv"
                behavior_summary.to_csv(csv_file, index=False)
                self._on_status_update(f"Behavior summary saved: {csv_file}")

            # Clear history
            self.arduino_worker.clear_ttl_history()
            self.arduino_worker.clear_ttl_event_history()

    # ===== Display Slots =====

    @Slot(np.ndarray)
    def _on_frame_ready(self, frame: np.ndarray):
        """
        Update video display with new frame.
        Converts numpy array to QPixmap and scales to fit display.
        """
        try:
            height, width = frame.shape[:2]
            self.last_frame_size = (width, height)

            # Convert numpy array to QImage
            if len(frame.shape) == 2:  # Grayscale
                bytes_per_line = width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:  # Color (if needed)
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Convert to pixmap and scale to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            if self.roi_rect or self.roi_preview:
                overlay = QPixmap(scaled_pixmap)
                painter = QPainter(overlay)
                pen = QPen(QColor(245, 158, 11), 2)
                painter.setPen(pen)
                roi = self.roi_preview or self.roi_rect
                rect = self._roi_to_display_rect(roi, overlay)
                painter.drawRect(rect)
                painter.end()
                scaled_pixmap = overlay

            self.video_label.setPixmap(scaled_pixmap)

        except Exception as e:
            self._on_error_occurred(f"Display error: {str(e)}")

    @Slot(str)
    def _on_status_update(self, message: str):
        """Update status bar with message."""
        self.status_bar.showMessage(message, 5000)  # Show for 5 seconds

    @Slot(float)
    def _on_fps_update(self, fps: float):
        """Update FPS display."""
        self.label_fps.setText(f"FPS: {fps:.1f}")

    @Slot(int)
    def _on_buffer_update(self, buffer_percent: int):
        """Update buffer usage display."""
        self.label_buffer.setText(f"Buffer: {buffer_percent}%")

        # Change color based on buffer usage
        if buffer_percent > 80:
            self.label_buffer.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        elif buffer_percent > 50:
            self.label_buffer.setStyleSheet("QLabel { color: orange; font-weight: bold; }")
        else:
            self.label_buffer.setStyleSheet("")

    @Slot(str)
    def _on_error_occurred(self, error_message: str):
        """Handle error messages."""
        self.status_bar.showMessage(f"ERROR: {error_message}", 10000)
        print(f"Error: {error_message}")

    def closeEvent(self, event):
        """Handle window close event - cleanup resources."""
        if self.is_camera_connected:
            self._disconnect_camera()

        if self.is_arduino_connected:
            self._stop_arduino_generation()
            self.arduino_worker.stop()
            self.arduino_worker.wait()

        event.accept()

    def _reset_ttl_plot(self):
        """Reset TTL plot data and time base."""
        for data in self.ttl_plot_data.values():
            data.clear()
        self.time_data.clear()
        self.plot_start_time = datetime.now()
        self.ttl_plot.setXRange(0, self.ttl_window_seconds)
        self.behavior_plot.setXRange(0, self.ttl_window_seconds)
        for label in self.ttl_state_labels.values():
            label.setText("LOW")
            label.setStyleSheet("color: #9ca3af;")
        for label in self.ttl_count_labels.values():
            label.setText("0")

    def _stop_arduino_generation(self):
        """Ensure Arduino TTL generation is stopped and UI updated."""
        if self.is_testing_ttl:
            self.arduino_worker.stop_test()
            self.is_testing_ttl = False
            self.btn_test_ttl.setText("Test TTL / Behavior")
            self.btn_test_ttl.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")

        self.arduino_worker.stop_recording()
        self.label_ttl_status.setText("TTL: IDLE")
        self.label_ttl_status.setStyleSheet("background-color: gray; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
        self.label_behavior_status.setText("Behavior: IDLE")
        self.label_behavior_status.setStyleSheet("background-color: #374151; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
