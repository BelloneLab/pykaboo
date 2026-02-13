"""
Enhanced Main GUI Window
PySide6-based interface for Basler camera control with Arduino integration.
"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QLineEdit,
                               QStatusBar, QGroupBox, QSpinBox, QDoubleSpinBox,
                               QFileDialog, QScrollArea, QFormLayout, QTextEdit,
                               QSplitter, QFrame, QSlider)
from PySide6.QtCore import Qt, Slot, QTimer, QSettings, QEvent, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import numpy as np
from datetime import datetime
import pyqtgraph as pg
from collections import deque
import json
from pathlib import Path
from pypylon import pylon
import cv2
from typing import Optional
from camera_worker import CameraWorker
from arduino_output import ArduinoOutputWorker


class MainWindow(QMainWindow):
    """
    Enhanced main application window with camera control, Arduino integration,
    and comprehensive settings.
    """

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
        self.gate_data = deque(maxlen=self.ttl_max_points)
        self.sync_data = deque(maxlen=self.ttl_max_points)
        self.barcode0_data = deque(maxlen=self.ttl_max_points)
        self.time_data = deque(maxlen=self.ttl_max_points)
        self.plot_start_time = datetime.now()

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
                font-family: "Segoe UI";
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

        # === LEFT PANEL: Metadata ===
        left_panel = self._create_metadata_panel()
        splitter.addWidget(left_panel)

        # === CENTER PANEL: Video and Controls ===
        center_panel = self._create_center_panel()
        splitter.addWidget(center_panel)

        # === RIGHT PANEL: Arduino and TTL Plot ===
        right_panel = self._create_arduino_panel()
        splitter.addWidget(right_panel)

        # Set initial sizes
        splitter.setSizes([300, 900, 400])

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
        """Create right panel for Arduino controls and TTL plot."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Arduino TTL Generator")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Arduino connection
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

        # Connect button
        self.btn_arduino_connect = QPushButton("Connect Arduino")
        self.btn_arduino_connect.clicked.connect(self._on_arduino_connect_clicked)
        arduino_layout.addWidget(self.btn_arduino_connect)

        # Status
        self.label_arduino_status = QLabel("Status: Disconnected")
        self.label_arduino_status.setStyleSheet("color: gray;")
        arduino_layout.addWidget(self.label_arduino_status)

        arduino_group.setLayout(arduino_layout)
        layout.addWidget(arduino_group)

        # Pin Configuration Display
        pin_group = QGroupBox("Pin Configuration")
        pin_layout = QFormLayout()

        self.label_gate_pin = QLabel("Not connected")
        pin_layout.addRow("Gate Pin:", self.label_gate_pin)

        self.label_sync_pin = QLabel("Not connected")
        pin_layout.addRow("1Hz Sync Pin:", self.label_sync_pin)

        self.label_barcode_pins = QLabel("Not connected")
        pin_layout.addRow("Barcode Pins:", self.label_barcode_pins)

        pin_group.setLayout(pin_layout)
        layout.addWidget(pin_group)

        # Camera input line labels (optional)
        line_group = QGroupBox("Camera Input Labels (Optional)")
        line_layout = QFormLayout()
        line_options = ["None", "Gate", "TTL 1Hz", "Barcode"]

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
        layout.addWidget(line_group)

        # Test TTL button
        self.btn_test_ttl = QPushButton("Test TTLs")
        self.btn_test_ttl.clicked.connect(self._on_test_ttl_clicked)
        self.btn_test_ttl.setEnabled(False)
        self.btn_test_ttl.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        self.btn_test_ttl.setMinimumHeight(40)
        layout.addWidget(self.btn_test_ttl)

        # TTL status indicator
        self.label_ttl_status = QLabel("TTL: IDLE")
        self.label_ttl_status.setStyleSheet("background-color: gray; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
        self.label_ttl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_ttl_status)

        # TTL Plot
        plot_group = QGroupBox("TTL Signals (Real-time)")
        plot_layout = QVBoxLayout()

        self.ttl_plot = pg.PlotWidget()
        pg.setConfigOptions(antialias=True)
        self.ttl_plot.setBackground((18, 27, 36))
        self.ttl_plot.setMouseEnabled(x=False, y=False)
        self.ttl_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ttl_plot.setLabel('bottom', 'Time (s)')
        self.ttl_plot.setYRange(-0.6, 3.6)
        self.ttl_plot.setXRange(0, self.ttl_window_seconds)
        self.ttl_plot.setLimits(xMin=0)
        self.ttl_plot.setDownsampling(auto=True, mode='peak')
        self.ttl_plot.addLegend()

        axis_left = self.ttl_plot.getAxis('left')
        axis_left.setTextPen(pg.mkPen('#b9c6d3'))
        axis_left.setPen(pg.mkPen('#6c7a89'))
        axis_left.setTicks([[
            (3, 'Gate'),
            (2, 'Sync 1Hz'),
            (1, 'Barcode'),
        ]])
        axis_bottom = self.ttl_plot.getAxis('bottom')
        axis_bottom.setTextPen(pg.mkPen('#b9c6d3'))
        axis_bottom.setPen(pg.mkPen('#6c7a89'))

        self.gate_curve = self.ttl_plot.plot(pen=pg.mkPen('#22c55e', width=2), name='Gate', stepMode=True)
        self.sync_curve = self.ttl_plot.plot(pen=pg.mkPen('#38bdf8', width=2), name='Sync 1Hz', stepMode=True)
        self.barcode0_curve = self.ttl_plot.plot(pen=pg.mkPen('#f97316', width=2), name='Barcode', stepMode=True)

        plot_layout.addWidget(self.ttl_plot)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group, stretch=1)

        return panel

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
        """Initialize Arduino worker thread."""
        self.arduino_worker = ArduinoOutputWorker()

        # Connect signals
        self.arduino_worker.port_list_updated.connect(self._on_port_list_updated)
        self.arduino_worker.connection_status.connect(self._on_arduino_connection_status)
        self.arduino_worker.ttl_states_updated.connect(self._on_ttl_states_updated)
        self.arduino_worker.pin_config_received.connect(self._on_pin_config_received)
        self.arduino_worker.error_occurred.connect(self._on_error_occurred)

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
        except Exception:
            pass

        # USB cameras
        for index in range(10):
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
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
        if label == "TTL 1Hz":
            return "ttl_1hz"
        if label == "Barcode":
            return "barcode"
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
                if not self.arduino_worker.start_recording():
                    self._on_status_update("Warning: Arduino TTLs failed to start; recording will continue.")
                    self.label_ttl_status.setText("TTL: START FAILED")
                    self.label_ttl_status.setStyleSheet("background-color: #b45309; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
                else:
                    # Reset and clear plot for new recording
                    self._reset_ttl_plot()

                    self.label_ttl_status.setText("TTL: RECORDING")
                    self.label_ttl_status.setStyleSheet("background-color: green; color: white; padding: 10px; font-weight: bold; font-size: 16px;")

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
            f"{base_path}_ttl_events.csv",
            f"{base_path}_ttl_counts.csv",
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
        """Handle Arduino connect/disconnect."""
        if not self.is_arduino_connected:
            port = self.combo_arduino_port.currentText()

            if self.arduino_worker.connect_to_port(port):
                self.is_arduino_connected = True
                self.arduino_worker.start()
                self.btn_arduino_connect.setText("Disconnect Arduino")
                self.btn_arduino_connect.setStyleSheet("QPushButton { background-color: #f44336; }")
                self.btn_test_ttl.setEnabled(True)
        else:
            if self.is_testing_ttl:
                self.arduino_worker.stop_test()
                self.is_testing_ttl = False
                self.btn_test_ttl.setText("Test TTLs")
                self.btn_test_ttl.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")

            self.arduino_worker.stop()
            self.arduino_worker.wait()
            self.is_arduino_connected = False
            self.btn_arduino_connect.setText("Connect Arduino")
            self.btn_arduino_connect.setStyleSheet("")
            self.btn_test_ttl.setEnabled(False)

            # Reset pin labels
            self.label_gate_pin.setText("Not connected")
            self.label_sync_pin.setText("Not connected")
            self.label_barcode_pins.setText("Not connected")
            self.label_ttl_status.setText("TTL: IDLE")
            self.label_ttl_status.setStyleSheet("background-color: gray; color: white; padding: 10px; font-weight: bold; font-size: 16px;")

    @Slot(bool, str)
    def _on_arduino_connection_status(self, connected, message):
        """Handle Arduino connection status."""
        if connected:
            self.label_arduino_status.setText(f"Status: {message}")
            self.label_arduino_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.label_arduino_status.setText(f"Status: {message}")
            self.label_arduino_status.setStyleSheet("color: red;")

    @Slot(dict)
    def _on_pin_config_received(self, config):
        """Handle pin configuration from Arduino."""
        if 'gate' in config:
            gate_pins = config['gate']
            if isinstance(gate_pins, list):
                self.label_gate_pin.setText(f"Pins {', '.join(str(pin) for pin in gate_pins)}")
            else:
                self.label_gate_pin.setText(f"Pin {gate_pins}")
        if 'sync' in config:
            sync_pins = config['sync']
            if isinstance(sync_pins, list):
                self.label_sync_pin.setText(f"Pins {', '.join(str(pin) for pin in sync_pins)}")
            else:
                self.label_sync_pin.setText(f"Pin {sync_pins}")
        if 'barcode' in config:
            pins_str = ", ".join([str(p) for p in config['barcode']])
            self.label_barcode_pins.setText(f"Pins {pins_str}")

    @Slot(dict)
    def _on_ttl_states_updated(self, states):
        """Handle TTL state updates."""
        # Update plot
        current_time = (datetime.now() - self.plot_start_time).total_seconds()
        self.time_data.append(current_time)
        amplitude = 0.35
        gate_level = 3.0
        sync_level = 2.0
        barcode0_level = 1.0
        self.gate_data.append(gate_level + amplitude if states['gate'] else gate_level - amplitude)
        self.sync_data.append(sync_level + amplitude if states['sync'] else sync_level - amplitude)
        self.barcode0_data.append(barcode0_level + amplitude if states['barcode0'] else barcode0_level - amplitude)

        # Update curves
        times = np.fromiter(self.time_data, dtype=float)
        if times.size == 0:
            return
        if times.size == 1:
            step = 0.03
        else:
            step = max(0.01, times[-1] - times[-2])
        times_step = np.append(times, times[-1] + step)

        self.gate_curve.setData(times_step, np.fromiter(self.gate_data, dtype=float))
        self.sync_curve.setData(times_step, np.fromiter(self.sync_data, dtype=float))
        self.barcode0_curve.setData(times_step, np.fromiter(self.barcode0_data, dtype=float))

        end_time = times[-1]
        start_time = max(0.0, end_time - self.ttl_window_seconds)
        self.ttl_plot.setXRange(start_time, end_time)

    @Slot()
    def _on_test_ttl_clicked(self):
        """Handle test TTL button click."""
        if not self.is_testing_ttl:
            # Start test
            if self.arduino_worker.start_test():
                self.is_testing_ttl = True
                self.btn_test_ttl.setText("Stop Test")
                self.btn_test_ttl.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
                self.label_ttl_status.setText("TTL: TESTING")
                self.label_ttl_status.setStyleSheet("background-color: blue; color: white; padding: 10px; font-weight: bold; font-size: 16px;")
                self._reset_ttl_plot()
        else:
            # Stop test
            if self.arduino_worker.stop_test():
                self.is_testing_ttl = False
                self.btn_test_ttl.setText("Test TTLs")
                self.btn_test_ttl.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
                self.label_ttl_status.setText("TTL: IDLE")
                self.label_ttl_status.setStyleSheet("background-color: gray; color: white; padding: 10px; font-weight: bold; font-size: 16px;")

    @Slot(dict)
    def _on_frame_recorded(self, frame_metadata: dict):
        """
        Called for each recorded camera frame.
        Samples TTL state synchronized with camera acquisition.
        """
        if self.is_arduino_connected and self.arduino_worker.is_generating:
            # Sample TTL state for this frame
            self.arduino_worker.sample_ttl_state(frame_metadata)

    def _save_arduino_ttl_data(self, filepath: str):
        """Save Arduino TTL history to CSV."""
        if self.is_arduino_connected:
            import pandas as pd

            ttl_history = self.arduino_worker.get_ttl_history()
            if ttl_history:
                df = pd.DataFrame(ttl_history)
                df = self._apply_line_label_suffixes(df)
                csv_file = filepath + "_ttl_states.csv"
                df.to_csv(csv_file, index=False)
                self._on_status_update(f"TTL states saved: {csv_file}")

            ttl_events = self.arduino_worker.get_ttl_event_history()
            if ttl_events:
                df = pd.DataFrame(ttl_events)
                csv_file = filepath + "_ttl_events.csv"
                df.to_csv(csv_file, index=False)
                self._on_status_update(f"TTL events saved: {csv_file}")

            ttl_counts = self.arduino_worker.get_ttl_pulse_counts()
            if ttl_counts:
                df = pd.DataFrame([ttl_counts])
                csv_file = filepath + "_ttl_counts.csv"
                df.to_csv(csv_file, index=False)
                self._on_status_update(f"TTL counts saved: {csv_file}")

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
        self.gate_data.clear()
        self.sync_data.clear()
        self.barcode0_data.clear()
        self.time_data.clear()
        self.plot_start_time = datetime.now()
        self.ttl_plot.setXRange(0, self.ttl_window_seconds)

    def _stop_arduino_generation(self):
        """Ensure Arduino TTL generation is stopped and UI updated."""
        if self.is_testing_ttl:
            self.arduino_worker.stop_test()
            self.is_testing_ttl = False
            self.btn_test_ttl.setText("Test TTLs")
            self.btn_test_ttl.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")

        self.arduino_worker.stop_recording()
        self.label_ttl_status.setText("TTL: IDLE")
        self.label_ttl_status.setStyleSheet("background-color: gray; color: white; padding: 10px; font-weight: bold; font-size: 16px;")


