"""
Enhanced Main GUI Window
PySide6-based interface for Basler, FLIR, and USB camera control with Arduino integration.
"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QLineEdit,
                               QStatusBar, QGroupBox, QSpinBox, QDoubleSpinBox,
                               QFileDialog, QScrollArea, QFormLayout, QTextEdit,
                               QFrame, QSlider, QGridLayout,
                               QCheckBox, QToolButton, QDialog, QStackedWidget,
                               QDialogButtonBox, QStyle, QToolBar, QToolTip,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QAbstractItemView, QMessageBox, QSizePolicy)
from PySide6.QtCore import Qt, Slot, QTimer, QSettings, QSize, QPointF, QRectF, QEvent
from PySide6.QtGui import (QAction, QIcon, QPixmap, QPainter, QColor, QPen,
                           QBrush, QPainterPath, QLinearGradient, QShortcut,
                           QKeySequence)
import numpy as np
from datetime import datetime
import pyqtgraph as pg
from collections import deque
import json
from pathlib import Path
import os
import re
import cv2
from typing import Optional, Dict, List
from camera_backends import (
    discover_basler_cameras,
    discover_flir_cameras,
    discover_usb_cameras,
    get_camera_backend_diagnostics,
)
from camera_worker import CameraWorker
from arduino_output import ArduinoOutputWorker
from live_detection_logic import LiveRuleEngine
from live_detection_panel import LiveDetectionPanel
from live_detection_types import BehaviorROI, LiveDetectionResult, LiveTriggerRule, PreviewFramePacket
from live_inference_worker import LiveInferenceConfig, LiveInferenceWorker


class HoverLabelToolButton(QToolButton):
    """Compact icon rail button with an instant hover label."""

    def __init__(self, hover_label: str, tooltip_alignment: str = "right", parent=None):
        super().__init__(parent)
        self.hover_label = hover_label
        self.tooltip_alignment = tooltip_alignment

    def enterEvent(self, event):
        super().enterEvent(event)
        if not self.hover_label:
            return
        anchor = self.mapToGlobal(self.rect().center())
        offset_x = 26 if self.tooltip_alignment != "left" else -26
        QToolTip.showText(anchor + QPointF(offset_x, 0).toPoint(), self.hover_label, self, self.rect())

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)


class MainWindow(QMainWindow):
    """
    Enhanced main application window with camera control, Arduino integration,
    and comprehensive settings.
    """

    DISPLAY_SIGNAL_ORDER = ["gate", "sync", "barcode", "lever", "cue", "reward", "iti"]
    DISPLAY_SIGNAL_META = {
        "gate": {"state_key": "gate", "group": "ttl", "name": "Gate", "role": "Output", "default_pins": [3], "color": "#22c55e"},
        "sync": {"state_key": "sync", "group": "ttl", "name": "Sync", "role": "Output", "default_pins": [9], "color": "#38bdf8"},
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
        self.settings = QSettings("CamApp Live Detection", "CamApp Live Detection")
        self._migrate_legacy_settings()
        self.last_save_folder = self.settings.value('last_save_folder', '.') or '.'
        self.default_fps = float(self.settings.value('camera_fps', 30.0))
        self.default_width = int(self.settings.value('camera_width', 1080))
        self.default_height = int(self.settings.value('camera_height', 1080))
        self.default_image_format = self.settings.value('image_format', 'Mono8')
        self.frame_drop_monitor_visible = str(self.settings.value("frame_drop_monitor_visible", 1)).strip().lower() not in (
            "0", "false", "no", "off"
        )
        self.signal_display_config = self._load_signal_display_config()

        # Recording state
        self.recording_start_time = None
        self.current_recording_filepath = None
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self._update_recording_time)
        self.space_record_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_record_shortcut.setContext(Qt.WindowShortcut)
        self.space_record_shortcut.activated.connect(self._on_space_record_shortcut)

        # ROI state
        self.roi_rect = None
        self.roi_draw_mode = False
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
        self.behavior_state_labels: Dict[str, QLabel] = {}
        self.behavior_count_labels: Dict[str, QLabel] = {}
        self.ttl_counts_layout: Optional[QGridLayout] = None
        self.behavior_counts_layout: Optional[QGridLayout] = None
        self.pin_value_labels: Dict[str, QLabel] = {}
        self.pin_name_labels: Dict[str, QLabel] = {}
        self.behavior_pin_edits: Dict[str, QLineEdit] = {}
        self.behavior_role_boxes: Dict[str, QComboBox] = {}
        self.signal_label_edits: Dict[str, QLineEdit] = {}
        self.signal_enabled_checks: Dict[str, QCheckBox] = {}
        self.sync_param_button: Optional[QToolButton] = None
        self.barcode_param_button: Optional[QToolButton] = None
        self.live_image_view: Optional[pg.ImageView] = None
        self.live_header_status: Optional[QLabel] = None
        self.live_header_resolution: Optional[QLabel] = None
        self.live_header_mode: Optional[QLabel] = None
        self.live_header_roi: Optional[QLabel] = None
        self.live_status_badge: Optional[QLabel] = None
        self.active_camera_pixel_format_node = ""
        self.active_camera_bit_depth_node = ""
        self.label_ttl_status: Optional[QLabel] = None
        self.label_behavior_status: Optional[QLabel] = None
        self.label_frame_drop_summary: Optional[QLabel] = None
        self.frame_drop_log: Optional[QTextEdit] = None
        self.frame_drop_panel: Optional[QWidget] = None
        self.btn_toggle_frame_drop_panel: Optional[QPushButton] = None
        self.live_placeholder_auto_ranged = False
        self.live_frame_auto_ranged = False
        self.roi_item: Optional[pg.RectROI] = None
        self.live_detection_panel: Optional[LiveDetectionPanel] = None
        self.live_inference_worker: Optional[LiveInferenceWorker] = None
        self.live_detection_enabled = False
        self.live_detection_last_result: Optional[LiveDetectionResult] = None
        self.live_preview_scene = None
        self.live_preview_packet: Optional[PreviewFramePacket] = None
        self.live_rule_engine = LiveRuleEngine()
        self.live_rois: Dict[str, BehaviorROI] = {}
        self.live_rules: List[LiveTriggerRule] = []
        self.live_output_mapping: Dict[str, List[int]] = {f"DO{i}": [] for i in range(1, 9)}
        self.live_active_rule_ids: List[str] = []
        self.live_output_states: Dict[str, bool] = {f"DO{i}": False for i in range(1, 9)}
        self.live_roi_draw_mode = ""
        self.live_roi_draw_points: List[tuple[float, float]] = []
        self.live_roi_circle_center: Optional[tuple[float, float]] = None
        self.live_roi_drawing_name = ""
        self.frame_drop_events = deque(maxlen=4)
        self.last_frame_drop_stats: Dict[str, object] = {}
        self.last_frame_drop_log_signature = None
        self.metadata_dock: Optional[QWidget] = None
        self.camera_dock: Optional[QWidget] = None
        self.session_dock: Optional[QWidget] = None
        self.behavior_setup_dock: Optional[QWidget] = None
        self.monitor_dock: Optional[QWidget] = None
        self.planner_dock: Optional[QWidget] = None
        self.dock_area: Optional[QWidget] = None
        self.workspace_toolbar: Optional[QToolBar] = None
        self.left_panel_shell: Optional[QFrame] = None
        self.right_panel_shell: Optional[QFrame] = None
        self.acquisition_workspace_card: Optional[QFrame] = None
        self.recording_workspace_card: Optional[QFrame] = None
        self.workspace_controls_content: Optional[QWidget] = None
        self.left_panel_stack: Optional[QStackedWidget] = None
        self.right_panel_stack: Optional[QStackedWidget] = None
        self.left_panel_title: Optional[QLabel] = None
        self.right_panel_title: Optional[QLabel] = None
        self.left_nav_buttons: Dict[str, QToolButton] = {}
        self.right_nav_buttons: Dict[str, QToolButton] = {}
        self.left_panel_pages: Dict[str, QWidget] = {}
        self.right_panel_pages: Dict[str, QWidget] = {}
        self.current_left_panel_key: Optional[str] = None
        self.current_right_panel_key: Optional[str] = None
        self.btn_toggle_acquisition_panel: Optional[QPushButton] = None
        self.btn_toggle_recording_panel: Optional[QPushButton] = None
        self.btn_record: Optional[QPushButton] = None
        self.planner_table: Optional[QTableWidget] = None
        self.planner_default_columns = [
            "Status",
            "Trial",
            "Arena",
            "Animal ID",
            "Experiment",
            "Condition",
            "Start Delay (s)",
            "Duration (s)",
            "Comments",
        ]
        self.planner_custom_columns: List[str] = []
        self.planner_next_trial_number = 1
        self.active_planner_row: Optional[int] = None
        self.planner_panel_widget: Optional[QWidget] = None
        self.planner_host_layout = None
        self.planner_dialog: Optional[QDialog] = None
        self.planner_detached = False
        self.planner_reattaching = False
        self.advanced_dialog: Optional[QDialog] = None
        self.filename_order_boxes: List[QComboBox] = []
        self._filename_field_syncing = False
        self._custom_filename_override = str(self.settings.value("recording_filename_override", "") or "").strip()
        self.meta_trial: Optional[QLineEdit] = None
        self.meta_condition: Optional[QLineEdit] = None
        self.meta_arena: Optional[QLineEdit] = None
        self.label_session_summary: Optional[QLabel] = None
        self.label_session_details: Optional[QLabel] = None
        self.label_session_total_count: Optional[QLabel] = None
        self.label_session_pending_count: Optional[QLabel] = None
        self.label_session_acquiring_count: Optional[QLabel] = None
        self.label_session_acquired_count: Optional[QLabel] = None
        self.label_recording_plan_summary: Optional[QLabel] = None
        self.label_recording_plan_details: Optional[QLabel] = None

        # Metadata
        self.metadata = {}

        self.setWindowTitle("CamApp Live Detection")
        self.setGeometry(50, 50, 1600, 900)
        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

        # Workspace theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #070d15;
                color: #eef6ff;
            }
            QWidget {
                background-color: transparent;
                color: #eef6ff;
                font-family: "Arial Narrow", Arial, "Segoe UI";
                font-size: 11px;
            }
            QWidget#AppShell {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #060b13, stop:0.45 #09131e, stop:1 #050912);
            }
            QFrame#SideRail {
                background-color: #07111c;
                border: 1px solid #1a2a40;
                border-radius: 26px;
            }
            QFrame#PanelShell {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a1420, stop:1 #0d1725);
                border: 1px solid #203149;
                border-radius: 28px;
            }
            QFrame#PanelHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #101a29, stop:1 #122031);
                border: 1px solid #253852;
                border-radius: 20px;
            }
            QFrame#WorkspaceCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0b1521, stop:1 #0d1724);
                border: 1px solid #22364f;
                border-radius: 24px;
            }
            QFrame#WorkspaceSubCard {
                background-color: #0e1825;
                border: 1px solid #28405d;
                border-radius: 22px;
            }
            QGroupBox {
                border: 1px solid #243952;
                border-radius: 18px;
                margin-top: 16px;
                padding-top: 12px;
                background-color: #0d1723;
                font-weight: 600;
                color: #e7f2ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                background-color: #0d1723;
                color: #8dd0ff;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2488ff, stop:1 #2563eb);
                color: white;
                border: 1px solid #5aa7ff;
                border-radius: 15px;
                padding: 5px 12px;
                font-weight: 700;
                font-size: 12px;
                min-height: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #33b1ff, stop:1 #3b82f6);
            }
            QPushButton:disabled {
                background: #1a2637;
                color: #63778f;
                border-color: #26384f;
            }
            QPushButton#ghostButton {
                background: #101b29;
                border: 1px solid #33506f;
                color: #d8e8fa;
            }
            QPushButton#ghostButton:hover {
                background: #152436;
            }
            QPushButton#toggleButton {
                background: #101b29;
                border: 1px solid #33506f;
                color: #9fd9ff;
                border-radius: 14px;
                padding: 5px 12px;
            }
            QPushButton#toggleButton:hover {
                background: #152436;
                border-color: #46739f;
                color: #eef6ff;
            }
            QPushButton#toggleButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #123456, stop:1 #205b85);
                border: 1px solid #71c2ff;
                color: #eef8ff;
            }
            QPushButton#successButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3dbb67, stop:1 #69cf4a);
                border: 1px solid #8bf28a;
                color: #06110a;
            }
            QPushButton#dangerButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff5b70, stop:1 #ff3f98);
                border: 1px solid #ff8fb2;
            }
            QPushButton#violetButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7a52ff, stop:1 #cf4cff);
                border: 1px solid #e296ff;
            }
            QPushButton#orangeButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff9547, stop:1 #ff5f45);
                border: 1px solid #ffc38d;
            }
            QToolButton#navButton {
                background: #0d1725;
                border: 1px solid #22344e;
                border-radius: 18px;
                padding: 8px;
                min-width: 44px;
                max-width: 44px;
                min-height: 44px;
                max-height: 44px;
                color: #9bb4d2;
                font-weight: 700;
                font-size: 11px;
            }
            QToolButton#navButton:hover {
                background: #122033;
                border-color: #39577c;
                color: #eef6ff;
            }
            QToolButton#navButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #122943, stop:1 #163a63);
                border: 1px solid #66b7ff;
                color: #ffffff;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QTableWidget {
                background-color: #07101a;
                border: 1px solid #243851;
                border-radius: 12px;
                color: #eef6ff;
                padding: 5px 8px;
                min-height: 20px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #07101a;
                color: #eef6ff;
                border: 1px solid #243851;
                selection-background-color: #173150;
                selection-color: #ffffff;
                outline: 0;
            }
            QComboBox QAbstractItemView::item {
                min-height: 24px;
                padding: 6px 10px;
                color: #eef6ff;
                background: transparent;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #173150;
                color: #ffffff;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 5px;
                border: 1px solid #33506f;
                background-color: #07101a;
            }
            QCheckBox::indicator:checked {
                background-color: #2488ff;
                border-color: #5aa7ff;
            }
            QLabel {
                color: #dce8f4;
            }
            QStatusBar {
                background-color: #08101a;
                color: #dbe7f3;
                border-top: 1px solid #18283a;
            }
            QScrollBar:vertical {
                background-color: #0b1522;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background-color: #27415c;
                min-height: 20px;
                border-radius: 5px;
            }
            QHeaderView::section {
                background-color: #101c2b;
                color: #9dd9ff;
                border: none;
                border-right: 1px solid #203246;
                border-bottom: 1px solid #203246;
                padding: 6px;
                font-weight: 600;
            }
            QTableWidget {
                gridline-color: #1c3045;
                alternate-background-color: #0a1420;
                selection-background-color: #123a66;
                selection-color: #ffffff;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)

        self._init_ui()
        self._load_ui_settings()
        self._setup_worker()
        self._setup_arduino_worker()
        self._load_live_detection_settings()
        self._load_metadata()
        self._scan_cameras()

    def _migrate_legacy_settings(self):
        """Copy saved settings from the previous app identity once."""
        legacy_settings = QSettings("BaslerCam", "CameraApp")
        existing_keys = set(self.settings.allKeys())
        copied = False

        for key in legacy_settings.allKeys():
            if key in existing_keys:
                continue
            self.settings.setValue(key, legacy_settings.value(key))
            copied = True

        if copied:
            self.settings.sync()

    def _init_ui(self):
        """Initialize the tool-rail workspace."""
        self.dock_area = QWidget()
        self.dock_area.setObjectName("AppShell")
        self.setCentralWidget(self.dock_area)

        root_layout = QHBoxLayout(self.dock_area)
        root_layout.setContentsMargins(16, 16, 16, 12)
        root_layout.setSpacing(14)

        left_rail = self._create_nav_rail("left")
        self.left_panel_shell, self.left_panel_title, self.left_panel_stack = self._create_side_panel_shell("Session", "left")
        right_rail = self._create_nav_rail("right")
        self.right_panel_shell, self.right_panel_title, self.right_panel_stack = self._create_side_panel_shell("Arduino", "right")

        camera_page = self._create_camera_connection_panel()
        settings_page = self._create_general_settings_panel()
        session_page = self._wrap_scroll_dock_widget(self._create_session_hub_panel())
        file_page = self._create_file_tools_panel()
        ttl_page = self._create_ttl_monitor_panel()
        behavior_page = self._create_behavior_monitor_panel()
        arduino_page = self._wrap_scroll_dock_widget(self._create_behavior_setup_panel())
        live_detection_page = self._wrap_scroll_dock_widget(self._create_live_detection_panel())
        self._rebuild_monitor_visuals(reset_plot=True)

        self.left_panel_pages = {
            "camera": camera_page,
            "settings": settings_page,
            "session": session_page,
            "file": file_page,
        }
        self.right_panel_pages = {
            "arduino": arduino_page,
            "ttl": ttl_page,
            "behavior": behavior_page,
            "live_detection": live_detection_page,
        }

        for page in self.left_panel_pages.values():
            self.left_panel_stack.addWidget(page)
        for page in self.right_panel_pages.values():
            self.right_panel_stack.addWidget(page)

        center_workspace = self._create_center_workspace()

        root_layout.addWidget(left_rail)
        root_layout.addWidget(self.left_panel_shell)
        root_layout.addWidget(center_workspace, 1)
        root_layout.addWidget(right_rail)
        root_layout.addWidget(self.right_panel_shell)
        self._update_side_panel_bounds()

        self.metadata_dock = self.left_panel_shell
        self.camera_dock = camera_page
        self.session_dock = center_workspace
        self.behavior_setup_dock = arduino_page
        self.monitor_dock = ttl_page
        self.planner_dock = session_page

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

        self.btn_toggle_metadata = QPushButton("Session")
        self.btn_toggle_metadata.setObjectName("ghostButton")
        self.btn_toggle_metadata.clicked.connect(self._toggle_metadata_panel)
        self.btn_toggle_metadata.setMaximumHeight(24)
        self.status_bar.addPermanentWidget(self.btn_toggle_metadata)

    def _create_nav_rail(self, side: str) -> QFrame:
        """Create one vertical navigation rail."""
        rail = QFrame()
        rail.setObjectName("SideRail")
        rail.setFixedWidth(62)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(7, 12, 7, 12)
        layout.setSpacing(10)

        specs = []
        button_store = self.left_nav_buttons if side == "left" else self.right_nav_buttons
        if side == "left":
            specs = [
                ("camera", "Camera", "camera", "#33c8ff", "Camera Connection"),
                ("settings", "Settings", "settings", "#d86cff", "General Settings"),
                ("session", "Session", "session", "#6fe06e", "Metadata and Planner"),
                ("file", "Files", "folder", "#ff9a43", "File Tools"),
            ]
        else:
            specs = [
                ("arduino", "Arduino", "chip", "#8f7cff", "Arduino Setup"),
                ("ttl", "TTL", "pulse", "#3fd5ff", "TTL Monitor"),
                ("behavior", "Behavior", "behavior", "#ff6c9e", "Behavior Monitor"),
                ("live_detection", "Live", "pulse", "#6fe06e", "Live Detection"),
            ]

        for key, label, icon_kind, accent, title in specs:
            button = self._create_nav_button(label, icon_kind, accent, side=side)
            button.clicked.connect(
                lambda checked=False, s=side, panel_key=key, panel_title=title: self._toggle_side_panel(s, panel_key, panel_title)
            )
            layout.addWidget(button)
            button_store[key] = button

        layout.addStretch()
        return rail

    def _create_side_panel_shell(self, title: str, side: str):
        """Create a collapsible side shell with header and stacked pages."""
        shell = QFrame()
        shell.setObjectName("PanelShell")
        shell.setMinimumWidth(320)
        shell.setMaximumWidth(440)
        shell.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        shell.setVisible(False)

        layout = QVBoxLayout(shell)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("PanelHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #eef6ff;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        hide_button = QPushButton("Hide")
        hide_button.setObjectName("ghostButton")
        hide_button.setMaximumWidth(76)
        hide_button.clicked.connect(lambda: self._hide_side_panel(side))
        header_layout.addWidget(hide_button)

        stack = QStackedWidget()
        stack.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(header)
        layout.addWidget(stack, 1)
        return shell, title_label, stack

    def _hide_side_panel(self, side: str):
        """Hide one side shell and clear the active nav state."""
        shell = self.left_panel_shell if side == "left" else self.right_panel_shell
        buttons = self.left_nav_buttons if side == "left" else self.right_nav_buttons
        if shell is not None:
            shell.setVisible(False)
        for button in buttons.values():
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        if side == "left":
            self.current_left_panel_key = None
            self.settings.setValue("metadata_panel_visible", 0)
        else:
            self.current_right_panel_key = None

    def _update_side_panel_bounds(self):
        """Adjust side-drawer widths to remain readable across window sizes."""
        window_width = max(0, self.width())
        if window_width >= 1850:
            left_bounds = (360, 460)
            right_bounds = (360, 460)
        elif window_width >= 1600:
            left_bounds = (330, 420)
            right_bounds = (330, 410)
        else:
            left_bounds = (300, 360)
            right_bounds = (300, 360)

        if self.left_panel_shell is not None:
            self.left_panel_shell.setMinimumWidth(left_bounds[0])
            self.left_panel_shell.setMaximumWidth(left_bounds[1])
        if self.right_panel_shell is not None:
            self.right_panel_shell.setMinimumWidth(right_bounds[0])
            self.right_panel_shell.setMaximumWidth(right_bounds[1])

    def _ensure_side_panel_fit(self, side: str):
        """Keep side panels usable on narrower windows by collapsing the opposite drawer."""
        if self.width() >= 1760:
            return
        other_side = "right" if side == "left" else "left"
        other_shell = self.right_panel_shell if side == "left" else self.left_panel_shell
        if other_shell is not None and other_shell.isVisible():
            self._hide_side_panel(other_side)

    def _create_nav_button(self, label: str, icon_kind: str, accent: str, side: str = "left") -> QToolButton:
        """Create one compact navigation button with hover-only labeling."""
        button = HoverLabelToolButton(label, tooltip_alignment="right" if side == "left" else "left")
        button.setObjectName("navButton")
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        button.setIcon(self._build_modern_icon(icon_kind, accent))
        button.setIconSize(QSize(24, 24))
        button.setText("")
        button.setToolTip(label)
        button.setStatusTip(label)
        button.setAccessibleName(label)
        return button

    def _toggle_side_panel(self, side: str, panel_key: str, title: str):
        """Show or hide a contextual side panel."""
        if side == "left":
            shell = self.left_panel_shell
            stack = self.left_panel_stack
            title_label = self.left_panel_title
            pages = self.left_panel_pages
            buttons = self.left_nav_buttons
            current_key = self.current_left_panel_key
        else:
            shell = self.right_panel_shell
            stack = self.right_panel_stack
            title_label = self.right_panel_title
            pages = self.right_panel_pages
            buttons = self.right_nav_buttons
            current_key = self.current_right_panel_key

        if shell is None or stack is None or title_label is None or panel_key not in pages:
            return

        should_hide = shell.isVisible() and current_key == panel_key
        if should_hide:
            shell.setVisible(False)
            for button in buttons.values():
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
            if side == "left":
                self.current_left_panel_key = None
                self.settings.setValue("metadata_panel_visible", 0)
            else:
                self.current_right_panel_key = None
            return

        self._ensure_side_panel_fit(side)
        shell.setVisible(True)
        stack.setCurrentWidget(pages[panel_key])
        title_label.setText(title)
        for key, button in buttons.items():
            button.blockSignals(True)
            button.setChecked(key == panel_key)
            button.blockSignals(False)

        if side == "left":
            self.current_left_panel_key = panel_key
            self.settings.setValue("metadata_panel_visible", 1)
        else:
            self.current_right_panel_key = panel_key

    def _create_center_workspace(self) -> QWidget:
        """Build the central live-view and recording workspace."""
        container = QWidget()
        container.setMinimumWidth(0)
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        live_card = QFrame()
        live_card.setObjectName("WorkspaceCard")
        live_card.setMinimumWidth(0)
        live_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        live_layout = QVBoxLayout(live_card)
        live_layout.setContentsMargins(12, 12, 12, 12)
        live_layout.addWidget(self._create_live_view_panel(), 1)

        self.acquisition_workspace_card = QFrame()
        self.acquisition_workspace_card.setObjectName("WorkspaceCard")
        self.acquisition_workspace_card.setMinimumWidth(0)
        self.acquisition_workspace_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        acquisition_layout = QVBoxLayout(self.acquisition_workspace_card)
        acquisition_layout.setContentsMargins(14, 14, 14, 14)
        acquisition_layout.addWidget(self._create_camera_settings())

        self.recording_workspace_card = QFrame()
        self.recording_workspace_card.setObjectName("WorkspaceCard")
        self.recording_workspace_card.setMinimumWidth(0)
        self.recording_workspace_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        recording_layout = QVBoxLayout(self.recording_workspace_card)
        recording_layout.setContentsMargins(14, 14, 14, 14)
        recording_layout.addWidget(self._create_control_panel())

        if self.btn_record is None:
            self.btn_record = QPushButton("Start Recording")
            self._set_button_icon(self.btn_record, "record", "#07260e", "successButton")
            self.btn_record.clicked.connect(self._on_record_clicked)
            self.btn_record.setEnabled(False)
            self.btn_record.setMinimumHeight(42)
            self.btn_record.setMinimumWidth(220)

        controls_shell = QFrame()
        controls_shell.setObjectName("WorkspaceCard")
        controls_shell.setMinimumWidth(0)
        controls_shell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        controls_layout = QVBoxLayout(controls_shell)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(10)

        controls_toolbar = QHBoxLayout()
        controls_toolbar.setSpacing(10)

        controls_title = QLabel("Workspace Controls")
        controls_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #eef6ff;")
        controls_toolbar.addWidget(controls_title)

        controls_hint = QLabel("Open the panels only when you need them so preview keeps the space.")
        controls_hint.setStyleSheet("color: #8fa6bf;")
        controls_toolbar.addWidget(controls_hint)
        controls_toolbar.addStretch()

        self.btn_toggle_acquisition_panel = QPushButton("Acquisition")
        self._set_button_icon(self.btn_toggle_acquisition_panel, "settings", "#7cc7ff", "toggleButton")
        self.btn_toggle_acquisition_panel.setCheckable(True)
        self.btn_toggle_acquisition_panel.setChecked(False)
        self.btn_toggle_acquisition_panel.toggled.connect(self._update_workspace_controls_visibility)
        controls_toolbar.addWidget(self.btn_toggle_acquisition_panel)

        self.btn_toggle_recording_panel = QPushButton("Recording")
        self._set_button_icon(self.btn_toggle_recording_panel, "session", "#9bf57f", "toggleButton")
        self.btn_toggle_recording_panel.setCheckable(True)
        self.btn_toggle_recording_panel.setChecked(False)
        self.btn_toggle_recording_panel.toggled.connect(self._update_workspace_controls_visibility)
        controls_toolbar.addWidget(self.btn_toggle_recording_panel)

        controls_toolbar.addSpacing(8)
        controls_toolbar.addWidget(self.btn_record)
        controls_layout.addLayout(controls_toolbar)

        self.workspace_controls_content = QWidget()
        controls_row = QHBoxLayout(self.workspace_controls_content)
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(14)
        controls_row.addWidget(self.acquisition_workspace_card, 1)
        controls_row.addWidget(self.recording_workspace_card, 1)
        controls_row.setStretch(0, 4)
        controls_row.setStretch(1, 5)
        controls_layout.addWidget(self.workspace_controls_content)

        layout.addWidget(live_card, 1)
        layout.addWidget(controls_shell, 0)
        self._update_workspace_controls_visibility()
        return container

    def _update_workspace_controls_visibility(self):
        """Show or hide bottom workspace panels while keeping record controls visible."""
        acquisition_visible = bool(
            self.btn_toggle_acquisition_panel is not None and self.btn_toggle_acquisition_panel.isChecked()
        )
        recording_visible = bool(
            self.btn_toggle_recording_panel is not None and self.btn_toggle_recording_panel.isChecked()
        )

        if self.acquisition_workspace_card is not None:
            self.acquisition_workspace_card.setVisible(acquisition_visible)
        if self.recording_workspace_card is not None:
            self.recording_workspace_card.setVisible(recording_visible)
        if self.workspace_controls_content is not None:
            self.workspace_controls_content.setVisible(acquisition_visible or recording_visible)

    def _create_metric_tile(self, title: str, value: str, accent: str):
        """Create a compact dashboard tile used for planner/session counts."""
        tile = QFrame()
        tile.setObjectName("WorkspaceSubCard")
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #8fa6bf; font-size: 11px; font-weight: 600;")
        value_label = QLabel(value)
        value_label.setStyleSheet(f"color: {accent}; font-size: 20px; font-weight: 800;")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addStretch()
        return tile, value_label

    def _paint_modern_icon(self, painter: QPainter, kind: str, accent: str):
        """Paint one icon glyph into a normalized 32x32 canvas."""
        pen = QPen(QColor(accent), 2.3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        if kind == "camera":
            painter.drawRoundedRect(QRectF(5, 9, 22, 15), 5, 5)
            painter.drawEllipse(QPointF(16, 16.5), 4.5, 4.5)
            painter.drawRect(QRectF(9, 6, 6, 4))
        elif kind == "settings":
            for y, knob_x in ((9, 12), (16, 20), (23, 14)):
                painter.drawLine(6, y, 26, y)
                painter.setBrush(QColor(accent))
                painter.drawEllipse(QPointF(knob_x, y), 2.4, 2.4)
                painter.setBrush(Qt.NoBrush)
        elif kind == "session":
            painter.drawRoundedRect(QRectF(8, 6, 16, 20), 4, 4)
            painter.drawLine(12, 11, 20, 11)
            painter.drawLine(12, 16, 20, 16)
            painter.drawLine(12, 21, 18, 21)
        elif kind == "folder":
            path = QPainterPath()
            path.moveTo(6, 12)
            path.lineTo(11, 12)
            path.lineTo(14, 9)
            path.lineTo(26, 9)
            path.lineTo(24, 24)
            path.lineTo(6, 24)
            path.closeSubpath()
            painter.drawPath(path)
        elif kind == "chip":
            painter.drawRoundedRect(QRectF(9, 9, 14, 14), 3, 3)
            for offset in (9, 14, 19, 24):
                painter.drawLine(offset, 5, offset, 9)
                painter.drawLine(offset, 23, offset, 27)
                painter.drawLine(5, offset, 9, offset)
                painter.drawLine(23, offset, 27, offset)
        elif kind == "pulse":
            path = QPainterPath()
            path.moveTo(5, 18)
            path.lineTo(10, 18)
            path.lineTo(13, 12)
            path.lineTo(17, 23)
            path.lineTo(20, 10)
            path.lineTo(24, 18)
            path.lineTo(27, 18)
            painter.drawPath(path)
        elif kind == "behavior":
            painter.drawEllipse(QPointF(10, 10), 2.6, 2.6)
            painter.drawEllipse(QPointF(22, 16), 2.6, 2.6)
            painter.drawEllipse(QPointF(11, 23), 2.6, 2.6)
            painter.drawLine(12, 12, 20, 15)
            painter.drawLine(12, 21, 20, 17)
            painter.drawLine(10, 13, 10, 20)
        elif kind == "plus":
            painter.drawLine(16, 7, 16, 25)
            painter.drawLine(7, 16, 25, 16)
        elif kind == "import":
            painter.drawLine(8, 24, 24, 24)
            painter.drawLine(16, 8, 16, 21)
            painter.drawLine(11, 16, 16, 21)
            painter.drawLine(21, 16, 16, 21)
        elif kind == "export":
            painter.drawLine(8, 24, 24, 24)
            painter.drawLine(16, 21, 16, 8)
            painter.drawLine(11, 13, 16, 8)
            painter.drawLine(21, 13, 16, 8)
        elif kind == "record":
            painter.setBrush(QColor(accent))
            painter.drawEllipse(QPointF(16, 16), 7, 7)
        elif kind == "play":
            path = QPainterPath()
            path.moveTo(11, 9)
            path.lineTo(24, 16)
            path.lineTo(11, 23)
            path.closeSubpath()
            painter.fillPath(path, QBrush(QColor(accent)))
        elif kind == "check":
            painter.drawLine(8, 17, 14, 23)
            painter.drawLine(14, 23, 24, 9)
        else:
            painter.drawRoundedRect(QRectF(7, 7, 18, 18), 5, 5)

    def _build_modern_icon(self, kind: str, accent: str) -> QIcon:
        """Build a multi-size icon so Qt stays crisp on higher-DPI displays."""
        icon = QIcon()
        for size in (16, 18, 20, 24, 28, 32, 48, 64):
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHints(
                QPainter.Antialiasing
                | QPainter.SmoothPixmapTransform
                | QPainter.TextAntialiasing
            )
            painter.scale(size / 32.0, size / 32.0)
            self._paint_modern_icon(painter, kind, accent)
            painter.end()
            icon.addPixmap(pixmap)
        return icon

    def _wrap_scroll_dock_widget(self, widget: QWidget) -> QWidget:
        """Wrap tall configuration widgets so docks remain usable on smaller screens."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(widget)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        return container

    def _set_button_icon(self, button: QPushButton, kind: str, accent: str, tone: Optional[str] = None):
        """Apply a custom icon and optional tone name to a push button."""
        button.setIcon(self._build_modern_icon(kind, accent))
        button.setIconSize(QSize(18, 18))
        button.setObjectName(tone or "")
        button.style().unpolish(button)
        button.style().polish(button)

    def _create_camera_connection_panel(self) -> QWidget:
        """Create the left-side camera connection page."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        hero = QFrame()
        hero.setObjectName("WorkspaceCard")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Camera Connection")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #eef6ff;")
        subtitle = QLabel("Choose a Basler, FLIR, or USB source, then arm the live workspace.")
        subtitle.setStyleSheet("color: #8fa6bf;")
        subtitle.setWordWrap(True)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)

        form = QFormLayout()
        self.combo_camera = QComboBox()
        form.addRow("Source:", self.combo_camera)
        hero_layout.addLayout(form)

        button_row = QHBoxLayout()
        self.btn_scan_cameras = QPushButton("Refresh")
        self._set_button_icon(self.btn_scan_cameras, "import", "#3fd5ff", "ghostButton")
        self.btn_scan_cameras.clicked.connect(self._scan_cameras)
        button_row.addWidget(self.btn_scan_cameras)

        self.btn_connect = QPushButton("Connect Camera")
        self._set_button_icon(self.btn_connect, "play", "#eef6ff")
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        self.btn_connect.setMinimumHeight(36)
        button_row.addWidget(self.btn_connect, 1)
        hero_layout.addLayout(button_row)

        self.label_camera_source_hint = QLabel("No source connected")
        self.label_camera_source_hint.setStyleSheet("color: #8fa6bf;")
        hero_layout.addWidget(self.label_camera_source_hint)

        layout.addWidget(hero)
        layout.addStretch()
        return panel

    def _create_general_settings_panel(self) -> QWidget:
        """Create the left-side general settings page."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        card = QFrame()
        card.setObjectName("WorkspaceCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(12)

        title = QLabel("General Settings")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #eef6ff;")
        subtitle = QLabel("Control filename assembly and open the advanced camera popup.")
        subtitle.setStyleSheet("color: #8fa6bf;")
        subtitle.setWordWrap(True)
        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)

        order_group = QGroupBox("Filename Order")
        order_layout = QFormLayout(order_group)
        self.filename_order_boxes = []
        for index in range(4):
            combo = QComboBox()
            combo.addItems(self._filename_field_labels())
            combo.currentTextChanged.connect(self._on_filename_order_changed)
            self.filename_order_boxes.append(combo)
            order_layout.addRow(f"Part {index + 1}:", combo)
        card_layout.addWidget(order_group)

        preview_group = QGroupBox("Filename Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.label_filename_formula = QLabel("")
        self.label_filename_formula.setWordWrap(True)
        self.label_filename_formula.setStyleSheet("color: #9fd9ff; font-weight: 600;")
        preview_layout.addWidget(self.label_filename_formula)
        card_layout.addWidget(preview_group)

        behavior_defaults_group = QGroupBox("Behavior / TTL Defaults")
        behavior_defaults_layout = QVBoxLayout(behavior_defaults_group)
        behavior_defaults_hint = QLabel(
            "Define the default board pins, signal labels/roles, and camera line labels from one popup."
        )
        behavior_defaults_hint.setWordWrap(True)
        behavior_defaults_hint.setStyleSheet("color: #8fa6bf;")
        behavior_defaults_layout.addWidget(behavior_defaults_hint)

        self.btn_open_behavior_defaults = QPushButton("Behavior Defaults Menu")
        self._set_button_icon(self.btn_open_behavior_defaults, "settings", "#7cc7ff", "ghostButton")
        self.btn_open_behavior_defaults.clicked.connect(self._open_behavior_defaults_dialog)
        behavior_defaults_layout.addWidget(self.btn_open_behavior_defaults)
        card_layout.addWidget(behavior_defaults_group)

        self.btn_open_advanced_settings = QPushButton("Advanced Camera Menu")
        self._set_button_icon(self.btn_open_advanced_settings, "settings", "#d86cff", "violetButton")
        self.btn_open_advanced_settings.clicked.connect(self._toggle_advanced_settings)
        card_layout.addWidget(self.btn_open_advanced_settings)

        card_layout.addStretch()
        layout.addWidget(card)
        layout.addStretch()
        return panel

    def _create_session_hub_panel(self) -> QWidget:
        """Create the merged metadata and planner workspace page."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        overview = QFrame()
        overview.setObjectName("WorkspaceCard")
        overview_layout = QVBoxLayout(overview)
        overview_layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Session Planner")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #eef6ff;")
        subtitle = QLabel("Planner rows now drive the active recording session and filename.")
        subtitle.setStyleSheet("color: #8fa6bf;")
        subtitle.setWordWrap(True)
        overview_layout.addWidget(title)
        overview_layout.addWidget(subtitle)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(10)
        total_tile, self.label_session_total_count = self._create_metric_tile("Trials", "0", "#eef6ff")
        pending_tile, self.label_session_pending_count = self._create_metric_tile("Pending", "0", "#ffd89c")
        acquiring_tile, self.label_session_acquiring_count = self._create_metric_tile("Active", "0", "#9dd9ff")
        acquired_tile, self.label_session_acquired_count = self._create_metric_tile("Done", "0", "#7ef0ac")
        for tile in (total_tile, pending_tile, acquiring_tile, acquired_tile):
            stats_row.addWidget(tile, 1)
        overview_layout.addLayout(stats_row)

        self.metadata_panel = self._create_metadata_panel()
        self.metadata_panel.hide()

        summary_card = QFrame()
        summary_card.setObjectName("WorkspaceSubCard")
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(14, 14, 14, 14)
        summary_layout.setSpacing(6)

        summary_title = QLabel("Selected Trial")
        summary_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #eef6ff;")
        self.label_session_summary = QLabel("No trial selected")
        self.label_session_summary.setStyleSheet("color: #9fd9ff; font-weight: 700;")
        self.label_session_details = QLabel("Select or edit a planner row, then record directly from that plan.")
        self.label_session_details.setWordWrap(True)
        self.label_session_details.setStyleSheet("color: #8fa6bf;")
        summary_layout.addWidget(summary_title)
        summary_layout.addWidget(self.label_session_summary)
        summary_layout.addWidget(self.label_session_details)
        overview_layout.addWidget(summary_card)

        planner_host = QFrame()
        planner_host.setObjectName("WorkspaceSubCard")
        self.planner_host_layout = QVBoxLayout(planner_host)
        self.planner_host_layout.setContentsMargins(14, 14, 14, 14)
        self.planner_host_layout.setSpacing(0)
        self.planner_panel_widget = self._create_trial_planner_panel()
        self.planner_host_layout.addWidget(self.planner_panel_widget)
        overview_layout.addWidget(planner_host, 1)

        layout.addWidget(overview, 1)
        return panel

    def _create_file_tools_panel(self) -> QWidget:
        """Create the file/session storage utility page."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        card = QFrame()
        card.setObjectName("WorkspaceCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(12)

        title = QLabel("File Tools")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #eef6ff;")
        subtitle = QLabel("Choose the save root and export reusable session presets.")
        subtitle.setStyleSheet("color: #8fa6bf;")
        subtitle.setWordWrap(True)
        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)

        save_group = QGroupBox("Recording Folder")
        save_layout = QVBoxLayout(save_group)
        self.label_file_save_folder = QLabel(self.last_save_folder)
        self.label_file_save_folder.setWordWrap(True)
        self.label_file_save_folder.setStyleSheet("color: #9fd9ff;")
        save_layout.addWidget(self.label_file_save_folder)
        btn_browse_root = QPushButton("Browse Save Folder")
        self._set_button_icon(btn_browse_root, "folder", "#ff9a43", "orangeButton")
        btn_browse_root.clicked.connect(self._browse_save_folder)
        save_layout.addWidget(btn_browse_root)
        card_layout.addWidget(save_group)

        actions_group = QGroupBox("Templates")
        actions_layout = QVBoxLayout(actions_group)
        btn_save_meta = QPushButton("Save Metadata Template")
        self._set_button_icon(btn_save_meta, "check", "#33d5ff", "ghostButton")
        btn_save_meta.clicked.connect(self._save_metadata_template)
        actions_layout.addWidget(btn_save_meta)
        card_layout.addWidget(actions_group)

        card_layout.addStretch()
        layout.addWidget(card)
        layout.addStretch()
        return panel

    def _create_ttl_monitor_panel(self) -> QWidget:
        """Create the dedicated TTL monitor page."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        status_card = QFrame()
        status_card.setObjectName("WorkspaceCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(16, 16, 16, 16)
        status_layout.setSpacing(10)

        title = QLabel("TTL Generator")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #eef6ff;")
        subtitle = QLabel("Watch gate, sync, and barcode timing independently from behavioral channels.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #8fa6bf;")
        status_layout.addWidget(title)
        status_layout.addWidget(subtitle)

        chip_row = QHBoxLayout()
        self.label_ttl_status = self._make_panel_chip("TTL: IDLE", "default")
        chip_row.addWidget(self.label_ttl_status)
        chip_row.addStretch()
        status_layout.addLayout(chip_row)

        self.ttl_counts_group = QGroupBox("TTL Summary")
        self.ttl_counts_layout = QGridLayout()
        self.ttl_counts_layout.setHorizontalSpacing(18)
        self.ttl_counts_layout.setVerticalSpacing(8)
        self.ttl_counts_group.setLayout(self.ttl_counts_layout)
        status_layout.addWidget(self.ttl_counts_group)

        self.ttl_plot_group = QGroupBox("TTL Generator Signals")
        ttl_plot_layout = QVBoxLayout()
        self.ttl_plot = pg.PlotWidget()
        self.ttl_plot.setBackground((8, 16, 26))
        self.ttl_plot.setMouseEnabled(x=False, y=False)
        self.ttl_plot.showGrid(x=True, y=True, alpha=0.16)
        self.ttl_plot.setLabel("bottom", "Time (s)")
        self.ttl_plot.setXRange(0, self.ttl_window_seconds)
        self.ttl_plot.setLimits(xMin=0)
        self.ttl_plot.setDownsampling(auto=True, mode="peak")
        self.ttl_plot.setMinimumHeight(240)
        ttl_plot_layout.addWidget(self.ttl_plot)
        self.ttl_plot_group.setLayout(ttl_plot_layout)
        status_layout.addWidget(self.ttl_plot_group, 1)

        layout.addWidget(status_card, 1)
        return panel

    def _make_panel_chip(self, text: str, tone: str = "default") -> QLabel:
        """Create a compact badge label for live-view headers and status areas."""
        palette = {
            "default": ("#0d2236", "#223a56", "#dbe7f3"),
            "accent": ("#0f3253", "#1d4f78", "#9dd9ff"),
            "success": ("#123324", "#1e5636", "#7ef0ac"),
            "warning": ("#3c2510", "#8a5a1c", "#ffd89c"),
            "danger": ("#3a1717", "#7b2323", "#ffb3b3"),
        }
        bg, border, fg = palette.get(tone, palette["default"])
        label = QLabel(text)
        label.setStyleSheet(
            f"QLabel {{ background-color: {bg}; border: 1px solid {border}; "
            f"border-radius: 10px; padding: 4px 10px; color: {fg}; font-weight: 600; }}"
        )
        return label

    def _set_status_chip(self, label: Optional[QLabel], text: str, tone: str = "default"):
        """Apply a themed status chip style to an existing label."""
        if label is None:
            return
        replacement = self._make_panel_chip(text, tone)
        label.setText(replacement.text())
        label.setStyleSheet(replacement.styleSheet())

    def _set_ttl_status(self, state: str, tone: str = "default"):
        self._set_status_chip(self.label_ttl_status, f"TTL: {state}", tone)

    def _set_behavior_status(self, state: str, tone: str = "default"):
        self._set_status_chip(self.label_behavior_status, f"Behavior: {state}", tone)

    def _create_live_view_panel(self) -> QWidget:
        """Build the dock widget that hosts the pyqtgraph ImageView live view."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QFrame()
        header.setStyleSheet("QFrame { background-color: #0f1b2a; border: 1px solid #203246; border-radius: 14px; }")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(8)

        title = QLabel("CamApp Live Detection")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #edf4ff;")
        header_layout.addWidget(title)

        self.btn_toggle_frame_drop_panel = QPushButton("Frame Drop")
        self._set_button_icon(self.btn_toggle_frame_drop_panel, "pulse", "#7cc7ff", "toggleButton")
        self.btn_toggle_frame_drop_panel.setCheckable(True)
        self.btn_toggle_frame_drop_panel.setChecked(bool(self.frame_drop_monitor_visible))
        self.btn_toggle_frame_drop_panel.toggled.connect(self._update_frame_drop_panel_visibility)
        header_layout.addWidget(self.btn_toggle_frame_drop_panel)

        self.live_status_badge = self._make_panel_chip("Offline", "warning")
        self.live_header_status = self._make_panel_chip("No camera connected", "default")
        self.live_header_resolution = self._make_panel_chip("-- x --", "default")
        self.live_header_mode = self._make_panel_chip(self.default_image_format, "accent")
        self.live_header_roi = self._make_panel_chip("Full Frame", "default")

        header_layout.addStretch()
        header_layout.addWidget(self.live_status_badge)
        header_layout.addWidget(self.live_header_status)
        header_layout.addWidget(self.live_header_resolution)
        header_layout.addWidget(self.live_header_mode)
        header_layout.addWidget(self.live_header_roi)
        layout.addWidget(header)
        self.frame_drop_panel = self._create_frame_drop_panel()
        layout.addWidget(self.frame_drop_panel)
        self._update_frame_drop_panel_visibility(bool(self.frame_drop_monitor_visible))

        plot_item = pg.PlotItem()
        plot_item.hideAxis("left")
        plot_item.hideAxis("bottom")
        plot_item.setMenuEnabled(False)
        plot_item.vb.setAspectLocked(True)
        plot_item.vb.setBackgroundColor("#050b12")

        self.live_image_view = pg.ImageView(view=plot_item)
        self.live_image_view.ui.roiBtn.hide()
        self.live_image_view.ui.menuBtn.hide()
        self.live_image_view.ui.histogram.hide()
        self.live_image_view.ui.roiPlot.hide()
        self.live_image_view.ui.normGroup.hide()
        self.live_image_view.setMinimumWidth(0)
        self.live_image_view.setMinimumHeight(420)
        self.live_image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.live_preview_scene = self.live_image_view.getView().scene()
        if self.live_preview_scene is not None:
            self.live_preview_scene.installEventFilter(self)
        layout.addWidget(self.live_image_view, stretch=1)

        self._show_live_placeholder("CamApp Live Detection", "Connect a camera to begin preview")
        return panel

    def _create_frame_drop_panel(self) -> QWidget:
        """Create a compact live panel for frame-drop statistics."""
        panel = QFrame()
        panel.setObjectName("WorkspaceSubCard")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        title = QLabel("Frame Drop Monitor")
        title.setStyleSheet("color: #8dd0ff; font-size: 11px; font-weight: 700;")
        self.label_frame_drop_summary = QLabel("")
        self.label_frame_drop_summary.setStyleSheet("color: #dce8f4; font-weight: 700;")
        text_layout.addWidget(title)
        text_layout.addWidget(self.label_frame_drop_summary)

        self.frame_drop_log = QTextEdit()
        self.frame_drop_log.setReadOnly(True)
        self.frame_drop_log.setLineWrapMode(QTextEdit.NoWrap)
        self.frame_drop_log.setFocusPolicy(Qt.NoFocus)
        self.frame_drop_log.setFixedHeight(54)
        self.frame_drop_log.setMinimumWidth(240)
        self.frame_drop_log.setMaximumWidth(320)
        self.frame_drop_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.frame_drop_log.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frame_drop_log.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frame_drop_log.setStyleSheet(
            """
            QTextEdit {
                background-color: #07111b;
                border: 1px solid #24405f;
                border-radius: 12px;
                color: #9dd9ff;
                font-family: "Arial Narrow", Arial, "Consolas";
                font-size: 10px;
                padding: 4px 6px;
            }
            """
        )

        layout.addLayout(text_layout, 1)
        layout.addWidget(self.frame_drop_log)
        self._reset_frame_drop_display()
        return panel

    def _update_frame_drop_panel_visibility(self, visible: Optional[bool] = None):
        """Show or hide the frame-drop monitor strip from the live-view header toggle."""
        if visible is None:
            visible = bool(
                self.btn_toggle_frame_drop_panel is not None and self.btn_toggle_frame_drop_panel.isChecked()
            )
        visible = bool(visible)
        self.frame_drop_monitor_visible = visible
        self.settings.setValue("frame_drop_monitor_visible", int(visible))

        if self.btn_toggle_frame_drop_panel is not None and self.btn_toggle_frame_drop_panel.isChecked() != visible:
            self.btn_toggle_frame_drop_panel.blockSignals(True)
            self.btn_toggle_frame_drop_panel.setChecked(visible)
            self.btn_toggle_frame_drop_panel.blockSignals(False)

        if self.frame_drop_panel is not None:
            self.frame_drop_panel.setVisible(visible)

    def _create_trial_planner_panel(self) -> QWidget:
        """Build the multi-trial recording planner dock."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QFrame()
        header.setStyleSheet("QFrame { background-color: #0f1b2a; border: 1px solid #203246; border-radius: 14px; }")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(8)

        title = QLabel("Recording Planner")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #edf4ff;")
        subtitle = QLabel("Edit metadata directly in the table. Select a row to make it the active recording plan.")
        subtitle.setStyleSheet("color: #8da6bf;")
        subtitle.setWordWrap(True)
        header_text = QVBoxLayout()
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header_layout.addLayout(header_text)
        header_layout.addStretch()

        self.btn_planner_fit = QPushButton("Fit")
        self._set_button_icon(self.btn_planner_fit, "settings", "#33d5ff", "ghostButton")
        self.btn_planner_fit.clicked.connect(self._fit_planner_columns)
        header_layout.addWidget(self.btn_planner_fit)

        self.btn_planner_detach = QPushButton("Detach")
        self._set_button_icon(self.btn_planner_detach, "export", "#ffb35d", "orangeButton")
        self.btn_planner_detach.clicked.connect(self._toggle_planner_detach)
        header_layout.addWidget(self.btn_planner_detach)
        layout.addWidget(header)

        button_grid = QGridLayout()
        button_grid.setHorizontalSpacing(8)
        button_grid.setVerticalSpacing(8)

        self.btn_planner_add_trials = QPushButton("Add Trials")
        self._set_button_icon(self.btn_planner_add_trials, "plus", "#35d2ff")
        self.btn_planner_add_trials.clicked.connect(self._add_planner_trials)

        self.btn_planner_add_variable = QPushButton("Add Variable")
        self._set_button_icon(self.btn_planner_add_variable, "session", "#d86cff", "violetButton")
        self.btn_planner_add_variable.clicked.connect(self._add_planner_variable)

        self.btn_planner_import = QPushButton("Import CSV")
        self._set_button_icon(self.btn_planner_import, "import", "#33d5ff", "ghostButton")
        self.btn_planner_import.clicked.connect(self._import_planner_trials)

        self.btn_planner_export = QPushButton("Export CSV")
        self._set_button_icon(self.btn_planner_export, "export", "#ffb35d", "ghostButton")
        self.btn_planner_export.clicked.connect(self._export_planner_trials)

        self.btn_planner_apply = QPushButton("Use Selected")
        self._set_button_icon(self.btn_planner_apply, "check", "#6fe06e", "ghostButton")
        self.btn_planner_apply.clicked.connect(self._apply_selected_planner_trial)

        self.btn_planner_remove = QPushButton("Remove")
        self._set_button_icon(self.btn_planner_remove, "record", "#ff6c9e", "dangerButton")
        self.btn_planner_remove.clicked.connect(self._remove_selected_planner_trials)

        planner_buttons = [
            self.btn_planner_add_trials,
            self.btn_planner_add_variable,
            self.btn_planner_import,
            self.btn_planner_export,
            self.btn_planner_apply,
            self.btn_planner_remove,
        ]
        planner_positions = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        ]
        for button, (row, col) in zip(planner_buttons, planner_positions):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button_grid.addWidget(button, row, col)
        layout.addLayout(button_grid)

        self.planner_table = QTableWidget(0, 0)
        self.planner_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.planner_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.planner_table.setAlternatingRowColors(True)
        self.planner_table.setWordWrap(False)
        self.planner_table.verticalHeader().setVisible(False)
        self.planner_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.planner_table.horizontalHeader().setStretchLastSection(False)
        self.planner_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.planner_table.setMinimumHeight(300)
        self.planner_table.itemSelectionChanged.connect(self._on_planner_selection_changed)
        self.planner_table.itemChanged.connect(self._on_planner_item_changed)
        layout.addWidget(self.planner_table, stretch=1)

        footer = QFrame()
        footer.setStyleSheet("QFrame { background-color: #0f1b2a; border: 1px solid #203246; border-radius: 14px; }")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(14, 10, 14, 10)
        footer_layout.setSpacing(8)
        self.label_planner_summary = QLabel("No trial selected")
        self.label_planner_summary.setStyleSheet("color: #9dd9ff; font-weight: 600;")
        footer_layout.addWidget(self.label_planner_summary)
        footer_layout.addStretch()
        layout.addWidget(footer)

        self._refresh_planner_columns()
        self._append_planner_trial()
        self.planner_table.selectRow(0)
        self._fit_planner_columns()
        return panel

    def _create_metadata_panel(self) -> QWidget:
        """Create metadata inputs used by the merged session desk."""
        panel = QFrame()
        panel.setObjectName("WorkspaceSubCard")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("Recording Metadata")
        title.setStyleSheet("font-weight: 700; font-size: 15px; color: #eef6ff;")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_widget = QWidget()
        self.metadata_layout = QFormLayout(scroll_widget)
        self.metadata_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.metadata_layout.setSpacing(10)

        self.meta_animal_id = QLineEdit()
        self.meta_animal_id.setPlaceholderText("e.g., Mouse001")
        self.metadata_layout.addRow("Animal ID:", self.meta_animal_id)

        self.meta_trial = QLineEdit()
        self.meta_trial.setPlaceholderText("e.g., 01")
        self.metadata_layout.addRow("Trial:", self.meta_trial)

        self.meta_experiment = QLineEdit()
        self.meta_experiment.setPlaceholderText("e.g., Behavior Test")
        self.metadata_layout.addRow("Experiment:", self.meta_experiment)

        self.meta_condition = QLineEdit()
        self.meta_condition.setPlaceholderText("e.g., saline")
        self.metadata_layout.addRow("Condition:", self.meta_condition)

        self.meta_arena = QLineEdit()
        self.meta_arena.setPlaceholderText("e.g., Arena 1")
        self.meta_arena.setText("Arena 1")
        self.metadata_layout.addRow("Arena:", self.meta_arena)

        self.meta_date = QLineEdit()
        self.meta_date.setText(datetime.now().strftime('%Y-%m-%d'))
        self.meta_date.setReadOnly(True)
        self.metadata_layout.addRow("Date:", self.meta_date)

        self.meta_notes = QTextEdit()
        self.meta_notes.setPlaceholderText("Additional notes...")
        self.meta_notes.setMaximumHeight(100)
        self.metadata_layout.addRow("Notes:", self.meta_notes)

        self.custom_metadata_fields = {}
        self.custom_metadata_fields["Trial"] = self.meta_trial
        self.custom_metadata_fields["Condition"] = self.meta_condition
        self.custom_metadata_fields["Arena"] = self.meta_arena

        for widget in (self.meta_animal_id, self.meta_trial, self.meta_experiment, self.meta_condition, self.meta_arena):
            widget.textChanged.connect(self._update_filename_preview)
            widget.textChanged.connect(self._save_recording_form_state)
        self.meta_notes.textChanged.connect(self._save_recording_form_state)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        btn_add_field = QPushButton("+ Add Custom Field")
        self._set_button_icon(btn_add_field, "plus", "#3fd5ff", "ghostButton")
        btn_add_field.clicked.connect(self._add_custom_metadata_field)
        layout.addWidget(btn_add_field)

        btn_save_meta = QPushButton("Save Metadata Template")
        self._set_button_icon(btn_save_meta, "check", "#6fe06e", "ghostButton")
        btn_save_meta.clicked.connect(self._save_metadata_template)
        layout.addWidget(btn_save_meta)

        return panel

    def _create_center_panel(self) -> QWidget:
        """Legacy compatibility shim for older code paths."""
        return self._create_live_view_panel()

    def _create_camera_settings(self) -> QGroupBox:
        """Create camera settings group."""
        settings_group = QGroupBox("Acquisition")
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
        self._set_button_icon(btn_apply_res, "check", "#6fe06e")
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

        self.btn_advanced = QPushButton("Advanced Controls")
        self._set_button_icon(self.btn_advanced, "settings", "#d86cff", "violetButton")
        self.btn_advanced.clicked.connect(self._toggle_advanced_settings)
        self.btn_advanced.setMinimumHeight(34)
        settings_container.addWidget(self.btn_advanced)

        self.advanced_dialog = QDialog(self)
        self.advanced_dialog.setWindowTitle("Advanced Camera Controls")
        self.advanced_dialog.setModal(False)
        self.advanced_dialog.resize(560, 420)
        dialog_layout = QVBoxLayout(self.advanced_dialog)

        self.performance_group = QGroupBox("Preview and Pipeline")
        performance_layout = QFormLayout()

        self.check_preview_enabled = QCheckBox("Enable live preview")
        self.check_preview_enabled.setChecked(True)
        self.check_preview_enabled.toggled.connect(self._on_preview_enabled_changed)
        performance_layout.addRow("", self.check_preview_enabled)

        self.spin_preview_fps = QDoubleSpinBox()
        self.spin_preview_fps.setRange(1.0, 60.0)
        self.spin_preview_fps.setDecimals(1)
        self.spin_preview_fps.setSingleStep(1.0)
        self.spin_preview_fps.setValue(25.0)
        self.spin_preview_fps.setSuffix(" fps")
        self.spin_preview_fps.valueChanged.connect(self._on_preview_fps_changed)
        performance_layout.addRow("Preview FPS:", self.spin_preview_fps)

        self.spin_preview_width = QSpinBox()
        self.spin_preview_width.setRange(0, 4096)
        self.spin_preview_width.setSingleStep(64)
        self.spin_preview_width.setSpecialValueText("Full resolution")
        self.spin_preview_width.setValue(1280)
        self.spin_preview_width.valueChanged.connect(self._on_preview_width_changed)
        performance_layout.addRow("Preview Max Width:", self.spin_preview_width)

        self.spin_frame_buffer = QSpinBox()
        self.spin_frame_buffer.setRange(8, 512)
        self.spin_frame_buffer.setSingleStep(8)
        self.spin_frame_buffer.setValue(128)
        self.spin_frame_buffer.setSuffix(" frames")
        self.spin_frame_buffer.valueChanged.connect(self._on_frame_buffer_size_changed)
        performance_layout.addRow("Frame Buffer:", self.spin_frame_buffer)

        self.spin_metadata_stats_interval = QSpinBox()
        self.spin_metadata_stats_interval.setRange(0, 240)
        self.spin_metadata_stats_interval.setSingleStep(1)
        self.spin_metadata_stats_interval.setSpecialValueText("Off")
        self.spin_metadata_stats_interval.setValue(25)
        self.spin_metadata_stats_interval.valueChanged.connect(self._on_metadata_stats_interval_changed)
        performance_layout.addRow("Raw Stats Every:", self.spin_metadata_stats_interval)

        self.performance_group.setLayout(performance_layout)
        dialog_layout.addWidget(self.performance_group)

        self.advanced_group = QGroupBox("Advanced Video")
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
        self._set_button_icon(self.btn_center_offsets, "check", "#33d5ff", "ghostButton")
        self.btn_center_offsets.clicked.connect(self._center_offsets)
        advanced_layout.addRow("", self.btn_center_offsets)

        self.spin_gain = QDoubleSpinBox()
        self.spin_gain.valueChanged.connect(self._on_gain_changed)
        advanced_layout.addRow("Gain:", self.spin_gain)

        self.combo_white_balance_auto = QComboBox()
        self.combo_white_balance_auto.addItems(["Continuous", "Off"])
        self.combo_white_balance_auto.currentTextChanged.connect(self._on_white_balance_auto_changed)
        advanced_layout.addRow("White Balance:", self.combo_white_balance_auto)

        self.spin_white_balance_red = QDoubleSpinBox()
        self.spin_white_balance_red.setDecimals(4)
        self.spin_white_balance_red.valueChanged.connect(self._on_white_balance_red_changed)
        advanced_layout.addRow("WB Red:", self.spin_white_balance_red)

        self.spin_white_balance_blue = QDoubleSpinBox()
        self.spin_white_balance_blue.setDecimals(4)
        self.spin_white_balance_blue.valueChanged.connect(self._on_white_balance_blue_changed)
        advanced_layout.addRow("WB Blue:", self.spin_white_balance_blue)

        self.spin_brightness = QDoubleSpinBox()
        self.spin_brightness.valueChanged.connect(self._on_brightness_changed)
        advanced_layout.addRow("Brightness:", self.spin_brightness)

        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.valueChanged.connect(self._on_contrast_changed)
        advanced_layout.addRow("Contrast:", self.spin_contrast)

        self.combo_camera_pixel_format = QComboBox()
        self.combo_camera_pixel_format.currentTextChanged.connect(self._on_camera_pixel_format_changed)
        advanced_layout.addRow("Pixel Format:", self.combo_camera_pixel_format)

        self.combo_camera_bit_depth = QComboBox()
        self.combo_camera_bit_depth.currentTextChanged.connect(self._on_camera_bit_depth_changed)
        advanced_layout.addRow("Bit Depth:", self.combo_camera_bit_depth)

        roi_button_layout = QHBoxLayout()
        self.btn_draw_roi = QPushButton("Draw ROI")
        self._set_button_icon(self.btn_draw_roi, "session", "#33d5ff", "ghostButton")
        self.btn_draw_roi.clicked.connect(self._toggle_roi_draw)
        roi_button_layout.addWidget(self.btn_draw_roi)
        self.btn_clear_roi = QPushButton("Clear ROI")
        self._set_button_icon(self.btn_clear_roi, "record", "#ff6c9e", "dangerButton")
        self.btn_clear_roi.clicked.connect(self._clear_roi)
        roi_button_layout.addWidget(self.btn_clear_roi)
        advanced_layout.addRow("ROI:", roi_button_layout)

        self.advanced_group.setLayout(advanced_layout)
        dialog_layout.addWidget(self.advanced_group)

        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Close)
        dialog_buttons.rejected.connect(self.advanced_dialog.close)
        dialog_layout.addWidget(dialog_buttons)

        self._set_advanced_controls_enabled(False)

        settings_group.setLayout(settings_container)
        return settings_group

    def _create_control_panel(self) -> QGroupBox:
        """Create recording control panel."""
        control_group = QGroupBox("Recording")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(12)

        session_strip = QFrame()
        session_strip.setObjectName("WorkspaceSubCard")
        session_strip_layout = QVBoxLayout(session_strip)
        session_strip_layout.setContentsMargins(14, 12, 14, 12)
        session_strip_layout.setSpacing(4)
        session_strip_title = QLabel("Active Session")
        session_strip_title.setStyleSheet("color: #8fa6bf; font-size: 11px; font-weight: 600;")
        self.label_recording_plan_summary = QLabel("No trial selected")
        self.label_recording_plan_summary.setStyleSheet("color: #eef6ff; font-size: 15px; font-weight: 700;")
        self.label_recording_plan_details = QLabel("Select a planner row in Session to drive filename and recording context.")
        self.label_recording_plan_details.setWordWrap(True)
        self.label_recording_plan_details.setStyleSheet("color: #8fa6bf;")
        session_strip_layout.addWidget(session_strip_title)
        session_strip_layout.addWidget(self.label_recording_plan_summary)
        session_strip_layout.addWidget(self.label_recording_plan_details)
        control_layout.addWidget(session_strip)

        self.label_recording_camera_hint = QLabel("Camera source is managed from the left Camera panel.")
        self.label_recording_camera_hint.setWordWrap(True)
        self.label_recording_camera_hint.setStyleSheet("color: #8fa6bf;")
        control_layout.addWidget(self.label_recording_camera_hint)

        recording_layout = QHBoxLayout()
        self.edit_save_folder = QLineEdit()
        self.edit_save_folder.setText(self.last_save_folder)
        self.edit_save_folder.setReadOnly(True)
        recording_layout.addWidget(QLabel("Save to:"))
        recording_layout.addWidget(self.edit_save_folder)

        btn_browse = QPushButton("Browse...")
        self._set_button_icon(btn_browse, "folder", "#ff9a43", "orangeButton")
        btn_browse.clicked.connect(self._browse_save_folder)
        recording_layout.addWidget(btn_browse)
        control_layout.addLayout(recording_layout)

        filename_layout = QHBoxLayout()
        filename_layout.addWidget(QLabel("Filename preview:"))

        self.edit_filename = QLineEdit()
        self.edit_filename.setPlaceholderText("Type a custom filename or leave blank for auto-generated")
        self.edit_filename.textEdited.connect(self._on_filename_text_edited)
        self.edit_filename.editingFinished.connect(self._on_filename_editing_finished)
        filename_layout.addWidget(self.edit_filename, stretch=2)
        control_layout.addLayout(filename_layout)

        self.label_filename_hint = QLabel("Type a custom filename here, or leave it empty to follow General Settings.")
        self.label_filename_hint.setStyleSheet("color: #8fa6bf;")
        control_layout.addWidget(self.label_filename_hint)

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

        self.spin_hours.valueChanged.connect(self._on_recording_length_controls_changed)
        self.spin_minutes.valueChanged.connect(self._on_recording_length_controls_changed)
        self.spin_seconds.valueChanged.connect(self._on_recording_length_controls_changed)
        self.check_unlimited.currentTextChanged.connect(self._on_recording_length_controls_changed)

        control_layout.addStretch()

        control_group.setLayout(control_layout)
        return control_group

    def _create_behavior_setup_panel(self) -> QWidget:
        """Create the dock content for Arduino connection and signal mapping."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        arduino_group = QGroupBox("Arduino Connection")
        arduino_layout = QVBoxLayout()

        port_layout = QHBoxLayout()
        self.combo_arduino_port = QComboBox()
        port_layout.addWidget(QLabel("Port:"))
        port_layout.addWidget(self.combo_arduino_port)

        btn_scan = QPushButton("Scan")
        self._set_button_icon(btn_scan, "import", "#33d5ff", "ghostButton")
        btn_scan.clicked.connect(self._scan_arduino_ports)
        port_layout.addWidget(btn_scan)
        arduino_layout.addLayout(port_layout)

        self.btn_arduino_connect = QPushButton("Connect Arduino")
        self._set_button_icon(self.btn_arduino_connect, "play", "#eef6ff")
        self.btn_arduino_connect.clicked.connect(self._on_arduino_connect_clicked)
        arduino_layout.addWidget(self.btn_arduino_connect)

        self.label_arduino_status = QLabel("Status: Disconnected")
        self.label_arduino_status.setStyleSheet("color: #8da6bf;")
        arduino_layout.addWidget(self.label_arduino_status)

        arduino_group.setLayout(arduino_layout)
        layout.addWidget(arduino_group)

        pin_group = QGroupBox("Board Pin Defaults")
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
        layout.addWidget(pin_group)

        config_group = QGroupBox("Signal Mapping")
        config_layout = QVBoxLayout()
        config_grid = QGridLayout()
        config_grid.setHorizontalSpacing(6)
        config_grid.setVerticalSpacing(6)
        config_grid.setColumnStretch(1, 2)
        config_grid.setColumnStretch(2, 1)
        config_grid.setColumnStretch(3, 1)
        config_grid.addWidget(QLabel("Use"), 0, 0)
        config_grid.addWidget(QLabel("Label"), 0, 1)
        config_grid.addWidget(QLabel("Role"), 0, 2)
        config_grid.addWidget(QLabel("Pins"), 0, 3)
        config_grid.addWidget(QLabel("Cfg"), 0, 4)

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
        self.btn_apply_behavior_config = QPushButton("Apply Mapping")
        self._set_button_icon(self.btn_apply_behavior_config, "check", "#6fe06e")
        self.btn_apply_behavior_config.clicked.connect(lambda _: self._apply_behavior_pin_configuration(persist=True))
        config_layout.addWidget(self.btn_apply_behavior_config)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        line_group = QGroupBox("Camera Input Labels")
        line_layout = QFormLayout()
        line_options = self._line_label_choice_list()

        self.combo_line1_label = QComboBox()
        self.combo_line1_label.setEditable(True)
        self.combo_line1_label.addItems(line_options)
        self.combo_line1_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(1, v))
        line_layout.addRow("Line 1:", self.combo_line1_label)

        self.combo_line2_label = QComboBox()
        self.combo_line2_label.setEditable(True)
        self.combo_line2_label.addItems(line_options)
        self.combo_line2_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(2, v))
        line_layout.addRow("Line 2:", self.combo_line2_label)

        self.combo_line3_label = QComboBox()
        self.combo_line3_label.setEditable(True)
        self.combo_line3_label.addItems(line_options)
        self.combo_line3_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(3, v))
        line_layout.addRow("Line 3:", self.combo_line3_label)

        self.combo_line4_label = QComboBox()
        self.combo_line4_label.setEditable(True)
        self.combo_line4_label.addItems(line_options)
        self.combo_line4_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(4, v))
        line_layout.addRow("Line 4:", self.combo_line4_label)

        line_group.setLayout(line_layout)
        layout.addWidget(line_group)

        self.btn_test_ttl = QPushButton("Test TTL / Behavior")
        self._set_button_icon(self.btn_test_ttl, "pulse", "#33d5ff", "violetButton")
        self.btn_test_ttl.clicked.connect(self._on_test_ttl_clicked)
        self.btn_test_ttl.setEnabled(False)
        self.btn_test_ttl.setMinimumHeight(36)
        layout.addWidget(self.btn_test_ttl)

        layout.addStretch()
        return panel

    def _create_behavior_monitor_panel(self) -> QWidget:
        """Create the dedicated behavior signal page."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        card = QFrame()
        card.setObjectName("WorkspaceCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(12)

        title = QLabel("Behavior Signals")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #eef6ff;")
        subtitle = QLabel("Monitor lever, cue, reward, and ITI channels separately from the generator timeline.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #8fa6bf;")
        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)

        chip_row = QHBoxLayout()
        self.label_behavior_status = self._make_panel_chip("Behavior: IDLE", "default")
        chip_row.addWidget(self.label_behavior_status)
        chip_row.addStretch()
        card_layout.addLayout(chip_row)

        self.behavior_counts_group = QGroupBox("Behavior Summary")
        self.behavior_counts_layout = QGridLayout()
        self.behavior_counts_layout.setHorizontalSpacing(18)
        self.behavior_counts_layout.setVerticalSpacing(8)
        self.behavior_counts_group.setLayout(self.behavior_counts_layout)
        card_layout.addWidget(self.behavior_counts_group)

        self.behavior_plot_group = QGroupBox("Behavior Timeline")
        behavior_plot_layout = QVBoxLayout()
        self.behavior_plot = pg.PlotWidget()
        self.behavior_plot.setBackground((8, 16, 26))
        self.behavior_plot.setMouseEnabled(x=False, y=False)
        self.behavior_plot.showGrid(x=True, y=True, alpha=0.16)
        self.behavior_plot.setLabel("bottom", "Time (s)")
        self.behavior_plot.setXRange(0, self.ttl_window_seconds)
        self.behavior_plot.setLimits(xMin=0)
        self.behavior_plot.setDownsampling(auto=True, mode="peak")
        self.behavior_plot.setMinimumHeight(300)
        behavior_plot_layout.addWidget(self.behavior_plot)
        self.behavior_plot_group.setLayout(behavior_plot_layout)
        card_layout.addWidget(self.behavior_plot_group, 1)

        layout.addWidget(card, 1)
        return panel

    def _create_live_detection_panel(self) -> QWidget:
        """Create the live segmentation + TTL trigger control surface."""
        self.live_detection_panel = LiveDetectionPanel()
        self.live_detection_panel.toggle_detection_requested.connect(self._on_live_detection_toggled)
        self.live_detection_panel.start_roi_draw_requested.connect(self._start_live_roi_draw)
        self.live_detection_panel.finish_polygon_requested.connect(self._finish_live_polygon_roi)
        self.live_detection_panel.remove_roi_requested.connect(self._remove_live_roi)
        self.live_detection_panel.clear_rois_requested.connect(self._clear_live_rois)
        self.live_detection_panel.output_mapping_changed.connect(self._apply_live_output_mapping)
        self.live_detection_panel.add_rule_requested.connect(self._add_live_rule)
        self.live_detection_panel.remove_rule_requested.connect(self._remove_live_rule)
        return self.live_detection_panel

    def _create_arduino_panel(self) -> QWidget:
        """Legacy compatibility shim for older code paths."""
        return self._wrap_scroll_dock_widget(self._create_behavior_setup_panel())

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

    def _display_signal_key_for_worker_signal(self, key: str) -> str:
        """Map worker/raw signal names back to one display signal key."""
        signal_key = str(key).strip().lower()
        if signal_key in ("barcode0", "barcode1"):
            return "barcode"
        if signal_key in self.DISPLAY_SIGNAL_META:
            return signal_key
        for display_key in self.DISPLAY_SIGNAL_ORDER:
            if self._state_key_for_display(display_key) == signal_key:
                return display_key
        return signal_key

    def _slugify_export_label(self, label: str, fallback: str) -> str:
        """Convert one user label into a stable CSV-safe column fragment."""
        cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(label).strip().lower()).strip("_")
        return cleaned or str(fallback).strip().lower() or "signal"

    def _signal_export_definitions(self) -> Dict[str, Dict[str, str]]:
        """Build unique export column definitions from current user labels."""
        role_map = self._current_behavior_roles() if self.behavior_role_boxes else self._default_behavior_roles()
        pin_map = self._current_behavior_pin_map() if self.behavior_pin_edits else self._default_behavior_pin_map()
        definitions: Dict[str, Dict[str, str]] = {}
        used_slugs: set[str] = set()

        for key in self.DISPLAY_SIGNAL_ORDER:
            label = self._signal_label(key).strip() or str(self.DISPLAY_SIGNAL_META[key]["name"])
            base_slug = self._slugify_export_label(label, key)
            slug = base_slug
            index = 2
            while slug in used_slugs:
                slug = f"{base_slug}_{index}"
                index += 1
            used_slugs.add(slug)
            definitions[key] = {
                "key": key,
                "state_key": self._state_key_for_display(key),
                "label": label,
                "slug": slug,
                "state_column": slug,
                "count_column": f"{slug}_count",
                "role": str(role_map.get(key, self.DISPLAY_SIGNAL_META[key]["role"])),
                "pins": self._format_pin_list(pin_map.get(key, [])),
            }

        return definitions

    def _coerce_binary_series(self, df, candidates: List[str]):
        """Return the first matching column coerced to a clean binary series."""
        import pandas as pd

        for column in candidates:
            if column in df.columns:
                return pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int).clip(0, 1)
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    def _resolve_display_signal_series(self, df, key: str):
        """Resolve one logical signal into a binary series from raw export columns."""
        definitions = self._signal_export_definitions()
        state_column = definitions.get(key, {}).get("state_column", "")

        if key == "barcode":
            barcode_aggregate = self._coerce_binary_series(
                df,
                [state_column, "barcode_state", "barcode_ttl", "barcode"],
            )
            barcode_pin0 = self._coerce_binary_series(df, ["barcode0_state", "barcode_pin0_ttl", "barcode0"])
            barcode_pin1 = self._coerce_binary_series(df, ["barcode1_state", "barcode_pin1_ttl", "barcode1"])
            return (barcode_aggregate | barcode_pin0 | barcode_pin1).astype(int)

        candidates = {
            "gate": [state_column, "gate_state", "gate_ttl", "gate"],
            "sync": [state_column, "sync_state", "sync_1hz_ttl", "sync_10hz_ttl", "sync"],
            "lever": [state_column, "lever_state", "lever_ttl", "lever"],
            "cue": [state_column, "cue_state", "cue_ttl", "cue"],
            "reward": [state_column, "reward_state", "reward_ttl", "reward"],
            "iti": [state_column, "iti_state", "iti_ttl", "iti"],
        }
        return self._coerce_binary_series(df, candidates.get(key, [state_column, f"{key}_state", key]))

    def _resolve_display_signal_count(self, key: str, ttl_counts: Dict) -> int:
        """Resolve one logical signal count from the worker pulse-count payload."""
        if key == "barcode":
            return max(int(ttl_counts.get("barcode0", 0)), int(ttl_counts.get("barcode1", 0)))
        state_key = self._state_key_for_display(key)
        return int(ttl_counts.get(state_key, 0))

    def _resolve_display_signal_count_series(self, df, key: str):
        """Resolve one logical signal count series from exported sample columns."""
        import pandas as pd

        definitions = self._signal_export_definitions()
        preferred = definitions.get(key, {}).get("count_column", "")
        if key == "barcode":
            candidates = [preferred, "barcode_count"]
        else:
            state_key = self._state_key_for_display(key)
            candidates = [preferred, f"{key}_count", f"{state_key}_count"]

        for column in candidates:
            if column in df.columns:
                return pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
        return None

    def _reorder_signal_export_columns(self, df):
        """Move label-driven signal columns forward in exported CSV files."""
        if df is None or df.empty:
            return df

        definitions = self._signal_export_definitions()
        preferred: List[str] = []
        metadata_columns = [
            "frame_id",
            "timestamp_camera",
            "timestamp_software",
            "timestamp_arduino_ms",
            "exposure_time_us",
            "passive_mode",
            "signal",
            "signal_key",
            "signal_label",
            "signal_role",
            "signal_pins",
            "state_column",
            "count_column",
            "edge",
            "state",
            "count",
        ]
        for column in metadata_columns:
            if column in df.columns and column not in preferred:
                preferred.append(column)

        for line in range(1, 5):
            for column in (f"line{line}_status",):
                if column in df.columns and column not in preferred:
                    preferred.append(column)
            for column in df.columns:
                if column.startswith(f"line{line}_status_") and column not in preferred:
                    preferred.append(column)

        for key in self.DISPLAY_SIGNAL_ORDER:
            spec = definitions.get(key, {})
            for column in (spec.get("state_column"), spec.get("count_column"), f"{key}_state"):
                if column and column in df.columns and column not in preferred:
                    preferred.append(column)

        for column in ("ttl_state", "behavior_state", "ttl_state_vector", "behavior_state_vector"):
            if column in df.columns and column not in preferred:
                preferred.append(column)

        remaining = [column for column in df.columns if column not in preferred]
        return df[preferred + remaining]

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

    def _set_signal_state_label(self, label: QLabel, active: bool):
        """Render a HIGH/LOW pill for signal state rows."""
        if active:
            label.setText("HIGH")
            label.setStyleSheet(
                "QLabel { background-color: #113626; border: 1px solid #1f6c44; "
                "border-radius: 10px; padding: 2px 8px; color: #89f0b2; font-weight: 700; }"
            )
        else:
            label.setText("LOW")
            label.setStyleSheet(
                "QLabel { background-color: #0f1a28; border: 1px solid #29415d; "
                "border-radius: 10px; padding: 2px 8px; color: #9fb1c7; font-weight: 700; }"
            )

    def _set_signal_count_label(self, label: QLabel, count_value: int):
        """Render a count label with stronger contrast once pulses are present."""
        label.setText(str(int(count_value)))
        label.setStyleSheet(
            "QLabel { color: %s; font-weight: 700; }"
            % ("#eef6ff" if int(count_value) > 0 else "#93a7bf")
        )

    def _populate_signal_count_grid(
        self,
        layout: Optional[QGridLayout],
        keys: List[str],
        state_labels: Dict[str, QLabel],
        count_labels: Dict[str, QLabel],
    ):
        """Build one count table for a signal group."""
        if layout is None:
            return
        state_labels.clear()
        count_labels.clear()
        self._clear_layout(layout)

        headers = [("Signal", Qt.AlignLeft), ("State", Qt.AlignCenter), ("Count", Qt.AlignRight)]
        for column, (text, alignment) in enumerate(headers):
            header = QLabel(text)
            header.setAlignment(alignment | Qt.AlignVCenter)
            header.setStyleSheet("color: #8dd0ff; font-weight: 700;")
            layout.addWidget(header, 0, column)

        for row, key in enumerate(keys, start=1):
            signal_label = QLabel(self._signal_label(key))
            signal_label.setStyleSheet("color: #eef6ff; font-weight: 600;")

            state_label = QLabel()
            state_label.setAlignment(Qt.AlignCenter)
            state_label.setMinimumWidth(74)
            self._set_signal_state_label(state_label, False)

            count_label = QLabel()
            count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._set_signal_count_label(count_label, 0)

            layout.addWidget(signal_label, row, 0)
            layout.addWidget(state_label, row, 1)
            layout.addWidget(count_label, row, 2)
            state_labels[key] = state_label
            count_labels[key] = count_label

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)

    def _reset_signal_count_widgets(self):
        """Reset both TTL and behavior count rows to their idle state."""
        for label_map in (self.ttl_state_labels, self.behavior_state_labels):
            for label in label_map.values():
                self._set_signal_state_label(label, False)
        for count_map in (self.ttl_count_labels, self.behavior_count_labels):
            for label in count_map.values():
                self._set_signal_count_label(label, 0)

    def _rebuild_monitor_visuals(self, reset_plot: bool = False):
        """Rebuild count rows and plot axes from current signal configuration."""
        ttl_keys = self._active_signal_keys(group="ttl")
        behavior_keys = self._active_signal_keys(group="behavior")
        self._populate_signal_count_grid(self.ttl_counts_layout, ttl_keys, self.ttl_state_labels, self.ttl_count_labels)
        self._populate_signal_count_grid(
            self.behavior_counts_layout,
            behavior_keys,
            self.behavior_state_labels,
            self.behavior_count_labels,
        )

        self.ttl_output_curves.clear()
        self.behavior_curves.clear()
        self.ttl_output_levels.clear()
        self.behavior_levels.clear()

        self.ttl_plot.clear()
        self.behavior_plot.clear()
        self.ttl_plot.setXRange(0, self.ttl_window_seconds)
        self.behavior_plot.setXRange(0, self.ttl_window_seconds)

        ttl_ticks = []
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

    def _current_barcode_output_pins(self) -> List[int]:
        """Return the current barcode output pin list from UI/settings."""
        defaults = self._default_behavior_pin_map().get("barcode", []).copy()

        try:
            pin_map = self._current_behavior_pin_map() if self.behavior_pin_edits else {}
        except Exception:
            pin_map = {}

        pins = [int(pin) for pin in pin_map.get("barcode", [])]
        if pins:
            return pins

        raw_setting = self.settings.value("behavior_pin_barcode", self._format_pin_list(defaults))
        try:
            parsed = self._parse_pin_text(str(raw_setting))
        except Exception:
            parsed = []
        return parsed if parsed else defaults

    def _apply_barcode_output_pins(self, pins: List[int], persist: bool = True):
        """Apply one or two mirrored barcode output pins through the normal mapping path."""
        normalized = []
        seen = set()
        for pin in pins:
            try:
                parsed = int(pin)
            except Exception:
                continue
            if parsed in seen:
                continue
            seen.add(parsed)
            normalized.append(parsed)

        if not normalized:
            normalized = self._default_behavior_pin_map().get("barcode", []).copy()

        pin_text = self._format_pin_list(normalized)
        pin_edit = self.behavior_pin_edits.get("barcode")
        if pin_edit is not None:
            pin_edit.setText(pin_text)

        self.settings.setValue("barcode_mirror_enabled", int(len(normalized) > 1))
        self.settings.setValue("barcode_mirror_pin", int(normalized[1]) if len(normalized) > 1 else -1)

        if pin_edit is not None:
            self._apply_behavior_pin_configuration(persist=persist)
            return

        if persist:
            self.settings.setValue("behavior_pin_barcode", pin_text)
            self.settings.sync()

        if self.arduino_worker:
            self.arduino_worker.set_manual_pin_config({"barcode": normalized})

        if "barcode" in self.pin_value_labels:
            self.pin_value_labels["barcode"].setText(pin_text)

    def _factory_behavior_defaults_snapshot(self) -> Dict[str, Dict]:
        """Return the app-default behavior/TTL configuration."""
        default_roles = self._default_behavior_roles()
        default_pins = self._default_behavior_pin_map()
        signals = {}
        for key in self.DISPLAY_SIGNAL_ORDER:
            signals[key] = {
                "enabled": True,
                "label": str(self.DISPLAY_SIGNAL_META[key]["name"]),
                "role": str(default_roles.get(key, self.DISPLAY_SIGNAL_META[key]["role"])),
                "pins": default_pins.get(key, []).copy(),
            }
        return {
            "signals": signals,
            "line_labels": {line: "None" for line in range(1, 5)},
        }

    def _behavior_defaults_snapshot(self) -> Dict[str, Dict]:
        """Collect the current persisted/live behavior defaults."""
        snapshot = self._factory_behavior_defaults_snapshot()
        pin_map = self._current_behavior_pin_map() if self.behavior_pin_edits else self._default_behavior_pin_map()
        role_map = self._current_behavior_roles() if self.behavior_role_boxes else self._default_behavior_roles()

        for key in self.DISPLAY_SIGNAL_ORDER:
            default_entry = snapshot["signals"][key]

            enabled_check = self.signal_enabled_checks.get(key)
            if enabled_check is not None:
                enabled_value = bool(enabled_check.isChecked())
            else:
                enabled_raw = self.settings.value(f"behavior_signal_enabled_{key}", int(default_entry["enabled"]))
                enabled_value = str(enabled_raw).strip().lower() not in ("0", "false", "no", "off")

            label_edit = self.signal_label_edits.get(key)
            if label_edit is not None:
                label_value = label_edit.text().strip() or default_entry["label"]
            else:
                label_value = str(self.settings.value(f"behavior_signal_label_{key}", default_entry["label"])).strip()
                if not label_value:
                    label_value = default_entry["label"]

            snapshot["signals"][key] = {
                "enabled": bool(enabled_value),
                "label": label_value,
                "role": str(role_map.get(key, default_entry["role"])),
                "pins": [int(pin) for pin in pin_map.get(key, default_entry["pins"])],
            }

        for line in range(1, 5):
            combo = getattr(self, f"combo_line{line}_label", None)
            if combo is not None:
                value = combo.currentText()
            else:
                value = str(self.settings.value(f"line_label_{line}", "None"))
            snapshot["line_labels"][line] = value if value else "None"

        return snapshot

    def _apply_behavior_defaults_snapshot(self, snapshot: Dict[str, Dict], persist: bool = True):
        """Apply one behavior-default snapshot back into the main UI and worker."""
        signals = snapshot.get("signals", {})
        line_labels = snapshot.get("line_labels", {})

        for key in self.DISPLAY_SIGNAL_ORDER:
            entry = signals.get(key, {})
            default_label = str(self.DISPLAY_SIGNAL_META[key]["name"])
            label_value = str(entry.get("label", default_label)).strip() or default_label
            role_value = str(entry.get("role", self.DISPLAY_SIGNAL_META[key]["role"]))
            pins = [int(pin) for pin in entry.get("pins", [])]
            enabled_value = bool(entry.get("enabled", True))

            label_edit = self.signal_label_edits.get(key)
            if label_edit is not None:
                label_edit.setText(label_value)

            role_box = self.behavior_role_boxes.get(key)
            if role_box is not None:
                role_box.blockSignals(True)
                if role_value in [role_box.itemText(i) for i in range(role_box.count())]:
                    role_box.setCurrentText(role_value)
                role_box.blockSignals(False)

            pin_edit = self.behavior_pin_edits.get(key)
            if pin_edit is not None:
                pin_edit.setText(self._format_pin_list(pins))

            enabled_check = self.signal_enabled_checks.get(key)
            if enabled_check is not None:
                enabled_check.setChecked(enabled_value)

            self.signal_display_config.setdefault(key, {})
            self.signal_display_config[key]["name"] = label_value
            self.signal_display_config[key]["enabled"] = enabled_value

            if persist:
                self.settings.setValue(f"behavior_pin_{key}", self._format_pin_list(pins))
                self.settings.setValue(f"behavior_role_{key}", role_value)
                self.settings.setValue(f"behavior_signal_label_{key}", label_value)
                self.settings.setValue(f"behavior_signal_enabled_{key}", int(enabled_value))

        for line in range(1, 5):
            value = str(line_labels.get(line, "None")) or "None"
            combo = getattr(self, f"combo_line{line}_label", None)
            if combo is not None:
                combo.blockSignals(True)
                if value in [combo.itemText(i) for i in range(combo.count())]:
                    combo.setCurrentText(value)
                else:
                    combo.setCurrentText("None")
                combo.blockSignals(False)
                value = combo.currentText()
            if persist:
                self.settings.setValue(f"line_label_{line}", value)

        if persist:
            self.settings.sync()

        if self.behavior_pin_edits:
            self._apply_behavior_pin_configuration(persist=persist)
        else:
            self._refresh_pin_display_from_map({
                key: [int(pin) for pin in signals.get(key, {}).get("pins", [])]
                for key in self.BEHAVIOR_PIN_KEYS
            })

        self._apply_line_label_map_to_worker()

    def _open_behavior_defaults_dialog(self):
        """Open a popup to edit behavior/TTL defaults from General Settings."""
        snapshot = self._behavior_defaults_snapshot()
        line_options = ["None", "Gate", "Sync", "Barcode", "Lever", "Cue", "Reward", "ITI"]

        dialog = QDialog(self)
        dialog.setWindowTitle("Behavior / TTL Defaults")
        dialog.resize(820, 760)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        intro = QLabel(
            "Define the default board pins and labels for the built-in Arduino/TTL channels. "
            "If a GenICam camera is connected, you can also edit the camera line labels and allowed line modes/sources here."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8fa6bf;")
        layout.addWidget(intro)

        summary_group = QGroupBox("Board Pin Defaults")
        summary_layout = QFormLayout(summary_group)
        summary_name_labels = {}
        summary_value_labels = {}
        for key in self.BEHAVIOR_PIN_KEYS:
            name_label = QLabel("")
            value_label = QLabel("")
            summary_name_labels[key] = name_label
            summary_value_labels[key] = value_label
            summary_layout.addRow(name_label, value_label)
        layout.addWidget(summary_group)

        mapping_group = QGroupBox("Signal Mapping")
        mapping_layout = QVBoxLayout(mapping_group)
        mapping_grid = QGridLayout()
        mapping_grid.setHorizontalSpacing(6)
        mapping_grid.setVerticalSpacing(6)
        mapping_grid.setColumnStretch(1, 2)
        mapping_grid.setColumnStretch(2, 1)
        mapping_grid.setColumnStretch(3, 1)
        mapping_grid.addWidget(QLabel("Use"), 0, 0)
        mapping_grid.addWidget(QLabel("Label"), 0, 1)
        mapping_grid.addWidget(QLabel("Role"), 0, 2)
        mapping_grid.addWidget(QLabel("Pins"), 0, 3)

        dialog_enabled_checks = {}
        dialog_label_edits = {}
        dialog_role_boxes = {}
        dialog_pin_edits = {}

        def refresh_pin_summary():
            for signal_key in self.BEHAVIOR_PIN_KEYS:
                label_text = dialog_label_edits[signal_key].text().strip() or str(self.DISPLAY_SIGNAL_META[signal_key]["name"])
                pin_text = dialog_pin_edits[signal_key].text().strip() or "-"
                summary_name_labels[signal_key].setText(f"{label_text}:")
                summary_value_labels[signal_key].setText(pin_text)

        for row, key in enumerate(self.DISPLAY_SIGNAL_ORDER, start=1):
            entry = snapshot["signals"][key]
            enabled_check = QCheckBox()
            enabled_check.setChecked(bool(entry.get("enabled", True)))

            label_edit = QLineEdit(str(entry.get("label", self.DISPLAY_SIGNAL_META[key]["name"])))
            label_edit.setPlaceholderText("Signal label")

            role_box = QComboBox()
            role_box.addItems(["Input", "Output"])
            role_box.setCurrentText(str(entry.get("role", self.DISPLAY_SIGNAL_META[key]["role"])))

            entry_pins = [int(pin) for pin in entry.get("pins", [])]
            pin_edit = QLineEdit(self._format_pin_list(entry_pins) if entry_pins else "")
            pin_edit.setPlaceholderText("e.g. 8, 9")

            row_widgets = [label_edit, role_box, pin_edit]
            for widget in row_widgets:
                widget.setEnabled(enabled_check.isChecked())
            enabled_check.toggled.connect(
                lambda checked, widgets=row_widgets: [widget.setEnabled(checked) for widget in widgets]
            )

            label_edit.textChanged.connect(refresh_pin_summary)
            pin_edit.textChanged.connect(refresh_pin_summary)

            dialog_enabled_checks[key] = enabled_check
            dialog_label_edits[key] = label_edit
            dialog_role_boxes[key] = role_box
            dialog_pin_edits[key] = pin_edit

            mapping_grid.addWidget(enabled_check, row, 0, alignment=Qt.AlignCenter)
            mapping_grid.addWidget(label_edit, row, 1)
            mapping_grid.addWidget(role_box, row, 2)
            mapping_grid.addWidget(pin_edit, row, 3)

        mapping_layout.addLayout(mapping_grid)
        mapping_note = QLabel(
            "These rows correspond to the built-in hardware-backed Arduino/TTL channels. "
            "Rename them freely, switch input/output roles, and use comma-separated pins for mirrored outputs such as barcode `18, 19`."
        )
        mapping_note.setWordWrap(True)
        mapping_note.setStyleSheet("color: #8fa6bf;")
        mapping_layout.addWidget(mapping_note)
        layout.addWidget(mapping_group)

        line_group = QGroupBox("Camera Inputs / Outputs")
        line_layout = QGridLayout(line_group)
        line_layout.setHorizontalSpacing(6)
        line_layout.setVerticalSpacing(6)
        line_layout.setColumnStretch(1, 2)
        line_layout.setColumnStretch(2, 1)
        line_layout.setColumnStretch(3, 1)
        line_layout.addWidget(QLabel("Line"), 0, 0)
        line_layout.addWidget(QLabel("Label"), 0, 1)
        line_layout.addWidget(QLabel("Mode"), 0, 2)
        line_layout.addWidget(QLabel("Source"), 0, 3)

        dialog_line_entries = self._camera_line_entries_for_dialog()
        dialog_line_controls = {}
        line_choice_items = self._line_label_choice_list()

        for row, entry in enumerate(dialog_line_entries, start=1):
            selector_label = QLabel(str(entry.get("display_name", f"Line {row}")))

            label_box = QComboBox()
            label_box.setEditable(True)
            label_box.addItems(line_choice_items)
            label_box.setCurrentText(str(entry.get("label", "None")) or "None")

            mode_box = QComboBox()
            mode_options = [str(option) for option in entry.get("mode_options", []) if str(option).strip()]
            if mode_options:
                mode_box.addItems(mode_options)
                current_mode = str(entry.get("mode", "")).strip()
                if current_mode in mode_options:
                    mode_box.setCurrentText(current_mode)
            else:
                mode_box.addItem("N/A")
                mode_box.setEnabled(False)

            source_box = QComboBox()
            source_options = [str(option) for option in entry.get("source_options", []) if str(option).strip()]
            if source_options:
                source_box.addItems(source_options)
                current_source = str(entry.get("source", "")).strip()
                if current_source in source_options:
                    source_box.setCurrentText(current_source)
            else:
                source_box.addItem("N/A")
                source_box.setEnabled(False)

            if not bool(entry.get("live", False)):
                mode_box.setEnabled(False)
                source_box.setEnabled(False)

            def sync_source_enabled(mode_widget=mode_box, source_widget=source_box, options=source_options):
                if not options:
                    source_widget.setEnabled(False)
                    return
                mode_value = mode_widget.currentText().strip().lower()
                source_widget.setEnabled(mode_value == "output")

            mode_box.currentTextChanged.connect(sync_source_enabled)
            sync_source_enabled()

            dialog_line_controls[str(entry.get("selector", f"Line{row}"))] = {
                "entry": entry,
                "label": label_box,
                "mode": mode_box,
                "source": source_box,
            }

            line_layout.addWidget(selector_label, row, 0)
            line_layout.addWidget(label_box, row, 1)
            line_layout.addWidget(mode_box, row, 2)
            line_layout.addWidget(source_box, row, 3)

        line_note = QLabel(
            "Mode/source controls are enabled only when a compatible Basler or FLIR GenICam camera is connected. "
            "Label fields stay editable so export column names can still be customized."
        )
        line_note.setWordWrap(True)
        line_note.setStyleSheet("color: #8fa6bf;")
        line_layout.addWidget(line_note, len(dialog_line_entries) + 1, 0, 1, 4)
        layout.addWidget(line_group)

        action_row = QHBoxLayout()
        btn_restore = QPushButton("Restore App Defaults")
        self._set_button_icon(btn_restore, "import", "#33d5ff", "ghostButton")
        action_row.addWidget(btn_restore)
        action_row.addStretch()
        layout.addLayout(action_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        def apply_snapshot_to_dialog(dialog_snapshot: Dict[str, Dict]):
            for signal_key in self.DISPLAY_SIGNAL_ORDER:
                entry = dialog_snapshot["signals"][signal_key]
                dialog_enabled_checks[signal_key].setChecked(bool(entry.get("enabled", True)))
                dialog_label_edits[signal_key].setText(
                    str(entry.get("label", self.DISPLAY_SIGNAL_META[signal_key]["name"]))
                )
                role_value = str(entry.get("role", self.DISPLAY_SIGNAL_META[signal_key]["role"]))
                role_box = dialog_role_boxes[signal_key]
                role_box.blockSignals(True)
                if role_value in [role_box.itemText(i) for i in range(role_box.count())]:
                    role_box.setCurrentText(role_value)
                role_box.blockSignals(False)
                entry_pins = [int(pin) for pin in entry.get("pins", [])]
                dialog_pin_edits[signal_key].setText(self._format_pin_list(entry_pins) if entry_pins else "")

            for line_entry in dialog_line_entries:
                selector = str(line_entry.get("selector", ""))
                control = dialog_line_controls.get(selector)
                if control is None:
                    continue
                line_number = int(line_entry.get("line_number", 0))
                value = str(dialog_snapshot["line_labels"].get(line_number, "None"))
                control["label"].blockSignals(True)
                control["label"].setCurrentText(value if value else "None")
                control["label"].blockSignals(False)

            refresh_pin_summary()

        def restore_dialog_defaults():
            apply_snapshot_to_dialog(self._factory_behavior_defaults_snapshot())
            for line_entry in dialog_line_entries:
                selector = str(line_entry.get("selector", ""))
                control = dialog_line_controls.get(selector)
                if control is None:
                    continue
                entry = control["entry"]
                mode_options = [control["mode"].itemText(i) for i in range(control["mode"].count())]
                source_options = [control["source"].itemText(i) for i in range(control["source"].count())]
                default_mode = str(entry.get("mode", "")).strip()
                default_source = str(entry.get("source", "")).strip()
                if control["mode"].isEnabled() and default_mode in mode_options:
                    control["mode"].setCurrentText(default_mode)
                if control["source"].isEnabled() and default_source in source_options:
                    control["source"].setCurrentText(default_source)

        btn_restore.clicked.connect(restore_dialog_defaults)
        apply_snapshot_to_dialog(snapshot)

        while True:
            if dialog.exec() != QDialog.Accepted:
                return

            updated_snapshot = {"signals": {}, "line_labels": {}}
            try:
                for key in self.DISPLAY_SIGNAL_ORDER:
                    label_value = dialog_label_edits[key].text().strip() or str(self.DISPLAY_SIGNAL_META[key]["name"])
                    role_value = dialog_role_boxes[key].currentText()
                    pins = self._parse_pin_text(dialog_pin_edits[key].text())
                    if not pins:
                        pins = self._default_behavior_pin_map().get(key, []).copy()
                    updated_snapshot["signals"][key] = {
                        "enabled": bool(dialog_enabled_checks[key].isChecked()),
                        "label": label_value,
                        "role": role_value,
                        "pins": pins,
                    }

                line_defaults = {}
                for line_entry in dialog_line_entries:
                    selector = str(line_entry.get("selector", ""))
                    control = dialog_line_controls.get(selector)
                    if control is None:
                        continue
                    line_number = int(line_entry.get("line_number", 0))
                    label_value = str(control["label"].currentText()).strip() or "None"
                    updated_snapshot["line_labels"][line_number] = label_value
                    line_defaults[selector] = {
                        "label": label_value,
                        "mode": str(control["mode"].currentText()).strip() if control["mode"].isEnabled() else "",
                        "source": str(control["source"].currentText()).strip() if control["source"].isEnabled() else "",
                    }
            except Exception as exc:
                self._on_error_occurred(f"Invalid behavior default configuration: {str(exc)}")
                continue

            self._save_camera_line_defaults(line_defaults)
            self._apply_behavior_defaults_snapshot(updated_snapshot, persist=True)
            self._refresh_line_label_combo_options()
            if self.is_camera_connected:
                self._apply_saved_camera_line_defaults()
            self._on_status_update("Behavior and camera line defaults updated")
            return

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
            self.settings.sync()

        if self.arduino_worker:
            self.arduino_worker.set_manual_pin_config(pin_map)
            self.arduino_worker.set_signal_roles(role_map)

        self._refresh_line_label_combo_options()

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

        barcode_pins = self._current_barcode_output_pins()
        worker_pins = params.get("output_pins", [])
        if not barcode_pins and isinstance(worker_pins, list):
            barcode_pins = [int(pin) for pin in worker_pins if str(pin).strip()]
        if not barcode_pins:
            barcode_pins = self._default_behavior_pin_map().get("barcode", [18]).copy()

        primary_pin = int(barcode_pins[0])
        saved_mirror_pin = int(self.settings.value("barcode_mirror_pin", -1))
        mirror_enabled = len(barcode_pins) > 1 or int(self.settings.value("barcode_mirror_enabled", 0)) == 1
        mirror_pin = int(barcode_pins[1]) if len(barcode_pins) > 1 else (
            saved_mirror_pin if saved_mirror_pin >= 0 and saved_mirror_pin != primary_pin else primary_pin + 1
        )

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
        spin_interval.setRange(0.000, 60.0)
        spin_interval.setSuffix(" s")
        spin_interval.setValue(float(params.get("interval_s", 5.0)))
        form.addRow("Gap After Code:", spin_interval)

        spin_primary_pin = QSpinBox()
        spin_primary_pin.setRange(0, 99)
        spin_primary_pin.setValue(primary_pin)
        form.addRow("Primary Output Pin:", spin_primary_pin)

        check_mirror_pin = QCheckBox("Drive the same barcode on a second output pin")
        check_mirror_pin.setChecked(bool(mirror_enabled))
        form.addRow("Mirror Output:", check_mirror_pin)

        spin_mirror_pin = QSpinBox()
        spin_mirror_pin.setRange(0, 99)
        spin_mirror_pin.setValue(max(0, mirror_pin))
        spin_mirror_pin.setEnabled(check_mirror_pin.isChecked())
        check_mirror_pin.toggled.connect(spin_mirror_pin.setEnabled)
        form.addRow("Mirror Pin:", spin_mirror_pin)

        mirror_note = QLabel("Primary and mirror outputs carry the same barcode waveform.")
        mirror_note.setWordWrap(True)
        mirror_note.setStyleSheet("color: #8fa6bf;")
        form.addRow("", mirror_note)

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
        primary_pin_val = int(spin_primary_pin.value())
        mirror_enabled = bool(check_mirror_pin.isChecked())
        mirror_pin_val = int(spin_mirror_pin.value())
        word_duration = start_hi_val + start_lo_val + (bits_val * bit_val)
        cycle_duration = word_duration + interval_val

        if mirror_enabled and mirror_pin_val == primary_pin_val:
            self._on_error_occurred("Mirror pin must be different from the primary barcode output pin.")
            return

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

        barcode_pins = [primary_pin_val]
        if mirror_enabled:
            barcode_pins.append(mirror_pin_val)
        self._apply_barcode_output_pins(barcode_pins, persist=True)

        self._on_status_update(
            "Barcode params updated: "
            f"bits={bits_val}, start={start_hi_val:.3f}/{start_lo_val:.3f}s, "
            f"bit={bit_val:.3f}s, gap={interval_val:.3f}s, "
            f"word={word_duration:.3f}s, cycle={cycle_duration:.3f}s, "
            f"pins={self._format_pin_list(barcode_pins)}"
        )

    def _setup_worker(self):
        """Initialize the camera worker thread and connect signals."""
        self.worker = CameraWorker()

        # Connect worker signals to GUI slots
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.preview_packet_ready.connect(self._on_preview_packet_ready)
        self.worker.status_update.connect(self._on_status_update)
        self.worker.fps_update.connect(self._on_fps_update)
        self.worker.buffer_update.connect(self._on_buffer_update)
        self.worker.error_occurred.connect(self._on_error_occurred)
        self.worker.recording_stopped.connect(self._on_recording_stopped)
        self.worker.frame_drop_stats_updated.connect(self._on_frame_drop_stats_updated)

        # Connect frame recording to TTL sampling (sync TTLs with camera frames)
        self.worker.frame_recorded.connect(self._on_frame_recorded)
        self._apply_line_label_map_to_worker()
        self._apply_pipeline_settings_to_worker()
        self._setup_live_detection_worker()

    def _setup_live_detection_worker(self):
        if self.live_inference_worker is not None:
            return
        self.live_inference_worker = LiveInferenceWorker(self)
        self.live_inference_worker.result_ready.connect(self._on_live_detection_result)
        self.live_inference_worker.status_changed.connect(self._on_live_detection_status_changed)
        self.live_inference_worker.error_occurred.connect(self._on_error_occurred)

    def _apply_pipeline_settings_to_worker(self):
        """Push preview and buffering preferences into the camera worker."""
        if not self.worker:
            return
        self.worker.set_preview_enabled(self.check_preview_enabled.isChecked())
        self.worker.set_preview_fps(self.spin_preview_fps.value())
        self.worker.set_preview_max_width(self.spin_preview_width.value())
        self.worker.set_frame_buffer_size(self.spin_frame_buffer.value())
        self.worker.set_metadata_stats_interval(self.spin_metadata_stats_interval.value())

    def _get_max_record_seconds(self) -> int:
        """Return the configured recording limit in seconds, or 0 if unlimited/disabled."""
        if self.check_unlimited.currentText() != "Limited":
            return 0
        return (
            self.spin_hours.value() * 3600
            + self.spin_minutes.value() * 60
            + self.spin_seconds.value()
        )

    def _get_target_record_frames(self) -> Optional[int]:
        """
        Convert the configured max duration into an exact frame target.

        The target uses the worker's effective recording FPS so FLIR recordings
        stay aligned with the actual encoded video length.
        """
        max_seconds = self._get_max_record_seconds()
        if max_seconds <= 0:
            return None
        return max(1, int(round(self._get_recording_reference_fps() * max_seconds)))

    def _get_recording_reference_fps(self) -> float:
        """Return the FPS used for recording duration math."""
        if self.worker is not None:
            worker_fps_candidates = []
            recording_output_fps = getattr(self.worker, "recording_output_fps", None)
            if self.worker.is_recording and recording_output_fps:
                worker_fps_candidates.append(recording_output_fps)
            worker_fps_candidates.extend([
                getattr(self.worker, "camera_reported_fps", None),
                getattr(self.worker, "fps_target", None),
            ])
            for candidate in worker_fps_candidates:
                try:
                    fps_value = float(candidate)
                except (TypeError, ValueError):
                    continue
                if fps_value > 0:
                    return fps_value
        return max(1.0, float(self.spin_fps.value()))

    def _active_planner_row_index(self) -> Optional[int]:
        """Return the selected planner row when available, otherwise the active row."""
        if self.planner_table is not None and self.planner_table.selectionModel() is not None:
            selected_rows = self.planner_table.selectionModel().selectedRows()
            if selected_rows:
                return selected_rows[0].row()
        if self.active_planner_row is None:
            return None
        if self.planner_table is None:
            return None
        if 0 <= self.active_planner_row < self.planner_table.rowCount():
            return self.active_planner_row
        return None

    def _sync_active_trial_duration_cell(self):
        """Mirror the current recording-length controls into the active planner row."""
        if self.planner_table is None:
            return
        row = self._active_planner_row_index()
        if row is None:
            return
        duration_seconds = self._get_max_record_seconds()
        duration_text = str(duration_seconds if duration_seconds > 0 else 0)
        headers = self._planner_headers()
        if "Duration (s)" not in headers:
            return
        duration_column = headers.index("Duration (s)")
        item = self.planner_table.item(row, duration_column)
        if item is not None and item.text().strip() == duration_text:
            return
        self.planner_table.blockSignals(True)
        self._set_planner_cell(row, "Duration (s)", duration_text)
        self.planner_table.blockSignals(False)
        self._update_planner_summary()

    def _apply_recording_frame_limit(self):
        """Push the configured recording cap into the worker immediately."""
        if self.worker is None:
            return
        target_frames = self._get_target_record_frames()
        self.worker.set_recording_frame_limit(target_frames)

    def _update_recording_limit_inputs_enabled(self):
        """Disable duration spin boxes when unlimited recording is selected."""
        limited = self.check_unlimited.currentText() == "Limited"
        for widget in (self.spin_hours, self.spin_minutes, self.spin_seconds):
            widget.setEnabled(limited)

    def _on_recording_length_controls_changed(self, *_args):
        """Keep UI, planner state, and active worker limits aligned."""
        self._update_recording_limit_inputs_enabled()
        self._sync_active_trial_duration_cell()
        self._apply_recording_frame_limit()
        self._refresh_recording_session_summary()
        if self.recording_start_time:
            self._update_recording_time()

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
        """Scan for Basler, FLIR, and generic USB cameras."""
        self.combo_camera.clear()
        cameras = []

        try:
            basler_cameras = discover_basler_cameras()
        except Exception as e:
            basler_cameras = []
            if hasattr(self, "status_bar"):
                self._on_status_update(f"Basler scan error: {str(e)}")

        flir_cameras, reserved_usb_indices = discover_flir_cameras()
        usb_cameras = discover_usb_cameras(skip_indices=reserved_usb_indices)
        backend_diagnostics = get_camera_backend_diagnostics()

        for camera_info in basler_cameras + flir_cameras + usb_cameras:
            self.combo_camera.addItem(camera_info.get("label", "Camera"), camera_info)
            cameras.append(camera_info)

        if not any(cam.get("backend") == "spinnaker" for cam in flir_cameras):
            pyspin_diag = backend_diagnostics.get("pyspin", "")
            if pyspin_diag and hasattr(self, "status_bar"):
                self._on_status_update(f"FLIR Spinnaker unavailable: {pyspin_diag}")

        if not cameras:
            self.combo_camera.addItem("No cameras detected", None)
            return

        last_type = self.settings.value('last_camera_type', '')
        last_backend = self.settings.value('last_camera_backend', '')
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
                if (
                    data.get('type') == last_type
                    and str(data.get('backend', '')) == str(last_backend)
                    and int(data.get('index', -1)) == last_index
                ):
                    self.combo_camera.setCurrentIndex(i)
                    break

    # ... (continue with slot implementations)

    # ===== Metadata Methods =====

    def _add_custom_metadata_field(self):
        """Add a custom metadata field."""
        from PySide6.QtWidgets import QInputDialog

        field_name, ok = QInputDialog.getText(self, "Add Custom Field", "Field Name:")
        if ok and field_name:
            if field_name in self.custom_metadata_fields:
                return
            field_edit = QLineEdit()
            field_edit.setPlaceholderText(f"Enter {field_name}...")
            field_edit.textChanged.connect(self._update_filename_preview)
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
                    if 'trial' in template:
                        self.meta_trial.setText(template['trial'])
                    if 'experiment' in template:
                        self.meta_experiment.setText(template['experiment'])
                    if 'condition' in template:
                        self.meta_condition.setText(template['condition'])
                    if 'arena' in template:
                        self.meta_arena.setText(template['arena'])
            except:
                pass
        self._update_filename_preview()

    def _filename_field_labels(self) -> List[str]:
        return ["Animal ID", "Trial", "Experiment", "Condition", "Arena", "Date", "(skip)"]

    def _filename_label_to_key(self, label: str) -> str:
        mapping = {
            "Animal ID": "animal_id",
            "Trial": "trial",
            "Experiment": "experiment",
            "Condition": "condition",
            "Arena": "arena",
            "Date": "date",
            "(skip)": "",
        }
        return mapping.get(label, "")

    def _filename_key_to_label(self, key: str) -> str:
        mapping = {
            "animal_id": "Animal ID",
            "trial": "Trial",
            "experiment": "Experiment",
            "condition": "Condition",
            "arena": "Arena",
            "date": "Date",
            "": "(skip)",
        }
        return mapping.get(key, "(skip)")

    def _default_filename_order(self) -> List[str]:
        return ["animal_id", "trial", "experiment", "condition"]

    def _set_filename_order_controls(self):
        """Populate filename order combo boxes from persisted settings."""
        order = []
        for index in range(4):
            key = str(self.settings.value(f"filename_part_{index + 1}", self._default_filename_order()[index]))
            order.append(key)
        if not self.filename_order_boxes:
            return
        for combo, key in zip(self.filename_order_boxes, order):
            label = self._filename_key_to_label(key)
            combo.blockSignals(True)
            if label in [combo.itemText(i) for i in range(combo.count())]:
                combo.setCurrentText(label)
            else:
                combo.setCurrentText("(skip)")
            combo.blockSignals(False)
        self._update_filename_preview()

    def _selected_filename_order(self) -> List[str]:
        if not self.filename_order_boxes:
            return self._default_filename_order()
        order = []
        for combo in self.filename_order_boxes:
            key = self._filename_label_to_key(combo.currentText())
            if key:
                order.append(key)
        return order

    def _on_filename_order_changed(self, *_args):
        """Persist filename order and refresh the live preview."""
        for index, combo in enumerate(self.filename_order_boxes, start=1):
            self.settings.setValue(f"filename_part_{index}", self._filename_label_to_key(combo.currentText()))
        self._update_filename_preview()

    def _metadata_token_values(self) -> Dict[str, str]:
        return {
            "animal_id": self.meta_animal_id.text().strip() if self.meta_animal_id else "",
            "trial": self.meta_trial.text().strip() if self.meta_trial else "",
            "experiment": self.meta_experiment.text().strip() if self.meta_experiment else "",
            "condition": self.meta_condition.text().strip() if self.meta_condition else "",
            "arena": self.meta_arena.text().strip() if self.meta_arena else "",
            "date": self.meta_date.text().strip() if self.meta_date else datetime.now().strftime("%Y-%m-%d"),
        }

    def _sanitize_filename_part(self, value: str) -> str:
        cleaned = []
        for char in value.strip():
            if char.isalnum() or char in ("-", "_"):
                cleaned.append(char)
            elif char.isspace():
                cleaned.append("_")
        return "".join(cleaned).strip("_")

    def _set_filename_field_text(self, text: str):
        """Update the filename field without treating it as a user edit."""
        if not hasattr(self, "edit_filename") or self.edit_filename is None:
            return
        self._filename_field_syncing = True
        try:
            self.edit_filename.setText(text)
        finally:
            self._filename_field_syncing = False

    def _save_recording_form_state(self):
        """Persist the last recording-form values so they survive app relaunches."""
        if self.meta_animal_id is not None:
            self.settings.setValue("recording_meta_animal_id", self.meta_animal_id.text().strip())
        if self.meta_trial is not None:
            self.settings.setValue("recording_meta_trial", self.meta_trial.text().strip())
        if hasattr(self, "meta_experiment") and self.meta_experiment is not None:
            self.settings.setValue("recording_meta_experiment", self.meta_experiment.text().strip())
        if self.meta_condition is not None:
            self.settings.setValue("recording_meta_condition", self.meta_condition.text().strip())
        if self.meta_arena is not None:
            self.settings.setValue("recording_meta_arena", self.meta_arena.text().strip())
        if hasattr(self, "meta_notes") and self.meta_notes is not None:
            self.settings.setValue("recording_meta_notes", self.meta_notes.toPlainText())
        self.settings.setValue("recording_filename_override", self._custom_filename_override)

    def _load_recording_form_state(self):
        """Restore the last recording-form values from settings."""
        if self.meta_animal_id is not None:
            self.meta_animal_id.setText(str(self.settings.value("recording_meta_animal_id", "")))
        if self.meta_trial is not None:
            self.meta_trial.setText(str(self.settings.value("recording_meta_trial", "")))
        if hasattr(self, "meta_experiment") and self.meta_experiment is not None:
            self.meta_experiment.setText(str(self.settings.value("recording_meta_experiment", "")))
        if self.meta_condition is not None:
            self.meta_condition.setText(str(self.settings.value("recording_meta_condition", "")))
        if self.meta_arena is not None:
            saved_arena = str(self.settings.value("recording_meta_arena", self.meta_arena.text().strip() or "Arena 1"))
            self.meta_arena.setText(saved_arena or "Arena 1")
        if hasattr(self, "meta_notes") and self.meta_notes is not None:
            self.meta_notes.setPlainText(str(self.settings.value("recording_meta_notes", "")))

        self._custom_filename_override = str(self.settings.value("recording_filename_override", "") or "").strip()

    def _compose_generated_recording_basename(self) -> str:
        values = self._metadata_token_values()
        ordered_parts = []
        for key in self._selected_filename_order():
            token = self._sanitize_filename_part(values.get(key, ""))
            if token:
                ordered_parts.append(token)
        has_primary_identity = any(
            self._sanitize_filename_part(values.get(key, ""))
            for key in ("animal_id", "experiment", "condition")
        )
        if ordered_parts and (len(ordered_parts) > 1 or has_primary_identity):
            return "_".join(ordered_parts)
        return f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _current_custom_filename_override(self) -> str:
        """Return the sanitized custom filename override, if the user set one."""
        raw_override = self._custom_filename_override
        if hasattr(self, "edit_filename") and self.edit_filename is not None and self.edit_filename.hasFocus():
            raw_override = self.edit_filename.text()
        return self._sanitize_filename_part(str(raw_override))

    def _compose_recording_basename(self) -> str:
        custom_override = self._current_custom_filename_override()
        if custom_override:
            return custom_override
        return self._compose_generated_recording_basename()

    def _on_filename_text_edited(self, text: str):
        """Persist a user-entered filename override without losing edit focus."""
        if self._filename_field_syncing:
            return
        self._custom_filename_override = str(text)
        self._save_recording_form_state()
        self._refresh_recording_session_summary()

    def _on_filename_editing_finished(self):
        """Normalize the filename override after the user leaves the field."""
        if self._filename_field_syncing or not hasattr(self, "edit_filename") or self.edit_filename is None:
            return
        self._custom_filename_override = self._sanitize_filename_part(self.edit_filename.text())
        self._save_recording_form_state()
        self._update_filename_preview()

    def _update_filename_preview(self, *_args):
        """Refresh the generated filename preview and formula label."""
        generated_basename = self._compose_generated_recording_basename()
        basename = self._compose_recording_basename()
        if hasattr(self, "edit_filename") and self.edit_filename is not None:
            if not self.edit_filename.hasFocus():
                self._set_filename_field_text(basename)
        if hasattr(self, "label_filename_formula") and self.label_filename_formula is not None:
            custom_override = self._current_custom_filename_override()
            if custom_override:
                self.label_filename_formula.setText(
                    f"Custom filename override\nGenerated fallback: {generated_basename}\nPreview: {basename}"
                )
            else:
                readable = " / ".join(
                    self._filename_key_to_label(key)
                    for key in self._selected_filename_order()
                ) or "No filename parts selected"
                self.label_filename_formula.setText(f"{readable}\nPreview: {basename}")
        self._save_recording_form_state()
        self._refresh_recording_session_summary()

    def _planner_status_style(self, status: str):
        palette = {
            "Pending": ("#3c2510", "#ffd89c"),
            "Acquiring": ("#102b43", "#9dd9ff"),
            "Acquired": ("#123324", "#7ef0ac"),
        }
        return palette.get(status, ("#111827", "#dbe7f3"))

    def _set_planner_row_status(self, row: int, status: str):
        """Write and tint the planner status cell."""
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return
        self.planner_table.blockSignals(True)
        self._set_planner_cell(row, "Status", status)
        self.planner_table.blockSignals(False)
        headers = self._planner_headers()
        if "Status" not in headers:
            return
        item = self.planner_table.item(row, headers.index("Status"))
        if item is None:
            return
        bg, fg = self._planner_status_style(status)
        item.setBackground(QColor(bg))
        item.setForeground(QColor(fg))

    def _find_planner_row_for_current_session(self) -> Optional[int]:
        """Resolve the planner row associated with the current metadata selection."""
        if self.planner_table is None:
            return None
        selected = self.planner_table.selectionModel().selectedRows()
        if selected:
            return selected[0].row()

        current_trial = self.meta_trial.text().strip() if self.meta_trial else ""
        current_animal = self.meta_animal_id.text().strip() if self.meta_animal_id else ""
        for row in range(self.planner_table.rowCount()):
            payload = self._planner_row_payload(row)
            if current_trial and payload.get("Trial", "").strip() == current_trial:
                if not current_animal or payload.get("Animal ID", "").strip() == current_animal:
                    return row
        return None

    def _sync_active_trial_status(self, status: str):
        """Update the active trial status in the planner table."""
        row = self.active_planner_row
        if row is None:
            row = self._find_planner_row_for_current_session()
        if row is None:
            return
        self.active_planner_row = row
        self._set_planner_row_status(row, status)
        self._update_planner_summary()

    def _on_planner_item_changed(self, item: QTableWidgetItem):
        """React to planner table edits."""
        if item is None:
            return
        if self._planner_headers()[item.column()] == "Status":
            self._set_planner_row_status(item.row(), item.text().strip() or "Pending")
        self._update_planner_summary()

    def _planner_headers(self) -> List[str]:
        return self.planner_default_columns + self.planner_custom_columns

    def _planner_default_duration_seconds(self) -> int:
        if not hasattr(self, "spin_hours") or self.spin_hours is None:
            return (
                int(self.settings.value("max_hours", 0)) * 3600
                + int(self.settings.value("max_minutes", 5)) * 60
                + int(self.settings.value("max_seconds", 0))
            )
        if hasattr(self, "check_unlimited") and self.check_unlimited is not None:
            if self.check_unlimited.currentText() != "Limited":
                return 0
        return (
            int(self.spin_hours.value()) * 3600
            + int(self.spin_minutes.value()) * 60
            + int(self.spin_seconds.value())
        )

    def _planner_status_totals(self) -> Dict[str, int]:
        """Return planner row counts grouped by acquisition state."""
        totals = {"total": 0, "Pending": 0, "Acquiring": 0, "Acquired": 0}
        if self.planner_table is None:
            return totals
        totals["total"] = self.planner_table.rowCount()
        for row in range(self.planner_table.rowCount()):
            status = self._planner_row_payload(row).get("Status", "") or "Pending"
            if status not in totals:
                totals[status] = 0
            totals[status] += 1
        return totals

    def _refresh_session_metrics(self):
        """Update the planner dashboard count tiles."""
        totals = self._planner_status_totals()
        label_map = {
            "total": self.label_session_total_count,
            "Pending": self.label_session_pending_count,
            "Acquiring": self.label_session_acquiring_count,
            "Acquired": self.label_session_acquired_count,
        }
        for key, label in label_map.items():
            if label is not None:
                label.setText(str(int(totals.get(key, 0))))

    def _current_session_payload(self) -> Dict[str, str]:
        """Build a lightweight session summary from the selected planner row or hidden metadata."""
        selected_rows = []
        if self.planner_table is not None and self.planner_table.selectionModel() is not None:
            selected_rows = self.planner_table.selectionModel().selectedRows()
        if selected_rows:
            return self._planner_row_payload(selected_rows[0].row())
        return {
            "Status": "Pending",
            "Trial": self.meta_trial.text().strip() if self.meta_trial is not None else "",
            "Animal ID": self.meta_animal_id.text().strip() if self.meta_animal_id is not None else "",
            "Experiment": self.meta_experiment.text().strip() if hasattr(self, "meta_experiment") and self.meta_experiment is not None else "",
            "Condition": self.meta_condition.text().strip() if self.meta_condition is not None else "",
            "Arena": self.meta_arena.text().strip() if self.meta_arena is not None else "",
        }

    def _refresh_recording_session_summary(self):
        """Keep the recording card aligned with the active planner row and filename preview."""
        if self.label_recording_plan_summary is None or self.label_recording_plan_details is None:
            return
        payload = self._current_session_payload()
        trial = payload.get("Trial", "").strip() or "No trial"
        animal = payload.get("Animal ID", "").strip() or "No subject"
        status = payload.get("Status", "Pending").strip() or "Pending"
        experiment = payload.get("Experiment", "").strip() or "No experiment"
        condition = payload.get("Condition", "").strip() or "No condition"
        arena = payload.get("Arena", "").strip() or "No arena"
        filename = self.edit_filename.text().strip() if hasattr(self, "edit_filename") and self.edit_filename is not None else ""
        max_length_seconds = self._get_max_record_seconds()
        max_length_text = "Unlimited" if max_length_seconds <= 0 else self._format_duration_hms(max_length_seconds)
        self.label_recording_plan_summary.setText(f"{status}  |  Trial {trial}  |  {animal}")
        if filename:
            self.label_recording_plan_details.setText(
                f"{experiment}  |  {condition}  |  {arena}  |  Max {max_length_text}\nNext file: {filename}"
            )
        else:
            self.label_recording_plan_details.setText(
                f"{experiment}  |  {condition}  |  {arena}  |  Max {max_length_text}"
            )

    def _refresh_planner_columns(self):
        """Rebuild planner headers after custom-variable changes."""
        if self.planner_table is None:
            return
        headers = self._planner_headers()
        self.planner_table.blockSignals(True)
        self.planner_table.setColumnCount(len(headers))
        self.planner_table.setHorizontalHeaderLabels(headers)
        for index, header in enumerate(headers):
            width = 130
            if header == "Status":
                width = 96
            if header in ("Comments", "Experiment"):
                width = 180
            elif header in ("Start Delay (s)", "Duration (s)"):
                width = 120
            self.planner_table.setColumnWidth(index, width)
        self.planner_table.blockSignals(False)

    def _set_planner_cell(self, row: int, header: str, value: str):
        """Write one planner cell, creating the item when needed."""
        if self.planner_table is None:
            return
        headers = self._planner_headers()
        if header not in headers:
            return
        column = headers.index(header)
        item = self.planner_table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.planner_table.setItem(row, column, item)
        item.setText(str(value))

    def _append_planner_trial(self, seed: Optional[Dict[str, str]] = None):
        """Append one trial row to the planner table."""
        if self.planner_table is None:
            return
        seed = seed or {}
        row = self.planner_table.rowCount()
        self.planner_table.insertRow(row)

        trial_value = str(seed.get("Trial", self.planner_next_trial_number))
        defaults = {
            "Status": str(seed.get("Status", "Pending")),
            "Trial": trial_value,
            "Arena": str(seed.get("Arena", self.meta_arena.text().strip() if self.meta_arena else "Arena 1")),
            "Animal ID": str(seed.get("Animal ID", self.meta_animal_id.text().strip())),
            "Experiment": str(seed.get("Experiment", self.meta_experiment.text().strip())),
            "Condition": str(seed.get("Condition", self.meta_condition.text().strip() if self.meta_condition else "")),
            "Start Delay (s)": str(seed.get("Start Delay (s)", "0")),
            "Duration (s)": str(seed.get("Duration (s)", self._planner_default_duration_seconds())),
            "Comments": str(seed.get("Comments", "")),
        }
        self.planner_table.blockSignals(True)
        for header in self._planner_headers():
            self._set_planner_cell(row, header, seed.get(header, defaults.get(header, "")))
        self.planner_table.blockSignals(False)
        self._set_planner_row_status(row, defaults["Status"])

        try:
            self.planner_next_trial_number = max(self.planner_next_trial_number, int(trial_value) + 1)
        except Exception:
            self.planner_next_trial_number += 1

    def _planner_row_payload(self, row: int) -> Dict[str, str]:
        """Return one planner row as a dict."""
        payload = {}
        if self.planner_table is None:
            return payload
        for column, header in enumerate(self._planner_headers()):
            item = self.planner_table.item(row, column)
            payload[header] = item.text().strip() if item else ""
        return payload

    def _add_planner_trials(self):
        """Append one or more trials to the planner."""
        from PySide6.QtWidgets import QInputDialog

        count, ok = QInputDialog.getInt(self, "Add Trials", "Number of trials:", 1, 1, 500, 1)
        if not ok:
            return
        for _ in range(int(count)):
            self._append_planner_trial()
        self._fit_planner_columns()
        self._update_planner_summary()

    def _add_planner_variable(self):
        """Add a user-defined metadata column to the planner."""
        from PySide6.QtWidgets import QInputDialog

        variable_name, ok = QInputDialog.getText(self, "Add Variable", "Variable name:")
        if not ok or not variable_name.strip():
            return
        variable_name = variable_name.strip()
        if variable_name in self._planner_headers():
            self._on_error_occurred(f"Planner variable already exists: {variable_name}")
            return
        self.planner_custom_columns.append(variable_name)
        self._refresh_planner_columns()
        self._fit_planner_columns()
        self._update_planner_summary()

    def _remove_selected_planner_trials(self):
        """Remove the selected planner rows."""
        if self.planner_table is None:
            return
        rows = sorted({index.row() for index in self.planner_table.selectionModel().selectedRows()}, reverse=True)
        if not rows:
            return
        if self.active_planner_row in rows:
            self.active_planner_row = None
        for row in rows:
            self.planner_table.removeRow(row)
        self._update_planner_summary()

    def _import_planner_trials(self):
        """Load planner rows from a CSV file."""
        if self.planner_table is None:
            return
        import csv

        filepath, _ = QFileDialog.getOpenFileName(self, "Import Trial Plan", self.last_save_folder, "CSV Files (*.csv)")
        if not filepath:
            return
        with open(filepath, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        if not fieldnames:
            self._on_error_occurred("Selected CSV has no header row.")
            return

        extras = [field for field in fieldnames if field not in self.planner_default_columns]
        self.planner_custom_columns = extras
        self._refresh_planner_columns()
        self.planner_table.setRowCount(0)
        self.planner_next_trial_number = 1
        for row in rows:
            self._append_planner_trial({key: row.get(key, "") for key in self._planner_headers()})
        if self.planner_table.rowCount() > 0:
            self.planner_table.selectRow(0)
        self._fit_planner_columns()
        self._on_status_update(f"Imported planner CSV: {Path(filepath).name}")
        self._update_planner_summary()

    def _export_planner_trials(self):
        """Export the planner table to CSV."""
        if self.planner_table is None:
            return
        import csv

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Trial Plan",
            str(Path(self.last_save_folder) / "camapp_trial_plan.csv"),
            "CSV Files (*.csv)",
        )
        if not filepath:
            return

        headers = self._planner_headers()
        with open(filepath, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in range(self.planner_table.rowCount()):
                writer.writerow(self._planner_row_payload(row))
        self._on_status_update(f"Planner exported: {Path(filepath).name}")

    def _ensure_custom_metadata_field(self, field_name: str) -> QLineEdit:
        """Return an existing custom metadata field or create one in the form."""
        if field_name in self.custom_metadata_fields:
            return self.custom_metadata_fields[field_name]
        field_edit = QLineEdit()
        field_edit.setPlaceholderText(f"Enter {field_name}...")
        field_edit.textChanged.connect(self._update_filename_preview)
        self.metadata_layout.addRow(f"{field_name}:", field_edit)
        self.custom_metadata_fields[field_name] = field_edit
        return field_edit

    def _fit_planner_columns(self):
        """Resize planner columns for readability."""
        if self.planner_table is None:
            return
        self.planner_table.resizeColumnsToContents()
        self.planner_table.horizontalHeader().setStretchLastSection(False)

    def _toggle_planner_detach(self):
        """Detach the planner into a larger floating window or reattach it."""
        if self.planner_panel_widget is None or self.planner_host_layout is None:
            return
        if self.planner_detached:
            self._reattach_planner_panel()
            return

        self.planner_dialog = QDialog(self)
        self.planner_dialog.setWindowTitle("CamApp Live Detection Planner")
        self.planner_dialog.resize(1500, 900)
        self.planner_dialog.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        dialog_layout = QVBoxLayout(self.planner_dialog)
        dialog_layout.setContentsMargins(12, 12, 12, 12)

        self.planner_panel_widget.setParent(None)
        dialog_layout.addWidget(self.planner_panel_widget)
        self.planner_dialog.finished.connect(self._on_planner_dialog_finished)
        self.planner_detached = True
        self.btn_planner_detach.setText("Reattach")
        self._set_button_icon(self.btn_planner_detach, "import", "#6fe06e", "successButton")
        self.planner_dialog.show()
        self.planner_dialog.raise_()
        self.planner_dialog.activateWindow()
        self._fit_planner_columns()

    def _on_planner_dialog_finished(self, _result: int):
        """Reattach the planner if the floating window closes."""
        if self.planner_detached and not self.planner_reattaching:
            self._reattach_planner_panel()

    def _reattach_planner_panel(self):
        """Return the planner panel to the session shell."""
        if not self.planner_detached or self.planner_panel_widget is None or self.planner_host_layout is None:
            return

        self.planner_reattaching = True
        self.planner_panel_widget.setParent(None)
        self.planner_host_layout.addWidget(self.planner_panel_widget)
        self.planner_detached = False
        if self.planner_dialog is not None:
            try:
                self.planner_dialog.finished.disconnect(self._on_planner_dialog_finished)
            except Exception:
                pass
            self.planner_dialog.close()
            self.planner_dialog.deleteLater()
            self.planner_dialog = None
        self.btn_planner_detach.setText("Detach")
        self._set_button_icon(self.btn_planner_detach, "export", "#ffb35d", "orangeButton")
        self.planner_reattaching = False

    def _load_planner_row_into_metadata(self, row: int, announce: bool = False, apply_duration: bool = True):
        """Copy one planner row into the hidden metadata/session fields."""
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return

        payload = self._planner_row_payload(row)
        animal_id = payload.get("Animal ID", "")
        experiment = payload.get("Experiment", "")
        comments = payload.get("Comments", "")
        condition = payload.get("Condition", "")

        self.meta_animal_id.setText(animal_id)
        self.meta_trial.setText(payload.get("Trial", ""))
        self.meta_experiment.setText(experiment)
        self.meta_condition.setText(condition)
        self.meta_arena.setText(payload.get("Arena", ""))
        self.meta_notes.setPlainText(comments)

        for header, value in payload.items():
            if header in self.planner_default_columns:
                continue
            self._ensure_custom_metadata_field(header).setText(value)

        if apply_duration:
            try:
                duration_seconds = int(float(payload.get("Duration (s)", "0") or 0))
                if duration_seconds > 0:
                    self.check_unlimited.setCurrentText("Limited")
                    self.spin_hours.setValue(duration_seconds // 3600)
                    self.spin_minutes.setValue((duration_seconds % 3600) // 60)
                    self.spin_seconds.setValue(duration_seconds % 60)
                else:
                    self.check_unlimited.setCurrentText("Unlimited")
            except Exception:
                pass

        self.active_planner_row = row
        self._update_filename_preview()

        if self.label_session_summary is not None:
            self.label_session_summary.setText(
                f"Trial {payload.get('Trial', '?')}  |  {payload.get('Animal ID', '').strip() or 'No subject'}"
            )
        if self.label_session_details is not None:
            self.label_session_details.setText(
                f"{payload.get('Status', 'Pending')}  |  "
                f"{payload.get('Experiment', '').strip() or 'No experiment'}  |  "
                f"{payload.get('Condition', '').strip() or 'No condition'}  |  "
                f"{payload.get('Arena', '').strip() or 'No arena'}"
            )

        self._refresh_recording_session_summary()

        if announce:
            self._on_status_update(f"Loaded planner trial {payload.get('Trial', '?')} into the session form.")

    def _on_planner_selection_changed(self):
        """Make the selected planner row the active recording trial."""
        if self.planner_table is None:
            return
        selected_rows = self.planner_table.selectionModel().selectedRows()
        if selected_rows:
            self._load_planner_row_into_metadata(selected_rows[0].row(), announce=False)
        self._update_planner_summary()

    def _apply_selected_planner_trial(self):
        """Load the selected planner row into the active metadata/session form."""
        if self.planner_table is None:
            return
        selected_rows = self.planner_table.selectionModel().selectedRows()
        if not selected_rows:
            self._on_error_occurred("Select a trial row to load into the session form.")
            return
        self._load_planner_row_into_metadata(selected_rows[0].row(), announce=True)

    def _update_planner_summary(self):
        """Refresh the footer summary for the planner dock."""
        if self.planner_table is None:
            return
        self._refresh_session_metrics()
        selected_rows = self.planner_table.selectionModel().selectedRows()
        total_rows = self.planner_table.rowCount()
        if not selected_rows:
            self.label_planner_summary.setText(f"{total_rows} planned trial(s)")
            if self.label_session_summary is not None:
                self.label_session_summary.setText("No trial selected")
            if self.label_session_details is not None:
                self.label_session_details.setText("Select or edit a planner row, then record directly from that plan.")
            self._refresh_recording_session_summary()
            return
        payload = self._planner_row_payload(selected_rows[0].row())
        self.label_planner_summary.setText(
            f"{payload.get('Status', 'Pending')}  |  Trial {payload.get('Trial', '?')}  |  "
            f"{payload.get('Animal ID', 'No subject')}  |  {payload.get('Experiment', 'No experiment')}"
        )
        self._refresh_recording_session_summary()

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

        self.check_preview_enabled.blockSignals(True)
        self.check_preview_enabled.setChecked(int(self.settings.value('preview_enabled', 1)) == 1)
        self.check_preview_enabled.blockSignals(False)

        self.spin_preview_fps.blockSignals(True)
        self.spin_preview_fps.setValue(float(self.settings.value('preview_fps', 25.0)))
        self.spin_preview_fps.blockSignals(False)

        self.spin_preview_width.blockSignals(True)
        self.spin_preview_width.setValue(int(self.settings.value('preview_width', 1280)))
        self.spin_preview_width.blockSignals(False)

        self.spin_frame_buffer.blockSignals(True)
        self.spin_frame_buffer.setValue(int(self.settings.value('frame_buffer_size', 128)))
        self.spin_frame_buffer.blockSignals(False)

        self.spin_metadata_stats_interval.blockSignals(True)
        self.spin_metadata_stats_interval.setValue(int(self.settings.value('metadata_stats_interval', 25)))
        self.spin_metadata_stats_interval.blockSignals(False)

        self.spin_hours.setValue(int(self.settings.value('max_hours', 0)))
        self.spin_minutes.setValue(int(self.settings.value('max_minutes', 5)))
        self.spin_seconds.setValue(int(self.settings.value('max_seconds', 0)))

        unlimited = int(self.settings.value('max_unlimited', 0))
        self.check_unlimited.setCurrentIndex(1 if unlimited else 0)

        self.edit_save_folder.setText(self.last_save_folder)
        if hasattr(self, "label_file_save_folder") and self.label_file_save_folder is not None:
            self.label_file_save_folder.setText(self.last_save_folder)

        self._load_line_label_settings()
        self._load_behavior_panel_settings()
        self._load_recording_form_state()
        self._set_filename_order_controls()

        metadata_visible = int(self.settings.value("metadata_panel_visible", 1))
        if metadata_visible:
            self._toggle_side_panel("left", "session", "Metadata and Planner")
        else:
            if self.metadata_dock is not None:
                self.metadata_dock.setVisible(False)
        self.btn_toggle_metadata.setText("Session")

        self.spin_fps.valueChanged.connect(lambda v: self._save_ui_setting('camera_fps', v))
        self.spin_exposure.valueChanged.connect(lambda v: self._save_ui_setting('exposure_ms', v))
        self.spin_width.valueChanged.connect(lambda v: self._save_ui_setting('camera_width', v))
        self.spin_height.valueChanged.connect(lambda v: self._save_ui_setting('camera_height', v))
        self.combo_encoder.currentIndexChanged.connect(lambda v: self._save_ui_setting('encoder_index', v))
        self.combo_image_format.currentTextChanged.connect(lambda v: self._save_ui_setting('image_format', v))
        self.check_preview_enabled.toggled.connect(lambda v: self._save_ui_setting('preview_enabled', 1 if v else 0))
        self.spin_preview_fps.valueChanged.connect(lambda v: self._save_ui_setting('preview_fps', v))
        self.spin_preview_width.valueChanged.connect(lambda v: self._save_ui_setting('preview_width', v))
        self.spin_frame_buffer.valueChanged.connect(lambda v: self._save_ui_setting('frame_buffer_size', v))
        self.spin_metadata_stats_interval.valueChanged.connect(lambda v: self._save_ui_setting('metadata_stats_interval', v))
        self.spin_hours.valueChanged.connect(lambda v: self._save_ui_setting('max_hours', v))
        self.spin_minutes.valueChanged.connect(lambda v: self._save_ui_setting('max_minutes', v))
        self.spin_seconds.valueChanged.connect(lambda v: self._save_ui_setting('max_seconds', v))
        self.check_unlimited.currentIndexChanged.connect(lambda v: self._save_ui_setting('max_unlimited', 1 if v == 1 else 0))
        self._update_filename_preview()
        self._update_planner_summary()
        self._on_recording_length_controls_changed()
        self._update_preview_control_state()

    @Slot()
    def _toggle_metadata_panel(self):
        """Toggle the merged session panel from the status bar."""
        self._toggle_side_panel("left", "session", "Metadata and Planner")

    def _load_line_label_settings(self):
        """Load saved camera input label selections."""
        label_defaults = {
            1: self.settings.value('line_label_1', 'None'),
            2: self.settings.value('line_label_2', 'None'),
            3: self.settings.value('line_label_3', 'None'),
            4: self.settings.value('line_label_4', 'None'),
        }
        self._refresh_line_label_combo_options()
        for line, value in label_defaults.items():
            combo = getattr(self, f"combo_line{line}_label", None)
            if not combo:
                continue
            value = "Sync" if str(value) == "TTL 1Hz" else str(value)
            combo.blockSignals(True)
            combo.setCurrentText(value if value else "None")
            combo.blockSignals(False)

        self._apply_line_label_map_to_worker()

    def _save_ui_setting(self, key: str, value):
        """Persist a UI setting."""
        self.settings.setValue(key, value)

    def _load_live_detection_settings(self):
        """Restore live-detection model, ROI, rule, and DO mapping settings."""
        if self.live_detection_panel is None:
            return

        config_payload = {
            "model_key": str(self.settings.value("live_model_key", "rfdetr-seg-medium")),
            "checkpoint_path": str(self.settings.value("live_checkpoint_path", "") or ""),
            "threshold": float(self.settings.value("live_threshold", 0.35)),
            "selected_class_ids": self._parse_int_csv(self.settings.value("live_selected_classes", "0")),
            "identity_mode": str(self.settings.value("live_identity_mode", "tracker")),
            "expected_mouse_count": int(self.settings.value("live_expected_mouse_count", 1)),
        }

        model_index = self.live_detection_panel.combo_model_key.findData(config_payload["model_key"])
        if model_index >= 0:
            self.live_detection_panel.combo_model_key.setCurrentIndex(model_index)
        self.live_detection_panel.edit_checkpoint.setText(config_payload["checkpoint_path"])
        self.live_detection_panel.spin_threshold.setValue(config_payload["threshold"])
        self.live_detection_panel.edit_selected_classes.setText(
            ",".join(str(value) for value in config_payload["selected_class_ids"])
        )
        identity_index = self.live_detection_panel.combo_identity_mode.findData(config_payload["identity_mode"])
        if identity_index >= 0:
            self.live_detection_panel.combo_identity_mode.setCurrentIndex(identity_index)
        self.live_detection_panel.spin_expected_mice.setValue(config_payload["expected_mouse_count"])

        try:
            rois_payload = json.loads(str(self.settings.value("live_rois_json", "[]") or "[]"))
        except Exception:
            rois_payload = []
        self.live_rois = {}
        for entry in rois_payload:
            try:
                roi = BehaviorROI.from_dict(entry)
            except Exception:
                continue
            self.live_rois[roi.name] = roi

        try:
            rules_payload = json.loads(str(self.settings.value("live_rules_json", "[]") or "[]"))
        except Exception:
            rules_payload = []
        self.live_rules = []
        for entry in rules_payload:
            try:
                self.live_rules.append(LiveTriggerRule.from_dict(entry))
            except Exception:
                continue

        try:
            output_mapping_payload = json.loads(str(self.settings.value("live_output_map_json", "{}") or "{}"))
        except Exception:
            output_mapping_payload = {}
        self.live_output_mapping = self._normalize_live_output_mapping(output_mapping_payload)

        self.live_rule_engine.set_rois(self.live_rois)
        self.live_rule_engine.set_rules(self.live_rules)
        self.live_detection_panel.set_output_mapping(self.live_output_mapping)
        self.live_detection_panel.set_status("Idle")
        self._refresh_live_panel_state()

        if self.arduino_worker is not None:
            try:
                self.arduino_worker.configure_live_output_mapping(self.live_output_mapping)
            except Exception as exc:
                self._on_error_occurred(str(exc))

    def _persist_live_detection_settings(self):
        if self.live_detection_panel is None:
            return
        config = self.live_detection_panel.detection_config()
        self.settings.setValue("live_model_key", config["model_key"])
        self.settings.setValue("live_checkpoint_path", config["checkpoint_path"])
        self.settings.setValue("live_threshold", float(config["threshold"]))
        self.settings.setValue(
            "live_selected_classes",
            ",".join(str(value) for value in config["selected_class_ids"]),
        )
        self.settings.setValue("live_identity_mode", config["identity_mode"])
        self.settings.setValue("live_expected_mouse_count", int(config["expected_mouse_count"]))
        self.settings.setValue(
            "live_rois_json",
            json.dumps([roi.to_dict() for roi in self.live_rois.values()]),
        )
        self.settings.setValue(
            "live_rules_json",
            json.dumps([rule.to_dict() for rule in self.live_rules]),
        )
        self.settings.setValue("live_output_map_json", json.dumps(self.live_output_mapping))
        self.settings.sync()

    def _parse_int_csv(self, raw_value) -> List[int]:
        values: List[int] = []
        raw_text = str(raw_value or "").replace(";", ",")
        for token in raw_text.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token))
            except ValueError:
                continue
        return values

    def _normalize_live_output_mapping(self, payload: Dict) -> Dict[str, List[int]]:
        normalized: Dict[str, List[int]] = {f"DO{i}": [] for i in range(1, 9)}
        for output_id, raw_pins in dict(payload or {}).items():
            key = str(output_id or "").strip().upper()
            if key not in normalized:
                continue
            pins: List[int] = []
            if isinstance(raw_pins, (list, tuple)):
                source = raw_pins
            else:
                source = str(raw_pins or "").replace(";", ",").split(",")
            for entry in source:
                try:
                    pins.append(int(str(entry).strip()))
                except Exception:
                    continue
            normalized[key] = pins
        return normalized

    def _build_live_inference_config(self) -> LiveInferenceConfig:
        config = self.live_detection_panel.detection_config() if self.live_detection_panel else {}
        return LiveInferenceConfig(
            model_key=str(config.get("model_key", "rfdetr-seg-medium")),
            checkpoint_path=str(config.get("checkpoint_path", "") or ""),
            threshold=float(config.get("threshold", 0.35)),
            selected_class_ids=list(config.get("selected_class_ids", [])),
            identity_mode=str(config.get("identity_mode", "tracker")),
            expected_mouse_count=max(1, int(config.get("expected_mouse_count", 1))),
        )

    @Slot(object)
    def _on_preview_packet_ready(self, packet: object):
        if not isinstance(packet, PreviewFramePacket):
            return
        self.live_preview_packet = packet
        if self.live_detection_enabled and self.live_inference_worker is not None:
            self.live_inference_worker.submit_preview(packet)

    @Slot(bool)
    def _on_live_detection_toggled(self, enabled: bool):
        if self.live_detection_panel is None or self.live_inference_worker is None:
            return

        if enabled:
            if not self.check_preview_enabled.isChecked():
                self._on_error_occurred("Enable preview before starting live detection.")
                self.live_detection_panel.set_detection_running(False)
                return
            self.live_detection_enabled = True
            self.live_rule_engine.clear_runtime_state()
            self.live_inference_worker.start_inference(self._build_live_inference_config())
            self._persist_live_detection_settings()
            if self.arduino_worker is not None:
                try:
                    self.arduino_worker.configure_live_output_mapping(self.live_output_mapping)
                except Exception as exc:
                    self._on_error_occurred(str(exc))
            self.live_detection_panel.set_status("Waiting for frames")
            if self.live_preview_packet is not None:
                self.live_inference_worker.submit_preview(self.live_preview_packet)
            return

        self.live_detection_enabled = False
        self.live_active_rule_ids = []
        self.live_output_states = {f"DO{i}": False for i in range(1, 9)}
        if self.live_inference_worker is not None:
            self.live_inference_worker.stop_inference()
        if self.arduino_worker is not None:
            self.arduino_worker.clear_live_outputs()
        self.live_rule_engine.clear_runtime_state()
        self.live_detection_panel.set_status("Idle")
        self._refresh_live_panel_state()
        self._persist_live_detection_settings()

    @Slot(str)
    def _on_live_detection_status_changed(self, message: str):
        if self.live_detection_panel is not None:
            self.live_detection_panel.set_status(message)
        self._on_status_update(message)

    @Slot(object)
    def _on_live_detection_result(self, result: object):
        if not isinstance(result, LiveDetectionResult):
            return
        self.live_detection_last_result = result
        now_ms = int(result.timestamp_s * 1000.0)
        evaluation = self.live_rule_engine.evaluate(result, now_ms)
        self.live_active_rule_ids = list(evaluation.active_rule_ids)
        self.live_output_states = dict(evaluation.output_states)
        if self.arduino_worker is not None and self.is_arduino_connected:
            for output_id, state in self.live_output_states.items():
                self.arduino_worker.set_live_output_level(output_id, bool(state))
            for output_id, duration_ms in evaluation.triggered_pulses:
                self.arduino_worker.start_live_output_pulse(output_id, int(duration_ms))
        if self.live_detection_panel is not None:
            self.live_detection_panel.set_status(
                f"{len(result.tracked_mice)} mice, {result.inference_ms:.1f} ms"
            )
        self._refresh_live_panel_state()

    @Slot(dict)
    def _apply_live_output_mapping(self, mapping: Dict):
        self.live_output_mapping = self._normalize_live_output_mapping(mapping)
        if self.arduino_worker is not None:
            try:
                self.arduino_worker.configure_live_output_mapping(self.live_output_mapping)
            except Exception as exc:
                self._on_error_occurred(str(exc))
                return
        if self.live_detection_panel is not None:
            self.live_detection_panel.set_output_mapping(self.live_output_mapping)
        self._persist_live_detection_settings()

    @Slot(object)
    def _add_live_rule(self, payload: object):
        rule = payload if isinstance(payload, LiveTriggerRule) else None
        if rule is None:
            return
        if rule.rule_type == "roi_occupancy" and rule.roi_name not in self.live_rois:
            self._on_error_occurred(f"Unknown ROI: {rule.roi_name}")
            return
        self.live_rules.append(rule)
        self.live_rule_engine.set_rules(self.live_rules)
        self._refresh_live_panel_state()
        self._persist_live_detection_settings()

    @Slot(str)
    def _remove_live_rule(self, rule_id: str):
        self.live_rules = [rule for rule in self.live_rules if rule.rule_id != rule_id]
        self.live_rule_engine.set_rules(self.live_rules)
        self._refresh_live_panel_state()
        self._persist_live_detection_settings()

    def _refresh_live_panel_state(self):
        if self.live_detection_panel is None:
            return
        self.live_rule_engine.set_rois(self.live_rois)
        self.live_detection_panel.set_rois(self.live_rois)
        self.live_detection_panel.set_rules(self.live_rules, self.live_active_rule_ids)
        self.live_detection_panel.set_active_outputs(self.live_output_states)

    def _line_label_choice_list(self) -> List[str]:
        """Return suggested editable labels for camera line assignments."""
        choices = ["None"]
        seen = {"none"}

        for key in self.DISPLAY_SIGNAL_ORDER:
            label = self._signal_label(key).strip()
            lowered = label.lower()
            if label and lowered not in seen:
                choices.append(label)
                seen.add(lowered)

        for line in range(1, 5):
            saved = str(self.settings.value(f"line_label_{line}", "None")).strip()
            lowered = saved.lower()
            if saved and lowered not in seen:
                choices.append(saved)
                seen.add(lowered)

        raw_catalog = str(self.settings.value("camera_line_label_catalog", "") or "")
        for token in [value.strip() for value in raw_catalog.split("|")]:
            lowered = token.lower()
            if token and lowered not in seen:
                choices.append(token)
                seen.add(lowered)

        return choices

    def _refresh_line_label_combo_options(self):
        """Refresh editable line-label suggestions without losing current values."""
        choices = self._line_label_choice_list()
        for line in range(1, 5):
            combo = getattr(self, f"combo_line{line}_label", None)
            if combo is None:
                continue
            current = combo.currentText().strip()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(choices)
            combo.setCurrentText(current if current else "None")
            combo.blockSignals(False)

    def _load_camera_line_defaults(self) -> Dict[str, Dict[str, str]]:
        """Load persisted line-mode/source defaults keyed by camera selector name."""
        raw_value = str(self.settings.value("camera_line_defaults_json", "") or "").strip()
        if not raw_value:
            return {}
        try:
            payload = json.loads(raw_value)
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}

        normalized: Dict[str, Dict[str, str]] = {}
        for selector, config in payload.items():
            if not isinstance(config, dict):
                continue
            normalized[str(selector)] = {
                "label": str(config.get("label", "")).strip(),
                "mode": str(config.get("mode", "")).strip(),
                "source": str(config.get("source", "")).strip(),
            }
        return normalized

    def _save_camera_line_defaults(self, line_defaults: Dict[str, Dict[str, str]]):
        """Persist line defaults and keep the editable label catalog in sync."""
        serializable = {}
        label_catalog = set()
        for selector, config in (line_defaults or {}).items():
            selector_name = str(selector).strip()
            if not selector_name or not isinstance(config, dict):
                continue
            label = str(config.get("label", "")).strip()
            mode = str(config.get("mode", "")).strip()
            source = str(config.get("source", "")).strip()
            serializable[selector_name] = {"label": label, "mode": mode, "source": source}
            if label:
                label_catalog.add(label)

        self.settings.setValue("camera_line_defaults_json", json.dumps(serializable))
        self.settings.setValue("camera_line_label_catalog", "|".join(sorted(label_catalog, key=str.lower)))

    def _update_line_label_catalog_from_ui(self):
        """Persist custom line-label suggestions from the current UI selections."""
        catalog = set()
        for line in range(1, 5):
            combo = getattr(self, f"combo_line{line}_label", None)
            if combo is None:
                continue
            value = combo.currentText().strip()
            if value and value.lower() != "none":
                catalog.add(value)

        raw_catalog = str(self.settings.value("camera_line_label_catalog", "") or "")
        for token in [value.strip() for value in raw_catalog.split("|")]:
            if token and token.lower() != "none":
                catalog.add(token)

        self.settings.setValue("camera_line_label_catalog", "|".join(sorted(catalog, key=str.lower)))

    def _camera_line_entries_for_dialog(self) -> List[Dict[str, object]]:
        """Build camera-line rows for the defaults dialog from live camera capabilities when available."""
        saved_defaults = self._load_camera_line_defaults()
        live_capabilities = []
        if self.worker is not None and self.is_camera_connected and self.worker.is_genicam_camera():
            try:
                live_capabilities = self.worker.get_camera_line_capabilities()
            except Exception:
                live_capabilities = []

        entries: List[Dict[str, object]] = []
        if live_capabilities:
            for index, capability in enumerate(live_capabilities[:4], start=1):
                selector = str(capability.get("selector", f"Line{index}"))
                saved = saved_defaults.get(selector, {})
                entries.append({
                    "selector": selector,
                    "display_name": selector,
                    "label": str(saved.get("label", "")).strip() or str(self.settings.value(f"line_label_{index}", "None")),
                    "mode": str(saved.get("mode", "")).strip() or str(capability.get("mode", "")).strip(),
                    "mode_options": [str(value) for value in capability.get("mode_options", []) if str(value).strip()],
                    "source": str(saved.get("source", "")).strip() or str(capability.get("source", "")).strip(),
                    "source_options": [str(value) for value in capability.get("source_options", []) if str(value).strip()],
                    "live": True,
                    "line_number": index,
                })
            return entries

        for index in range(1, 5):
            entries.append({
                "selector": f"Line{index}",
                "display_name": f"Line {index}",
                "label": str(self.settings.value(f"line_label_{index}", "None")),
                "mode": "",
                "mode_options": [],
                "source": "",
                "source_options": [],
                "live": False,
                "line_number": index,
            })
        return entries

    def _apply_saved_camera_line_defaults(self):
        """Apply persisted line-mode/source defaults to the connected GenICam camera."""
        if self.worker is None or not self.is_camera_connected or not self.worker.is_genicam_camera():
            return

        saved_defaults = self._load_camera_line_defaults()
        if not saved_defaults:
            return

        capabilities = self.worker.get_camera_line_capabilities()
        pending_configs = []
        for capability in capabilities[:4]:
            selector = str(capability.get("selector", "")).strip()
            if not selector:
                continue
            saved = saved_defaults.get(selector, {})
            mode = str(saved.get("mode", "")).strip()
            source = str(saved.get("source", "")).strip()
            config = {"selector": selector}
            include = False
            if mode and mode in capability.get("mode_options", []):
                config["mode"] = mode
                include = True
            if source and source in capability.get("source_options", []):
                config["source"] = source
                include = True
            if include:
                pending_configs.append(config)

        if pending_configs:
            self.worker.apply_camera_line_configuration(pending_configs)

    def _on_line_label_changed(self, line_number: int, value: str):
        """Handle camera input label changes."""
        normalized = str(value).strip() or "None"
        self._save_ui_setting(f'line_label_{line_number}', normalized)
        self._update_line_label_catalog_from_ui()
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
        text = str(label).strip()
        if not text or text.lower() == "none":
            return ""
        if text == "TTL 1Hz":
            text = "Sync"
        return self._slugify_export_label(text, "line")

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
            'trial': self.meta_trial.text(),
            'experiment': self.meta_experiment.text(),
            'condition': self.meta_condition.text(),
            'arena': self.meta_arena.text(),
            'date': self.meta_date.text(),
            'notes': self.meta_notes.toPlainText(),
            'timestamp': datetime.now().isoformat(),
            'filename_preview': self._compose_recording_basename(),
            'filename_order': self._selected_filename_order(),
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

    def _frame_drop_reference_fps(self) -> float:
        """Return the currently configured FPS reference for frame-drop display."""
        if hasattr(self, "spin_fps") and self.spin_fps is not None:
            try:
                return float(self.spin_fps.value())
            except Exception:
                pass
        return float(self.default_fps)

    def _reset_frame_drop_display(self, recording_active: bool = False):
        """Reset the compact frame-drop monitor for a new or idle session."""
        self.frame_drop_events.clear()
        self.last_frame_drop_log_signature = None
        reference_fps = self._frame_drop_reference_fps()
        state = "REC" if recording_active else "Standby"
        self.last_frame_drop_stats = {
            "active": recording_active,
            "recorded_frames": 0,
            "estimated_dropped_frames": 0,
            "estimated_total_frames": 0,
            "drop_percent": 0.0,
            "reference_fps": reference_fps,
            "average_interval_ms": 0.0,
            "max_gap_ms": 0.0,
            "last_interval_ms": 0.0,
            "elapsed_seconds": 0.0,
            "timestamp_source": "software",
        }

        if self.label_frame_drop_summary is not None:
            summary_color = "#89f0b2" if recording_active else "#dce8f4"
            self.label_frame_drop_summary.setStyleSheet(f"color: {summary_color}; font-weight: 700;")
            self.label_frame_drop_summary.setText(
                f"{state} | drop 0 (0.00%) | frames 0 | ref {reference_fps:.1f} fps"
            )

        if self.frame_drop_log is not None:
            initial_line = (
                "00:00:00 | monitoring armed"
                if recording_active
                else "Idle | stats are saved to *_metadata.txt after recording"
            )
            self.frame_drop_log.setPlainText(initial_line)

    def _save_recording_text_metadata(self, base_path: str):
        """Write a human-readable metadata summary with frame-drop statistics."""
        metadata_txt_file = Path(f"{base_path}_metadata.txt")
        metadata = dict(self.metadata) if self.metadata else self._collect_metadata()
        stats = dict(self.last_frame_drop_stats)
        if not stats and self.worker:
            stats = dict(self.worker.last_recording_stats)

        try:
            lines = [
                "CamApp Live Detection Metadata Summary",
                f"generated_at: {datetime.now().isoformat()}",
                f"recording_base_path: {base_path}",
                "",
                "Session Metadata",
            ]

            for key, value in metadata.items():
                value_text = "" if value is None else str(value)
                value_lines = value_text.splitlines() or [""]
                if len(value_lines) == 1:
                    lines.append(f"{key}: {value_lines[0]}")
                else:
                    lines.append(f"{key}:")
                    for value_line in value_lines:
                        lines.append(f"  {value_line}")

            lines.extend(
                [
                    "",
                    "Frame Drop Statistics",
                    f"reference_fps: {float(stats.get('reference_fps', 0.0)):.3f}",
                    f"recorded_frames: {int(stats.get('recorded_frames', 0))}",
                    f"estimated_dropped_frames: {int(stats.get('estimated_dropped_frames', 0))}",
                    f"estimated_total_frames: {int(stats.get('estimated_total_frames', 0))}",
                    f"drop_percent: {float(stats.get('drop_percent', 0.0)):.3f}",
                    f"average_interval_ms: {float(stats.get('average_interval_ms', 0.0)):.3f}",
                    f"max_gap_ms: {float(stats.get('max_gap_ms', 0.0)):.3f}",
                    f"last_interval_ms: {float(stats.get('last_interval_ms', 0.0)):.3f}",
                    f"timestamp_source: {stats.get('timestamp_source', 'software')}",
                    "method: estimated from software timestamp gaps against the recording reference fps",
                ]
            )

            with open(metadata_txt_file, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            self._on_status_update(f"Metadata text saved: {metadata_txt_file.name}")
        except Exception as exc:
            self._on_error_occurred(f"Metadata text save error: {exc}")

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
                self._set_button_icon(self.btn_connect, "record", "#ffffff", "dangerButton")
                self.btn_record.setEnabled(True)
                self._save_ui_setting('last_camera_type', camera_info.get('type', ''))
                self._save_ui_setting('last_camera_backend', camera_info.get('backend', ''))
                self._save_ui_setting('last_camera_index', camera_info.get('index', ''))
                camera_name = self.combo_camera.currentText().strip() or "Camera"
                self.label_camera_source_hint.setText(f"Connected: {camera_name}")
                self.label_recording_camera_hint.setText(f"Ready to record from {camera_name}.")
                self._update_live_header(
                    status_text=f"{camera_info.get('type', 'camera').upper()} online",
                    badge_text="Preview",
                    badge_tone="accent",
                )

                # Apply initial settings
                self._on_fps_changed(self.spin_fps.value())
                self._on_exposure_changed(self.spin_exposure.value())
                self._on_resolution_changed()
                if (
                    camera_info.get('backend') == "spinnaker"
                    and self.worker is not None
                    and getattr(self.worker, "spinnaker_is_color", False)
                    and self.combo_image_format.currentText() != "BGR8"
                ):
                    self.combo_image_format.setCurrentText("BGR8")
                self._on_image_format_changed(self.combo_image_format.currentText())
                self._apply_saved_camera_line_defaults()

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
        self._set_button_icon(self.btn_connect, "play", "#eef6ff")
        self.btn_record.setEnabled(False)
        self.label_camera_source_hint.setText("No source connected")
        self.label_recording_camera_hint.setText("Camera source is managed from the left Camera panel.")

        self._clear_roi()
        self._show_live_placeholder("Camera Disconnected", "Reconnect a Basler, FLIR, or USB source")
        self._update_live_header(
            status_text="No camera connected",
            resolution_text="-- x --",
            badge_text="Offline",
            badge_tone="warning",
        )
        self._reset_frame_drop_display()
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
            if self.planner_table is not None and self.planner_table.rowCount() > 0:
                selected_rows = self.planner_table.selectionModel().selectedRows()
                if selected_rows:
                    # Preserve the duration currently shown in the recording controls.
                    self._load_planner_row_into_metadata(
                        selected_rows[0].row(),
                        announce=False,
                        apply_duration=False,
                    )

            filename = self._compose_recording_basename()

            if not filename:
                self._on_error_occurred("Please enter metadata or filename")
                return

            save_folder = Path(self.edit_save_folder.text())
            filepath = str(self._get_unique_recording_path(save_folder, filename))
            self.current_recording_filepath = filepath
            self.edit_filename.setText(Path(filepath).name)
            self.active_planner_row = self._find_planner_row_for_current_session()
            self._sync_active_trial_status("Acquiring")

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
            self.worker.set_recording_frame_limit(None)
            self._reset_frame_drop_display(recording_active=True)

            if self.is_arduino_connected:
                self._apply_behavior_pin_configuration(persist=True)
                if not self.arduino_worker.start_recording():
                    self._on_status_update("Warning: Arduino TTLs failed to start; recording will continue.")
                    self._set_ttl_status("START FAILED", "warning")
                    self._set_behavior_status("IDLE", "default")
                else:
                    # Reset and clear plot for new recording
                    self._reset_ttl_plot()

                    self._set_ttl_status("RECORDING", "danger")
                    self._set_behavior_status("ARMED", "accent")

            if not self.worker.start_recording(filepath):
                self._reset_frame_drop_display()
                if self.is_arduino_connected:
                    self._stop_arduino_generation()
                self._sync_active_trial_status("Pending")
                self.current_recording_filepath = None
                return

            self._apply_recording_frame_limit()

            self.recording_start_time = datetime.now()
            self.recording_timer.start(1000)

            self.btn_record.setText("Stop Recording")
            self._set_button_icon(self.btn_record, "record", "#ffffff", "dangerButton")
            self.label_recording.setText("Recording")
            self.label_recording.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            self._update_live_header(badge_text="REC", badge_tone="danger")

            self.btn_connect.setEnabled(False)
            self.edit_filename.setEnabled(False)
            if self.is_arduino_connected:
                self.btn_test_ttl.setEnabled(False)
        else:
            self.worker.stop_recording()
            self.recording_timer.stop()

            if self.is_arduino_connected:
                self._stop_arduino_generation()

    def _focused_widget_blocks_space_record(self) -> bool:
        """Return True when space should remain with the currently focused editor."""
        widget = self.focusWidget()
        if widget is None:
            return False
        return isinstance(
            widget,
            (
                QLineEdit,
                QTextEdit,
                QSpinBox,
                QDoubleSpinBox,
                QComboBox,
                QTableWidget,
            ),
        )

    @Slot()
    def _on_space_record_shortcut(self):
        """Start recording from the keyboard without stealing spaces from editors."""
        if self._focused_widget_blocks_space_record():
            return
        if not self.is_camera_connected or self.worker is None:
            return
        if self.worker.is_recording:
            return
        if self.btn_record is not None and self.btn_record.isEnabled():
            self._on_record_clicked()

    @Slot()
    def _on_recording_stopped(self):
        """Handle recording stopped signal."""
        self.btn_record.setText("Start Recording")
        self._set_button_icon(self.btn_record, "record", "#07260e", "successButton")
        self.label_recording.setText("Not Recording")
        self.label_recording.setStyleSheet("")
        self.label_recording_time.setText("00:00:00")
        self.recording_timer.stop()
        self.recording_start_time = None
        self._update_live_header(badge_text="Preview" if self.is_camera_connected else "Offline",
                                 badge_tone="accent" if self.is_camera_connected else "warning")
        filepath = self.current_recording_filepath
        if not filepath and self.worker and self.worker.recording_filename:
            filepath = self.worker.recording_filename

        # Re-enable controls
        self.btn_connect.setEnabled(True)
        self.edit_filename.setEnabled(True)
        if self.is_arduino_connected:
            self.btn_test_ttl.setEnabled(True)

        if filepath:
            self._save_recording_text_metadata(filepath)

        if self.is_arduino_connected:
            self._stop_arduino_generation()

            if not filepath:
                save_folder = Path(self.edit_save_folder.text())
                fallback_name = self.edit_filename.text().strip() or "recording"
                filepath = str(self._get_unique_recording_path(save_folder, fallback_name))

            # Save TTL history
            self._save_arduino_ttl_data(filepath)
            self.current_recording_filepath = None

            # Reset TTL status
            self._set_ttl_status("IDLE", "default")
            self._set_behavior_status("IDLE", "default")

        self._sync_active_trial_status("Acquired")
        self.current_recording_filepath = None
        self._update_filename_preview()

    def _update_recording_time(self):
        """Update recording time display."""
        if self.recording_start_time:
            elapsed_seconds = self._current_recording_elapsed_seconds()
            elapsed_text = self._format_duration_hms(elapsed_seconds)
            remaining_seconds = self._current_recording_remaining_seconds()
            if remaining_seconds is None:
                self.label_recording_time.setText(elapsed_text)
            else:
                remaining_text = self._format_duration_hms(remaining_seconds)
                self.label_recording_time.setText(f"{elapsed_text} | {remaining_text} left")

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
            f"{base_path}_metadata.txt",
            f"{base_path}_ttl_states.csv",
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
            if hasattr(self, "label_file_save_folder") and self.label_file_save_folder is not None:
                self.label_file_save_folder.setText(folder)
            self.settings.setValue('last_save_folder', folder)

    # ===== Camera Settings Slots =====

    @Slot(float)
    def _on_fps_changed(self, value):
        """Handle FPS change."""
        if not self.worker:
            return
        actual_fps = None
        actual_exposure_ms = None
        throughput_suffix = ""
        try:
            if self.worker.camera_type == "flir" and self.worker.flir_camera:
                flir_cap = getattr(self.worker.flir_camera, "cap", None)
                if flir_cap is not None:
                    flir_cap.set(cv2.CAP_PROP_FPS, float(value))
                    self.worker.set_target_fps(value)
            else:
                actual_fps = self.worker.set_camera_frame_rate(value)
                if actual_fps is None and not self.worker.is_genicam_camera():
                    self.worker.set_target_fps(value)
                actual_exposure_ms = self.worker.get_camera_exposure_ms()

            if actual_fps and abs(float(actual_fps) - float(value)) > 0.01:
                self.spin_fps.blockSignals(True)
                self.spin_fps.setValue(float(actual_fps))
                self.spin_fps.blockSignals(False)

            if actual_exposure_ms is not None and abs(float(actual_exposure_ms) - float(self.spin_exposure.value())) > 0.01:
                self.spin_exposure.blockSignals(True)
                self.spin_exposure.setValue(float(actual_exposure_ms))
                self.spin_exposure.blockSignals(False)

            if self.worker.is_spinnaker_camera():
                throughput_limit = self.worker._read_numeric_node("DeviceLinkThroughputLimit")
                throughput_max = self.worker._read_numeric_node("DeviceMaxThroughput")
                if (
                    throughput_limit is not None
                    and throughput_max is not None
                    and throughput_limit > 0
                    and throughput_max > throughput_limit
                ):
                    throughput_suffix = f", transport limit {int(throughput_limit)}"

            if actual_fps and abs(float(actual_fps) - float(value)) > 0.01:
                message = f"FPS set to {float(actual_fps):.3f} (requested {value:.3f})"
                if actual_exposure_ms is not None:
                    message += f", exposure {float(actual_exposure_ms):.2f} ms"
                self._on_status_update(message + throughput_suffix)
            else:
                self._on_status_update(f"FPS set to {value:.3f}{throughput_suffix}")
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
                self.spin_width.blockSignals(True)
                self.spin_height.blockSignals(True)
                self.spin_width.setValue(actual_width)
                self.spin_height.setValue(actual_height)
                self.spin_width.blockSignals(False)
                self.spin_height.blockSignals(False)
                self._on_status_update(f"Resolution set to {actual_width}x{actual_height}")
                self._update_live_header(resolution_text=f"{actual_width} x {actual_height}")
                self._update_advanced_controls_state()
            except Exception as e:
                self._on_error_occurred(f"Failed to set resolution: {str(e)}")
            return

        if self.worker.camera_type == "flir" and self.worker.flir_camera:
            flir_cap = getattr(self.worker.flir_camera, "cap", None)
            if flir_cap is None:
                self._on_status_update("Resolution is fixed for this FLIR backend")
                return
            try:
                width = int(self.spin_width.value())
                height = int(self.spin_height.value())
                flir_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                flir_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(flir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(flir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.worker.update_resolution(actual_width, actual_height)
                self.spin_width.blockSignals(True)
                self.spin_height.blockSignals(True)
                self.spin_width.setValue(actual_width)
                self.spin_height.setValue(actual_height)
                self.spin_width.blockSignals(False)
                self.spin_height.blockSignals(False)
                self._on_status_update(f"Resolution set to {actual_width}x{actual_height}")
                self._update_live_header(resolution_text=f"{actual_width} x {actual_height}")
                self._update_advanced_controls_state()
            except Exception as e:
                self._on_error_occurred(f"Failed to set resolution: {str(e)}")
            return

        if self.worker and self.worker.is_genicam_camera():
            try:
                applied = self.worker.set_camera_resolution(
                    int(self.spin_width.value()),
                    int(self.spin_height.value()),
                )
                if applied is None:
                    raise RuntimeError("Width/Height not supported by camera")
                width, height = applied
                self.spin_width.blockSignals(True)
                self.spin_height.blockSignals(True)
                self.spin_width.setValue(int(width))
                self.spin_height.setValue(int(height))
                self.spin_width.blockSignals(False)
                self.spin_height.blockSignals(False)
                status_message = f"Resolution set to {width}x{height}"
                actual_fps = self.worker.sync_camera_fps()
                if actual_fps is not None:
                    self.spin_fps.blockSignals(True)
                    self.spin_fps.setValue(float(actual_fps))
                    self.spin_fps.blockSignals(False)
                    status_message += f", current FPS {float(actual_fps):.3f}"
                if self.worker.is_spinnaker_camera():
                    throughput_limit = self.worker._read_numeric_node("DeviceLinkThroughputLimit")
                    throughput_max = self.worker._read_numeric_node("DeviceMaxThroughput")
                    if (
                        throughput_limit is not None
                        and throughput_max is not None
                        and throughput_limit > 0
                        and throughput_max > throughput_limit
                    ):
                        status_message += f", transport limit {int(throughput_limit)}"
                self._on_status_update(status_message)
                self._update_live_header(resolution_text=f"{width} x {height}")
                self._update_advanced_controls_state()
            except Exception as e:
                self._on_error_occurred(f"Failed to set resolution: {str(e)}")

    @Slot(float)
    def _on_exposure_changed(self, value):
        """Handle exposure time change."""
        if self.worker and self.worker.is_genicam_camera():
            try:
                applied_ms = self.worker.set_camera_exposure_ms(value)
                if applied_ms is None:
                    raise RuntimeError("Exposure control not supported by camera")
                actual_fps = self.worker.sync_camera_fps()
                if actual_fps and abs(float(actual_fps) - float(self.spin_fps.value())) > 0.01:
                    self.spin_fps.blockSignals(True)
                    self.spin_fps.setValue(float(actual_fps))
                    self.spin_fps.blockSignals(False)
                    self._on_status_update(f"Exposure set to {applied_ms:.2f} ms, FPS now {float(actual_fps):.3f}")
                else:
                    self._on_status_update(f"Exposure set to {applied_ms:.2f} ms")
            except Exception as e:
                self._on_error_occurred(f"Failed to set exposure: {str(e)}")

    @Slot(str)
    def _on_image_format_changed(self, format_text: str):
        """Handle image format change."""
        if self.worker:
            self.worker.set_image_format(format_text)
            self._save_ui_setting('image_format', format_text)
        self._update_live_header(mode_text=format_text)

    def _update_preview_control_state(self):
        """Keep preview controls aligned with the preview enable toggle."""
        preview_enabled = self.check_preview_enabled.isChecked()
        self.spin_preview_fps.setEnabled(preview_enabled)
        self.spin_preview_width.setEnabled(preview_enabled)

    def _on_preview_enabled_changed(self, enabled: bool):
        """Enable or disable live preview while leaving acquisition running."""
        self._update_preview_control_state()
        if self.worker:
            self.worker.set_preview_enabled(enabled)
        if enabled:
            self._on_status_update(f"Preview enabled at {self.spin_preview_fps.value():.1f} fps")
        else:
            self._on_status_update("Preview disabled; acquisition and recording continue")

    def _on_preview_fps_changed(self, value: float):
        """Set the preview cadence independent of the recording rate."""
        if self.worker:
            self.worker.set_preview_fps(value)
        if self.check_preview_enabled.isChecked():
            self._on_status_update(f"Preview FPS target: {value:.1f}")

    def _on_preview_width_changed(self, value: int):
        """Set the preview downscale target before frames reach the GUI."""
        if self.worker:
            self.worker.set_preview_max_width(value)
        if value <= 0:
            self._on_status_update("Preview width: full resolution")
        else:
            self._on_status_update(f"Preview max width: {int(value)} px")

    def _on_frame_buffer_size_changed(self, value: int):
        """Resize the internal queue and camera-side stream buffers."""
        if self.worker:
            self.worker.set_frame_buffer_size(value)
        self._on_status_update(f"Frame buffer set to {int(value)} frames")

    def _on_metadata_stats_interval_changed(self, value: int):
        """Control how often raw frame statistics are computed."""
        if self.worker:
            self.worker.set_metadata_stats_interval(value)
        if value <= 0:
            self._on_status_update("Raw frame statistics disabled")
        else:
            self._on_status_update(f"Raw frame statistics every {int(value)} frames")

    def _toggle_advanced_settings(self):
        """Open or close the advanced camera popup."""
        if self.advanced_dialog is None:
            return
        if self.advanced_dialog.isVisible():
            self.advanced_dialog.close()
        else:
            self.advanced_dialog.show()
            self.advanced_dialog.raise_()
            self.advanced_dialog.activateWindow()

    def _update_advanced_controls_state(self):
        """Update advanced controls based on camera availability."""
        if not self.worker or not self.worker.is_genicam_camera():
            self._refresh_camera_native_format_controls()
            self._set_advanced_controls_enabled(False)
            return

        self._set_advanced_controls_enabled(True)

        self._configure_int_node("OffsetX", self.slider_offset_x, self.spin_offset_x)
        self._configure_int_node("OffsetY", self.slider_offset_y, self.spin_offset_y)

        self._configure_float_node("Gain", self.spin_gain)
        self._configure_white_balance_controls()
        if not self._configure_float_node("Brightness", self.spin_brightness):
            self._configure_float_node("BlackLevel", self.spin_brightness)
        if not self._configure_float_node("Contrast", self.spin_contrast):
            self._configure_float_node("Gamma", self.spin_contrast)
        self._refresh_camera_native_format_controls(apply_saved=True)

        if self.spin_offset_x.isEnabled() and self.spin_offset_y.isEnabled():
            if self.settings.contains('offset_x') and self.settings.contains('offset_y'):
                try:
                    self._on_offset_x_changed(int(self.settings.value('offset_x')))
                    self._on_offset_y_changed(int(self.settings.value('offset_y')))
                except Exception:
                    self._center_offsets()
            else:
                self._center_offsets()

        if self.spin_gain.isEnabled() and self.settings.contains('gain'):
            try:
                self._on_gain_changed(float(self.settings.value('gain')))
            except Exception:
                pass
        if self.spin_brightness.isEnabled() and self.settings.contains('brightness'):
            try:
                self._on_brightness_changed(float(self.settings.value('brightness')))
            except Exception:
                pass
        if self.spin_contrast.isEnabled() and self.settings.contains('contrast'):
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
            self.spin_gain,
            self.combo_white_balance_auto,
            self.spin_white_balance_red,
            self.spin_white_balance_blue,
            self.spin_brightness,
            self.spin_contrast,
            self.combo_camera_pixel_format,
            self.combo_camera_bit_depth,
        ):
            widget.setEnabled(enabled)

    def _refresh_camera_native_format_controls(self, apply_saved: bool = False):
        """Populate advanced camera-native pixel-format and bit-depth controls."""
        if not self.worker or not self.worker.is_genicam_camera():
            self.active_camera_pixel_format_node = ""
            self.active_camera_bit_depth_node = ""
            for combo, placeholder in (
                (self.combo_camera_pixel_format, "Unavailable"),
                (self.combo_camera_bit_depth, "Unavailable"),
            ):
                combo.blockSignals(True)
                combo.clear()
                combo.addItem(placeholder)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
                combo.setEnabled(False)
                combo.setToolTip("")
            return

        pixel_info = self.worker.get_camera_pixel_format_options()
        self.active_camera_pixel_format_node = str(pixel_info.get("node_name", "") or "")
        if apply_saved:
            saved_pixel_format = str(self.settings.value("camera_native_pixel_format", "") or "").strip()
            pixel_options = [str(value).strip() for value in pixel_info.get("options", []) if str(value).strip()]
            current_pixel_format = str(pixel_info.get("current", "") or "").strip()
            if (
                saved_pixel_format
                and saved_pixel_format in pixel_options
                and saved_pixel_format != current_pixel_format
                and bool(pixel_info.get("writable", False))
            ):
                applied = self.worker.set_camera_pixel_format(saved_pixel_format)
                if applied is not None:
                    pixel_info = self.worker.get_camera_pixel_format_options()
                    self.active_camera_pixel_format_node = str(pixel_info.get("node_name", "") or "")
        self._configure_camera_enum_combo(
            combo=self.combo_camera_pixel_format,
            info=pixel_info,
            settings_key="camera_native_pixel_format",
            tooltip_prefix="Native camera pixel format",
            apply_saved=apply_saved,
        )

        bit_depth_info = self.worker.get_camera_bit_depth_options()
        self.active_camera_bit_depth_node = str(bit_depth_info.get("node_name", "") or "")
        if apply_saved:
            saved_bit_depth = str(self.settings.value("camera_native_bit_depth", "") or "").strip()
            depth_options = [str(value).strip() for value in bit_depth_info.get("options", []) if str(value).strip()]
            current_bit_depth = str(bit_depth_info.get("current", "") or "").strip()
            if (
                saved_bit_depth
                and saved_bit_depth in depth_options
                and saved_bit_depth != current_bit_depth
                and bool(bit_depth_info.get("writable", False))
            ):
                applied = self.worker.set_camera_bit_depth(saved_bit_depth)
                if applied is not None:
                    bit_depth_info = self.worker.get_camera_bit_depth_options()
                    self.active_camera_bit_depth_node = str(bit_depth_info.get("node_name", "") or "")
        tooltip_prefix = "Camera bit depth"
        if self.active_camera_bit_depth_node:
            tooltip_prefix = f"Camera bit depth ({self.active_camera_bit_depth_node})"
        self._configure_camera_enum_combo(
            combo=self.combo_camera_bit_depth,
            info=bit_depth_info,
            settings_key="camera_native_bit_depth",
            tooltip_prefix=tooltip_prefix,
            apply_saved=apply_saved,
        )

    def _configure_camera_enum_combo(
        self,
        combo: QComboBox,
        info: Dict[str, object],
        settings_key: str,
        tooltip_prefix: str,
        apply_saved: bool = False,
    ):
        """Fill one advanced combo from worker-reported enum capabilities."""
        options = [str(value).strip() for value in info.get("options", []) if str(value).strip()]
        current = str(info.get("current", "") or "").strip()
        writable = bool(info.get("writable", False))
        node_name = str(info.get("node_name", "") or "").strip()

        combo.blockSignals(True)
        combo.clear()
        if options:
            combo.addItems(options)
            target_value = current if current in options else options[0]
            if apply_saved:
                saved_value = str(self.settings.value(settings_key, "") or "").strip()
                if saved_value and saved_value in options:
                    target_value = saved_value
            combo.setCurrentText(target_value)
        else:
            combo.addItem("Unavailable")
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

        combo.setEnabled(bool(options) and writable)
        combo.setToolTip(f"{tooltip_prefix}: {node_name}" if node_name else tooltip_prefix)

    def _on_camera_pixel_format_changed(self, pixel_format: str):
        """Apply a camera-native pixel format from the advanced dialog."""
        pixel_format = str(pixel_format or "").strip()
        if not pixel_format or pixel_format == "Unavailable":
            return
        if not self.worker or not self.worker.is_genicam_camera():
            return

        applied = self.worker.set_camera_pixel_format(pixel_format)
        if applied is None:
            self._on_error_occurred("Failed to set native pixel format")
            self._refresh_camera_native_format_controls(apply_saved=False)
            return

        self._save_ui_setting("camera_native_pixel_format", applied)
        self._refresh_camera_native_format_controls(apply_saved=False)
        self._update_live_header(mode_text=self.combo_image_format.currentText())
        self._on_status_update(f"Camera pixel format: {applied}")

    def _on_camera_bit_depth_changed(self, bit_depth: str):
        """Apply a camera-native bit depth from the advanced dialog."""
        bit_depth = str(bit_depth or "").strip()
        if not bit_depth or bit_depth == "Unavailable":
            return
        if not self.worker or not self.worker.is_genicam_camera():
            return

        applied = self.worker.set_camera_bit_depth(bit_depth)
        if applied is None:
            self._on_error_occurred("Failed to set camera bit depth")
            self._refresh_camera_native_format_controls(apply_saved=False)
            return

        self._save_ui_setting("camera_native_bit_depth", applied)
        self._refresh_camera_native_format_controls(apply_saved=False)
        self._on_status_update(f"Camera bit depth: {applied}")

    def _configure_white_balance_controls(self):
        """Populate white-balance controls when the camera exposes them."""
        auto_node = self._get_camera_node("BalanceWhiteAuto")
        ratio_node = self._get_camera_node("BalanceRatio")
        if auto_node is None or ratio_node is None or self.worker is None:
            self.combo_white_balance_auto.setEnabled(False)
            self.spin_white_balance_red.setEnabled(False)
            self.spin_white_balance_blue.setEnabled(False)
            return False

        auto_mode = self.worker._read_enum_node_symbolic("BalanceWhiteAuto") or "Continuous"
        if auto_mode not in {"Continuous", "Off"}:
            auto_mode = "Continuous"
        self.combo_white_balance_auto.blockSignals(True)
        self.combo_white_balance_auto.setCurrentText(auto_mode)
        self.combo_white_balance_auto.blockSignals(False)
        self.combo_white_balance_auto.setEnabled(self.worker._node_is_writable(auto_node))

        min_ratio = 0.25
        max_ratio = 4.0
        ratio_inc = 0.0001
        try:
            min_ratio = float(ratio_node.GetMin())
            max_ratio = float(ratio_node.GetMax())
        except Exception:
            pass
        try:
            ratio_inc = max(float(ratio_node.GetInc()), 0.0001)
        except Exception:
            pass

        for selector, spin in (("Red", self.spin_white_balance_red), ("Blue", self.spin_white_balance_blue)):
            ratio_value = self.worker.get_camera_white_balance_ratio(selector)
            spin.blockSignals(True)
            spin.setRange(min_ratio, max_ratio)
            spin.setSingleStep(ratio_inc)
            if ratio_value is not None:
                spin.setValue(float(ratio_value))
            spin.blockSignals(False)

        manual_enabled = auto_mode == "Off" and self.worker._node_is_writable(ratio_node)
        self.spin_white_balance_red.setEnabled(manual_enabled)
        self.spin_white_balance_blue.setEnabled(manual_enabled)
        return True

    def _configure_int_node(self, node_name: str, slider: QSlider, spin: QSpinBox):
        node = self._get_camera_node(node_name)
        if not node or (self.worker and not self.worker._node_is_readable(node)):
            slider.setEnabled(False)
            spin.setEnabled(False)
            return False
        if self.worker and not self.worker._node_is_writable(node):
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
        if self.worker and not self.worker._node_is_writable(node):
            spin.setEnabled(False)
            return False

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
        if not self.worker or not self.worker.is_genicam_camera():
            return None
        node = self.worker._get_camera_node(node_name)
        if node is None or not self.worker._node_is_readable(node):
            return None
        return node

    def _clamp_int_node(self, node_name: str, value: int):
        if not self.worker:
            return None
        return self.worker._clamp_numeric_node_value(node_name, value, integer=True)

    def _set_camera_int_node(self, node_name: str, value: int):
        if not self.worker:
            return
        if self.worker._write_numeric_node(node_name, value, integer=True) is None:
            self._on_error_occurred(f"Failed to set {node_name}: unsupported by camera")

    def _set_camera_float_node(self, node_name: str, value: float):
        if not self.worker:
            return
        if self.worker._write_numeric_node(node_name, value, integer=False) is None:
            self._on_error_occurred(f"Failed to set {node_name}: unsupported by camera")

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
        if self.worker and self.worker.is_genicam_camera():
            applied = self.worker.set_camera_gain(value)
            if applied is None:
                self._on_error_occurred("Failed to set Gain: unsupported by camera")
            else:
                self._on_status_update(f"Gain set to {applied:.2f} dB")
        self._save_ui_setting('gain', value)

    def _on_white_balance_auto_changed(self, mode: str):
        if not self.worker or not self.worker.is_genicam_camera():
            return
        if not self.worker.set_camera_white_balance_auto(mode):
            self._on_error_occurred("Failed to set White Balance mode")
            return
        self._configure_white_balance_controls()
        self._on_status_update(f"White balance set to {mode}")

    def _on_white_balance_red_changed(self, value: float):
        if not self.worker or not self.worker.is_genicam_camera():
            return
        applied = self.worker.set_camera_white_balance_ratio("Red", value)
        if applied is None:
            self._on_error_occurred("Failed to set red white-balance ratio")
            return
        self._configure_white_balance_controls()
        self._on_status_update(f"WB Red set to {applied:.4f}")

    def _on_white_balance_blue_changed(self, value: float):
        if not self.worker or not self.worker.is_genicam_camera():
            return
        applied = self.worker.set_camera_white_balance_ratio("Blue", value)
        if applied is None:
            self._on_error_occurred("Failed to set blue white-balance ratio")
            return
        self._configure_white_balance_controls()
        self._on_status_update(f"WB Blue set to {applied:.4f}")

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
        """Toggle an editable RectROI overlay on the live ImageView."""
        if self.live_image_view is None or self.last_frame_size is None:
            self._on_error_occurred("Preview a camera frame before editing ROI.")
            return

        view_box = self.live_image_view.getView()
        if not self.roi_draw_mode:
            if self.roi_item is None:
                width, height = self.last_frame_size
                roi_w = max(64, int(width * 0.45))
                roi_h = max(64, int(height * 0.45))
                roi_x = max(0, int((width - roi_w) / 2))
                roi_y = max(0, int((height - roi_h) / 2))
                self.roi_item = pg.RectROI(
                    [roi_x, roi_y],
                    [roi_w, roi_h],
                    pen=pg.mkPen("#f59e0b", width=2),
                    movable=True,
                    resizable=True,
                )
                self.roi_item.addScaleHandle([1, 1], [0, 0])
                self.roi_item.addScaleHandle([0, 0], [1, 1])
                self.roi_item.addScaleHandle([1, 0], [0, 1])
                self.roi_item.addScaleHandle([0, 1], [1, 0])
                view_box.addItem(self.roi_item)
            self.roi_draw_mode = True
            self.btn_draw_roi.setText("Apply ROI")
            self.btn_draw_roi.setStyleSheet("QPushButton { background-color: #f59e0b; color: white; font-weight: bold; }")
            self._update_live_header(roi_text="Editing ROI")
            return

        if self.roi_item is not None:
            pos = self.roi_item.pos()
            size = self.roi_item.size()
            self.roi_rect = {
                "x": max(0, int(round(pos.x()))),
                "y": max(0, int(round(pos.y()))),
                "w": max(1, int(round(size.x()))),
                "h": max(1, int(round(size.y()))),
            }
            if self.worker:
                self.worker.set_roi(self.roi_rect)
            self.roi_item.setPen(pg.mkPen("#22c55e", width=2))
            self._update_live_header(roi_text=f"ROI {self.roi_rect['w']} x {self.roi_rect['h']}")

        self.roi_draw_mode = False
        self.btn_draw_roi.setText("Edit ROI")
        self.btn_draw_roi.setStyleSheet("")

    def _clear_roi(self):
        """Clear ROI and reset cropping."""
        self.roi_rect = None
        self.roi_draw_mode = False
        if self.roi_item is not None and self.live_image_view is not None:
            try:
                self.live_image_view.getView().removeItem(self.roi_item)
            except Exception:
                pass
            self.roi_item = None
        self.btn_draw_roi.setText("Draw ROI")
        self.btn_draw_roi.setStyleSheet("")
        self._update_live_header(roi_text="Full Frame")
        if self.worker:
            self.worker.set_roi(None)

    @Slot(str)
    def _start_live_roi_draw(self, shape: str):
        if self.live_image_view is None or self.last_frame_size is None:
            self._on_error_occurred("Preview a live frame before drawing behavioural ROIs.")
            return
        self.live_roi_draw_mode = str(shape or "").strip().lower()
        self.live_roi_draw_points = []
        self.live_roi_circle_center = None
        self.live_roi_drawing_name = (
            self.live_detection_panel.current_roi_name()
            if self.live_detection_panel is not None
            else f"ROI {len(self.live_rois) + 1}"
        )
        self._on_status_update(
            f"Drawing {self.live_roi_draw_mode or 'roi'} '{self.live_roi_drawing_name}'. "
            "Use left-clicks on the live view; right-click closes polygons."
        )

    @Slot()
    def _finish_live_polygon_roi(self):
        if self.live_roi_draw_mode != "polygon" or len(self.live_roi_draw_points) < 3:
            return
        self._register_live_roi("polygon", list(self.live_roi_draw_points))

    @Slot(str)
    def _remove_live_roi(self, roi_name: str):
        self.live_rois.pop(str(roi_name), None)
        self.live_rules = [
            rule
            for rule in self.live_rules
            if not (rule.rule_type == "roi_occupancy" and rule.roi_name == str(roi_name))
        ]
        self.live_rule_engine.set_rois(self.live_rois)
        self.live_rule_engine.set_rules(self.live_rules)
        self._refresh_live_panel_state()
        self._persist_live_detection_settings()

    @Slot()
    def _clear_live_rois(self):
        self.live_rois.clear()
        self.live_rules = [rule for rule in self.live_rules if rule.rule_type != "roi_occupancy"]
        self.live_rule_engine.set_rois(self.live_rois)
        self.live_rule_engine.set_rules(self.live_rules)
        self.live_roi_draw_mode = ""
        self.live_roi_draw_points = []
        self.live_roi_circle_center = None
        self._refresh_live_panel_state()
        self._persist_live_detection_settings()

    def _live_roi_color(self, index: int) -> tuple[int, int, int]:
        palette = [
            (255, 220, 120),
            (120, 240, 170),
            (120, 200, 255),
            (255, 140, 170),
            (220, 180, 255),
            (255, 190, 110),
        ]
        return palette[index % len(palette)]

    def _register_live_roi(self, roi_type: str, data):
        roi_name = self.live_roi_drawing_name or (
            self.live_detection_panel.current_roi_name()
            if self.live_detection_panel is not None
            else f"ROI {len(self.live_rois) + 1}"
        )
        roi = BehaviorROI(
            name=roi_name,
            roi_type=roi_type,
            data=data,
            color=self._live_roi_color(len(self.live_rois)),
        )
        self.live_rois[roi.name] = roi
        self.live_rule_engine.set_rois(self.live_rois)
        self.live_roi_draw_mode = ""
        self.live_roi_draw_points = []
        self.live_roi_circle_center = None
        self._refresh_live_panel_state()
        self._persist_live_detection_settings()

    def _map_scene_to_live_image(self, scene_pos) -> Optional[tuple[float, float]]:
        if self.live_image_view is None or self.last_frame_size is None:
            return None
        view = self.live_image_view.getView()
        point = view.mapSceneToView(scene_pos)
        x = float(point.x())
        y = float(point.y())
        width, height = self.last_frame_size
        if x < 0 or y < 0 or x > float(width) or y > float(height):
            return None
        return x, y

    def _handle_live_roi_click(self, x: float, y: float, double_click: bool = False):
        if self.live_roi_draw_mode == "rectangle":
            self.live_roi_draw_points.append((x, y))
            if len(self.live_roi_draw_points) >= 2:
                (x1, y1), (x2, y2) = self.live_roi_draw_points[:2]
                self._register_live_roi(
                    "rectangle",
                    [(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))],
                )
            return

        if self.live_roi_draw_mode == "circle":
            if self.live_roi_circle_center is None:
                self.live_roi_circle_center = (x, y)
            else:
                cx, cy = self.live_roi_circle_center
                radius = float(np.hypot(x - cx, y - cy))
                self._register_live_roi("circle", [(cx, cy, radius)])
            return

        if self.live_roi_draw_mode == "polygon":
            self.live_roi_draw_points.append((x, y))
            if double_click and len(self.live_roi_draw_points) >= 3:
                self._register_live_roi("polygon", list(self.live_roi_draw_points))

    def eventFilter(self, obj, event):
        if obj is self.live_preview_scene and self.live_roi_draw_mode:
            if event.type() in (QEvent.GraphicsSceneMousePress, QEvent.GraphicsSceneMouseDoubleClick):
                if event.button() == Qt.RightButton and self.live_roi_draw_mode == "polygon":
                    self._finish_live_polygon_roi()
                    return True
                if event.button() == Qt.LeftButton:
                    mapped = self._map_scene_to_live_image(event.scenePos())
                    if mapped is not None:
                        self._handle_live_roi_click(
                            mapped[0],
                            mapped[1],
                            double_click=event.type() == QEvent.GraphicsSceneMouseDoubleClick,
                        )
                        return True
        return super().eventFilter(obj, event)

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
                try:
                    self.arduino_worker.configure_live_output_mapping(self.live_output_mapping)
                except Exception as exc:
                    self._on_error_occurred(str(exc))
                self.btn_arduino_connect.setText("Disconnect Arduino")
                self._set_button_icon(self.btn_arduino_connect, "record", "#ffffff", "dangerButton")
                self.btn_test_ttl.setEnabled(True)
        else:
            if self.is_testing_ttl:
                self.arduino_worker.stop_test()
                self.is_testing_ttl = False
                self.btn_test_ttl.setText("Test TTL / Behavior")
                self._set_button_icon(self.btn_test_ttl, "pulse", "#33d5ff", "violetButton")

            self.arduino_worker.clear_live_outputs()
            self.arduino_worker.stop()
            self.arduino_worker.wait()
            self.is_arduino_connected = False
            self.btn_arduino_connect.setText("Connect Arduino")
            self._set_button_icon(self.btn_arduino_connect, "play", "#eef6ff")
            self.btn_test_ttl.setEnabled(False)

            self._apply_behavior_pin_configuration(persist=False)
            self._reset_ttl_plot()
            self._set_ttl_status("IDLE", "default")
            self._set_behavior_status("IDLE", "default")

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
                self._set_button_icon(self.btn_arduino_connect, "play", "#eef6ff")
                self.btn_test_ttl.setEnabled(False)
                self.btn_test_ttl.setText("Test TTL / Behavior")
                self._set_button_icon(self.btn_test_ttl, "pulse", "#33d5ff", "violetButton")
                self._reset_ttl_plot()
                self._set_ttl_status("IDLE", "default")
                self._set_behavior_status("IDLE", "default")

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
            if key.upper().startswith("DO"):
                self.live_output_mapping[key.upper()] = pins
        if self.live_detection_panel is not None:
            self.live_detection_panel.set_output_mapping(self.live_output_mapping)

    def _ttl_count_key_for_signal(self, key: str) -> str:
        """Map signal key to arduino count key."""
        if key in ("barcode", "barcode0", "barcode1"):
            return "barcode_count"
        return f"{key}_count"

    def _count_value_for_display_signal(self, key: str, states: Dict, pulse_counts: Dict) -> int:
        """Resolve the most accurate count value for one display signal."""
        count_key = self._ttl_count_key_for_signal(key)
        state_key = self._state_key_for_display(key)
        if states.get("passive_mode"):
            if key == "barcode":
                return max(int(pulse_counts.get("barcode0", 0)), int(pulse_counts.get("barcode1", 0)))
            return int(pulse_counts.get(state_key, 0))
        if count_key in states:
            return int(states.get(count_key, 0))
        if key == "barcode":
            return max(int(pulse_counts.get("barcode0", 0)), int(pulse_counts.get("barcode1", 0)))
        return int(pulse_counts.get(state_key, 0))

    def _update_signal_monitor_counts(
        self,
        states: Dict,
        pulse_counts: Dict,
        state_labels: Dict[str, QLabel],
        count_labels: Dict[str, QLabel],
    ):
        """Update state and count rows for one signal group."""
        for key, state_label in state_labels.items():
            state_key = self._state_key_for_display(key)
            state = bool(states.get(state_key, False))
            if state_label:
                self._set_signal_state_label(state_label, state)

            count_label = count_labels.get(key)
            if count_label is None:
                continue
            self._set_signal_count_label(count_label, self._count_value_for_display_signal(key, states, pulse_counts))

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
                self._set_ttl_status("MONITORING", "accent")
        elif self.label_ttl_status.text() == "TTL: MONITORING" and self.is_testing_ttl:
            self._set_ttl_status("TESTING", "warning")

        behavior_active = any(
            bool(states.get(self._state_key_for_display(key), False))
            for key in self._active_signal_keys(group="behavior")
        )
        if behavior_active:
            self._set_behavior_status("ACTIVE", "success")
        elif states.get("passive_mode"):
            self._set_behavior_status("MONITORING", "accent")
        elif self.is_testing_ttl:
            self._set_behavior_status("ARMED", "accent")
        else:
            self._set_behavior_status("IDLE", "default")

        pulse_counts = states.get("pulse_counts", {})
        self._update_signal_monitor_counts(states, pulse_counts, self.ttl_state_labels, self.ttl_count_labels)
        self._update_signal_monitor_counts(
            states,
            pulse_counts,
            self.behavior_state_labels,
            self.behavior_count_labels,
        )

    @Slot()
    def _on_test_ttl_clicked(self):
        """Handle test TTL button click."""
        if not self.is_testing_ttl:
            # Start test
            self._apply_behavior_pin_configuration(persist=True)
            if self.arduino_worker.start_test():
                self.is_testing_ttl = True
                self.btn_test_ttl.setText("Stop Test / Monitor")
                self._set_button_icon(self.btn_test_ttl, "record", "#ffffff", "dangerButton")
                self._set_ttl_status("TESTING", "warning")
                self._set_behavior_status("ARMED", "accent")
                self._reset_ttl_plot()
            else:
                self._on_error_occurred("Failed to start test/monitor mode on Arduino.")
        else:
            # Stop test
            self.arduino_worker.stop_test()
            self.is_testing_ttl = False
            self.btn_test_ttl.setText("Test TTL / Behavior")
            self._set_button_icon(self.btn_test_ttl, "pulse", "#33d5ff", "violetButton")
            self._set_ttl_status("IDLE", "default")
            self._set_behavior_status("IDLE", "default")

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
        if df is None or df.empty:
            return df

        df = df.copy()
        definitions = self._signal_export_definitions()
        resolved = {
            key: self._resolve_display_signal_series(df, key)
            for key in self.DISPLAY_SIGNAL_ORDER
        }

        for key, series in resolved.items():
            logical_column = f"{key}_state"
            if logical_column not in df.columns:
                df[logical_column] = series
            labeled_column = definitions[key]["state_column"]
            if labeled_column not in df.columns:
                df[labeled_column] = series

            count_series = self._resolve_display_signal_count_series(df, key)
            labeled_count_column = definitions[key]["count_column"]
            if count_series is not None and labeled_count_column not in df.columns:
                df[labeled_count_column] = count_series

        ttl_active = (resolved["gate"] | resolved["sync"] | resolved["barcode"]).astype(int)
        behavior_active = (resolved["lever"] | resolved["cue"] | resolved["reward"] | resolved["iti"]).astype(int)

        df["ttl_state"] = np.where(ttl_active > 0, "HIGH", "LOW")
        df["behavior_state"] = np.where(behavior_active > 0, "ACTIVE", "IDLE")
        ttl_labels = {key: definitions[key]["label"] for key in ("gate", "sync", "barcode")}
        behavior_labels = {key: definitions[key]["label"] for key in ("lever", "cue", "reward", "iti")}
        df["ttl_state_vector"] = (
            f"{ttl_labels['gate']}=" + resolved["gate"].astype(str)
            + f"|{ttl_labels['sync']}=" + resolved["sync"].astype(str)
            + f"|{ttl_labels['barcode']}=" + resolved["barcode"].astype(str)
        )
        df["behavior_state_vector"] = (
            f"{behavior_labels['lever']}=" + resolved["lever"].astype(str)
            + f"|{behavior_labels['cue']}=" + resolved["cue"].astype(str)
            + f"|{behavior_labels['reward']}=" + resolved["reward"].astype(str)
            + f"|{behavior_labels['iti']}=" + resolved["iti"].astype(str)
        )

        return self._reorder_signal_export_columns(df)

    def _build_behavior_summary_df(self, source_df, ttl_counts: Dict) -> "pd.DataFrame":
        """Build behavior summary (counts and cumulative HIGH durations)."""
        import pandas as pd

        signals = ["lever", "cue", "reward", "iti"]
        definitions = self._signal_export_definitions()
        rows = []

        if source_df is None or source_df.empty or "timestamp_software" not in source_df.columns:
            for signal in signals:
                definition = definitions.get(signal, {})
                rows.append(
                    {
                        "signal": definition.get("label", signal),
                        "signal_key": signal,
                        "signal_role": definition.get("role", self.DISPLAY_SIGNAL_META[signal]["role"]),
                        "signal_pins": definition.get("pins", "-"),
                        "state_column": definition.get("state_column", signal),
                        "count_column": definition.get("count_column", f"{signal}_count"),
                        "count": self._resolve_display_signal_count(signal, ttl_counts),
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
            definition = definitions.get(signal, {})
            state = self._resolve_display_signal_series(df, signal)
            rises = int(((state == 1) & (state.shift(1, fill_value=0) == 0)).sum())
            duration_high = float((dt * state).sum())
            duty_cycle = (100.0 * duration_high / total_duration) if total_duration > 0 else 0.0
            count_value = self._resolve_display_signal_count(signal, ttl_counts)
            if count_value <= 0:
                count_value = rises

            rows.append(
                {
                    "signal": definition.get("label", signal),
                    "signal_key": signal,
                    "signal_role": definition.get("role", self.DISPLAY_SIGNAL_META[signal]["role"]),
                    "signal_pins": definition.get("pins", "-"),
                    "state_column": definition.get("state_column", signal),
                    "count_column": definition.get("count_column", f"{signal}_count"),
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
        - *_ttl_counts.csv: final pulse counters
        - *_behavior_summary.csv: count + cumulative HIGH duration summary
        """
        if self.is_arduino_connected:
            import pandas as pd

            ttl_counts = self.arduino_worker.get_ttl_pulse_counts() or {}
            export_definitions = self._signal_export_definitions()
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

            if ttl_counts:
                count_row = dict(ttl_counts)
                for key, definition in export_definitions.items():
                    count_row[definition["count_column"]] = self._resolve_display_signal_count(key, ttl_counts)
                df = pd.DataFrame([count_row])
                df = self._reorder_signal_export_columns(df)
                csv_file = filepath + "_ttl_counts.csv"
                df.to_csv(csv_file, index=False)
                self._on_status_update(f"TTL counts saved: {csv_file}")

            summary_source = df_history if df_history is not None else df_live
            if summary_source is not None:
                behavior_summary = self._build_behavior_summary_df(summary_source, ttl_counts)
                csv_file = filepath + "_behavior_summary.csv"
                behavior_summary.to_csv(csv_file, index=False)
                self._on_status_update(f"Behavior summary saved: {csv_file}")

            # Clear history
            self.arduino_worker.clear_ttl_history()
            self.arduino_worker.clear_ttl_event_history()

    # ===== Display Slots =====

    @Slot(dict)
    def _on_frame_drop_stats_updated(self, stats: dict):
        """Refresh the compact frame-drop monitor from worker timing data."""
        self.last_frame_drop_stats = dict(stats or {})
        recorded_frames = int(self.last_frame_drop_stats.get("recorded_frames", 0))
        dropped_frames = int(self.last_frame_drop_stats.get("estimated_dropped_frames", 0))
        drop_percent = float(self.last_frame_drop_stats.get("drop_percent", 0.0))
        reference_fps = float(self.last_frame_drop_stats.get("reference_fps", self._frame_drop_reference_fps()))
        average_interval_ms = float(self.last_frame_drop_stats.get("average_interval_ms", 0.0))
        max_gap_ms = float(self.last_frame_drop_stats.get("max_gap_ms", 0.0))
        elapsed_seconds = int(round(float(self.last_frame_drop_stats.get("elapsed_seconds", 0.0))))
        is_active = bool(self.last_frame_drop_stats.get("active", False))

        if self.label_frame_drop_summary is not None:
            if dropped_frames == 0:
                summary_color = "#89f0b2"
            elif drop_percent < 1.0:
                summary_color = "#ffc86b"
            else:
                summary_color = "#ff98ae"
            state = "REC" if is_active else "LAST"
            self.label_frame_drop_summary.setStyleSheet(f"color: {summary_color}; font-weight: 700;")
            self.label_frame_drop_summary.setText(
                f"{state} | drop {dropped_frames} ({drop_percent:.2f}%) | frames {recorded_frames} | ref {reference_fps:.1f} fps"
            )

        log_signature = (recorded_frames, dropped_frames, round(drop_percent, 3), is_active)
        if log_signature == self.last_frame_drop_log_signature:
            return
        self.last_frame_drop_log_signature = log_signature

        log_line = (
            f"{self._format_duration_hms(elapsed_seconds)} | {recorded_frames}f | "
            f"drop {dropped_frames} ({drop_percent:.2f}%) | avg {average_interval_ms:.1f} ms | max {max_gap_ms:.1f} ms"
        )
        self.frame_drop_events.append(log_line)
        if self.frame_drop_log is not None:
            self.frame_drop_log.setPlainText("\n".join(self.frame_drop_events))

    def _update_live_header(
        self,
        status_text: Optional[str] = None,
        resolution_text: Optional[str] = None,
        mode_text: Optional[str] = None,
        roi_text: Optional[str] = None,
        badge_text: Optional[str] = None,
        badge_tone: str = "default",
    ):
        """Update the live-view header chips."""
        if status_text is not None and self.live_header_status is not None:
            self.live_header_status.setText(status_text)
        if resolution_text is not None and self.live_header_resolution is not None:
            self.live_header_resolution.setText(resolution_text)
        if mode_text is not None and self.live_header_mode is not None:
            self.live_header_mode.setText(mode_text)
        if roi_text is not None and self.live_header_roi is not None:
            self.live_header_roi.setText(roi_text)
        if badge_text is not None and self.live_status_badge is not None:
            replacement = self._make_panel_chip(badge_text, badge_tone)
            self.live_status_badge.setText(replacement.text())
            self.live_status_badge.setStyleSheet(replacement.styleSheet())

    def _apply_live_image(self, image_rgb: np.ndarray, auto_range: bool = False):
        """Push an RGB numpy frame into the ImageView."""
        if self.live_image_view is None:
            return
        levels = None
        if np.issubdtype(image_rgb.dtype, np.integer):
            dtype_info = np.iinfo(image_rgb.dtype)
            levels = (dtype_info.min, dtype_info.max)

        self.live_image_view.setImage(
            image_rgb,
            autoLevels=False,
            autoRange=auto_range,
            autoHistogramRange=auto_range,
            axes={"x": 1, "y": 0, "c": 2},
            levels=levels,
        )

    def _build_placeholder_frame(self, title: str, subtitle: str = "") -> np.ndarray:
        """Create a branded placeholder frame using OpenCV drawing primitives."""
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        x_gradient = np.linspace(14, 42, canvas.shape[1], dtype=np.uint8)
        y_gradient = np.linspace(8, 28, canvas.shape[0], dtype=np.uint8)[:, None]
        canvas[:, :, 0] = x_gradient
        canvas[:, :, 1] = np.clip((x_gradient * 0.65) + y_gradient, 0, 255).astype(np.uint8)
        canvas[:, :, 2] = np.clip((x_gradient * 0.35) + (y_gradient * 0.6), 0, 255).astype(np.uint8)

        overlay = canvas.copy()
        cv2.circle(overlay, (975, 190), 230, (48, 96, 148), -1)
        cv2.circle(overlay, (240, 560), 260, (20, 60, 108), -1)
        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)

        cv2.rectangle(canvas, (92, 84), (1188, 636), (8, 15, 23), thickness=-1)
        cv2.rectangle(canvas, (92, 84), (1188, 636), (30, 62, 95), thickness=2)
        cv2.rectangle(canvas, (128, 126), (1152, 594), (10, 20, 31), thickness=-1)

        cv2.rectangle(canvas, (158, 196), (362, 336), (15, 32, 49), thickness=-1)
        cv2.rectangle(canvas, (158, 196), (362, 336), (78, 165, 255), thickness=2)
        cv2.circle(canvas, (260, 266), 34, (78, 165, 255), thickness=3)
        cv2.rectangle(canvas, (214, 178), (266, 198), (78, 165, 255), thickness=2)
        cv2.line(canvas, (172, 370), (346, 370), (78, 165, 255), 4, cv2.LINE_AA)

        cv2.putText(canvas, "CamApp Live Detection", (278, 222), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (157, 217, 255), 4, cv2.LINE_AA)
        cv2.putText(canvas, title, (406, 314), cv2.FONT_HERSHEY_SIMPLEX, 1.06, (238, 244, 255), 3, cv2.LINE_AA)
        if subtitle:
            cv2.putText(canvas, subtitle, (406, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (164, 184, 205), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Live workspace is ready when a camera source is connected.", (406, 438),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (130, 154, 180), 2, cv2.LINE_AA)
        cv2.line(canvas, (406, 476), (732, 476), (61, 150, 255), 4, cv2.LINE_AA)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def _show_live_placeholder(self, title: str, subtitle: str = ""):
        """Display a placeholder frame in the live ImageView."""
        placeholder = self._build_placeholder_frame(title, subtitle)
        auto_range = not self.live_placeholder_auto_ranged
        self._apply_live_image(placeholder, auto_range=auto_range)
        self.live_placeholder_auto_ranged = True

    def _format_duration_hms(self, total_seconds: int) -> str:
        """Format a duration in whole seconds as HH:MM:SS."""
        total_seconds = max(0, int(total_seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _current_recording_elapsed_seconds(self) -> int:
        """Return elapsed encoded video time in whole seconds."""
        if self.worker is not None and self.worker.is_recording:
            reference_fps = self._get_recording_reference_fps()
            if reference_fps > 0:
                recorded_frames = max(0, int(getattr(self.worker, "frame_counter", 0)))
                return max(0, int(recorded_frames / reference_fps))
        if not self.recording_start_time:
            return 0
        return max(0, int((datetime.now() - self.recording_start_time).total_seconds()))

    def _current_recording_remaining_seconds(self) -> Optional[int]:
        """Return remaining configured recording time, or None for unlimited."""
        max_seconds = self._get_max_record_seconds()
        if max_seconds <= 0:
            return None
        return max(max_seconds - self._current_recording_elapsed_seconds(), 0)

    def _draw_recording_overlay(self, display_bgr: np.ndarray):
        """Draw a large elapsed/remaining recording HUD over the live frame."""
        height, width = display_bgr.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        elapsed_seconds = self._current_recording_elapsed_seconds()
        remaining_seconds = self._current_recording_remaining_seconds()
        max_seconds = self._get_max_record_seconds()

        title_text = "RECORDING"
        elapsed_text = f"Elapsed {self._format_duration_hms(elapsed_seconds)}"
        if remaining_seconds is None:
            remaining_text = "Remaining Unlimited"
        else:
            remaining_text = f"Remaining {self._format_duration_hms(remaining_seconds)}"

        title_scale = max(0.52, min(0.86, width / 2100))
        main_scale = max(0.82, min(1.32, width / 1200))
        sub_scale = max(0.58, min(0.96, width / 1750))
        title_thickness = max(1, int(round(title_scale * 2.1)))
        main_thickness = max(2, int(round(main_scale * 2.25)))
        sub_thickness = max(1, int(round(sub_scale * 2.0)))
        pad = max(18, int(width * 0.018))
        line_gap = max(10, int(height * 0.012))
        progress_height = max(10, int(height * 0.014)) if max_seconds > 0 else 0

        lines = [
            (title_text, title_scale, title_thickness, (112, 183, 255)),
            (elapsed_text, main_scale, main_thickness, (247, 250, 255)),
            (
                remaining_text,
                sub_scale,
                sub_thickness,
                (120, 225, 162) if remaining_seconds is None or remaining_seconds > 10 else (114, 194, 255),
            ),
        ]

        line_sizes = [cv2.getTextSize(text, font, scale, thickness)[0] for text, scale, thickness, _ in lines]
        max_text_width = max(size[0] for size in line_sizes)
        total_text_height = sum(size[1] for size in line_sizes) + line_gap * (len(lines) - 1)
        panel_width = max_text_width + pad * 2
        panel_height = total_text_height + pad * 2 + (progress_height + line_gap if progress_height else 0)

        x1 = max(24, int(width * 0.028))
        y1 = max(56, height - panel_height - max(24, int(height * 0.035)))
        x2 = min(width - 18, x1 + panel_width)
        y2 = min(height - 18, y1 + panel_height)

        overlay = display_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (7, 14, 24), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 78, 118), 2)
        cv2.addWeighted(overlay, 0.72, display_bgr, 0.28, 0, display_bgr)

        y_cursor = y1 + pad
        for (text, scale, thickness, color), (_, text_height) in zip(lines, line_sizes):
            baseline_y = y_cursor + text_height
            cv2.putText(display_bgr, text, (x1 + pad, baseline_y), font, scale, color, thickness, cv2.LINE_AA)
            y_cursor = baseline_y + line_gap

        if progress_height:
            progress_left = x1 + pad
            progress_right = x2 - pad
            progress_bottom = y2 - pad
            progress_top = progress_bottom - progress_height
            cv2.rectangle(display_bgr, (progress_left, progress_top), (progress_right, progress_bottom), (20, 36, 52), -1)
            progress = min(max(elapsed_seconds / max_seconds, 0.0), 1.0) if max_seconds > 0 else 0.0
            fill_color = (82, 211, 255) if remaining_seconds is None or remaining_seconds > 10 else (65, 153, 255)
            progress_fill = progress_left + int((progress_right - progress_left) * progress)
            if progress_fill > progress_left:
                cv2.rectangle(display_bgr, (progress_left, progress_top), (progress_fill, progress_bottom), fill_color, -1)

    def _decorate_live_frame(self, frame: np.ndarray) -> np.ndarray:
        """Render the live frame through an OpenCV overlay pipeline before display."""
        if frame.ndim == 2:
            display_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.worker and self.worker.is_recording:
            cv2.circle(display_bgr, (28, 28), 8, (32, 59, 240), -1)
            cv2.putText(display_bgr, "REC", (48, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            self._draw_recording_overlay(display_bgr)

        self._draw_live_detection_overlay(display_bgr)

        info_text = f"{display_bgr.shape[1]}x{display_bgr.shape[0]}  {self.combo_image_format.currentText()}"
        info_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        info_x = max(18, display_bgr.shape[1] - info_size[0] - 22)
        cv2.putText(display_bgr, info_text, (info_x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (195, 216, 236), 2, cv2.LINE_AA)
        return cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)

    def _draw_live_detection_overlay(self, display_bgr: np.ndarray):
        overlay = display_bgr.copy()
        for roi_name, roi in self.live_rois.items():
            color_bgr = (int(roi.color[2]), int(roi.color[1]), int(roi.color[0]))
            if roi.roi_type == "rectangle" and roi.data:
                x1, y1, x2, y2 = [int(round(value)) for value in roi.data[0]]
                cv2.rectangle(display_bgr, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(display_bgr, roi_name, (x1 + 6, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)
            elif roi.roi_type == "circle" and roi.data:
                cx, cy, radius = roi.data[0]
                cv2.circle(display_bgr, (int(round(cx)), int(round(cy))), int(round(radius)), color_bgr, 2)
                cv2.putText(display_bgr, roi_name, (int(cx) + 6, max(20, int(cy) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)
            elif roi.roi_type == "polygon" and roi.data:
                pts = np.array([(int(round(px)), int(round(py))) for px, py in roi.data], dtype=np.int32)
                if len(pts) >= 3:
                    cv2.polylines(display_bgr, [pts], True, color_bgr, 2, cv2.LINE_AA)
                    cv2.fillPoly(overlay, [pts], color_bgr)
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))
                    cv2.putText(display_bgr, roi_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)

        if self.live_rois:
            cv2.addWeighted(overlay, 0.12, display_bgr, 0.88, 0, display_bgr)

        if self.live_detection_last_result is not None:
            for mouse in self.live_detection_last_result.tracked_mice:
                color_bgr = (90 + (mouse.mouse_id * 40) % 140, 220 - (mouse.mouse_id * 35) % 120, 120 + (mouse.mouse_id * 55) % 110)
                if mouse.mask is not None and mouse.mask.size > 0 and mouse.mask.shape[:2] == display_bgr.shape[:2]:
                    mask_overlay = display_bgr.copy()
                    mask_overlay[mouse.mask.astype(bool)] = color_bgr
                    cv2.addWeighted(mask_overlay, 0.18, display_bgr, 0.82, 0, display_bgr)
                x1, y1, x2, y2 = [int(round(value)) for value in mouse.bbox]
                cv2.rectangle(display_bgr, (x1, y1), (x2, y2), color_bgr, 2)
                cx, cy = int(round(mouse.center[0])), int(round(mouse.center[1]))
                cv2.circle(display_bgr, (cx, cy), 4, color_bgr, -1)
                label = f"{mouse.label}  C{mouse.class_id}  {mouse.confidence:.2f}"
                cv2.putText(display_bgr, label, (x1 + 4, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2, cv2.LINE_AA)

        if self.live_roi_draw_mode:
            draw_color = (255, 255, 255)
            if self.live_roi_draw_mode == "polygon" and self.live_roi_draw_points:
                pts = np.array([(int(round(px)), int(round(py))) for px, py in self.live_roi_draw_points], dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(display_bgr, [pts], False, draw_color, 2, cv2.LINE_AA)
                for px, py in pts:
                    cv2.circle(display_bgr, (int(px), int(py)), 4, draw_color, -1)
            elif self.live_roi_draw_mode == "rectangle" and self.live_roi_draw_points:
                x1, y1 = self.live_roi_draw_points[0]
                cv2.circle(display_bgr, (int(round(x1)), int(round(y1))), 4, draw_color, -1)
            elif self.live_roi_draw_mode == "circle" and self.live_roi_circle_center is not None:
                cx, cy = self.live_roi_circle_center
                cv2.circle(display_bgr, (int(round(cx)), int(round(cy))), 4, draw_color, -1)

        if self.live_output_states:
            active_outputs = [output_id for output_id, state in sorted(self.live_output_states.items()) if bool(state)]
            if active_outputs:
                text = "TTL HIGH: " + ", ".join(active_outputs)
                cv2.putText(display_bgr, text, (18, max(64, display_bgr.shape[0] - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 180), 2, cv2.LINE_AA)

    @Slot(np.ndarray)
    def _on_frame_ready(self, frame: np.ndarray):
        """
        Update the docked ImageView with a frame from the worker.
        """
        try:
            height, width = frame.shape[:2]
            self.last_frame_size = (width, height)
            image_rgb = self._decorate_live_frame(frame)
            auto_range = not self.live_frame_auto_ranged
            self._apply_live_image(image_rgb, auto_range=auto_range)
            self.live_frame_auto_ranged = True
            self._update_live_header(
                status_text="Streaming",
                resolution_text=f"{width} x {height}",
                mode_text=self.combo_image_format.currentText(),
                badge_text="REC" if (self.worker and self.worker.is_recording) else "Preview",
                badge_tone="danger" if (self.worker and self.worker.is_recording) else "accent",
            )

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

    def resizeEvent(self, event):
        """Keep shell widths responsive as the main window changes size."""
        self._update_side_panel_bounds()
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Handle window close event - cleanup resources."""
        try:
            self._apply_behavior_pin_configuration(persist=True)
        except Exception:
            pass
        try:
            self._persist_live_detection_settings()
        except Exception:
            pass

        if self.is_camera_connected:
            self._disconnect_camera()

        if self.is_arduino_connected:
            self.arduino_worker.clear_live_outputs()
            self._stop_arduino_generation()
            self.arduino_worker.stop()
            self.arduino_worker.wait()

        if self.live_inference_worker is not None:
            self.live_inference_worker.shutdown()

        event.accept()

    def _reset_ttl_plot(self):
        """Reset TTL plot data and time base."""
        for data in self.ttl_plot_data.values():
            data.clear()
        self.time_data.clear()
        self.plot_start_time = datetime.now()
        self.ttl_plot.setXRange(0, self.ttl_window_seconds)
        self.behavior_plot.setXRange(0, self.ttl_window_seconds)
        for curve in list(self.ttl_output_curves.values()) + list(self.behavior_curves.values()):
            curve.setData([], [])
        self._reset_signal_count_widgets()

    def _stop_arduino_generation(self):
        """Ensure Arduino TTL generation is stopped and UI updated."""
        if self.is_testing_ttl:
            self.arduino_worker.stop_test()
            self.is_testing_ttl = False
            self.btn_test_ttl.setText("Test TTL / Behavior")
            self._set_button_icon(self.btn_test_ttl, "pulse", "#33d5ff", "violetButton")

        self.arduino_worker.stop_recording()
        self._set_ttl_status("IDLE", "default")
        self._set_behavior_status("IDLE", "default")
