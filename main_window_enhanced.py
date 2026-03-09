"""
Enhanced Main GUI Window
PySide6-based interface for Basler camera control with Arduino integration.
"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QLineEdit,
                               QStatusBar, QGroupBox, QSpinBox, QDoubleSpinBox,
                               QFileDialog, QScrollArea, QFormLayout, QTextEdit,
                               QFrame, QSlider, QGridLayout,
                               QCheckBox, QToolButton, QDialog, QStackedWidget,
                               QDialogButtonBox, QStyle, QToolBar,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QAbstractItemView, QMessageBox)
from PySide6.QtCore import Qt, Slot, QTimer, QSettings, QSize, QPointF, QRectF
from PySide6.QtGui import (QAction, QIcon, QPixmap, QPainter, QColor, QPen,
                           QBrush, QPainterPath, QLinearGradient)
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
        self.settings = QSettings("CamApp", "CamApp")
        self._migrate_legacy_settings()
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
        self.label_ttl_status: Optional[QLabel] = None
        self.label_behavior_status: Optional[QLabel] = None
        self.live_placeholder_auto_ranged = False
        self.live_frame_auto_ranged = False
        self.roi_item: Optional[pg.RectROI] = None
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

        self.setWindowTitle("CamApp")
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
                font-family: "Segoe UI Variable Text", "Bahnschrift", "Segoe UI";
                font-size: 12px;
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
                border-radius: 18px;
                padding: 11px 16px;
                font-weight: 700;
                font-size: 13px;
                min-height: 20px;
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
                border-radius: 22px;
                padding: 10px 6px;
                min-width: 78px;
                max-width: 78px;
                min-height: 78px;
                color: #9bb4d2;
                font-weight: 700;
                font-size: 12px;
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
                border-radius: 14px;
                color: #eef6ff;
                padding: 7px 10px;
                min-height: 22px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
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
        session_page = self._create_session_hub_panel()
        file_page = self._create_file_tools_panel()
        ttl_page = self._create_ttl_monitor_panel()
        behavior_page = self._create_behavior_monitor_panel()
        arduino_page = self._create_behavior_setup_panel()
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
        }

        for page in self.left_panel_pages.values():
            self.left_panel_stack.addWidget(page)
        for page in self.right_panel_pages.values():
            self.right_panel_stack.addWidget(page)

        center_workspace = self._create_center_workspace()

        root_layout.addWidget(left_rail)
        root_layout.addWidget(self.left_panel_shell)
        root_layout.addWidget(center_workspace, 1)
        root_layout.addWidget(self.right_panel_shell)
        root_layout.addWidget(right_rail)

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
        rail.setFixedWidth(96)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(12)

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
            ]

        for key, label, icon_kind, accent, title in specs:
            button = self._create_nav_button(label, icon_kind, accent)
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
        if side == "left":
            shell.setMinimumWidth(390)
            shell.setMaximumWidth(560)
        else:
            shell.setMinimumWidth(360)
            shell.setMaximumWidth(500)
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

    def _create_nav_button(self, label: str, icon_kind: str, accent: str) -> QToolButton:
        """Create one modern navigation button with a custom icon."""
        button = QToolButton()
        button.setObjectName("navButton")
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setIcon(self._build_modern_icon(icon_kind, accent))
        button.setIconSize(QSize(28, 28))
        button.setText(label)
        button.setToolTip(label)
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
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        live_card = QFrame()
        live_card.setObjectName("WorkspaceCard")
        live_layout = QVBoxLayout(live_card)
        live_layout.setContentsMargins(12, 12, 12, 12)
        live_layout.addWidget(self._create_live_view_panel(), 1)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(14)

        acquisition_card = QFrame()
        acquisition_card.setObjectName("WorkspaceCard")
        acquisition_layout = QVBoxLayout(acquisition_card)
        acquisition_layout.setContentsMargins(14, 14, 14, 14)
        acquisition_layout.addWidget(self._create_camera_settings())

        session_card = QFrame()
        session_card.setObjectName("WorkspaceCard")
        session_layout = QVBoxLayout(session_card)
        session_layout.setContentsMargins(14, 14, 14, 14)
        session_layout.addWidget(self._create_control_panel())

        controls_row.addWidget(acquisition_card, 1)
        controls_row.addWidget(session_card, 1)
        controls_row.setStretch(0, 4)
        controls_row.setStretch(1, 5)

        layout.addWidget(live_card, 5)
        layout.addLayout(controls_row, 2)
        return container

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

    def _build_modern_icon(self, kind: str, accent: str) -> QIcon:
        """Draw a simple modern line icon used by tool rails and key actions."""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
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

        painter.end()
        return QIcon(pixmap)

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
        subtitle = QLabel("Choose a Basler or USB source, then arm the live workspace.")
        subtitle.setStyleSheet("color: #8fa6bf;")
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
        self.btn_connect.setMinimumHeight(42)
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

        title = QLabel("CamApp Live View")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #edf4ff;")
        header_layout.addWidget(title)

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
        self.live_image_view.setMinimumSize(720, 480)
        layout.addWidget(self.live_image_view, stretch=1)

        self._show_live_placeholder("CamApp", "Connect a camera to begin preview")
        return panel

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
        header_text = QVBoxLayout()
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header_layout.addLayout(header_text)
        header_layout.addStretch()

        self.btn_planner_fit = QPushButton("Fit Columns")
        self._set_button_icon(self.btn_planner_fit, "settings", "#33d5ff", "ghostButton")
        self.btn_planner_fit.clicked.connect(self._fit_planner_columns)
        header_layout.addWidget(self.btn_planner_fit)

        self.btn_planner_detach = QPushButton("Detach Planner")
        self._set_button_icon(self.btn_planner_detach, "export", "#ffb35d", "orangeButton")
        self.btn_planner_detach.clicked.connect(self._toggle_planner_detach)
        header_layout.addWidget(self.btn_planner_detach)
        layout.addWidget(header)

        button_grid = QGridLayout()
        button_grid.setHorizontalSpacing(10)
        button_grid.setVerticalSpacing(10)

        self.btn_planner_add_trials = QPushButton("Add Trials")
        self._set_button_icon(self.btn_planner_add_trials, "plus", "#35d2ff")
        self.btn_planner_add_trials.setMinimumWidth(138)
        self.btn_planner_add_trials.clicked.connect(self._add_planner_trials)
        button_grid.addWidget(self.btn_planner_add_trials, 0, 0)

        self.btn_planner_add_variable = QPushButton("Add Variable")
        self._set_button_icon(self.btn_planner_add_variable, "session", "#d86cff", "violetButton")
        self.btn_planner_add_variable.setMinimumWidth(138)
        self.btn_planner_add_variable.clicked.connect(self._add_planner_variable)
        button_grid.addWidget(self.btn_planner_add_variable, 0, 1)

        self.btn_planner_import = QPushButton("Import CSV")
        self._set_button_icon(self.btn_planner_import, "import", "#33d5ff", "ghostButton")
        self.btn_planner_import.setMinimumWidth(138)
        self.btn_planner_import.clicked.connect(self._import_planner_trials)
        button_grid.addWidget(self.btn_planner_import, 0, 2)

        self.btn_planner_export = QPushButton("Export CSV")
        self._set_button_icon(self.btn_planner_export, "export", "#ffb35d", "ghostButton")
        self.btn_planner_export.setMinimumWidth(138)
        self.btn_planner_export.clicked.connect(self._export_planner_trials)
        button_grid.addWidget(self.btn_planner_export, 1, 0)

        self.btn_planner_apply = QPushButton("Use Selected Trial")
        self._set_button_icon(self.btn_planner_apply, "check", "#6fe06e", "ghostButton")
        self.btn_planner_apply.setMinimumWidth(138)
        self.btn_planner_apply.clicked.connect(self._apply_selected_planner_trial)
        button_grid.addWidget(self.btn_planner_apply, 1, 1)

        self.btn_planner_remove = QPushButton("Remove")
        self._set_button_icon(self.btn_planner_remove, "record", "#ff6c9e", "dangerButton")
        self.btn_planner_remove.setMinimumWidth(138)
        self.btn_planner_remove.clicked.connect(self._remove_selected_planner_trials)
        button_grid.addWidget(self.btn_planner_remove, 1, 2)
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
        settings_container.addWidget(self.btn_advanced)

        self.advanced_dialog = QDialog(self)
        self.advanced_dialog.setWindowTitle("Advanced Camera Controls")
        self.advanced_dialog.setModal(False)
        self.advanced_dialog.resize(560, 420)
        dialog_layout = QVBoxLayout(self.advanced_dialog)

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

        self.spin_brightness = QDoubleSpinBox()
        self.spin_brightness.valueChanged.connect(self._on_brightness_changed)
        advanced_layout.addRow("Brightness:", self.spin_brightness)

        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.valueChanged.connect(self._on_contrast_changed)
        advanced_layout.addRow("Contrast:", self.spin_contrast)

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
        self.edit_filename.setPlaceholderText("Generated from metadata")
        self.edit_filename.setReadOnly(True)
        filename_layout.addWidget(self.edit_filename, stretch=2)
        control_layout.addLayout(filename_layout)

        self.label_filename_hint = QLabel("Pattern follows the order set in General Settings.")
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

        self.btn_record = QPushButton("Start Recording")
        self._set_button_icon(self.btn_record, "record", "#07260e", "successButton")
        self.btn_record.clicked.connect(self._on_record_clicked)
        self.btn_record.setEnabled(False)
        self.btn_record.setMinimumHeight(50)
        control_layout.addWidget(self.btn_record)

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
        self.btn_apply_behavior_config = QPushButton("Apply Mapping")
        self._set_button_icon(self.btn_apply_behavior_config, "check", "#6fe06e")
        self.btn_apply_behavior_config.clicked.connect(lambda _: self._apply_behavior_pin_configuration(persist=True))
        config_layout.addWidget(self.btn_apply_behavior_config)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        line_group = QGroupBox("Camera Input Labels")
        line_layout = QFormLayout()
        line_options = ["None", "Gate", "Sync", "Barcode", "Lever", "Cue", "Reward", "ITI"]

        self.combo_line1_label = QComboBox()
        self.combo_line1_label.addItems(line_options)
        self.combo_line1_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(1, v))
        line_layout.addRow("Line 1:", self.combo_line1_label)

        self.combo_line2_label = QComboBox()
        self.combo_line2_label.addItems(line_options)
        self.combo_line2_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(2, v))
        line_layout.addRow("Line 2:", self.combo_line2_label)

        self.combo_line3_label = QComboBox()
        self.combo_line3_label.addItems(line_options)
        self.combo_line3_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(3, v))
        line_layout.addRow("Line 3:", self.combo_line3_label)

        self.combo_line4_label = QComboBox()
        self.combo_line4_label.addItems(line_options)
        self.combo_line4_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(4, v))
        line_layout.addRow("Line 4:", self.combo_line4_label)

        line_group.setLayout(line_layout)
        layout.addWidget(line_group)

        self.btn_test_ttl = QPushButton("Test TTL / Behavior")
        self._set_button_icon(self.btn_test_ttl, "pulse", "#33d5ff", "violetButton")
        self.btn_test_ttl.clicked.connect(self._on_test_ttl_clicked)
        self.btn_test_ttl.setEnabled(False)
        self.btn_test_ttl.setMinimumHeight(42)
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
        spin_interval.setRange(0.000, 60.0)
        spin_interval.setSuffix(" s")
        spin_interval.setValue(float(params.get("interval_s", 5.0)))
        form.addRow("Gap After Code:", spin_interval)

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
        word_duration = start_hi_val + start_lo_val + (bits_val * bit_val)
        cycle_duration = word_duration + interval_val

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
            f"bit={bit_val:.3f}s, gap={interval_val:.3f}s, "
            f"word={word_duration:.3f}s, cycle={cycle_duration:.3f}s"
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

        Using a frame target avoids short recordings when the camera reports a
        slightly different real FPS than the requested output FPS.
        """
        max_seconds = self._get_max_record_seconds()
        if max_seconds <= 0:
            return None
        return max(1, int(round(float(self.spin_fps.value()) * max_seconds)))

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

    def _compose_recording_basename(self) -> str:
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

    def _update_filename_preview(self, *_args):
        """Refresh the generated filename preview and formula label."""
        basename = self._compose_recording_basename()
        if hasattr(self, "edit_filename") and self.edit_filename is not None:
            self.edit_filename.setText(basename)
        if hasattr(self, "label_filename_formula") and self.label_filename_formula is not None:
            readable = " / ".join(
                self._filename_key_to_label(key)
                for key in self._selected_filename_order()
            ) or "No filename parts selected"
            self.label_filename_formula.setText(f"{readable}\nPreview: {basename}")
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
        self.label_recording_plan_summary.setText(f"{status}  |  Trial {trial}  |  {animal}")
        if filename:
            self.label_recording_plan_details.setText(
                f"{experiment}  |  {condition}  |  {arena}\nNext file: {filename}"
            )
        else:
            self.label_recording_plan_details.setText(
                f"{experiment}  |  {condition}  |  {arena}"
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
        self.planner_dialog.setWindowTitle("CamApp Planner")
        self.planner_dialog.resize(1500, 900)
        self.planner_dialog.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        dialog_layout = QVBoxLayout(self.planner_dialog)
        dialog_layout.setContentsMargins(12, 12, 12, 12)

        self.planner_panel_widget.setParent(None)
        dialog_layout.addWidget(self.planner_panel_widget)
        self.planner_dialog.finished.connect(self._on_planner_dialog_finished)
        self.planner_detached = True
        self.btn_planner_detach.setText("Reattach Planner")
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
        self.btn_planner_detach.setText("Detach Planner")
        self._set_button_icon(self.btn_planner_detach, "export", "#ffb35d", "orangeButton")
        self.planner_reattaching = False

    def _load_planner_row_into_metadata(self, row: int, announce: bool = False):
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

        try:
            duration_seconds = int(float(payload.get("Duration (s)", "0") or 0))
            if duration_seconds > 0:
                self.check_unlimited.setCurrentText("Limited")
                self.spin_hours.setValue(duration_seconds // 3600)
                self.spin_minutes.setValue((duration_seconds % 3600) // 60)
                self.spin_seconds.setValue(duration_seconds % 60)
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
        self.spin_hours.valueChanged.connect(lambda v: self._save_ui_setting('max_hours', v))
        self.spin_minutes.valueChanged.connect(lambda v: self._save_ui_setting('max_minutes', v))
        self.spin_seconds.valueChanged.connect(lambda v: self._save_ui_setting('max_seconds', v))
        self.check_unlimited.currentIndexChanged.connect(lambda v: self._save_ui_setting('max_unlimited', 1 if v == 1 else 0))
        self._update_filename_preview()
        self._update_planner_summary()

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
        self._set_button_icon(self.btn_connect, "play", "#eef6ff")
        self.btn_record.setEnabled(False)
        self.label_camera_source_hint.setText("No source connected")
        self.label_recording_camera_hint.setText("Camera source is managed from the left Camera panel.")

        self._clear_roi()
        self._show_live_placeholder("Camera Disconnected", "Reconnect a Basler or USB source")
        self._update_live_header(
            status_text="No camera connected",
            resolution_text="-- x --",
            badge_text="Offline",
            badge_tone="warning",
        )
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
                    self._load_planner_row_into_metadata(selected_rows[0].row(), announce=False)
                elif self.active_planner_row is None:
                    self.planner_table.selectRow(0)
                    self._load_planner_row_into_metadata(0, announce=False)

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
            self.worker.set_recording_frame_limit(self._get_target_record_frames())

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
                if self.is_arduino_connected:
                    self._stop_arduino_generation()
                self._sync_active_trial_status("Pending")
                self.current_recording_filepath = None
                return

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

        # Re-enable controls
        self.btn_connect.setEnabled(True)
        self.edit_filename.setEnabled(True)
        if self.is_arduino_connected:
            self.btn_test_ttl.setEnabled(True)

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
            if hasattr(self, "label_file_save_folder") and self.label_file_save_folder is not None:
                self.label_file_save_folder.setText(folder)
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
                actual_fps = self.worker.sync_camera_fps()
            elif self.worker.camera and self.worker.camera.IsOpen():
                self.worker.camera.AcquisitionFrameRate.SetValue(value)
                self.worker.set_target_fps(value)
                actual_fps = self.worker.sync_camera_fps()
            else:
                self.worker.set_target_fps(value)
                actual_fps = None

            if actual_fps and abs(float(actual_fps) - float(value)) > 0.01:
                self._on_status_update(f"FPS set to {value:.3f} (camera reports {actual_fps:.3f})")
            else:
                self._on_status_update(f"FPS set to {value:.3f}")
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
                self._update_live_header(resolution_text=f"{actual_width} x {actual_height}")
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
                self._update_live_header(resolution_text=f"{width} x {height}")
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
        self._update_live_header(mode_text=format_text)

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

    def eventFilter(self, obj, event):
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
                self.btn_arduino_connect.setText("Disconnect Arduino")
                self._set_button_icon(self.btn_arduino_connect, "record", "#ffffff", "dangerButton")
                self.btn_test_ttl.setEnabled(True)
        else:
            if self.is_testing_ttl:
                self.arduino_worker.stop_test()
                self.is_testing_ttl = False
                self.btn_test_ttl.setText("Test TTL / Behavior")
                self._set_button_icon(self.btn_test_ttl, "pulse", "#33d5ff", "violetButton")

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

        cv2.putText(canvas, "CamApp", (406, 222), cv2.FONT_HERSHEY_SIMPLEX, 1.85, (157, 217, 255), 4, cv2.LINE_AA)
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
        """Return elapsed recording time in whole seconds."""
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

        info_text = f"{display_bgr.shape[1]}x{display_bgr.shape[0]}  {self.combo_image_format.currentText()}"
        info_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        info_x = max(18, display_bgr.shape[1] - info_size[0] - 22)
        cv2.putText(display_bgr, info_text, (info_x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (195, 216, 236), 2, cv2.LINE_AA)
        return cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)

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
