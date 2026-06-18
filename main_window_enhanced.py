"""
Enhanced Main GUI Window
PySide6-based interface for Basler, FLIR, and USB camera control with Arduino integration.
"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QLineEdit,
                               QStatusBar, QGroupBox, QSpinBox, QDoubleSpinBox,
                               QFileDialog, QScrollArea, QFormLayout, QTextEdit,
                               QFrame, QSlider, QGridLayout,
                               QCheckBox, QToolButton, QDialog, QStackedWidget, QTabWidget,
                               QDialogButtonBox, QStyle, QToolBar, QToolTip,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QAbstractItemView, QMessageBox, QSizePolicy,
                               QKeySequenceEdit,
                               QMenu, QSplitter, QProgressBar)
from PySide6.QtCore import Qt, Slot, QTimer, QSettings, QSize, QPointF, QRectF, QEvent, QStandardPaths, QUrl
from PySide6.QtGui import (QIcon, QPixmap, QPainter, QColor, QPen,
                           QBrush, QPainterPath, QShortcut,
                           QDesktopServices,
                           QKeySequence, QGuiApplication)
import numpy as np
from datetime import datetime
import pyqtgraph as pg
from collections import deque
import json
from pathlib import Path
import os
import re
import sys
import time
import uuid
import cv2
from typing import Optional, Dict, List
from camera_backends import (
    discover_basler_cameras,
    discover_flir_cameras,
    discover_usb_cameras,
    get_camera_backend_diagnostics,
)
from camera_selection import (
    camera_matches_saved_selection,
    saved_camera_match_score,
    saved_camera_settings_available,
    saved_camera_settings_from_info,
)
from camera_worker import CameraWorker
from camera_stream_manager import CameraStreamManager
from camera_stream_tiles import AuxCameraTile
from metadata_normalization import (
    infer_timestamp_tick_scale,
    normalize_recording_timestamps,
)
from app_theme import ChipLabel, WorkspaceSplitter, build_app_stylesheet
from arduino_output import ArduinoOutputWorker, scan_serial_ports
from auxiliary_arduino import ArduinoDeviceManager
from live_detection_logic import (
    LiveRuleEngine,
    build_rule_label,
    format_roi_properties,
    normalize_output_id,
    occupied_roi_names,
    roi_geometry_properties,
)
from live_detection_metrics import format_live_detection_status, live_result_retention_ms
from live_overlay_utils import (
    clamp_mask_opacity,
    overlay_result_is_current,
    scale_live_detection_result_to_shape,
)
from live_overlay_quality import (
    identity_color_rgb,
    skeleton_for_keypoint_count,
)
from live_detection_panel import LiveDetectionPanel
from live_detection_types import (
    BehaviorROI,
    LiveDetectionResult,
    LiveTriggerRule,
    PreviewFramePacket,
    normalize_activation_pattern,
)
from live_inference_worker import LiveInferenceConfig, LiveInferenceWorker
from overlay_video_export import OverlayVideoFrameTask, OverlayVideoRecorder
from audio_recorder import UltrasoundPanel
from recording_timing import (
    build_recording_timing_warnings,
    capture_duration_seconds,
    encoded_video_duration_seconds,
    measured_capture_fps,
    percent_delta,
)
from user_flag_utils import project_user_flag_events

APP_NAME = "PyKaboo"
APP_SETTINGS_ORGANIZATION = "PyKaboo"
PLANNER_DURATION_HEADER = "Duration (HH:MM:SS)"
LEGACY_PLANNER_DURATION_HEADER = "Duration (s)"
PLANNER_RECORDING_BASE_ROLE = Qt.ItemDataRole.UserRole
PLANNER_MANUAL_PENDING_ROLE = Qt.ItemDataRole.UserRole + 1
PLANNER_MANUAL_ACQUIRED_ROLE = Qt.ItemDataRole.UserRole + 2
STARTUP_CAMERA_AUTOCONNECT_MAX_ATTEMPTS = 5
STARTUP_CAMERA_AUTOCONNECT_RETRY_MS = 1000


class ZeroPaddedSpinBox(QSpinBox):
    """Spin box that renders values with leading zeros."""

    def __init__(self, digits: int = 2, parent=None):
        super().__init__(parent)
        self._digits = max(1, int(digits))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def textFromValue(self, value: int) -> str:
        return f"{int(value):0{self._digits}d}"


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

    DISPLAY_SIGNAL_META = {
        "gate": {"state_key": "gate", "group": "ttl", "name": "Gate", "role": "Output", "default_pins": [3], "color": "#22c55e"},
        "sync": {"state_key": "sync", "group": "ttl", "name": "Sync", "role": "Output", "default_pins": [9], "color": "#38bdf8"},
        "barcode": {"state_key": "barcode0", "group": "ttl", "name": "Barcode", "role": "Output", "default_pins": [18], "color": "#f97316"},
        "lever": {"state_key": "lever", "group": "behavior", "name": "Lever", "role": "Input", "default_pins": [14], "color": "#facc15"},
        "cue": {"state_key": "cue", "group": "behavior", "name": "Cue LED", "role": "Output", "default_pins": [45], "color": "#34d399"},
        "reward": {"state_key": "reward", "group": "behavior", "name": "Reward LED", "role": "Output", "default_pins": [21], "color": "#60a5fa"},
        "iti": {"state_key": "iti", "group": "behavior", "name": "ITI LED", "role": "Output", "default_pins": [46], "color": "#ef4444"},
    }
    DISPLAY_SIGNAL_ORDER = ["gate", "sync", "barcode", "lever", "cue", "reward", "iti"]
    BEHAVIOR_PIN_KEYS = ["gate", "sync", "barcode", "lever", "cue", "reward", "iti"]
    CAMERA_LINE_KEYS = ("line1_status", "line2_status", "line3_status", "line4_status")
    LIVE_ROI_OCCUPIED_COLOR = (34, 197, 94)
    SIDE_PANEL_MIN_FLOOR = 200

    def __init__(self):
        super().__init__()

        self.worker: Optional[CameraWorker] = None
        self.camera_stream_manager: Optional[CameraStreamManager] = None
        self.aux_camera_tiles: list = []
        self.camera_grid_container: Optional[QWidget] = None
        self.camera_grid_layout: Optional[QGridLayout] = None
        self.connected_camera_info: Optional[Dict] = None
        self.live_tracking_mode_active = False
        self.btn_tracking_mode: Optional[QPushButton] = None
        self.arduino_worker: Optional[ArduinoOutputWorker] = None
        self.aux_arduino_manager: Optional[ArduinoDeviceManager] = None
        self.aux_device_widgets: Dict[str, Dict[str, object]] = {}
        self.aux_selected_device_id: Optional[str] = None
        self.aux_row_widgets: List[Dict[str, object]] = []
        self.is_camera_connected = False
        self.is_arduino_connected = False
        self.is_testing_ttl = False

        # Settings
        self.settings = QSettings(APP_SETTINGS_ORGANIZATION, APP_NAME)
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
        self.recording_start_anchor_locked = False
        self.current_recording_filepath = None
        self.recording_timer = QTimer(self)
        self.recording_timer.timeout.connect(self._update_recording_time)
        self.recording_duration_timer = QTimer(self)
        self.recording_duration_timer.setSingleShot(True)
        self.recording_duration_timer.timeout.connect(self._on_recording_duration_timeout)
        self.recording_first_frame_wallclock = None
        self.recording_last_frame_wallclock = None
        self.recording_stop_requested_at = None
        self.recording_stop_reason = ""
        self._audio_video_start_marked = False
        self.active_recording_timing_audit: Dict[str, object] = {}
        self.last_recording_timing_audit: Dict[str, object] = {}
        self.last_audio_recording_metadata: Dict[str, object] = {}
        self.space_record_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_record_shortcut.setContext(Qt.WindowShortcut)
        self.space_record_shortcut.activated.connect(self._on_space_record_shortcut)
        self.user_flag_shortcut_bindings: list[QShortcut] = []

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
        self.camera_line_time_data = deque(maxlen=self.ttl_max_points)
        self.camera_line_plot_data: Dict[str, deque] = {
            key: deque(maxlen=self.ttl_max_points)
            for key in self.CAMERA_LINE_KEYS
        }
        self.ttl_output_curves: Dict[str, pg.PlotDataItem] = {}
        self.behavior_curves: Dict[str, pg.PlotDataItem] = {}
        self.camera_line_curves: Dict[str, pg.PlotDataItem] = {}
        self.ttl_output_levels: Dict[str, float] = {}
        self.behavior_levels: Dict[str, float] = {}
        self.camera_line_levels: Dict[str, float] = {}
        self.camera_line_plot_start_time_s: Optional[float] = None
        self.camera_line_last_signature = None
        # Throttle the pyqtgraph line-plot refresh during recording so that
        # high-FPS cameras don't flood the GUI thread (the plot curve + range
        # update is the single biggest per-frame cost on the main thread).
        self._camera_line_plot_last_paint_s: float = 0.0
        self._camera_line_plot_min_interval_s: float = 1.0 / 30.0
        self.ttl_state_labels: Dict[str, QLabel] = {}
        self.ttl_count_labels: Dict[str, QLabel] = {}
        self.behavior_state_labels: Dict[str, QLabel] = {}
        self.behavior_count_labels: Dict[str, QLabel] = {}
        self.ttl_counts_layout: Optional[QGridLayout] = None
        self.behavior_counts_layout: Optional[QGridLayout] = None
        self.pin_value_labels: Dict[str, QLabel] = {}
        self.pin_name_labels: Dict[str, QLabel] = {}
        self.pin_row_widgets: Dict[str, List[QWidget]] = {}
        self.behavior_pin_edits: Dict[str, QLineEdit] = {}
        self.behavior_role_boxes: Dict[str, QComboBox] = {}
        self.signal_label_edits: Dict[str, QLineEdit] = {}
        self.signal_enabled_checks: Dict[str, QCheckBox] = {}
        self.signal_mapping_row_widgets: Dict[str, List[QWidget]] = {}
        self.camera_line_row_widgets: Dict[int, List[QWidget]] = {}
        self.camera_line_selector_display_names: Dict[int, str] = self._build_camera_line_selector_display_names()
        self.sync_param_button: Optional[QToolButton] = None
        self.barcode_param_button: Optional[QToolButton] = None
        self.pin_defaults_group: Optional[QGroupBox] = None
        self.signal_mapping_group: Optional[QGroupBox] = None
        self.camera_line_labels_group: Optional[QGroupBox] = None
        self.hw_barcode_frame: Optional[QFrame] = None
        self.label_hw_barcode_mode: Optional[QLabel] = None
        self.label_hw_barcode_counter: Optional[QLabel] = None
        self.label_hw_barcode_bit: Optional[QLabel] = None
        self.label_hw_barcode_bitwidth: Optional[QLabel] = None
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
        self.audio_panel: Optional[UltrasoundPanel] = None
        self.live_inference_worker: Optional[LiveInferenceWorker] = None
        self.live_detection_enabled = False
        self.live_detection_last_result: Optional[LiveDetectionResult] = None
        self.live_detection_result_history: deque[LiveDetectionResult] = deque(maxlen=256)
        self.live_inference_frame_cache: Dict[int, PreviewFramePacket] = {}
        self.live_detection_results_by_frame: Dict[int, LiveDetectionResult] = {}
        self.live_overlay_pending_packets: Dict[int, PreviewFramePacket] = {}
        self.live_overlay_last_written_frame_index: Optional[int] = None
        self.live_synced_overlay_active = False
        self.live_synced_overlay_last_update_s = 0.0
        self.live_preview_scene = None
        self.live_preview_packet: Optional[PreviewFramePacket] = None
        self.live_preview_frame_index = -1
        self.live_preview_timestamp_s = 0.0
        self.live_rule_engine = LiveRuleEngine()
        self.live_rule_timer = QTimer(self)
        self.live_rule_timer.setInterval(10)
        self.live_rule_timer.timeout.connect(self._on_live_rule_timer_timeout)
        self.live_rois: Dict[str, BehaviorROI] = {}
        self.live_rules: List[LiveTriggerRule] = []
        self.live_output_mapping: Dict[str, List[int]] = {f"DO{i}": [] for i in range(1, 9)}
        self.live_active_rule_ids: List[str] = []
        self.live_output_states: Dict[str, bool] = {f"DO{i}": False for i in range(1, 9)}
        self.live_level_output_states: Dict[str, bool] = {f"DO{i}": False for i in range(1, 9)}
        self.latest_ttl_states: Dict[str, object] = {}
        self.user_flag_events: List[Dict[str, object]] = []
        self.user_flag_preview_text = ""
        self.user_flag_preview_until_s = 0.0
        self.live_roi_draw_mode = ""
        self.live_roi_draw_points: List[tuple[float, float]] = []
        self.live_roi_circle_center: Optional[tuple[float, float]] = None
        self.live_roi_drawing_name = ""
        self.live_circle_roi_items: Dict[str, pg.CircleROI] = {}
        self._syncing_live_circle_roi_item = False
        self.live_recording_detection_rows: List[Dict[str, object]] = []
        self.live_recording_frame_rows: Dict[int, Dict[str, object]] = {}
        self.live_recording_roi_states: Dict[str, bool] = {}
        self.live_recording_coco_images: Dict[int, Dict[str, object]] = {}
        self.live_recording_coco_annotations: List[Dict[str, object]] = []
        self.live_recording_coco_categories: Dict[int, Dict[str, object]] = {}
        self.live_recording_coco_next_annotation_id = 1
        self.live_overlay_video_enabled = False
        self.live_overlay_video_recorder: Optional[OverlayVideoRecorder] = None
        self.live_overlay_video_path = ""
        self.live_overlay_video_fps = 0.0
        self._startup_autoconnect_done = False
        self._startup_camera_autoconnect_attempts = 0
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
        self.center_splitter: Optional[QSplitter] = None
        self.preview_height_slider: Optional[QSlider] = None
        self.workspace_controls_scroll: Optional[QScrollArea] = None
        self.workspace_controls_row = None
        self._workspace_controls_stacked = False
        self.workspace_toolbar: Optional[QToolBar] = None
        self.workspace_root_layout: Optional[QHBoxLayout] = None
        self._responsive_layout_refresh_pending = False
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
            "Session",
            "Experiment",
            "Condition",
            "Start Delay (s)",
            PLANNER_DURATION_HEADER,
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
        self.action_planner_load_last = None
        self.btn_planner_menu: Optional[QToolButton] = None
        self._planner_fit_pending = False
        self._syncing_planner_to_recording = False
        self._syncing_recording_to_planner = False
        self._syncing_planner_recording_statuses = False
        self._planner_state_loading = False
        self._planner_autosave_enabled = False
        self.advanced_dialog: Optional[QDialog] = None
        self.filename_order_boxes: List[QComboBox] = []
        self._filename_field_syncing = False
        self._custom_filename_override = str(self.settings.value("recording_filename_override", "") or "").strip()
        self.check_organize_session_folders: Optional[QCheckBox] = None
        self.folder_order_boxes: List[QComboBox] = []
        self.folder_structure_group: Optional[QGroupBox] = None
        self.label_folder_structure_preview: Optional[QLabel] = None
        self.edit_path_preview: Optional[QLineEdit] = None
        self.btn_open_folder: Optional[QPushButton] = None
        self.btn_create_folders: Optional[QPushButton] = None
        self.user_flag_configs: List[Dict[str, object]] = []
        self.label_user_flag_summary: Optional[QLabel] = None
        self.label_user_flag_details: Optional[QLabel] = None
        self.label_user_flag_pin_summary: Optional[QLabel] = None
        self.btn_manage_user_flags: Optional[QPushButton] = None
        self.user_flag_dialog: Optional[QDialog] = None
        self.user_flag_table: Optional[QTableWidget] = None
        self.btn_add_user_flag: Optional[QPushButton] = None
        self.btn_edit_user_flag: Optional[QPushButton] = None
        self.btn_remove_user_flag: Optional[QPushButton] = None
        self.meta_session: Optional[QLineEdit] = None
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
        self.label_recording_session_header: Optional[QLabel] = None
        # Compact session summary shown in the live-view top bar, plus the
        # floating recording countdown overlay pinned to the live view.
        self.live_header_session: Optional[QLabel] = None
        self.recording_overlay: Optional[QFrame] = None

        # Metadata
        self.metadata = {}

        self.setWindowTitle(APP_NAME)
        self.setGeometry(50, 50, 1600, 900)
        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

        # Workspace theme: global palette + generated glyph assets.
        self.setStyleSheet(build_app_stylesheet())

        self._init_ui()
        self._load_ui_settings()
        self._setup_worker()
        self._setup_arduino_worker()
        self._load_live_detection_settings()
        self._load_metadata()
        self._scan_cameras()
        QTimer.singleShot(0, self._restore_center_splitter_sizes)
        QTimer.singleShot(250, self._auto_connect_startup_devices)

    def _migrate_legacy_settings(self):
        """Copy saved settings from the previous app identity once."""
        existing_keys = set(self.settings.allKeys())
        copied = False

        for organization, application in (
            ("CamApp Live Detection", "CamApp Live Detection"),
            ("BaslerCam", "CameraApp"),
        ):
            legacy_settings = QSettings(organization, application)
            for key in legacy_settings.allKeys():
                if key in existing_keys:
                    continue
                self.settings.setValue(key, legacy_settings.value(key))
                existing_keys.add(key)
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
        self.workspace_root_layout = root_layout

        left_rail = self._create_nav_rail("left")
        self.left_panel_shell, self.left_panel_title, self.left_panel_stack = self._create_side_panel_shell("Session", "left")
        right_rail = self._create_nav_rail("right")
        self.right_panel_shell, self.right_panel_title, self.right_panel_stack = self._create_side_panel_shell("Arduino", "right")

        camera_page = self._wrap_scroll_dock_widget(self._create_camera_connection_panel())
        settings_page = self._wrap_scroll_dock_widget(self._create_general_settings_panel())
        session_page = self._wrap_scroll_dock_widget(self._create_session_hub_panel())
        file_page = self._wrap_scroll_dock_widget(self._create_file_tools_panel())
        ttl_page = self._wrap_scroll_dock_widget(self._create_ttl_monitor_panel())
        behavior_page = self._wrap_scroll_dock_widget(self._create_behavior_monitor_panel())
        arduino_page = self._wrap_scroll_dock_widget(self._create_behavior_setup_panel())
        live_detection_page = self._wrap_scroll_dock_widget(self._create_live_detection_panel())
        audio_page = self._wrap_scroll_dock_widget(self._create_audio_recording_panel())
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
            "audio": audio_page,
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
        self.label_fps.setToolTip("Acquisition frame rate reported by the camera")
        self.label_buffer = QLabel("Buffer: 0%")
        self.label_buffer.setToolTip(
            "Frame buffer fill level. Sustained high values mean the encoder\n"
            "or disk cannot keep up with the camera."
        )
        self.label_recording = QLabel("Not Recording")
        self.label_recording_time = QLabel("00:00:00")
        self.label_recording_time.setToolTip("Elapsed recording time")
        for status_label in (
            self.label_fps,
            self.label_buffer,
            self.label_recording,
            self.label_recording_time,
        ):
            status_label.setObjectName("statusMetric")

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
                ("audio", "Audio", "mic", "#ffd166", "Ultrasound Audio"),
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
        shell.setMinimumWidth(240)
        shell.setMaximumWidth(420)
        shell.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        shell.setVisible(False)

        layout = QVBoxLayout(shell)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("PanelHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 8, 14, 8)
        header_layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #eef6ff;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        hide_button = QToolButton()
        hide_button.setObjectName("panelCloseButton")
        hide_button.setIcon(self._build_modern_icon("close", "#9fb8d4"))
        hide_button.setIconSize(QSize(14, 14))
        hide_button.setFixedSize(28, 28)
        hide_button.setCursor(Qt.PointingHandCursor)
        hide_button.setToolTip("Close this panel (its rail button reopens it)")
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
        self._update_side_panel_bounds()

    def _update_side_panel_bounds(self):
        """Adjust side-drawer widths to remain readable across window sizes."""
        window_width = max(0, self._workspace_available_width())
        session_active = self.current_left_panel_key == "session" and not self.planner_detached
        settings_active = self.current_left_panel_key == "settings"
        left_visible = self.left_panel_shell is not None and self.left_panel_shell.isVisible()
        right_visible = self.right_panel_shell is not None and self.right_panel_shell.isVisible()
        if window_width >= 1850:
            left_bounds = (360, 460)
            right_bounds = (360, 460)
        elif window_width >= 1600:
            left_bounds = (330, 420)
            right_bounds = (330, 410)
        elif window_width >= 1360:
            left_bounds = (280, 360)
            right_bounds = (260, 330)
        elif window_width >= 1180:
            left_bounds = (260, 330)
            right_bounds = (240, 300)
        else:
            left_bounds = (230, 300)
            right_bounds = (220, 280)

        if session_active:
            if not right_visible and window_width >= 1850:
                left_bounds = (760, 920)
            elif not right_visible and window_width >= 1600:
                left_bounds = (680, 840)
            elif not right_visible and window_width >= 1360:
                left_bounds = (460, 620)
            elif not right_visible:
                left_bounds = (320, 460)
            elif window_width >= 1850:
                left_bounds = (560, 720)
            elif window_width >= 1600:
                left_bounds = (500, 640)
            elif window_width >= 1360:
                left_bounds = (420, 540)
            else:
                left_bounds = (320, 460)
        elif settings_active:
            if not right_visible and window_width >= 1850:
                left_bounds = (620, 760)
            elif not right_visible and window_width >= 1600:
                left_bounds = (560, 700)
            elif not right_visible and window_width >= 1360:
                left_bounds = (420, 560)
            elif not right_visible:
                left_bounds = (320, 460)
            elif window_width >= 1850:
                left_bounds = (520, 660)
            elif window_width >= 1600:
                left_bounds = (460, 580)
            elif window_width >= 1360:
                left_bounds = (360, 480)
            else:
                left_bounds = (300, 420)

        # Decouple the comfortable width (what the panel renders at when there
        # is room) from the hard minimum (what it forces onto the window). The
        # panels render at their comfortable width while the window is wide
        # enough to hold both plus the preview; once it is not, the minimum
        # drops to a small floor so the window can keep shrinking instead of
        # clipping content. _auto_collapse_overflowing_panel handles the case
        # where even the floor will not fit.
        floor = self.SIDE_PANEL_MIN_FLOOR
        reserve = self._minimum_preview_reserve(window_width)
        fixed = self._workspace_fixed_width()
        needed_comfortable = fixed + reserve
        if left_visible:
            needed_comfortable += left_bounds[0]
        if right_visible:
            needed_comfortable += right_bounds[0]
        afford_comfortable = window_width >= needed_comfortable

        def resolve(bounds: tuple[int, int], visible: bool) -> tuple[int, int]:
            min_w, max_w = bounds
            if not visible or afford_comfortable:
                return min_w, max(min_w, max_w)
            return floor, max(floor, max_w)

        left_bounds = resolve(left_bounds, left_visible)
        right_bounds = resolve(right_bounds, right_visible)

        if self.left_panel_shell is not None:
            self.left_panel_shell.setMinimumWidth(left_bounds[0])
            self.left_panel_shell.setMaximumWidth(left_bounds[1])
            self.left_panel_shell.updateGeometry()
        if self.right_panel_shell is not None:
            self.right_panel_shell.setMinimumWidth(right_bounds[0])
            self.right_panel_shell.setMaximumWidth(right_bounds[1])
            self.right_panel_shell.updateGeometry()
        if self.workspace_root_layout is not None:
            self.workspace_root_layout.invalidate()
        if session_active:
            self._schedule_planner_column_fit()

    def _workspace_available_width(self) -> int:
        """Return the usable central-widget width in Qt logical pixels."""
        if self.dock_area is not None and self.dock_area.width() > 0:
            return int(self.dock_area.width())
        return int(self.width())

    def _workspace_fixed_width(self) -> int:
        """Width consumed by margins, rails, and gaps before side panels."""
        if self.workspace_root_layout is None:
            return 0
        margins = self.workspace_root_layout.contentsMargins()
        spacing = max(0, int(self.workspace_root_layout.spacing()))
        return margins.left() + margins.right() + 2 * 62 + 4 * spacing

    def _minimum_preview_reserve(self, window_width: int) -> int:
        """Reserve enough width for the live preview and recording controls."""
        if window_width >= 1850:
            return 720
        if window_width >= 1600:
            return 600
        if window_width >= 1360:
            return 500
        if window_width >= 1180:
            return 420
        return 360

    def _schedule_responsive_layout_refresh(self):
        """Run one layout pass after Qt finishes a resize or window-state change."""
        if self._responsive_layout_refresh_pending:
            return
        self._responsive_layout_refresh_pending = True
        QTimer.singleShot(0, self._run_responsive_layout_refresh)

    def _run_responsive_layout_refresh(self):
        self._responsive_layout_refresh_pending = False
        self._auto_collapse_overflowing_panel()
        self._update_side_panel_bounds()
        self._update_workspace_controls_orientation()
        if self.workspace_root_layout is not None:
            self.workspace_root_layout.activate()
        if self.live_image_view is not None:
            self.live_image_view.updateGeometry()
        self._schedule_planner_column_fit()

    def _two_panel_min_width(self) -> int:
        """Smallest window width that can still hold both drawers plus preview."""
        return (
            self._workspace_fixed_width()
            + 2 * self.SIDE_PANEL_MIN_FLOOR
            + self._minimum_preview_reserve(self._workspace_available_width())
        )

    def _auto_collapse_overflowing_panel(self):
        """When the window is too narrow for both drawers, close the right one.

        This prevents the centre workspace (and the Recording card inside it)
        from being squeezed until its controls clip off-screen. The left
        drawer is kept because it carries the session/planner context.
        """
        left_visible = self.left_panel_shell is not None and self.left_panel_shell.isVisible()
        right_visible = self.right_panel_shell is not None and self.right_panel_shell.isVisible()
        if not (left_visible and right_visible):
            return
        if self._workspace_available_width() >= self._two_panel_min_width():
            return
        self._hide_side_panel("right")

    def _update_workspace_controls_orientation(self):
        """Stack the Acquisition/Recording cards when the centre column is narrow."""
        row = self.workspace_controls_row
        if row is None:
            return
        column_width = 0
        if self.center_splitter is not None and self.center_splitter.width() > 0:
            column_width = self.center_splitter.width()
        elif self.workspace_controls_content is not None:
            column_width = self.workspace_controls_content.width()
        if column_width <= 0:
            return
        # Stack before the Recording card (the wider of the two) would be
        # squeezed below the width its buttons need.
        should_stack = column_width < 1020
        if should_stack == self._workspace_controls_stacked:
            return
        self._workspace_controls_stacked = should_stack
        if should_stack:
            row.setDirection(QHBoxLayout.TopToBottom)
            row.setSpacing(12)
            row.setStretch(0, 0)
            row.setStretch(1, 0)
        else:
            row.setDirection(QHBoxLayout.LeftToRight)
            row.setSpacing(14)
            row.setStretch(0, 4)
            row.setStretch(1, 5)

    def _ensure_side_panel_fit(self, side: str):
        """Keep side panels usable on narrower windows by collapsing the opposite drawer."""
        if self._workspace_available_width() >= self._two_panel_min_width():
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
        self._update_side_panel_bounds()
        if side == "left" and panel_key == "session":
            self._schedule_planner_column_fit()

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
        acquisition_layout.setContentsMargins(10, 10, 10, 10)
        acquisition_layout.addWidget(self._create_camera_settings())

        self.recording_workspace_card = QFrame()
        self.recording_workspace_card.setObjectName("WorkspaceCard")
        self.recording_workspace_card.setMinimumWidth(0)
        self.recording_workspace_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        recording_layout = QVBoxLayout(self.recording_workspace_card)
        recording_layout.setContentsMargins(10, 10, 10, 10)
        recording_layout.addWidget(self._create_control_panel())

        if self.btn_record is None:
            self.btn_record = QPushButton("Start Recording")
            self._set_button_icon(self.btn_record, "record", "#07260e", "successButton")
            self.btn_record.setToolTip("Start or stop recording on every connected stream (Spacebar)")
            self.btn_record.clicked.connect(self._on_record_clicked)
            self.btn_record.setEnabled(False)
            self.btn_record.setMinimumHeight(36)
            self.btn_record.setMinimumWidth(140)
            self.btn_record.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.btn_record.setMaximumWidth(220)

        controls_shell = QFrame()
        controls_shell.setObjectName("WorkspaceCard")
        controls_shell.setMinimumWidth(0)
        controls_shell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        controls_layout = QVBoxLayout(controls_shell)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(10)

        controls_toolbar = QHBoxLayout()
        controls_toolbar.setSpacing(8)

        # Slim toolbar: the old "Workspace Controls" title and resize hint were
        # pure chrome, so the panel toggles and the record button now sit on a
        # single compact row and stay pushed to the right.
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
        # Kept so the Acquisition/Recording cards can reflow from side-by-side
        # to stacked when the centre column gets narrow (see _update_workspace
        # _controls_orientation), which stops the Recording card clipping.
        self.workspace_controls_row = controls_row

        # The open panels live inside a scroll area so the operator can give
        # the preview as much (or as little) height as they want: when the
        # controls area is squeezed, it scrolls instead of clipping.
        self.workspace_controls_scroll = QScrollArea()
        self.workspace_controls_scroll.setWidgetResizable(True)
        self.workspace_controls_scroll.setFrameShape(QFrame.NoFrame)
        # Horizontal scroll is left "as needed" on purpose: with it forced off
        # and widgetResizable on, the scroll area would propagate the controls
        # row's full width as a window minimum (the window then refuses to
        # shrink). As-needed keeps the window freely resizable and guarantees
        # the Recording card is never clipped — it scrolls as a last resort,
        # though vertical stacking usually avoids that entirely.
        self.workspace_controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.workspace_controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.workspace_controls_scroll.setWidget(self.workspace_controls_content)
        controls_layout.addWidget(self.workspace_controls_scroll, 1)

        # Vertical splitter so the operator can freely resize the live preview
        # against the controls area by dragging the grip divider.
        self.center_splitter = WorkspaceSplitter(Qt.Vertical)
        self.center_splitter.setObjectName("workspaceSplitter")
        self.center_splitter.setChildrenCollapsible(False)
        self.center_splitter.setHandleWidth(12)
        live_card.setMinimumHeight(220)
        controls_shell.setMinimumHeight(64)
        self.center_splitter.addWidget(live_card)
        self.center_splitter.addWidget(controls_shell)
        self.center_splitter.setStretchFactor(0, 1)
        self.center_splitter.setStretchFactor(1, 0)
        self.center_splitter.splitterMoved.connect(self._on_center_splitter_moved)
        layout.addWidget(self.center_splitter, 1)
        self._update_workspace_controls_visibility()
        # The recording controls now exist, so populate the top-bar session chip
        # (early planner/metadata signals were skipped by the guard above).
        self._refresh_recording_session_summary()
        return container

    def _on_center_splitter_moved(self, *_args):
        """Persist the preview/controls split so it survives restarts."""
        if self.center_splitter is None:
            return
        sizes = self.center_splitter.sizes()
        if len(sizes) == 2 and min(sizes) >= 0:
            self._save_ui_setting("workspace_splitter_sizes", ",".join(str(s) for s in sizes))
        self._sync_preview_height_slider()

    def _is_any_workspace_panel_open(self) -> bool:
        """True when the Acquisition or Recording bottom panel is expanded."""
        acq = self.btn_toggle_acquisition_panel
        rec = self.btn_toggle_recording_panel
        return bool(
            (acq is not None and acq.isChecked())
            or (rec is not None and rec.isChecked())
        )

    def _restore_center_splitter_sizes(self):
        # The Acquisition/Recording panels start closed, so at launch the live
        # preview should fill the workspace. A split saved while a panel was
        # open (or after the preview was dragged smaller) would otherwise
        # restore a shrunken preview sitting above a large empty controls strip.
        # Only honour a saved split while a panel is actually open; with both
        # closed, maximise the preview by collapsing the controls to their
        # toolbar.
        if not self._is_any_workspace_panel_open():
            self._sync_center_splitter_to_controls()
            return
        raw_value = str(self.settings.value("workspace_splitter_sizes", "") or "")
        if not raw_value:
            self._sync_preview_height_slider()
            return
        try:
            sizes = [int(part) for part in raw_value.split(",")]
        except ValueError:
            return
        if len(sizes) == 2 and sum(sizes) > 0 and self.center_splitter is not None:
            self.center_splitter.setSizes(sizes)
        self._sync_preview_height_slider()

    def _on_preview_height_slider_changed(self, percent: int):
        """Resize the live preview to take the requested share of the column."""
        if self.center_splitter is None:
            return
        sizes = self.center_splitter.sizes()
        total = sum(sizes)
        if total <= 0 or len(sizes) != 2:
            return
        fraction = max(25, min(100, int(percent))) / 100.0
        preview_height = int(total * fraction)
        preview_height = max(220, min(preview_height, total - 64))
        self.center_splitter.setSizes([preview_height, total - preview_height])
        self._save_ui_setting(
            "workspace_splitter_sizes",
            f"{preview_height},{total - preview_height}",
        )

    def _sync_preview_height_slider(self):
        """Reflect the actual splitter split in the header slider."""
        if self.preview_height_slider is None or self.center_splitter is None:
            return
        sizes = self.center_splitter.sizes()
        total = sum(sizes)
        if total <= 0 or len(sizes) != 2:
            return
        percent = int(round(sizes[0] * 100.0 / total))
        self.preview_height_slider.blockSignals(True)
        self.preview_height_slider.setValue(max(25, min(100, percent)))
        self.preview_height_slider.blockSignals(False)

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
        any_panel_visible = acquisition_visible or recording_visible
        if self.workspace_controls_content is not None:
            self.workspace_controls_content.setVisible(any_panel_visible)
        if self.workspace_controls_scroll is not None:
            self.workspace_controls_scroll.setVisible(any_panel_visible)
        self._update_workspace_controls_orientation()
        if self.center_splitter is not None:
            QTimer.singleShot(0, self._sync_center_splitter_to_controls)

    def _sync_center_splitter_to_controls(self):
        """Re-balance the preview/controls split after panels open or close."""
        if self.center_splitter is None:
            return
        sizes = self.center_splitter.sizes()
        total = sum(sizes)
        if total <= 0 or len(sizes) != 2:
            return
        controls_widget = self.center_splitter.widget(1)
        if controls_widget is None:
            return
        hint = controls_widget.sizeHint().height()
        content = self.workspace_controls_content
        if content is not None and content.isVisible():
            # The scroll area reports a small hint; size to the real content
            # (plus toolbar row and card margins) so panels open fully.
            hint = max(hint, content.sizeHint().height() + 92)
        hint = max(64, min(hint, int(total * 0.62)))
        self.center_splitter.setSizes([max(220, total - hint), hint])
        self._sync_preview_height_slider()

    def _create_metric_tile(self, title: str, value: str, accent: str):
        """Create a compact dashboard tile used for planner/session counts."""
        tile = QFrame()
        tile.setObjectName("MetricTile")
        tile.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        tile.setFixedHeight(54)
        tile.setStyleSheet(
            "QFrame#MetricTile {"
            " background-color: #0d1827; border: 1px solid #223750;"
            f" border-left: 3px solid {accent}; border-radius: 11px; }}"
        )
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(11, 6, 11, 6)
        layout.setSpacing(1)

        title_label = QLabel(title.upper())
        title_label.setStyleSheet("color: #7f96ad; font-size: 9px; font-weight: 700;")
        value_label = QLabel(value)
        value_label.setStyleSheet(f"color: {accent}; font-size: 17px; font-weight: 800;")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return tile, value_label

    def _paint_modern_icon(self, painter: QPainter, kind: str, accent: str):
        """Paint one icon glyph into a normalized 32x32 canvas.

        All glyphs share a single 2.1px round-joined stroke so the icon set
        reads as one consistent family across the rails, buttons, and toolbars.
        """
        color = QColor(accent)
        pen = QPen(color, 2.1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        if kind == "camera":
            painter.drawRoundedRect(QRectF(4.5, 9.5, 23, 14.5), 4.5, 4.5)
            painter.drawRect(QRectF(10, 6.5, 7, 3.5))
            painter.drawEllipse(QPointF(16, 16.7), 4.4, 4.4)
        elif kind == "settings":
            # Two slider tracks with offset knobs: clean "controls" metaphor.
            painter.drawLine(7, 12, 25, 12)
            painter.drawLine(7, 20, 25, 20)
            painter.setBrush(QColor("#0b1521"))
            painter.drawEllipse(QPointF(12.5, 12), 3.0, 3.0)
            painter.drawEllipse(QPointF(20.5, 20), 3.0, 3.0)
            painter.setBrush(Qt.NoBrush)
        elif kind == "session":
            # Clipboard with a check: the planner / session metaphor.
            painter.drawRoundedRect(QRectF(7.5, 7, 17, 19), 3.5, 3.5)
            painter.drawRoundedRect(QRectF(12, 4.5, 8, 4.5), 1.8, 1.8)
            painter.drawLine(11.5, 18.5, 14.5, 21.5)
            painter.drawLine(14.5, 21.5, 20.5, 13.5)
        elif kind == "folder":
            path = QPainterPath()
            path.moveTo(5.5, 11.5)
            path.lineTo(12, 11.5)
            path.lineTo(14.5, 8.5)
            path.lineTo(26.5, 8.5)
            path.lineTo(26.5, 24)
            path.lineTo(5.5, 24)
            path.closeSubpath()
            painter.drawPath(path)
        elif kind == "chip":
            painter.drawRoundedRect(QRectF(9, 9, 14, 14), 3, 3)
            painter.drawRoundedRect(QRectF(13, 13, 6, 6), 1.5, 1.5)
            for offset in (12, 16, 20):
                painter.drawLine(offset, 5.5, offset, 9)
                painter.drawLine(offset, 23, offset, 26.5)
                painter.drawLine(5.5, offset, 9, offset)
                painter.drawLine(23, offset, 26.5, offset)
        elif kind == "pulse":
            path = QPainterPath()
            path.moveTo(4.5, 17)
            path.lineTo(10, 17)
            path.lineTo(13, 11)
            path.lineTo(17, 23.5)
            path.lineTo(20, 10)
            path.lineTo(23, 17)
            path.lineTo(27.5, 17)
            painter.drawPath(path)
        elif kind == "behavior":
            # Connected nodes (network) — three filled dots linked.
            painter.drawLine(10.5, 11, 21, 16)
            painter.drawLine(10.5, 22, 21, 16)
            painter.drawLine(10.5, 11, 10.5, 22)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(10.5, 11), 2.6, 2.6)
            painter.drawEllipse(QPointF(21, 16), 2.6, 2.6)
            painter.drawEllipse(QPointF(10.5, 22), 2.6, 2.6)
            painter.setBrush(Qt.NoBrush)
        elif kind == "plus":
            painter.drawLine(16, 7.5, 16, 24.5)
            painter.drawLine(7.5, 16, 24.5, 16)
        elif kind == "import":
            # Download into tray.
            painter.drawLine(16, 6.5, 16, 19)
            painter.drawLine(10.5, 13.5, 16, 19)
            painter.drawLine(21.5, 13.5, 16, 19)
            painter.drawLine(7.5, 23.5, 24.5, 23.5)
        elif kind == "export":
            # Upload from tray.
            painter.drawLine(16, 20, 16, 7.5)
            painter.drawLine(10.5, 13, 16, 7.5)
            painter.drawLine(21.5, 13, 16, 7.5)
            painter.drawLine(7.5, 23.5, 24.5, 23.5)
        elif kind == "record":
            painter.setBrush(color)
            painter.drawEllipse(QPointF(16, 16), 6.5, 6.5)
        elif kind == "play":
            path = QPainterPath()
            path.moveTo(11.5, 8.5)
            path.lineTo(24, 16)
            path.lineTo(11.5, 23.5)
            path.closeSubpath()
            painter.fillPath(path, QBrush(color))
        elif kind == "check":
            painter.drawLine(7.5, 16.5, 13.5, 22.5)
            painter.drawLine(13.5, 22.5, 24.5, 9.5)
        elif kind == "close":
            painter.drawLine(10, 10, 22, 22)
            painter.drawLine(22, 10, 10, 22)
        elif kind == "trash":
            painter.drawLine(7, 9.5, 25, 9.5)
            painter.drawLine(12.5, 9.5, 13.5, 6.5)
            painter.drawLine(13.5, 6.5, 18.5, 6.5)
            painter.drawLine(18.5, 6.5, 19.5, 9.5)
            path = QPainterPath()
            path.moveTo(9, 9.5)
            path.lineTo(10.3, 25)
            path.lineTo(21.7, 25)
            path.lineTo(23, 9.5)
            painter.drawPath(path)
            painter.drawLine(14, 13.5, 14, 21)
            painter.drawLine(18, 13.5, 18, 21)
        elif kind == "duplicate":
            painter.drawRoundedRect(QRectF(7.5, 7.5, 12.5, 12.5), 3, 3)
            painter.setBrush(QColor("#0b1521"))
            painter.drawRoundedRect(QRectF(12.5, 12.5, 12.5, 12.5), 3, 3)
            painter.setBrush(Qt.NoBrush)
        elif kind == "detach":
            painter.drawRoundedRect(QRectF(6, 11, 13, 13), 3, 3)
            painter.drawLine(15, 7, 26, 7)
            painter.drawLine(26, 7, 26, 18)
            painter.drawLine(26, 7, 16.5, 16.5)
        elif kind == "columns":
            painter.drawRoundedRect(QRectF(6.5, 7.5, 19, 17), 2.5, 2.5)
            painter.drawLine(13, 7.5, 13, 24.5)
            painter.drawLine(19, 7.5, 19, 24.5)
        elif kind == "fit":
            painter.drawLine(6, 11, 6, 7)
            painter.drawLine(6, 7, 10, 7)
            painter.drawLine(26, 11, 26, 7)
            painter.drawLine(26, 7, 22, 7)
            painter.drawLine(6, 21, 6, 25)
            painter.drawLine(6, 25, 10, 25)
            painter.drawLine(26, 21, 26, 25)
            painter.drawLine(26, 25, 22, 25)
            painter.drawLine(11, 16, 21, 16)
        elif kind == "mic":
            painter.drawRoundedRect(QRectF(12, 5, 8, 15), 4, 4)
            painter.drawArc(QRectF(8, 12, 16, 12), 200 * 16, 140 * 16)
            painter.drawLine(16, 22, 16, 27)
            painter.drawLine(12, 27, 20, 27)
        elif kind == "cameraplus":
            painter.drawRoundedRect(QRectF(4.5, 11, 16, 13), 4, 4)
            painter.drawRect(QRectF(8.5, 8, 6, 3))
            painter.drawEllipse(QPointF(12, 17.5), 3.6, 3.6)
            painter.drawLine(24, 7, 24, 13)
            painter.drawLine(21, 10, 27, 10)
        elif kind == "track":
            painter.drawEllipse(QPointF(16, 16), 8.5, 8.5)
            painter.drawLine(16, 3.5, 16, 8)
            painter.drawLine(16, 24, 16, 28.5)
            painter.drawLine(3.5, 16, 8, 16)
            painter.drawLine(24, 16, 28.5, 16)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(16, 16), 2.4, 2.4)
            painter.setBrush(Qt.NoBrush)
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
        widget.setMinimumWidth(0)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(0)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(widget)

        container = QWidget()
        container.setMinimumWidth(0)
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        return container

    def _make_header_action_button(self, icon_kind: str, accent: str, label: str, checkable: bool = False) -> QToolButton:
        """Build one compact icon button for the live-view header."""
        button = QToolButton()
        button.setObjectName("toolIconButton")
        button.setIcon(self._build_modern_icon(icon_kind, accent))
        button.setIconSize(QSize(18, 18))
        button.setFixedSize(34, 30)
        button.setCheckable(checkable)
        button.setCursor(Qt.PointingHandCursor)
        if label:
            button.setToolTip(label)
        return button

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
        hero_layout.setSpacing(10)

        subtitle = QLabel("Choose a Basler, FLIR, or USB source, then arm the live workspace.")
        subtitle.setStyleSheet("color: #8fa6bf;")
        subtitle.setWordWrap(True)
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

        sources_card = QFrame()
        sources_card.setObjectName("WorkspaceCard")
        sources_layout = QVBoxLayout(sources_card)
        sources_layout.setContentsMargins(16, 14, 16, 14)
        sources_layout.setSpacing(8)

        sources_title = QLabel("Detected Sources")
        sources_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #eef6ff;")
        sources_layout.addWidget(sources_title)

        self.camera_sources_list_layout = QVBoxLayout()
        self.camera_sources_list_layout.setSpacing(6)
        sources_layout.addLayout(self.camera_sources_list_layout)

        self.label_camera_scan_diagnostics = QLabel("Scan not run yet")
        self.label_camera_scan_diagnostics.setWordWrap(True)
        self.label_camera_scan_diagnostics.setStyleSheet("color: #6f859d; font-size: 10px;")
        sources_layout.addWidget(self.label_camera_scan_diagnostics)

        layout.addWidget(sources_card)
        layout.addStretch()
        return panel

    def _refresh_camera_sources_list(self, cameras: List[Dict]):
        """Rebuild the clickable detected-sources rows in the camera panel."""
        layout = getattr(self, "camera_sources_list_layout", None)
        if layout is None:
            return
        self._clear_layout(layout)

        if not cameras:
            empty = QLabel("No cameras found — check connections and press Refresh.")
            empty.setWordWrap(True)
            empty.setStyleSheet("color: #8fa6bf; font-size: 11px;")
            layout.addWidget(empty)
            return

        type_colors = {"basler": "#33c8ff", "flir": "#ff9a43", "usb": "#7ef0ac"}
        for combo_index, camera_info in enumerate(cameras):
            cam_type = str(camera_info.get("type", "usb"))
            accent = type_colors.get(cam_type, "#9dd9ff")
            row = QPushButton(f"  {camera_info.get('label', 'Camera')}")
            row.setObjectName("ghostButton")
            row.setCursor(Qt.PointingHandCursor)
            row.setStyleSheet(
                "QPushButton#ghostButton {"
                " text-align: left; padding: 8px 12px; border-radius: 9px;"
                f" background: rgba(157, 196, 240, 0.04); border-left: 3px solid {accent}; }}"
                "QPushButton#ghostButton:hover {"
                f" background: rgba(157, 196, 240, 0.11); border-color: {accent}; }}"
            )
            row.setToolTip(f"{cam_type.upper()} source. Click to select it, then press Connect.")
            row.clicked.connect(
                lambda checked=False, idx=combo_index: self.combo_camera.setCurrentIndex(idx)
            )
            layout.addWidget(row)

    def _create_general_settings_panel(self) -> QWidget:
        """Create the left-side general settings page."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        order_group = QGroupBox("Filename Order")
        order_layout = QFormLayout(order_group)
        self.filename_order_boxes = []
        for index in range(4):
            combo = QComboBox()
            combo.addItems(self._filename_field_labels())
            combo.currentTextChanged.connect(self._on_filename_order_changed)
            self.filename_order_boxes.append(combo)
            order_layout.addRow(f"Part {index + 1}:", combo)

        preview_group = QGroupBox("Filename Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.label_filename_formula = QLabel("")
        self.label_filename_formula.setWordWrap(True)
        self.label_filename_formula.setStyleSheet("color: #9fd9ff; font-weight: 600;")
        preview_layout.addWidget(self.label_filename_formula)

        storage_group = QGroupBox("Recording Storage")
        storage_layout = QVBoxLayout(storage_group)
        storage_hint = QLabel(
            "Optionally place each recording in nested subfolders under the selected save root. "
            "Choose which metadata fields become folder levels and in what order."
        )
        storage_hint.setWordWrap(True)
        storage_hint.setStyleSheet("color: #8fa6bf;")
        storage_layout.addWidget(storage_hint)

        self.check_organize_session_folders = QCheckBox("Organize recordings into nested metadata folders")
        self.check_organize_session_folders.toggled.connect(self._on_organize_recordings_toggled)
        storage_layout.addWidget(self.check_organize_session_folders)

        self.folder_structure_group = QGroupBox("Folder Structure")
        folder_structure_layout = QVBoxLayout(self.folder_structure_group)
        folder_structure_hint = QLabel(
            "Each level becomes one nested subfolder, top to bottom. Empty fields are skipped, "
            "so Animal ID / Experiment / Session yields paths like Mouse01 / OpenField / S1."
        )
        folder_structure_hint.setWordWrap(True)
        folder_structure_hint.setStyleSheet("color: #8fa6bf;")
        folder_structure_layout.addWidget(folder_structure_hint)

        folder_levels_form = QFormLayout()
        self.folder_order_boxes = []
        for index in range(4):
            combo = QComboBox()
            combo.addItems(self._filename_field_labels())
            combo.currentTextChanged.connect(self._on_folder_order_changed)
            self.folder_order_boxes.append(combo)
            folder_levels_form.addRow(f"Level {index + 1}:", combo)
        folder_structure_layout.addLayout(folder_levels_form)

        self.label_folder_structure_preview = QLabel("")
        self.label_folder_structure_preview.setWordWrap(True)
        self.label_folder_structure_preview.setStyleSheet("color: #9fd9ff; font-weight: 600;")
        folder_structure_layout.addWidget(self.label_folder_structure_preview)

        storage_layout.addWidget(self.folder_structure_group)

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

        user_flag_group = QGroupBox("User Flag")
        user_flag_layout = QVBoxLayout(user_flag_group)
        user_flag_hint = QLabel(
            "Assign a labeled manual marker to a shortcut. During recording it is stamped into "
            "the metadata CSV, and when a DO output is selected it can also emit a TTL pulse."
        )
        user_flag_hint.setWordWrap(True)
        user_flag_hint.setStyleSheet("color: #8fa6bf;")
        user_flag_layout.addWidget(user_flag_hint)

        self.label_user_flag_summary = QLabel("Flags: 0")
        self.label_user_flag_summary.setStyleSheet("color: #9fd9ff; font-weight: 700;")
        user_flag_layout.addWidget(self.label_user_flag_summary)

        self.label_user_flag_details = QLabel("No user flags configured")
        self.label_user_flag_details.setWordWrap(True)
        self.label_user_flag_details.setStyleSheet("color: #cfe8ff;")
        user_flag_layout.addWidget(self.label_user_flag_details)

        self.label_user_flag_pin_summary = QLabel("Pins: no TTL outputs selected")
        self.label_user_flag_pin_summary.setWordWrap(True)
        self.label_user_flag_pin_summary.setStyleSheet("color: #8fa6bf;")
        user_flag_layout.addWidget(self.label_user_flag_pin_summary)

        self.btn_manage_user_flags = QPushButton("Manage Flags")
        self._set_button_icon(self.btn_manage_user_flags, "settings", "#7cc7ff", "ghostButton")
        self.btn_manage_user_flags.clicked.connect(self._show_user_flag_dialog)
        user_flag_layout.addWidget(self.btn_manage_user_flags)

        self.btn_open_advanced_settings = QPushButton("Advanced Camera Menu")
        self._set_button_icon(self.btn_open_advanced_settings, "settings", "#d86cff", "ghostButton")
        self.btn_open_advanced_settings.clicked.connect(self._toggle_advanced_settings)

        def build_settings_section_page(title_text: str, subtitle_text: str) -> tuple[QWidget, QVBoxLayout]:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(0)

            card = QFrame()
            card.setObjectName("WorkspaceCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 16, 16, 16)
            card_layout.setSpacing(12)

            title = QLabel(title_text)
            title.setStyleSheet("font-size: 16px; font-weight: 700; color: #eef6ff;")
            subtitle = QLabel(subtitle_text)
            subtitle.setStyleSheet("color: #8fa6bf;")
            subtitle.setWordWrap(True)
            card_layout.addWidget(title)
            card_layout.addWidget(subtitle)

            page_layout.addWidget(card)
            page_layout.addStretch()
            return page, card_layout

        intro = QLabel(
            "Only the selected settings subsection is shown so the panel stays spacious and readable. "
            "Use the footer actions to save the current setup, update defaults, or move full presets in and out."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8fa6bf;")
        layout.addWidget(intro)

        section_tabs = QTabWidget()
        section_tabs.setObjectName("settingsSectionTabs")
        section_tabs.setTabPosition(QTabWidget.North)
        section_tabs.setDocumentMode(True)
        section_tabs.setUsesScrollButtons(False)
        section_tabs.setElideMode(Qt.ElideNone)

        section_specs: List[tuple[str, str, str, str]] = [
            ("run_naming", "Naming", "Run Naming", "Choose filename parts and preview the resulting folder and basename."),
            ("storage", "Storage", "Recording Storage", "Control how each recording is organized under the selected save root."),
            ("behavior", "Behavior", "Behavior / TTL", "Open the defaults menu for pins, roles, labels, and camera line names."),
            ("user_flag", "Flag", "User Flag", "Configure a manual marker shortcut and optional TTL pulse."),
            ("camera_tools", "Camera", "Camera Tools", "Open the advanced camera popup without crowding the main settings page."),
        ]

        run_naming_page, run_naming_layout = build_settings_section_page(
            section_specs[0][2],
            section_specs[0][3],
        )
        run_naming_layout.addWidget(order_group)
        run_naming_layout.addWidget(preview_group)
        run_naming_layout.addStretch()
        section_tabs.addTab(run_naming_page, section_specs[0][1])

        storage_page, storage_layout_page = build_settings_section_page(
            section_specs[1][2],
            section_specs[1][3],
        )
        storage_layout_page.addWidget(storage_group)
        storage_layout_page.addStretch()
        section_tabs.addTab(storage_page, section_specs[1][1])

        behavior_page, behavior_layout_page = build_settings_section_page(
            section_specs[2][2],
            section_specs[2][3],
        )
        behavior_layout_page.addWidget(behavior_defaults_group)
        behavior_layout_page.addStretch()
        section_tabs.addTab(behavior_page, section_specs[2][1])

        user_flag_page, user_flag_layout_page = build_settings_section_page(
            section_specs[3][2],
            section_specs[3][3],
        )
        user_flag_layout_page.addWidget(user_flag_group)
        user_flag_layout_page.addStretch()
        section_tabs.addTab(user_flag_page, section_specs[3][1])

        camera_tools_group = QGroupBox("Advanced Camera")
        camera_tools_layout = QVBoxLayout(camera_tools_group)
        camera_tools_hint = QLabel(
            "Launch the advanced camera popup for pixel format, bit depth, white balance, and sensor-specific controls."
        )
        camera_tools_hint.setWordWrap(True)
        camera_tools_hint.setStyleSheet("color: #8fa6bf;")
        camera_tools_layout.addWidget(camera_tools_hint)
        camera_tools_layout.addWidget(self.btn_open_advanced_settings)

        camera_tools_page, camera_tools_layout_page = build_settings_section_page(
            section_specs[4][2],
            section_specs[4][3],
        )
        camera_tools_layout_page.addWidget(camera_tools_group)
        camera_tools_layout_page.addStretch()
        section_tabs.addTab(camera_tools_page, section_specs[4][1])
        section_tabs.setCurrentIndex(0)
        layout.addWidget(section_tabs, 1)

        footer_card = QFrame()
        footer_card.setObjectName("WorkspaceSubCard")
        footer_layout = QVBoxLayout(footer_card)
        footer_layout.setContentsMargins(14, 12, 14, 12)
        footer_layout.setSpacing(10)

        footer_title = QLabel("Actions")
        footer_title.setStyleSheet("font-weight: 700; color: #eef6ff;")
        footer_hint = QLabel(
            "Save updates immediately, set the current setup as the next-launch default, or export/import a full preset."
        )
        footer_hint.setWordWrap(True)
        footer_hint.setStyleSheet("color: #8fa6bf;")
        footer_layout.addWidget(footer_title)
        footer_layout.addWidget(footer_hint)

        actions_row = QHBoxLayout()
        btn_save_settings = QPushButton("Save Settings")
        self._set_button_icon(btn_save_settings, "check", "#6fe06e", "ghostButton")
        btn_save_settings.clicked.connect(self._save_current_settings_snapshot)
        actions_row.addWidget(btn_save_settings)

        btn_set_default = QPushButton("Set As Default")
        self._set_button_icon(btn_set_default, "settings", "#ffb35d", "orangeButton")
        btn_set_default.clicked.connect(self._set_current_settings_as_default)
        actions_row.addWidget(btn_set_default)
        footer_layout.addLayout(actions_row)

        preset_row = QHBoxLayout()
        btn_save_preset = QPushButton("Export Preset")
        self._set_button_icon(btn_save_preset, "export", "#9fd9ff", "ghostButton")
        btn_save_preset.setToolTip(
            "Save all camera, trial planner, ROI, and pulse settings to a single .pkpreset file"
        )
        btn_save_preset.clicked.connect(self._save_preset_to_file)
        preset_row.addWidget(btn_save_preset)

        btn_load_preset = QPushButton("Import Preset")
        self._set_button_icon(btn_load_preset, "import", "#9fd9ff", "ghostButton")
        btn_load_preset.setToolTip(
            "Restore all camera, trial planner, ROI, and pulse settings from a .pkpreset file"
        )
        btn_load_preset.clicked.connect(self._load_preset_from_file)
        preset_row.addWidget(btn_load_preset)
        footer_layout.addLayout(preset_row)

        layout.addWidget(footer_card)
        return panel

    def _create_session_hub_panel(self) -> QWidget:
        """Create the merged metadata and planner workspace page."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        overview = QFrame()
        overview.setObjectName("WorkspaceCard")
        overview_layout = QVBoxLayout(overview)
        overview_layout.setContentsMargins(14, 12, 14, 12)
        overview_layout.setSpacing(8)

        # ── Compact header with inline stats ─────────────────────────
        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        title = QLabel("Session Planner")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #eef6ff;")
        header_row.addWidget(title)
        header_row.addStretch()

        # Inline stat chips instead of 4 large tiles
        stats_row = QHBoxLayout()
        stats_row.setSpacing(6)
        total_tile, self.label_session_total_count = self._create_metric_tile("Trials", "0", "#eef6ff")
        pending_tile, self.label_session_pending_count = self._create_metric_tile("Pending", "0", "#ffd89c")
        acquiring_tile, self.label_session_acquiring_count = self._create_metric_tile("Active", "0", "#9dd9ff")
        acquired_tile, self.label_session_acquired_count = self._create_metric_tile("Done", "0", "#7ef0ac")
        for tile in (total_tile, pending_tile, acquiring_tile, acquired_tile):
            tile.setFixedHeight(44)
            stats_row.addWidget(tile, 1)
        overview_layout.addLayout(header_row)
        overview_layout.addLayout(stats_row)

        self.metadata_panel = self._create_metadata_panel()
        self.metadata_panel.hide()

        # ── Selected trial — single compact line ─────────────────────
        selection_row = QHBoxLayout()
        selection_row.setSpacing(6)
        sel_icon = QLabel("\u25B6")
        sel_icon.setStyleSheet("color: #9fd9ff; font-size: 11px;")
        sel_icon.setFixedWidth(14)
        self.label_session_summary = QLabel("No trial selected")
        self.label_session_summary.setStyleSheet("color: #9fd9ff; font-weight: 600; font-size: 11px;")
        self.label_session_details = QLabel("")
        self.label_session_details.setWordWrap(True)
        self.label_session_details.setStyleSheet("color: #8fa6bf; font-size: 10px;")
        selection_row.addWidget(sel_icon)
        selection_row.addWidget(self.label_session_summary, 1)
        overview_layout.addLayout(selection_row)
        overview_layout.addWidget(self.label_session_details)

        # ── Planner ──────────────────────────────────────────────────
        planner_host = QFrame()
        planner_host.setObjectName("WorkspaceSubCard")
        self.planner_host_layout = QVBoxLayout(planner_host)
        self.planner_host_layout.setContentsMargins(10, 10, 10, 10)
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
        layout.setSpacing(6)

        status_card = QFrame()
        status_card.setObjectName("WorkspaceCard")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(10, 10, 10, 10)
        status_layout.setSpacing(6)

        chip_row = QHBoxLayout()
        chip_row.setContentsMargins(0, 0, 0, 0)
        chip_row.setSpacing(6)
        title = QLabel("TTL Generator")
        title.setStyleSheet("font-size: 13px; font-weight: 700; color: #eef6ff;")
        chip_row.addWidget(title)
        chip_row.addStretch()
        self.label_ttl_status = self._make_panel_chip("TTL: IDLE", "default")
        chip_row.addWidget(self.label_ttl_status)
        status_layout.addLayout(chip_row)

        self.ttl_counts_group = QGroupBox("TTL Summary")
        self.ttl_counts_layout = QGridLayout()
        self.ttl_counts_layout.setContentsMargins(8, 8, 8, 6)
        self.ttl_counts_layout.setHorizontalSpacing(10)
        self.ttl_counts_layout.setVerticalSpacing(2)
        self.ttl_counts_group.setLayout(self.ttl_counts_layout)
        status_layout.addWidget(self.ttl_counts_group)

        # Hardware barcode status strip
        self.hw_barcode_frame = QFrame()
        self.hw_barcode_frame.setObjectName("WorkspaceSubCard")
        hw_bc_layout = QHBoxLayout(self.hw_barcode_frame)
        hw_bc_layout.setContentsMargins(12, 6, 12, 6)
        hw_bc_layout.setSpacing(12)
        hw_bc_title = QLabel("HW Barcode")
        hw_bc_title.setStyleSheet("color: #f97316; font-weight: 700; font-size: 11px;")
        hw_bc_layout.addWidget(hw_bc_title)
        self.label_hw_barcode_mode = QLabel("OFF")
        self.label_hw_barcode_mode.setStyleSheet("color: #8fa6bf;")
        hw_bc_layout.addWidget(self.label_hw_barcode_mode)
        self.label_hw_barcode_counter = QLabel("Counter: —")
        self.label_hw_barcode_counter.setStyleSheet("color: #dce8f4; font-weight: 600;")
        hw_bc_layout.addWidget(self.label_hw_barcode_counter)
        self.label_hw_barcode_bit = QLabel("Bit: —")
        self.label_hw_barcode_bit.setStyleSheet("color: #dce8f4;")
        hw_bc_layout.addWidget(self.label_hw_barcode_bit)
        self.label_hw_barcode_bitwidth = QLabel("Width: — ms")
        self.label_hw_barcode_bitwidth.setStyleSheet("color: #8fa6bf;")
        hw_bc_layout.addWidget(self.label_hw_barcode_bitwidth)
        hw_bc_layout.addStretch()
        self.hw_barcode_frame.setVisible(False)
        status_layout.addWidget(self.hw_barcode_frame)

        self.ttl_plot_group = QGroupBox("TTL Generator Signals")
        ttl_plot_layout = QVBoxLayout()
        ttl_plot_layout.setContentsMargins(6, 8, 6, 6)
        ttl_plot_layout.setSpacing(2)
        self.ttl_plot = pg.PlotWidget()
        self.ttl_plot.setBackground((8, 16, 26))
        self.ttl_plot.setMouseEnabled(x=False, y=False)
        self.ttl_plot.showGrid(x=True, y=True, alpha=0.16)
        self.ttl_plot.setLabel("bottom", "Time (s)")
        self.ttl_plot.setXRange(0, self.ttl_window_seconds)
        self.ttl_plot.setLimits(xMin=0)
        self.ttl_plot.setDownsampling(auto=True, mode="peak")
        self.ttl_plot.setMinimumHeight(150)
        self.ttl_plot.setMaximumHeight(180)
        self.ttl_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ttl_plot.setStyleSheet("border: 1px solid #1c3046; border-radius: 8px;")
        ttl_plot_layout.addWidget(self.ttl_plot)
        self.ttl_plot_group.setLayout(ttl_plot_layout)
        status_layout.addWidget(self.ttl_plot_group)

        self.camera_line_plot_group = QGroupBox("Camera Chunk Line States")
        camera_line_plot_layout = QVBoxLayout()
        camera_line_plot_layout.setContentsMargins(6, 8, 6, 6)
        camera_line_plot_layout.setSpacing(2)
        self.camera_line_plot = pg.PlotWidget()
        self.camera_line_plot.setBackground((8, 16, 26))
        self.camera_line_plot.setMouseEnabled(x=False, y=False)
        self.camera_line_plot.showGrid(x=True, y=True, alpha=0.16)
        self.camera_line_plot.setLabel("bottom", "Time (s)")
        self.camera_line_plot.setXRange(0, self.ttl_window_seconds)
        self.camera_line_plot.setLimits(xMin=0)
        self.camera_line_plot.setDownsampling(auto=True, mode="peak")
        self.camera_line_plot.setMinimumHeight(120)
        self.camera_line_plot.setMaximumHeight(150)
        self.camera_line_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.camera_line_plot.setStyleSheet("border: 1px solid #1c3046; border-radius: 8px;")
        camera_line_plot_layout.addWidget(self.camera_line_plot)
        self.camera_line_plot_group.setLayout(camera_line_plot_layout)
        status_layout.addWidget(self.camera_line_plot_group)
        status_layout.addStretch(1)

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
        label = ChipLabel(text)
        label.setAlignment(Qt.AlignCenter)
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

    def _make_field_label(self, text: str) -> QLabel:
        """Create a muted form-field label so values stay visually dominant."""
        label = QLabel(text)
        label.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        return label

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
        header_layout.setContentsMargins(14, 8, 14, 8)
        header_layout.setSpacing(8)

        title = QLabel(APP_NAME)
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #edf4ff;")
        header_layout.addWidget(title)

        # Always-visible compact session summary (trial / subject / session) so
        # the operator keeps recording context without opening the Recording
        # panel; the full detail lives in the tooltip.
        self.live_header_session = self._make_panel_chip("No session", "default")
        self.live_header_session.setToolTip("Active planner trial, subject, and session")
        self.live_header_session.setMinimumWidth(0)
        self.live_header_session.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        header_layout.addWidget(self.live_header_session)

        # Compact icon actions keep the header narrow (so the workspace can
        # shrink) while staying legible through tooltips.
        self.btn_toggle_frame_drop_panel = self._make_header_action_button(
            "pulse", "#7cc7ff", "Frame-drop monitor", checkable=True
        )
        self.btn_toggle_frame_drop_panel.setChecked(bool(self.frame_drop_monitor_visible))
        self.btn_toggle_frame_drop_panel.toggled.connect(self._update_frame_drop_panel_visibility)
        header_layout.addWidget(self.btn_toggle_frame_drop_panel)

        self.btn_tracking_mode = self._make_header_action_button(
            "track", "#ff9bd2", "", checkable=True
        )
        self.btn_tracking_mode.setToolTip(
            "Tracking mode — runs the mask and pose models in parallel on every\n"
            "frame with identity tracking. Saves COCO JSON masks (with track ids)\n"
            "and a DLC-format pose CSV next to each recording. Configure the\n"
            "checkpoints once in the Live Detection panel."
        )
        self.btn_tracking_mode.toggled.connect(self._on_tracking_mode_toggled)
        header_layout.addWidget(self.btn_tracking_mode)

        self.btn_add_camera_stream = self._make_header_action_button(
            "cameraplus", "#9bf57f", "", checkable=False
        )
        self.btn_add_camera_stream.setToolTip(
            "Add another live camera stream (USB, FLIR, or Basler).\n"
            "Every connected stream records in sync with the main Start Recording button."
        )
        self.btn_add_camera_stream.clicked.connect(self._on_add_camera_stream_clicked)
        header_layout.addWidget(self.btn_add_camera_stream)

        self.live_status_badge = self._make_panel_chip("Offline", "warning")
        self.live_header_status = self._make_panel_chip("No camera connected", "default")
        self.live_header_resolution = self._make_panel_chip("-- x --", "default")
        self.live_header_mode = self._make_panel_chip(self.default_image_format, "accent")
        self.live_header_roi = self._make_panel_chip("Full Frame", "default")
        # The status chips are informational. Letting them shrink to nothing
        # (Preferred + zero minimum) keeps them at natural size when there is
        # room, but stops the header row from pinning the whole window wide.
        for chip in (
            self.live_header_status,
            self.live_header_resolution,
            self.live_header_mode,
            self.live_header_roi,
        ):
            chip.setMinimumWidth(0)
            chip.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        header_layout.addStretch()

        preview_height_label = QLabel("Preview")
        preview_height_label.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        preview_height_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        header_layout.addWidget(preview_height_label)

        self.preview_height_slider = QSlider(Qt.Horizontal)
        self.preview_height_slider.setRange(25, 100)
        self.preview_height_slider.setValue(100)
        self.preview_height_slider.setFixedWidth(92)
        self.preview_height_slider.setToolTip(
            "Preview height: share of the workspace the live view keeps.\n"
            "You can also drag the grip divider below the preview."
        )
        self.preview_height_slider.setStatusTip("Adjust how much vertical space the live preview uses")
        self.preview_height_slider.valueChanged.connect(self._on_preview_height_slider_changed)
        header_layout.addWidget(self.preview_height_slider)
        header_layout.addSpacing(6)

        header_layout.addWidget(self.live_status_badge)
        header_layout.addWidget(self.live_header_status)
        header_layout.addWidget(self.live_header_resolution)
        header_layout.addWidget(self.live_header_mode)
        header_layout.addWidget(self.live_header_roi)
        layout.addWidget(header)

        # Slim recording progress strip (visible only while a limited-length
        # recording is running): fills elapsed/limit with a warm gradient.
        self.recording_progress_bar = QProgressBar()
        self.recording_progress_bar.setObjectName("recordingProgressBar")
        self.recording_progress_bar.setRange(0, 1000)
        self.recording_progress_bar.setValue(0)
        self.recording_progress_bar.setTextVisible(False)
        self.recording_progress_bar.setFixedHeight(6)
        self.recording_progress_bar.setStyleSheet(
            "QProgressBar#recordingProgressBar { background: #101b2c;"
            " border: none; border-radius: 3px; }"
            "QProgressBar#recordingProgressBar::chunk {"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #ff5b70, stop:1 #ff9a43); border-radius: 3px; }"
        )
        self.recording_progress_bar.hide()
        layout.addWidget(self.recording_progress_bar)

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
        self.live_image_view.setMinimumHeight(200)
        self.live_image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.live_preview_scene = self.live_image_view.getView().scene()
        if self.live_preview_scene is not None:
            self.live_preview_scene.installEventFilter(self)
        self._build_recording_overlay(self.live_image_view)
        self.live_image_view.installEventFilter(self)

        # Adaptive multi-stream grid: the primary view fills the workspace by
        # default, and auxiliary camera tiles claim space as they are added.
        self.camera_grid_container = QWidget()
        self.camera_grid_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_grid_layout = QGridLayout(self.camera_grid_container)
        self.camera_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_grid_layout.setSpacing(10)
        layout.addWidget(self.camera_grid_container, stretch=1)
        self._relayout_camera_streams()

        self._show_live_placeholder(APP_NAME, "Connect a camera to begin preview")
        return panel

    def _build_recording_overlay(self, parent: QWidget) -> None:
        """Create the floating recording countdown pinned to the live view.

        A Qt overlay (rather than text burned into the frame) stays crisp and
        ticks every second from the recording timer even when the camera runs
        at a low frame rate. Hidden until a recording starts.
        """
        overlay = QFrame(parent)
        overlay.setObjectName("recordingOverlay")
        overlay.setStyleSheet(
            "QFrame#recordingOverlay { background-color: rgba(9, 15, 25, 215);"
            " border: 1px solid #7b2323; border-radius: 12px; }"
        )
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(14, 9, 14, 11)
        overlay_layout.setSpacing(3)

        rec_row = QHBoxLayout()
        rec_row.setSpacing(7)
        self.recording_overlay_dot = QLabel("●")
        self.recording_overlay_dot.setStyleSheet("color: #ff5b6e; font-size: 13px;")
        rec_row.addWidget(self.recording_overlay_dot)
        rec_title = QLabel("REC")
        rec_title.setStyleSheet("color: #ffb3b3; font-size: 12px; font-weight: 800; letter-spacing: 2px;")
        rec_row.addWidget(rec_title)
        rec_row.addStretch()
        overlay_layout.addLayout(rec_row)

        self.recording_overlay_time = QLabel("00:00 / 00:00")
        self.recording_overlay_time.setStyleSheet("color: #f6faff; font-size: 23px; font-weight: 800;")
        overlay_layout.addWidget(self.recording_overlay_time)

        self.recording_overlay_remaining = QLabel("Remaining --:--")
        self.recording_overlay_remaining.setStyleSheet("color: #9bf0bc; font-size: 12px; font-weight: 700;")
        overlay_layout.addWidget(self.recording_overlay_remaining)

        self.recording_overlay_progress = QProgressBar()
        self.recording_overlay_progress.setRange(0, 1000)
        self.recording_overlay_progress.setValue(0)
        self.recording_overlay_progress.setTextVisible(False)
        self.recording_overlay_progress.setFixedHeight(5)
        self.recording_overlay_progress.setStyleSheet(
            "QProgressBar { background: #16273a; border: none; border-radius: 2px; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #ff5b70, stop:1 #ff9a43); border-radius: 2px; }"
        )
        overlay_layout.addWidget(self.recording_overlay_progress)

        overlay.setFixedWidth(196)
        overlay.hide()
        self.recording_overlay = overlay

    def _position_recording_overlay(self) -> None:
        """Pin the recording overlay to the live view's top-right corner."""
        overlay = self.recording_overlay
        if overlay is None or self.live_image_view is None:
            return
        overlay.adjustSize()
        margin = 16
        x = max(margin, self.live_image_view.width() - overlay.width() - margin)
        overlay.move(x, margin)
        overlay.raise_()

    def _show_recording_overlay(self, visible: bool) -> None:
        """Show or hide the floating recording countdown."""
        overlay = self.recording_overlay
        if overlay is None:
            return
        if visible:
            self._update_recording_overlay(self._current_recording_elapsed_seconds())
            overlay.show()
            self._position_recording_overlay()
        else:
            overlay.hide()

    def _update_recording_overlay(self, elapsed_seconds: int) -> None:
        """Refresh the overlay's elapsed / remaining / progress readouts."""
        overlay = self.recording_overlay
        if overlay is None or not overlay.isVisible():
            return
        max_seconds = self._get_max_record_seconds()
        remaining_seconds = self._current_recording_remaining_seconds()
        elapsed_text = self._format_duration_hms(elapsed_seconds)
        if max_seconds > 0:
            self.recording_overlay_time.setText(f"{elapsed_text} / {self._format_duration_hms(max_seconds)}")
        else:
            self.recording_overlay_time.setText(elapsed_text)
        if remaining_seconds is None:
            self.recording_overlay_remaining.setText("Remaining Unlimited")
            self.recording_overlay_remaining.setStyleSheet("color: #9bf0bc; font-size: 12px; font-weight: 700;")
            self.recording_overlay_progress.hide()
        else:
            warn = remaining_seconds <= 10
            self.recording_overlay_remaining.setText(f"{self._format_duration_hms(remaining_seconds)} left")
            self.recording_overlay_remaining.setStyleSheet(
                f"color: {'#ffb06b' if warn else '#9bf0bc'}; font-size: 12px; font-weight: 700;"
            )
            self.recording_overlay_progress.show()
            if max_seconds > 0:
                ratio = min(max(elapsed_seconds / max_seconds, 0.0), 1.0)
                self.recording_overlay_progress.setValue(int(ratio * 1000))
        self._position_recording_overlay()

    # ===== Auxiliary camera streams =====

    def _primary_camera_info(self) -> Optional[Dict]:
        """Identity of the camera held by the primary worker, if connected."""
        if not self.is_camera_connected:
            return None
        return self.connected_camera_info

    def _scan_cameras_for_streams(self) -> List[Dict]:
        """Discover every attached camera for the auxiliary stream pickers."""
        cameras: List[Dict] = []
        try:
            cameras.extend(discover_basler_cameras())
        except Exception:
            pass
        try:
            flir_cameras, reserved_usb_indices = discover_flir_cameras()
        except Exception:
            flir_cameras, reserved_usb_indices = [], set()
        cameras.extend(flir_cameras)

        # Skip USB indices already claimed by a stream: probing an in-use MSMF
        # device is slow to fail and can disturb the running capture.
        skip_indices = set(reserved_usb_indices)
        if self.camera_stream_manager is not None:
            for key in self.camera_stream_manager.used_camera_keys():
                if key.startswith("usb") and "index=" in key:
                    try:
                        skip_indices.add(int(key.rsplit("index=", 1)[1]))
                    except ValueError:
                        pass
        try:
            cameras.extend(discover_usb_cameras(skip_indices=skip_indices))
        except Exception:
            pass
        return cameras

    def _on_add_camera_stream_clicked(self):
        """Create a new auxiliary stream tile in the live workspace."""
        if self.camera_stream_manager is None or not self.camera_stream_manager.can_add_stream():
            self._on_status_update(
                "Stream limit reached: up to "
                f"{CameraStreamManager.MAX_STREAMS + 1} simultaneous cameras."
            )
            return
        stream = self.camera_stream_manager.create_stream()
        if stream is None:
            return
        tile = AuxCameraTile(
            stream,
            scan_cameras=self._scan_cameras_for_streams,
            used_camera_keys=self.camera_stream_manager.used_camera_keys,
            request_remove=self._remove_camera_stream_tile,
            settings=self.settings,
        )
        self.aux_camera_tiles.append(tile)
        self._relayout_camera_streams()
        self._on_status_update(
            f"{stream.display_name} added — pick a source and press Connect. "
            "It will record in sync with the primary camera."
        )

    def _remove_camera_stream_tile(self, tile):
        """Disconnect and drop one auxiliary stream tile."""
        if tile.stream.is_recording:
            self._on_status_update("Stop the recording before removing a camera stream.")
            return
        if tile in self.aux_camera_tiles:
            self.aux_camera_tiles.remove(tile)
        if self.camera_stream_manager is not None:
            self.camera_stream_manager.remove_stream(tile.stream)
        tile.setParent(None)
        tile.deleteLater()
        self._relayout_camera_streams()

    def _relayout_camera_streams(self):
        """Arrange the primary view and auxiliary tiles in an adaptive grid."""
        if self.camera_grid_layout is None:
            return
        while self.camera_grid_layout.count():
            item = self.camera_grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self.camera_grid_container)
        for column in range(6):
            self.camera_grid_layout.setColumnStretch(column, 0)
        for row in range(6):
            self.camera_grid_layout.setRowStretch(row, 0)

        tiles = list(self.aux_camera_tiles)
        count = len(tiles)
        if count == 0:
            self.camera_grid_layout.addWidget(self.live_image_view, 0, 0)
            self.camera_grid_layout.setColumnStretch(0, 1)
            self.camera_grid_layout.setRowStretch(0, 1)
        elif count == 1:
            self.camera_grid_layout.addWidget(self.live_image_view, 0, 0)
            self.camera_grid_layout.addWidget(tiles[0], 0, 1)
            self.camera_grid_layout.setColumnStretch(0, 3)
            self.camera_grid_layout.setColumnStretch(1, 2)
            self.camera_grid_layout.setRowStretch(0, 1)
        elif count == 2:
            self.camera_grid_layout.addWidget(self.live_image_view, 0, 0, 2, 1)
            self.camera_grid_layout.addWidget(tiles[0], 0, 1)
            self.camera_grid_layout.addWidget(tiles[1], 1, 1)
            self.camera_grid_layout.setColumnStretch(0, 2)
            self.camera_grid_layout.setColumnStretch(1, 1)
            self.camera_grid_layout.setRowStretch(0, 1)
            self.camera_grid_layout.setRowStretch(1, 1)
        else:
            # Uniform grid for 4-12 streams: primary first, near-square shape.
            import math

            total = count + 1
            columns = int(math.ceil(math.sqrt(total)))
            rows = int(math.ceil(total / columns))
            widgets = [self.live_image_view] + tiles
            for index, widget in enumerate(widgets):
                self.camera_grid_layout.addWidget(widget, index // columns, index % columns)
            for column in range(columns):
                self.camera_grid_layout.setColumnStretch(column, 1)
            for row in range(rows):
                self.camera_grid_layout.setRowStretch(row, 1)
        for widget in [self.live_image_view] + tiles:
            widget.show()

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
        self.frame_drop_log.setMinimumWidth(0)
        self.frame_drop_log.setMaximumWidth(320)
        self.frame_drop_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.frame_drop_log.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frame_drop_log.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frame_drop_log.setStyleSheet(
            """
            QTextEdit {
                background-color: #07111b;
                border: 1px solid #24405f;
                border-radius: 10px;
                color: #9dd9ff;
                font-family: "Cascadia Mono", Consolas, "Courier New";
                font-size: 9px;
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
        layout.setSpacing(6)

        # ── Header: title + hint, with Fit/Detach as quiet tool buttons ─
        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("Recording Planner")
        title.setStyleSheet("font-size: 13px; font-weight: 700; color: #edf4ff;")
        hint = QLabel("Pick a row to load it into the session, then record.")
        hint.setStyleSheet("color: #7f96ad; font-size: 10px;")
        title_box.addWidget(title)
        title_box.addWidget(hint)
        header_row.addLayout(title_box)
        header_row.addStretch()

        self.btn_planner_fit = QToolButton()
        self.btn_planner_fit.setObjectName("toolIconButton")
        self.btn_planner_fit.setIcon(self._build_modern_icon("fit", "#9fd9ff"))
        self.btn_planner_fit.setIconSize(QSize(17, 17))
        self.btn_planner_fit.setFixedSize(30, 28)
        self.btn_planner_fit.setCursor(Qt.PointingHandCursor)
        self.btn_planner_fit.setToolTip("Fit columns to their contents")
        self.btn_planner_fit.clicked.connect(self._fit_planner_columns)
        header_row.addWidget(self.btn_planner_fit)

        self.btn_planner_detach = QPushButton("Detach")
        self._set_button_icon(self.btn_planner_detach, "detach", "#ffb35d", "ghostButton")
        self.btn_planner_detach.setToolTip("Pop the planner out into its own resizable window")
        self.btn_planner_detach.clicked.connect(self._toggle_planner_detach)
        self.btn_planner_detach.setFixedHeight(28)
        header_row.addWidget(self.btn_planner_detach)
        layout.addLayout(header_row)

        # ── Primary actions: the two everyday verbs, given real weight ──
        primary_row = QHBoxLayout()
        primary_row.setSpacing(8)

        self.btn_planner_add_trials = QPushButton("Add Trial")
        self._set_button_icon(self.btn_planner_add_trials, "plus", "#eaf6ff", "toggleButton")
        self.btn_planner_add_trials.setToolTip("Add one trial row (use Plan ▾ to add several at once)")
        self.btn_planner_add_trials.clicked.connect(self._add_one_planner_trial)

        self.btn_planner_apply = QPushButton("Load to Session")
        self._set_button_icon(self.btn_planner_apply, "play", "#06140b", "successButton")
        self.btn_planner_apply.setToolTip("Load the selected trial into the live session form")
        self.btn_planner_apply.clicked.connect(self._apply_selected_planner_trial)

        for btn in (self.btn_planner_add_trials, self.btn_planner_apply):
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setFixedHeight(32)
            primary_row.addWidget(btn)
        primary_row.setStretch(0, 2)
        primary_row.setStretch(1, 3)
        layout.addLayout(primary_row)

        # ── Row tools + Plan menu: edits on the left, file I/O grouped ──
        tools_row = QHBoxLayout()
        tools_row.setSpacing(6)

        self.btn_planner_duplicate = QPushButton("Duplicate")
        self._set_button_icon(self.btn_planner_duplicate, "duplicate", "#9fd9ff", "ghostButton")
        self.btn_planner_duplicate.setToolTip("Duplicate the selected trial row(s)")
        self.btn_planner_duplicate.clicked.connect(self._duplicate_selected_planner_trials)

        self.btn_planner_remove = QPushButton("Remove")
        self._set_button_icon(self.btn_planner_remove, "trash", "#ff8da6", "ghostButton")
        self.btn_planner_remove.setToolTip("Remove the selected trial row(s)")
        self.btn_planner_remove.clicked.connect(self._remove_selected_planner_trials)

        self.btn_planner_add_variable = QPushButton("Variable")
        self._set_button_icon(self.btn_planner_add_variable, "columns", "#cf9bff", "ghostButton")
        self.btn_planner_add_variable.setToolTip("Add a custom metadata column to the plan")
        self.btn_planner_add_variable.clicked.connect(self._add_planner_variable)

        for btn in (self.btn_planner_duplicate, self.btn_planner_remove, self.btn_planner_add_variable):
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setFixedHeight(27)
            tools_row.addWidget(btn)

        self.btn_planner_menu = QToolButton()
        self.btn_planner_menu.setText("Plan")
        self.btn_planner_menu.setObjectName("ghostMenuButton")
        self.btn_planner_menu.setIcon(self._build_modern_icon("folder", "#9fd9ff"))
        self.btn_planner_menu.setIconSize(QSize(16, 16))
        self.btn_planner_menu.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_planner_menu.setPopupMode(QToolButton.InstantPopup)
        self.btn_planner_menu.setCursor(Qt.PointingHandCursor)
        self.btn_planner_menu.setFixedHeight(27)
        self.btn_planner_menu.setToolTip("Bulk add, import, and export the trial plan")
        planner_menu = QMenu(self.btn_planner_menu)
        planner_menu.addAction("Add multiple trials…", self._add_planner_trials)
        planner_menu.addSeparator()
        planner_menu.addAction("Import plan from CSV…", self._import_planner_trials)
        planner_menu.addAction("Export plan to CSV…", self._export_planner_trials)
        self.action_planner_load_last = planner_menu.addAction("Load last plan", self._load_last_planner_trials)
        self.btn_planner_menu.setMenu(planner_menu)
        tools_row.addWidget(self.btn_planner_menu)
        self._update_planner_load_last_button_state()
        layout.addLayout(tools_row)

        # ── Table ────────────────────────────────────────────────────
        self.planner_table = QTableWidget(0, 0)
        self.planner_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.planner_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.planner_table.setAlternatingRowColors(True)
        self.planner_table.setWordWrap(False)
        self.planner_table.verticalHeader().setVisible(False)
        self.planner_table.verticalHeader().setDefaultSectionSize(24)
        self.planner_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.planner_table.horizontalHeader().setStretchLastSection(False)
        self.planner_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.planner_table.setMinimumHeight(200)
        self.planner_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.planner_table.viewport().installEventFilter(self)
        self.planner_table.customContextMenuRequested.connect(self._show_planner_context_menu)
        self.planner_table.itemSelectionChanged.connect(self._on_planner_selection_changed)
        self.planner_table.itemChanged.connect(self._on_planner_item_changed)
        self.planner_copy_shortcut = QShortcut(QKeySequence.Copy, self.planner_table)
        self.planner_copy_shortcut.activated.connect(self._copy_selected_planner_trials)
        self.planner_paste_shortcut = QShortcut(QKeySequence.Paste, self.planner_table)
        self.planner_paste_shortcut.activated.connect(self._paste_selected_planner_trials)
        layout.addWidget(self.planner_table, stretch=1)

        # ── Footer — single-line summary ─────────────────────────────
        self.label_planner_summary = QLabel("No trial selected")
        self.label_planner_summary.setStyleSheet(
            "color: #9dd9ff; font-weight: 600; font-size: 10px; padding: 4px 2px;"
        )
        layout.addWidget(self.label_planner_summary)

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

        self.meta_session = QLineEdit()
        self.meta_session.setPlaceholderText("e.g., Session01")
        self.metadata_layout.addRow("Session:", self.meta_session)

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

        for widget in (
            self.meta_animal_id,
            self.meta_session,
            self.meta_trial,
            self.meta_experiment,
            self.meta_condition,
            self.meta_arena,
        ):
            widget.textChanged.connect(self._update_filename_preview)
            widget.textChanged.connect(self._save_recording_form_state)
            widget.textChanged.connect(self._on_recording_metadata_controls_changed)
        self.meta_notes.textChanged.connect(self._save_recording_form_state)
        self.meta_notes.textChanged.connect(self._on_recording_metadata_controls_changed)

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
        settings_container.setSpacing(8)
        settings_layout = QFormLayout()
        # Tighter rows reclaim vertical space so the live preview keeps more room
        # when this panel is open alongside the Recording panel.
        settings_layout.setVerticalSpacing(6)
        settings_layout.setHorizontalSpacing(10)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        # FPS setting
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1.0, 200.0)
        self.spin_fps.setValue(self.default_fps)
        self.spin_fps.setSuffix(" fps")
        self.spin_fps.setToolTip(
            "Requested acquisition frame rate. The camera reports the rate it\n"
            "actually achieves; if exposure is too long, FPS drops below target."
        )
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
        self.spin_exposure.setToolTip(
            "Sensor exposure per frame. Must stay below 1000 / FPS milliseconds,\n"
            "otherwise the camera cannot reach the requested frame rate."
        )
        self.spin_exposure.valueChanged.connect(self._on_exposure_changed)
        settings_layout.addRow("Exposure Time:", self.spin_exposure)

        # Image format selection
        self.combo_image_format = QComboBox()
        self.combo_image_format.addItems(["Mono8", "BGR8"])
        if self.default_image_format in ("Mono8", "BGR8"):
            self.combo_image_format.setCurrentText(self.default_image_format)
        self.combo_image_format.setToolTip(
            "Mono8: grayscale, smallest files, best for tracking.\n"
            "BGR8: full color (color cameras only)."
        )
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
        self.combo_encoder.setToolTip(
            "Hardware encoder used by FFmpeg. NVIDIA GPU is fastest when available;\n"
            "libx264 works everywhere but uses CPU; QuickSync needs an Intel iGPU."
        )
        settings_layout.addRow("Video Encoder:", self.combo_encoder)

        settings_container.addLayout(settings_layout)

        self.btn_advanced = QPushButton("Advanced Controls")
        self._set_button_icon(self.btn_advanced, "settings", "#d86cff", "ghostButton")
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
        control_layout.setSpacing(8)

        # Compact readiness line (e.g. "Ready to record from FLIR …"). The full
        # planner session summary now lives in the live-view top bar, so the
        # bulky "Active Session" strip that used to sit here is gone.
        self.label_recording_camera_hint = QLabel("Camera source is managed from the left Camera panel.")
        self.label_recording_camera_hint.setStyleSheet("color: #8fa6bf; font-size: 11px;")
        control_layout.addWidget(self.label_recording_camera_hint)

        recording_layout = QHBoxLayout()
        self.edit_save_folder = QLineEdit()
        self.edit_save_folder.setText(self.last_save_folder)
        self.edit_save_folder.setReadOnly(True)
        self.edit_save_folder.textChanged.connect(self._update_filename_preview)
        recording_layout.addWidget(self._make_field_label("Save to:"))
        recording_layout.addWidget(self.edit_save_folder)

        btn_browse = QPushButton("Browse...")
        self._set_button_icon(btn_browse, "folder", "#ff9a43", "orangeButton")
        btn_browse.clicked.connect(self._browse_save_folder)
        recording_layout.addWidget(btn_browse)
        control_layout.addLayout(recording_layout)

        filename_layout = QHBoxLayout()
        filename_layout.addWidget(self._make_field_label("Filename preview:"))

        self.edit_filename = QLineEdit()
        self.edit_filename.setPlaceholderText("Type a custom filename or leave blank for auto-generated")
        self.edit_filename.textEdited.connect(self._on_filename_text_edited)
        self.edit_filename.editingFinished.connect(self._on_filename_editing_finished)
        filename_layout.addWidget(self.edit_filename, stretch=2)
        control_layout.addLayout(filename_layout)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self._make_field_label("Path preview:"))

        self.edit_path_preview = QLineEdit()
        self.edit_path_preview.setReadOnly(True)
        self.edit_path_preview.setPlaceholderText("Recording output base path")
        path_layout.addWidget(self.edit_path_preview, stretch=2)

        self.btn_open_folder = QPushButton("Open Folder")
        self._set_button_icon(self.btn_open_folder, "folder", "#9dd9ff", "ghostButton")
        self.btn_open_folder.clicked.connect(self._open_recording_output_folder)
        path_layout.addWidget(self.btn_open_folder)

        self.btn_create_folders = QPushButton("Create Folders")
        self._set_button_icon(self.btn_create_folders, "folder", "#ffb35d", "ghostButton")
        self.btn_create_folders.clicked.connect(self._create_recording_folders)
        path_layout.addWidget(self.btn_create_folders)
        control_layout.addLayout(path_layout)

        length_layout = QHBoxLayout()
        length_layout.addWidget(self._make_field_label("Max Length (HH:MM:SS):"))

        self.spin_hours = ZeroPaddedSpinBox(2)
        self.spin_hours.setRange(0, 99)
        self.spin_hours.setValue(0)
        self.spin_hours.setFixedWidth(80)
        length_layout.addWidget(self.spin_hours)

        length_layout.addWidget(QLabel(":"))

        self.spin_minutes = ZeroPaddedSpinBox(2)
        self.spin_minutes.setRange(0, 59)
        self.spin_minutes.setValue(5)
        self.spin_minutes.setFixedWidth(80)
        length_layout.addWidget(self.spin_minutes)

        length_layout.addWidget(QLabel(":"))

        self.spin_seconds = ZeroPaddedSpinBox(2)
        self.spin_seconds.setRange(0, 59)
        self.spin_seconds.setValue(0)
        self.spin_seconds.setFixedWidth(80)
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

        arduino_hint = QLabel("Plug the board in over USB, scan, pick its COM port, then connect.")
        arduino_hint.setWordWrap(True)
        arduino_hint.setStyleSheet("color: #8fa6bf; font-size: 10px;")
        arduino_layout.addWidget(arduino_hint)

        port_layout = QHBoxLayout()
        self.combo_arduino_port = QComboBox()
        self.combo_arduino_port.setToolTip(
            "Serial port of the Arduino running StandardFirmataBarcode.\n"
            "Press Scan if the list is empty or the board was just plugged in."
        )
        port_layout.addWidget(QLabel("Port:"))
        port_layout.addWidget(self.combo_arduino_port)

        btn_scan = QPushButton("Scan")
        btn_scan.setToolTip("Search for serial ports with a connected board")
        self._set_button_icon(btn_scan, "import", "#33d5ff", "ghostButton")
        btn_scan.clicked.connect(self._scan_arduino_ports)
        port_layout.addWidget(btn_scan)
        arduino_layout.addLayout(port_layout)

        self.btn_arduino_connect = QPushButton("Connect Arduino")
        self.btn_arduino_connect.setToolTip(
            "Open the selected port and start TTL / behavior I/O.\n"
            "Once connected, signals are sampled in sync with camera frames."
        )
        self._set_button_icon(self.btn_arduino_connect, "play", "#eef6ff")
        self.btn_arduino_connect.clicked.connect(self._on_arduino_connect_clicked)
        arduino_layout.addWidget(self.btn_arduino_connect)

        self.label_arduino_status = self._make_panel_chip("Disconnected", "default")
        arduino_layout.addWidget(self.label_arduino_status, alignment=Qt.AlignLeft)

        arduino_group.setLayout(arduino_layout)
        layout.addWidget(arduino_group)

        pin_group = QGroupBox("Board Pin Defaults")
        self.pin_defaults_group = pin_group
        pin_layout = QFormLayout()
        default_pin_map = self._default_behavior_pin_map()
        for key in self.BEHAVIOR_PIN_KEYS:
            name_label = QLabel(f"{self._signal_label(key)}:")
            value_label = QLabel(self._format_pin_list(default_pin_map.get(key, [])))
            self.pin_name_labels[key] = name_label
            self.pin_value_labels[key] = value_label
            self.pin_row_widgets[key] = [name_label, value_label]
            pin_layout.addRow(name_label, value_label)
        self.label_gate_pin = self.pin_value_labels["gate"]
        self.label_sync_pin = self.pin_value_labels["sync"]
        self.label_barcode_pins = self.pin_value_labels["barcode"]
        pin_group.setLayout(pin_layout)
        layout.addWidget(pin_group)

        config_group = QGroupBox("Signal Mapping")
        self.signal_mapping_group = config_group
        config_layout = QVBoxLayout()

        mapping_hint = QLabel(
            "Map each signal to board pins. Inputs are sampled per camera frame; "
            "Outputs are driven by the board. Press Apply Mapping to activate changes."
        )
        mapping_hint.setWordWrap(True)
        mapping_hint.setStyleSheet("color: #8fa6bf; font-size: 10px;")
        config_layout.addWidget(mapping_hint)

        config_grid = QGridLayout()
        config_grid.setHorizontalSpacing(6)
        config_grid.setVerticalSpacing(6)
        config_grid.setColumnStretch(1, 2)
        config_grid.setColumnStretch(2, 1)
        config_grid.setColumnStretch(3, 1)
        header_use = QLabel("Use")
        header_use.setToolTip("Enable or disable this signal entirely (UI, board I/O, and CSV export)")
        header_label = QLabel("Label")
        header_label.setToolTip("Display name used in plots and CSV column names")
        header_role = QLabel("Role")
        header_role.setToolTip("Input: the board reads this pin.\nOutput: the board drives this pin.")
        header_pins = QLabel("Pins")
        header_pins.setToolTip("Digital pin numbers on the board, comma separated (e.g. 8, 9)")
        header_cfg = QLabel("Cfg")
        header_cfg.setToolTip("Extra parameters for sync and barcode signals")
        config_grid.addWidget(header_use, 0, 0)
        config_grid.addWidget(header_label, 0, 1)
        config_grid.addWidget(header_role, 0, 2)
        config_grid.addWidget(header_pins, 0, 3)
        config_grid.addWidget(header_cfg, 0, 4)

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
                self.barcode_param_button.setToolTip("Edit barcode parameters (software / hardware Timer1 ISR)")
                self.barcode_param_button.clicked.connect(self._edit_barcode_parameters)
                param_cell = self.barcode_param_button

            row_widgets = [enabled_check, label_edit, role_box, pin_edit]
            if isinstance(param_cell, QWidget):
                row_widgets.append(param_cell)
            self.signal_mapping_row_widgets[key] = row_widgets
            for widget in row_widgets:
                if widget is enabled_check:
                    continue
                widget.setEnabled(enabled_check.isChecked())
            enabled_check.toggled.connect(
                lambda checked, widgets=row_widgets: [
                    widget.setEnabled(checked) for widget in widgets if widget is not None and widget is not widgets[0]
                ]
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
        self.camera_line_labels_group = line_group
        line_layout = QFormLayout()
        line_options = self._line_label_choice_list()

        line1_label = QLabel("Line 1:")
        self.combo_line1_label = QComboBox()
        self.combo_line1_label.setEditable(True)
        self.combo_line1_label.addItems(line_options)
        self.combo_line1_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(1, v))
        self.camera_line_row_widgets[1] = [line1_label, self.combo_line1_label]
        line_layout.addRow(line1_label, self.combo_line1_label)

        line2_label = QLabel("Line 2:")
        self.combo_line2_label = QComboBox()
        self.combo_line2_label.setEditable(True)
        self.combo_line2_label.addItems(line_options)
        self.combo_line2_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(2, v))
        self.camera_line_row_widgets[2] = [line2_label, self.combo_line2_label]
        line_layout.addRow(line2_label, self.combo_line2_label)

        line3_label = QLabel("Line 3:")
        self.combo_line3_label = QComboBox()
        self.combo_line3_label.setEditable(True)
        self.combo_line3_label.addItems(line_options)
        self.combo_line3_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(3, v))
        self.camera_line_row_widgets[3] = [line3_label, self.combo_line3_label]
        line_layout.addRow(line3_label, self.combo_line3_label)

        line4_label = QLabel("Line 4:")
        self.combo_line4_label = QComboBox()
        self.combo_line4_label.setEditable(True)
        self.combo_line4_label.addItems(line_options)
        self.combo_line4_label.currentTextChanged.connect(lambda v: self._on_line_label_changed(4, v))
        self.camera_line_row_widgets[4] = [line4_label, self.combo_line4_label]
        line_layout.addRow(line4_label, self.combo_line4_label)

        line_group.setLayout(line_layout)
        layout.addWidget(line_group)

        self.btn_test_ttl = QPushButton("Test TTL / Behavior")
        self.btn_test_ttl.setToolTip(
            "Dry-run the board without recording: drives outputs and plots inputs\n"
            "in the TTL Monitor so you can verify wiring before an experiment."
        )
        self._set_button_icon(self.btn_test_ttl, "pulse", "#33d5ff", "violetButton")
        self.btn_test_ttl.clicked.connect(self._on_test_ttl_clicked)
        self.btn_test_ttl.setEnabled(False)
        self.btn_test_ttl.setMinimumHeight(36)
        layout.addWidget(self.btn_test_ttl)

        layout.addWidget(self._create_auxiliary_devices_group())

        layout.addStretch()
        return panel

    def _create_auxiliary_devices_group(self) -> QWidget:
        """Build the 'Additional Arduino Devices' editor (extra outputs / inputs).

        The primary board above is untouched. Each auxiliary device is a separate
        board with a generic set of named Input/Output pins. Input pins are
        sampled per camera frame and logged as dev<id>_<label>_ttl columns;
        Output pins can be held high or pulsed from here.
        """
        group = QGroupBox("Additional Arduino Devices")
        outer = QVBoxLayout(group)

        hint = QLabel(
            "Add extra boards to drive more outputs or record more inputs. "
            "Input pins are logged per frame alongside the primary TTL columns; "
            "Output pins can be held or pulsed below."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8fa6bf; font-size: 10px;")
        outer.addWidget(hint)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Device:"))
        self.aux_combo_select = QComboBox()
        self.aux_combo_select.setToolTip("Select which auxiliary board to edit")
        self.aux_combo_select.currentIndexChanged.connect(self._on_aux_device_selected)
        selector_row.addWidget(self.aux_combo_select, 1)

        self.btn_aux_add = QPushButton("Add")
        self.btn_aux_add.setToolTip("Add a new auxiliary board")
        self._set_button_icon(self.btn_aux_add, "import", "#6fe06e", "ghostButton")
        self.btn_aux_add.clicked.connect(self._on_aux_add_device)
        selector_row.addWidget(self.btn_aux_add)

        self.btn_aux_remove = QPushButton("Remove")
        self.btn_aux_remove.setToolTip("Remove the selected auxiliary board")
        self._set_button_icon(self.btn_aux_remove, "trash", "#ff8f8f", "ghostButton")
        self.btn_aux_remove.clicked.connect(self._on_aux_remove_device)
        selector_row.addWidget(self.btn_aux_remove)
        outer.addLayout(selector_row)

        # Editor for the currently selected device (populated on selection).
        self.aux_editor_container = QWidget()
        self.aux_editor_layout = QVBoxLayout(self.aux_editor_container)
        self.aux_editor_layout.setContentsMargins(0, 0, 0, 0)
        self.aux_editor_layout.setSpacing(8)
        outer.addWidget(self.aux_editor_container)

        self.aux_empty_label = QLabel("No additional devices. Click Add to create one.")
        self.aux_empty_label.setWordWrap(True)
        self.aux_empty_label.setStyleSheet("color: #8fa6bf; font-style: italic;")
        outer.addWidget(self.aux_empty_label)

        if self.aux_arduino_manager is not None:
            self._rebuild_aux_device_selector()
        return group

    # ===== Auxiliary device UI helpers =====

    def _rebuild_aux_device_selector(self):
        """Repopulate the device dropdown from the manager roster."""
        combo = getattr(self, "aux_combo_select", None)
        if combo is None or self.aux_arduino_manager is None:
            return
        devices = self.aux_arduino_manager.devices()
        combo.blockSignals(True)
        combo.clear()
        for worker in devices:
            combo.addItem(worker.name, worker.device_id)
        combo.blockSignals(False)

        has_devices = bool(devices)
        self.aux_empty_label.setVisible(not has_devices)
        self.aux_editor_container.setVisible(has_devices)
        self.btn_aux_remove.setEnabled(has_devices)

        if not has_devices:
            self.aux_selected_device_id = None
            self._clear_aux_editor()
            return

        target = self.aux_selected_device_id
        index = combo.findData(target) if target is not None else -1
        if index < 0:
            index = 0
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self.aux_selected_device_id = combo.itemData(index)
        self._show_aux_device_editor(self.aux_selected_device_id)

    def _clear_aux_editor(self):
        layout = getattr(self, "aux_editor_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.aux_row_widgets = []

    def _on_aux_device_selected(self, index: int):
        combo = self.aux_combo_select
        if index < 0 or combo is None:
            return
        self.aux_selected_device_id = combo.itemData(index)
        self._show_aux_device_editor(self.aux_selected_device_id)

    def _on_aux_add_device(self):
        if self.aux_arduino_manager is None:
            return
        worker = self.aux_arduino_manager.add_device()
        self.aux_arduino_manager.save()
        self.aux_selected_device_id = worker.device_id
        self._rebuild_aux_device_selector()
        self._scan_aux_ports()

    def _on_aux_remove_device(self):
        if self.aux_arduino_manager is None or self.aux_selected_device_id is None:
            return
        device_id = self.aux_selected_device_id
        worker = self.aux_arduino_manager.get_device(device_id)
        name = worker.name if worker is not None else "this device"
        reply = QMessageBox.question(
            self, "Remove Device",
            f"Remove {name}? This disconnects the board and forgets its pin setup.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.aux_arduino_manager.remove_device(device_id)
        self.aux_selected_device_id = None
        self._rebuild_aux_device_selector()

    def _show_aux_device_editor(self, device_id):
        """Rebuild the per-device editor (name, port, connect, pin table)."""
        self._clear_aux_editor()
        if self.aux_arduino_manager is None:
            return
        worker = self.aux_arduino_manager.get_device(device_id)
        if worker is None:
            return

        # Name + port + connect row
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.aux_name_edit = QLineEdit(worker.name)
        self.aux_name_edit.setPlaceholderText("Device name")
        self.aux_name_edit.editingFinished.connect(self._on_aux_name_edited)
        name_row.addWidget(self.aux_name_edit, 1)
        name_container = QWidget()
        name_container.setLayout(name_row)
        self.aux_editor_layout.addWidget(name_container)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Port:"))
        self.aux_port_combo = QComboBox()
        self.aux_port_combo.setEditable(True)
        self.aux_port_combo.setToolTip("Serial port of this auxiliary board running Firmata")
        self.aux_port_combo.currentTextChanged.connect(self._on_aux_port_changed)
        port_row.addWidget(self.aux_port_combo, 1)
        btn_scan = QPushButton("Scan")
        self._set_button_icon(btn_scan, "import", "#33d5ff", "ghostButton")
        btn_scan.clicked.connect(self._scan_aux_ports)
        port_row.addWidget(btn_scan)
        port_container = QWidget()
        port_container.setLayout(port_row)
        self.aux_editor_layout.addWidget(port_container)

        connect_row = QHBoxLayout()
        self.aux_connect_btn = QPushButton("Connect")
        self.aux_connect_btn.clicked.connect(self._on_aux_connect_clicked)
        connect_row.addWidget(self.aux_connect_btn)
        self.aux_status_chip = self._make_panel_chip("Disconnected", "default")
        connect_row.addWidget(self.aux_status_chip)
        connect_row.addStretch()
        connect_container = QWidget()
        connect_container.setLayout(connect_row)
        self.aux_editor_layout.addWidget(connect_container)

        # Pin table
        self.aux_pin_table = QTableWidget(0, 5)
        self.aux_pin_table.setHorizontalHeaderLabels(["Pin", "Label", "Role", "Live", "Action"])
        self.aux_pin_table.verticalHeader().setVisible(False)
        self.aux_pin_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.aux_pin_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.aux_pin_table.setMinimumHeight(150)
        self.aux_editor_layout.addWidget(self.aux_pin_table)

        self.aux_row_widgets = []
        for pin_entry in worker.snapshot()["pins"]:
            self._append_aux_pin_row(pin_entry)

        pin_btn_row = QHBoxLayout()
        btn_add_pin = QPushButton("Add Pin")
        self._set_button_icon(btn_add_pin, "import", "#6fe06e", "ghostButton")
        btn_add_pin.clicked.connect(lambda: self._append_aux_pin_row(None))
        pin_btn_row.addWidget(btn_add_pin)
        btn_remove_pin = QPushButton("Remove Pin")
        self._set_button_icon(btn_remove_pin, "trash", "#ff8f8f", "ghostButton")
        btn_remove_pin.clicked.connect(self._on_aux_remove_pin_row)
        pin_btn_row.addWidget(btn_remove_pin)
        pin_btn_row.addStretch()
        btn_apply = QPushButton("Apply Pins")
        self._set_button_icon(btn_apply, "check", "#6fe06e")
        btn_apply.clicked.connect(self._apply_aux_pins)
        pin_btn_row.addWidget(btn_apply)
        pin_btn_container = QWidget()
        pin_btn_container.setLayout(pin_btn_row)
        self.aux_editor_layout.addWidget(pin_btn_container)

        self._populate_aux_ports(worker.port_name)
        self._refresh_aux_device_controls()

    def _append_aux_pin_row(self, pin_entry):
        table = getattr(self, "aux_pin_table", None)
        if table is None:
            return
        row = table.rowCount()
        table.insertRow(row)

        pin_spin = QSpinBox()
        pin_spin.setRange(0, 69)
        pin_spin.setValue(int((pin_entry or {}).get("pin", 2)))
        table.setCellWidget(row, 0, pin_spin)

        label_edit = QLineEdit(str((pin_entry or {}).get("label", "")))
        label_edit.setPlaceholderText("e.g. lickL")
        table.setCellWidget(row, 1, label_edit)

        role_combo = QComboBox()
        role_combo.addItems(["Input", "Output"])
        role_combo.setCurrentText(str((pin_entry or {}).get("role", "Input")))
        role_combo.currentTextChanged.connect(lambda _v: self._refresh_aux_action_cells())
        table.setCellWidget(row, 2, role_combo)

        live_label = QLabel("-")
        live_label.setAlignment(Qt.AlignCenter)
        table.setCellWidget(row, 3, live_label)

        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        hold_check = QCheckBox("Hold")
        hold_check.toggled.connect(lambda checked, r=row: self._on_aux_hold_toggled(r, checked))
        pulse_btn = QPushButton("Pulse")
        pulse_btn.clicked.connect(lambda _=False, r=row: self._on_aux_pulse_clicked(r))
        action_layout.addWidget(hold_check)
        action_layout.addWidget(pulse_btn)
        table.setCellWidget(row, 4, action_widget)

        self.aux_row_widgets.append({
            "pin_spin": pin_spin,
            "label_edit": label_edit,
            "role_combo": role_combo,
            "live_label": live_label,
            "hold_check": hold_check,
            "pulse_btn": pulse_btn,
        })
        self._refresh_aux_action_cells()

    def _refresh_aux_action_cells(self):
        """Enable Hold/Pulse only on Output rows, and only when connected."""
        worker = self._selected_aux_worker()
        connected = bool(worker is not None and worker.is_connected)
        for widgets in getattr(self, "aux_row_widgets", []):
            is_output = widgets["role_combo"].currentText() == "Output"
            widgets["hold_check"].setVisible(is_output)
            widgets["pulse_btn"].setVisible(is_output)
            widgets["hold_check"].setEnabled(is_output and connected)
            widgets["pulse_btn"].setEnabled(is_output and connected)

    def _on_aux_remove_pin_row(self):
        table = getattr(self, "aux_pin_table", None)
        if table is None:
            return
        row = table.currentRow()
        if row < 0:
            row = table.rowCount() - 1
        if row < 0:
            return
        table.removeRow(row)
        if 0 <= row < len(self.aux_row_widgets):
            self.aux_row_widgets.pop(row)

    def _collect_aux_pins_from_table(self) -> List[Dict[str, object]]:
        pins = []
        for widgets in getattr(self, "aux_row_widgets", []):
            pins.append({
                "pin": int(widgets["pin_spin"].value()),
                "label": widgets["label_edit"].text().strip(),
                "role": widgets["role_combo"].currentText(),
            })
        return pins

    def _selected_aux_worker(self):
        if self.aux_arduino_manager is None or self.aux_selected_device_id is None:
            return None
        return self.aux_arduino_manager.get_device(self.aux_selected_device_id)

    def _apply_aux_pins(self):
        worker = self._selected_aux_worker()
        if worker is None:
            return
        worker.set_pins(self._collect_aux_pins_from_table())
        self.aux_arduino_manager.save()
        self._on_status_update(f"{worker.name}: pin configuration applied")
        self._refresh_aux_action_cells()

    def _on_aux_name_edited(self):
        worker = self._selected_aux_worker()
        if worker is None:
            return
        new_name = self.aux_name_edit.text().strip() or worker.name
        worker.name = new_name
        self.aux_arduino_manager.save()
        index = self.aux_combo_select.findData(worker.device_id)
        if index >= 0:
            self.aux_combo_select.setItemText(index, new_name)

    def _populate_aux_ports(self, preferred: str = ""):
        combo = getattr(self, "aux_port_combo", None)
        if combo is None:
            return
        ports = scan_serial_ports() or []
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(ports)
        if preferred:
            match = -1
            for i in range(combo.count()):
                if combo.itemText(i).split(" - ")[0].strip().upper() == preferred.split(" - ")[0].strip().upper():
                    match = i
                    break
            if match >= 0:
                combo.setCurrentIndex(match)
            else:
                combo.setEditText(preferred)
        combo.blockSignals(False)

    def _scan_aux_ports(self):
        worker = self._selected_aux_worker()
        self._populate_aux_ports(worker.port_name if worker is not None else "")

    def _on_aux_port_changed(self, text: str):
        worker = self._selected_aux_worker()
        if worker is None or worker.is_connected:
            return
        worker.port_name = str(text).strip()
        self.aux_arduino_manager.save()

    def _on_aux_connect_clicked(self):
        worker = self._selected_aux_worker()
        if worker is None or self.aux_arduino_manager is None:
            return
        if not worker.is_connected:
            port = self.aux_port_combo.currentText().strip()
            if not port or port.startswith("No ports"):
                self._on_error_occurred(f"{worker.name}: pick a COM port first (press Scan).")
                return
            normalized = port.split(" - ")[0].strip().upper()
            primary_port = str(self.combo_arduino_port.currentText() or "").split(" - ")[0].strip().upper()
            if self.is_arduino_connected and normalized == primary_port:
                self._on_error_occurred(f"{port} is already used by the primary board.")
                return
            if normalized in self.aux_arduino_manager.used_ports(exclude_id=worker.device_id):
                self._on_error_occurred(f"{port} is already assigned to another auxiliary device.")
                return
            worker.port_name = port
            worker.set_pins(self._collect_aux_pins_from_table())
            self.aux_arduino_manager.save()
            self.aux_arduino_manager.connect_device(worker.device_id)
        else:
            self.aux_arduino_manager.disconnect_device(worker.device_id)
        self._refresh_aux_device_controls()

    def _refresh_aux_device_controls(self):
        worker = self._selected_aux_worker()
        if worker is None:
            return
        connected = worker.is_connected
        if getattr(self, "aux_connect_btn", None) is not None:
            self.aux_connect_btn.setText("Disconnect" if connected else "Connect")
            self._set_button_icon(
                self.aux_connect_btn,
                "record" if connected else "play",
                "#ffffff" if connected else "#eef6ff",
                "dangerButton" if connected else None,
            )
        if getattr(self, "aux_status_chip", None) is not None:
            self._set_status_chip(
                self.aux_status_chip,
                "Connected" if connected else "Disconnected",
                "success" if connected else "default",
            )
        self._refresh_aux_action_cells()

    def _on_aux_hold_toggled(self, row: int, checked: bool):
        worker = self._selected_aux_worker()
        if worker is None or row >= len(self.aux_row_widgets):
            return
        pin = int(self.aux_row_widgets[row]["pin_spin"].value())
        worker.set_output_level(pin, checked)

    def _on_aux_pulse_clicked(self, row: int):
        worker = self._selected_aux_worker()
        if worker is None or row >= len(self.aux_row_widgets):
            return
        pin = int(self.aux_row_widgets[row]["pin_spin"].value())
        worker.start_output_pulse(pin, duration_ms=200, count=1)

    @Slot(str, bool, str)
    def _on_aux_connection_status(self, device_id, connected, message):
        if str(device_id) == str(self.aux_selected_device_id):
            self._refresh_aux_device_controls()
        if message:
            self._on_status_update(str(message))

    @Slot(dict)
    def _on_aux_states_updated(self, payload):
        if not isinstance(payload, dict):
            return
        if str(payload.get("device_id")) != str(self.aux_selected_device_id):
            return
        pin_states = payload.get("pins", {})
        for widgets in getattr(self, "aux_row_widgets", []):
            label = widgets["label_edit"].text().strip()
            if label in pin_states:
                high = bool(pin_states[label])
                live = widgets["live_label"]
                live.setText("HIGH" if high else "LOW")
                live.setStyleSheet(
                    "color: #7ef0ac; font-weight: 700;" if high else "color: #6f8197;"
                )

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
        self.behavior_plot.setStyleSheet("border: 1px solid #1c3046; border-radius: 8px;")
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
        self.live_detection_panel.center_circle_roi_requested.connect(self._center_live_circle_roi)
        self.live_detection_panel.edit_roi_requested.connect(self._edit_live_roi)
        self.live_detection_panel.finish_polygon_requested.connect(self._finish_live_polygon_roi)
        self.live_detection_panel.remove_roi_requested.connect(self._remove_live_roi)
        self.live_detection_panel.clear_rois_requested.connect(self._clear_live_rois)
        self.live_detection_panel.output_mapping_changed.connect(self._apply_live_output_mapping)
        self.live_detection_panel.add_rule_requested.connect(self._add_live_rule)
        self.live_detection_panel.edit_rule_requested.connect(self._edit_live_rule)
        self.live_detection_panel.test_rule_requested.connect(self._test_live_rule)
        self.live_detection_panel.remove_rule_requested.connect(self._remove_live_rule)
        self.live_detection_panel.overlay_options_changed.connect(self._on_live_overlay_options_changed)
        return self.live_detection_panel

    def _create_arduino_panel(self) -> QWidget:
        """Legacy compatibility shim for older code paths."""
        return self._wrap_scroll_dock_widget(self._create_behavior_setup_panel())

    def _create_audio_recording_panel(self) -> QWidget:
        """Create the ultrasound audio capture + live waveform tool panel."""
        self.audio_panel = UltrasoundPanel()
        return self.audio_panel

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

    @staticmethod
    def _format_camera_line_selector(selector: str, fallback_line_number: Optional[int] = None) -> str:
        text = str(selector or "").strip()
        if text:
            match = re.fullmatch(r"Line\s*(\d+)", text, flags=re.IGNORECASE)
            if match:
                return f"Line {int(match.group(1))}"
            return text
        if fallback_line_number is not None:
            return f"Line {int(fallback_line_number)}"
        return "Line"

    @staticmethod
    def _default_camera_line_selectors_for_backend(
        camera_type: str = "",
        flir_backend: str = "",
    ) -> List[str]:
        if str(camera_type).strip().lower() == "flir":
            return [f"Line{index}" for index in range(0, 4)]
        return [f"Line{index}" for index in range(1, 5)]

    @classmethod
    def _build_camera_line_selector_display_names(
        cls,
        capabilities: Optional[List[Dict[str, object]]] = None,
        fallback_selectors: Optional[List[str]] = None,
    ) -> Dict[int, str]:
        display_names: Dict[int, str] = {}
        selectors = [str(value).strip() for value in (fallback_selectors or []) if str(value).strip()]
        for line_number in range(1, 5):
            selector = selectors[line_number - 1] if line_number <= len(selectors) else ""
            display_names[line_number] = cls._format_camera_line_selector(selector, fallback_line_number=line_number)
        for index, capability in enumerate(capabilities or [], start=1):
            if index > 4:
                break
            selector = str(capability.get("selector", "")).strip()
            display_names[index] = cls._format_camera_line_selector(selector, fallback_line_number=index)
        return display_names

    def _camera_line_backend_fallback_selectors(self) -> List[str]:
        camera_type = ""
        flir_backend = ""
        if self.worker is not None and self.is_camera_connected:
            camera_type = str(getattr(self.worker, "camera_type", "") or "")
            flir_backend = str(getattr(self.worker, "flir_backend", "") or "")
        return self._default_camera_line_selectors_for_backend(camera_type, flir_backend)

    def _refresh_camera_line_selector_display_names(
        self,
        capabilities: Optional[List[Dict[str, object]]] = None,
    ) -> None:
        if capabilities is None:
            capabilities = []
            if self.worker is not None and self.is_camera_connected and self.worker.is_genicam_camera():
                try:
                    capabilities = self.worker.get_camera_line_capabilities()
                except Exception:
                    capabilities = []
        self.camera_line_selector_display_names = self._build_camera_line_selector_display_names(
            capabilities,
            fallback_selectors=self._camera_line_backend_fallback_selectors(),
        )

    def _sync_connected_camera_line_labels(self) -> None:
        capabilities: List[Dict[str, object]] = []
        if self.worker is not None and self.is_camera_connected and self.worker.is_genicam_camera():
            try:
                capabilities = self.worker.get_camera_line_capabilities()
            except Exception:
                capabilities = []

        self._refresh_camera_line_selector_display_names(capabilities)

        if capabilities:
            saved_defaults = self._load_camera_line_defaults()
            for index, capability in enumerate(capabilities[:4], start=1):
                selector = str(capability.get("selector", "")).strip()
                label_value = str(saved_defaults.get(selector, {}).get("label", "")).strip()
                if not selector or not label_value:
                    continue

                combo = getattr(self, f"combo_line{index}_label", None)
                if combo is None:
                    continue

                normalized_label = "Sync" if label_value == "TTL 1Hz" else label_value
                combo.blockSignals(True)
                combo.setCurrentText(normalized_label)
                combo.blockSignals(False)

        self._apply_line_label_map_to_worker()
        self._refresh_behavior_panel_visibility()
        self._rebuild_monitor_visuals(reset_plot=False)

    def _camera_line_display_name(self, line_number: int) -> str:
        base_name = self.camera_line_selector_display_names.get(int(line_number), f"Line {line_number}")
        label = self._camera_line_label_text(line_number)
        if label and label.lower() != "none":
            return f"{base_name} ({label})"
        return base_name

    def _camera_line_export_base_column(self, line_number: int) -> str:
        """Return the CSV base column for the line name shown in the UI."""
        display_name = self.camera_line_selector_display_names.get(int(line_number), f"Line {line_number}")
        match = re.search(r"\bline\s*(\d+)\b", str(display_name), flags=re.IGNORECASE)
        if match:
            return f"line{int(match.group(1))}_status"
        slug = self._slugify_export_label(str(display_name), f"line{line_number}")
        return f"{slug}_status"

    def _camera_line_labeled_export_column(self, line_number: int, suffix: str) -> str:
        """Return the final CSV column name for one labeled camera line."""
        return f"{self._camera_line_export_base_column(line_number)}_{suffix}"

    def _camera_line_label_text(self, line_number: int) -> str:
        combo = getattr(self, f"combo_line{line_number}_label", None)
        if combo is not None:
            return combo.currentText().strip()
        return str(self.settings.value(f"line_label_{line_number}", "") or "").strip()

    def _camera_line_color(self, line_number: int) -> str:
        palette = {
            1: "#7ef0ac",
            2: "#5cc8ff",
            3: "#ffb35d",
            4: "#facc15",
        }
        return palette.get(int(line_number), "#94a3b8")

    def _current_camera_roi_text(self) -> str:
        if isinstance(self.roi_rect, dict):
            try:
                width = max(1, int(self.roi_rect.get("w", 0)))
                height = max(1, int(self.roi_rect.get("h", 0)))
            except Exception:
                return "Full Frame"
            return f"ROI {width} x {height}"
        return "Full Frame"

    def _sync_camera_roi_ui_state(self):
        if hasattr(self, "btn_draw_roi") and self.btn_draw_roi is not None and not self.roi_draw_mode:
            self.btn_draw_roi.setText("Edit ROI" if self.roi_rect else "Draw ROI")
            self.btn_draw_roi.setStyleSheet("")
        self._update_live_header(roi_text=self._current_camera_roi_text())

    def _persist_camera_roi_setting(self, sync: bool = True):
        """Persist the current camera ROI immediately."""
        self.settings.setValue(
            "camera_roi_json",
            json.dumps(self.roi_rect) if isinstance(self.roi_rect, dict) else "",
        )
        if sync:
            self.settings.sync()

    def _roi_preview_to_camera_scale(self) -> tuple[float, float]:
        """Return (scale_x, scale_y) from preview coordinates to camera resolution."""
        if self.last_frame_size is None or self.worker is None:
            return 1.0, 1.0
        preview_w, preview_h = self.last_frame_size
        cam_w = getattr(self.worker, 'width', 0) or 0
        cam_h = getattr(self.worker, 'height', 0) or 0
        if preview_w <= 0 or preview_h <= 0 or cam_w <= 0 or cam_h <= 0:
            return 1.0, 1.0
        return cam_w / float(preview_w), cam_h / float(preview_h)

    def _camera_roi_geometry_for_frame(self) -> Optional[tuple[int, int, int, int]]:
        """Return the saved camera-resolution ROI scaled to preview coordinates."""
        if not isinstance(self.roi_rect, dict) or self.last_frame_size is None:
            return None
        width, height = self.last_frame_size
        if width <= 0 or height <= 0:
            return None
        sx, sy = self._roi_preview_to_camera_scale()
        # roi_rect is stored in camera space → convert to preview space
        roi_x = max(0, min(int(round(self.roi_rect.get("x", 0) / sx)), max(0, width - 1)))
        roi_y = max(0, min(int(round(self.roi_rect.get("y", 0) / sy)), max(0, height - 1)))
        roi_w = max(1, min(int(round(self.roi_rect.get("w", width) / sx)), max(1, width - roi_x)))
        roi_h = max(1, min(int(round(self.roi_rect.get("h", height) / sy)), max(1, height - roi_y)))
        return roi_x, roi_y, roi_w, roi_h

    def _remove_camera_roi_item(self):
        """Remove the camera ROI overlay without clearing the saved ROI."""
        if self.roi_item is None:
            return
        if self.live_image_view is not None:
            try:
                self.live_image_view.getView().removeItem(self.roi_item)
            except Exception:
                pass
        self.roi_item = None

    def _sync_camera_roi_overlay(self):
        """Ensure the saved camera ROI is visible once preview frames are available."""
        if self.roi_draw_mode or self.live_image_view is None:
            return

        geometry = self._camera_roi_geometry_for_frame()
        if geometry is None:
            self._remove_camera_roi_item()
            return

        roi_x, roi_y, roi_w, roi_h = geometry
        if self.roi_item is None:
            view_box = self.live_image_view.getView()
            self.roi_item = pg.RectROI(
                [roi_x, roi_y],
                [roi_w, roi_h],
                pen=pg.mkPen("#22c55e", width=2),
                movable=True,
                resizable=True,
            )
            self.roi_item.addScaleHandle([1, 1], [0, 0])
            self.roi_item.addScaleHandle([0, 0], [1, 1])
            self.roi_item.addScaleHandle([1, 0], [0, 1])
            self.roi_item.addScaleHandle([0, 1], [1, 0])
            view_box.addItem(self.roi_item)
        else:
            self.roi_item.setPos([roi_x, roi_y], finish=False)
            self.roi_item.setSize([roi_w, roi_h], finish=False)
        self.roi_item.setPen(pg.mkPen("#22c55e", width=2))

    def _load_saved_camera_roi(self) -> Optional[Dict[str, int]]:
        raw_value = str(self.settings.value("camera_roi_json", "") or "").strip()
        if not raw_value:
            return None
        try:
            payload = json.loads(raw_value)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return {
                "x": max(0, int(round(float(payload.get("x", 0))))),
                "y": max(0, int(round(float(payload.get("y", 0))))),
                "w": max(1, int(round(float(payload.get("w", 1))))),
                "h": max(1, int(round(float(payload.get("h", 1))))),
            }
        except Exception:
            return None

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
                "enabled": bool(self.signal_display_config.get(key, {}).get("enabled", True)),
            }

        return definitions

    def _coerce_binary_series(self, df, candidates: List[str]):
        """Return the first matching column coerced to a clean binary series."""
        import pandas as pd

        for column in candidates:
            if column in df.columns:
                return pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int).clip(0, 1)
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    def _signal_label_aliases(self, key: str) -> set[str]:
        """Return CSV-safe aliases that may identify a logical signal on a camera line."""
        definitions = self._signal_export_definitions()
        definition = definitions.get(key, {})
        aliases = {
            key,
            self._state_key_for_display(key),
            str(definition.get("slug", "")),
            self._slugify_export_label(str(definition.get("label", "")), key),
            self._slugify_export_label(str(self.DISPLAY_SIGNAL_META.get(key, {}).get("name", "")), key),
        }
        aliases.update(
            {
                "gate": ["gate", "gate_beam"],
                "sync": ["sync", "sync_pulse", "ttl_1hz", "ttl1hz", "1hz", "imec_1hz", "imec1hz", "imec_sync", "barcode_sync"],
                "barcode": ["barcode", "barcode0", "barcode1", "barcode_data", "code", "code_out", "data"],
                "lever": ["lever", "lever_press"],
                "cue": ["cue", "cue_led"],
                "reward": ["reward", "reward_led", "reward_valve"],
                "iti": ["iti", "iti_led", "iti_light"],
            }.get(key, [])
        )
        return {
            self._slugify_export_label(alias, key)
            for alias in aliases
            if str(alias).strip()
        }

    def _camera_line_signal_columns(self, df, key: str) -> List[str]:
        """Find labeled camera-line columns that represent one logical signal."""
        if df is None or df.empty:
            return []

        aliases = self._signal_label_aliases(key)
        label_map = self._get_line_label_map()
        columns: List[str] = []

        for line_number in range(1, 5):
            base_column = f"line{line_number}_status"
            suffixes = []
            mapped_suffix = str(label_map.get(base_column, "") or "").strip()
            if mapped_suffix:
                suffixes.append(mapped_suffix)
            ui_suffix = self._line_label_suffix(self._camera_line_label_text(line_number))
            if ui_suffix and ui_suffix not in suffixes:
                suffixes.append(ui_suffix)

            for suffix in suffixes:
                if self._slugify_export_label(suffix, key) not in aliases:
                    continue
                exported_column = self._camera_line_labeled_export_column(line_number, suffix)
                if exported_column in df.columns and exported_column not in columns:
                    columns.append(exported_column)
                labeled_column = f"{base_column}_{suffix}"
                if labeled_column in df.columns and labeled_column not in columns:
                    columns.append(labeled_column)
                if base_column in df.columns and base_column not in columns:
                    columns.append(base_column)

        for column in df.columns:
            match = re.match(r"^line[1-4]_status_(.+)$", str(column))
            if not match:
                continue
            suffix = self._slugify_export_label(match.group(1), key)
            if suffix in aliases and column not in columns:
                columns.append(column)

        return columns

    def _resolve_camera_line_signal_series(self, df, key: str):
        """Resolve a signal from camera chunk line states when a line is labeled for it."""
        columns = self._camera_line_signal_columns(df, key)
        if not columns:
            return None

        import pandas as pd

        combined = None
        found_valid = False
        for column in columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            if not bool(numeric.notna().any()):
                continue
            found_valid = True
            binary = numeric.fillna(0).astype(int).clip(0, 1)
            combined = binary if combined is None else (combined | binary).astype(int)

        return combined if found_valid else None

    def _cumulative_rise_count_series(self, state_series):
        """Build a framewise cumulative pulse count from a binary state series."""
        import pandas as pd

        state = pd.to_numeric(state_series, errors="coerce").fillna(0).astype(int).clip(0, 1)
        rises = ((state == 1) & (state.shift(1, fill_value=0) == 0)).astype(int)
        return rises.cumsum().astype(int)

    def _resolve_display_signal_series(self, df, key: str):
        """Resolve one logical signal into a binary series from raw export columns."""
        definitions = self._signal_export_definitions()
        state_column = definitions.get(key, {}).get("state_column", "")
        camera_line_series = self._resolve_camera_line_signal_series(df, key)
        if camera_line_series is not None:
            return camera_line_series

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
        camera_line_series = self._resolve_camera_line_signal_series(df, key)
        if camera_line_series is not None:
            return self._cumulative_rise_count_series(camera_line_series)

        if key == "barcode":
            candidates = [preferred, "barcode_count"]
        else:
            state_key = self._state_key_for_display(key)
            candidates = [preferred, f"{key}_count", f"{state_key}_count"]

        for column in candidates:
            if column in df.columns:
                return pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
        return None

    def _has_frame_aligned_signal_sources(self, df) -> bool:
        """Return True when frame rows contain camera-line or raw TTL columns worth exporting."""
        if df is None or df.empty:
            return False

        for key in self._active_signal_keys():
            if self._resolve_camera_line_signal_series(df, key) is not None:
                return True

        raw_columns = {
            "gate_ttl",
            "sync_1hz_ttl",
            "sync_10hz_ttl",
            "barcode_pin0_ttl",
            "barcode_pin1_ttl",
            "lever_ttl",
            "cue_ttl",
            "reward_ttl",
            "iti_ttl",
        }
        return any(column in df.columns for column in raw_columns)

    def _reorder_signal_export_columns(self, df):
        """Move label-driven signal columns forward in exported CSV files."""
        if df is None or df.empty:
            return df

        definitions = self._signal_export_definitions()
        active_signal_keys = self._active_signal_keys()
        preferred: List[str] = []
        metadata_columns = [
            "frame_id",
            "timestamp_camera",
            "timestamp_software",
            "animal_id",
            "session",
            "trial",
            "experiment",
            "condition",
            "arena",
            "date",
            "filename_preview",
            "user_flag_label",
            "user_flag_shortcut",
            "user_flag_output",
            "user_flag_pulse_ms",
            "user_flag_event_timestamp_software",
            "user_flag_event",
            "user_flag_ttl",
            "user_flag_count",
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
            "live_detection_frame_id",
            "live_detection_timestamp_software",
            "live_detection_completed_timestamp_software",
            "live_detection_age_ms",
            "live_inference_ms",
            "live_predict_ms",
            "live_preprocess_ms",
            "live_postprocess_ms",
            "live_queue_wait_ms",
            "live_end_to_end_ms",
            "live_inference_input_width",
            "live_inference_input_height",
            "live_detection_count",
            "animal_track_id",
            "animal_center_x",
            "animal_center_y",
            "animal_confidence",
            "animal_class_id",
        ]
        for column in metadata_columns:
            if column in df.columns and column not in preferred:
                preferred.append(column)

        for line_number in self._active_camera_line_numbers():
            base_column = f"line{line_number}_status"
            suffix = self._get_line_label_map().get(base_column, "")
            candidates = []
            if suffix:
                candidates.append(self._camera_line_labeled_export_column(line_number, suffix))
                candidates.append(f"{base_column}_{suffix}")
            candidates.append(base_column)
            for selected_column in candidates:
                if selected_column in df.columns and selected_column not in preferred:
                    preferred.append(selected_column)
                    break

        for key in active_signal_keys:
            spec = definitions.get(key, {})
            for column in (spec.get("state_column"), spec.get("count_column")):
                if column and column in df.columns and column not in preferred:
                    preferred.append(column)

        for column in df.columns:
            if column.startswith("in_zone_roi_") and column not in preferred:
                preferred.append(column)
        for column in df.columns:
            if "_in_zone_roi_" in column and column not in preferred:
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

    def _active_camera_line_numbers(self) -> List[int]:
        active = []
        for line_number in range(1, 5):
            label = self._camera_line_label_text(line_number)
            if label and label.lower() != "none":
                active.append(line_number)
        return active

    def _active_camera_line_keys(self) -> List[str]:
        return [f"line{line_number}_status" for line_number in self._active_camera_line_numbers()]

    def _refresh_behavior_panel_visibility(self):
        """Show only selected behavior signals and labeled camera lines in the main panel."""
        active_signal_keys = set(self._active_signal_keys())
        for key in self.BEHAVIOR_PIN_KEYS:
            visible = key in active_signal_keys
            for widget in self.pin_row_widgets.get(key, []):
                if widget is not None:
                    widget.setVisible(visible)
            for widget in self.signal_mapping_row_widgets.get(key, []):
                if widget is not None:
                    widget.setVisible(visible)

        if self.pin_defaults_group is not None:
            self.pin_defaults_group.setVisible(bool(active_signal_keys))
        if self.signal_mapping_group is not None:
            self.signal_mapping_group.setVisible(bool(active_signal_keys))

        active_lines = set(self._active_camera_line_numbers())
        for line_number, widgets in self.camera_line_row_widgets.items():
            if widgets and isinstance(widgets[0], QLabel):
                widgets[0].setText(
                    f"{self.camera_line_selector_display_names.get(line_number, f'Line {line_number}')}:"
                )
            visible = line_number in active_lines
            for widget in widgets:
                if widget is not None:
                    widget.setVisible(visible)
        if self.camera_line_labels_group is not None:
            self.camera_line_labels_group.setVisible(bool(active_lines))

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
        label.setFixedHeight(18)
        if active:
            label.setText("HIGH")
            label.setStyleSheet(
                "QLabel { background-color: #113626; border: 1px solid #1f6c44; "
                "border-radius: 4px; padding: 1px 6px; color: #89f0b2; font-weight: 700; }"
            )
        else:
            label.setText("LOW")
            label.setStyleSheet(
                "QLabel { background-color: #0f1a28; border: 1px solid #29415d; "
                "border-radius: 4px; padding: 1px 6px; color: #9fb1c7; font-weight: 700; }"
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
            header.setFixedHeight(16)
            header.setStyleSheet("color: #8dd0ff; font-size: 10px; font-weight: 700;")
            layout.addWidget(header, 0, column)

        for row, key in enumerate(keys, start=1):
            signal_label = QLabel(self._signal_label(key))
            signal_label.setFixedHeight(18)
            signal_label.setStyleSheet("color: #eef6ff; font-weight: 600;")

            state_label = QLabel()
            state_label.setAlignment(Qt.AlignCenter)
            state_label.setMinimumWidth(66)
            self._set_signal_state_label(state_label, False)

            count_label = QLabel()
            count_label.setFixedHeight(18)
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
        self.camera_line_curves.clear()
        self.ttl_output_levels.clear()
        self.behavior_levels.clear()
        self.camera_line_levels.clear()

        self.ttl_plot.clear()
        self.behavior_plot.clear()
        self.camera_line_plot.clear()
        self.ttl_plot.setXRange(0, self.ttl_window_seconds)
        self.behavior_plot.setXRange(0, self.ttl_window_seconds)
        self.camera_line_plot.setXRange(0, self.ttl_window_seconds)

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

        active_camera_lines = self._active_camera_line_numbers()
        camera_line_ticks = []
        n_camera_lines = len(active_camera_lines)
        for index, line_number in enumerate(active_camera_lines, start=1):
            key = f"line{line_number}_status"
            level = float(n_camera_lines - index + 1)
            self.camera_line_levels[key] = level
            camera_line_ticks.append((level, self._camera_line_display_name(line_number)))
            self.camera_line_curves[key] = self.camera_line_plot.plot(
                pen=pg.mkPen(self._camera_line_color(line_number), width=2),
                name=self._camera_line_display_name(line_number),
                stepMode=True,
            )
        self.camera_line_plot_group.setVisible(bool(active_camera_lines))
        self.camera_line_plot.setYRange(-0.6, max(1.0, float(n_camera_lines) + 0.6))
        camera_line_axis_left = self.camera_line_plot.getAxis("left")
        camera_line_axis_left.setTextPen(pg.mkPen("#b9c6d3"))
        camera_line_axis_left.setPen(pg.mkPen("#6c7a89"))
        camera_line_axis_left.setTicks([camera_line_ticks] if camera_line_ticks else [[]])
        camera_line_axis_bottom = self.camera_line_plot.getAxis("bottom")
        camera_line_axis_bottom.setTextPen(pg.mkPen("#b9c6d3"))
        camera_line_axis_bottom.setPen(pg.mkPen("#6c7a89"))

        if reset_plot:
            self.time_data.clear()
            self.camera_line_time_data.clear()
            self.plot_start_time = datetime.now()
            self.camera_line_plot_start_time_s = None
            self.camera_line_last_signature = None
            for series in self.ttl_plot_data.values():
                series.clear()
            for series in self.camera_line_plot_data.values():
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
                self._sync_connected_camera_line_labels()
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
        self._refresh_behavior_panel_visibility()
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
        """Open dialog for barcode parameters — software or hardware (Timer1 ISR) mode."""
        hw_enabled = False
        hw_bit_width_ms = 80
        if self.arduino_worker:
            params = self.arduino_worker.get_barcode_parameters()
            hw_enabled = self.arduino_worker.get_hw_barcode_enabled()
            hw_bit_width_ms = self.arduino_worker.get_hw_barcode_bit_width_ms()
        else:
            params = {
                "bits": int(self.settings.value("barcode_bits", 32)),
                "start_pulse_s": float(self.settings.value("barcode_start_pulse_s", 0.1)),
                "start_low_s": float(self.settings.value("barcode_start_low_s", 0.1)),
                "bit_s": float(self.settings.value("barcode_bit_s", 0.1)),
                "interval_s": float(self.settings.value("barcode_interval_s", 5.0)),
            }
            hw_enabled = bool(int(self.settings.value("hw_barcode_enabled", 0)))
            hw_bit_width_ms = int(self.settings.value("hw_barcode_bit_width_ms", 80))

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

        # ── Hardware barcode mode ──
        check_hw = QCheckBox("Use hardware Timer1 ISR barcode (StandardFirmataBarcode)")
        check_hw.setChecked(hw_enabled)
        hw_note = QLabel(
            "When enabled, the Arduino Timer1 ISR drives D8 (word sync) and D9 (data, LSB-first). "
            "Python only sends start/stop commands — no jitter from the host. "
            "Flash the Arduino with StandardFirmataBarcode.ino first."
        )
        hw_note.setWordWrap(True)
        hw_note.setStyleSheet("color: #8fa6bf;")
        form.addRow("Hardware Barcode:", check_hw)
        form.addRow("", hw_note)

        spin_hw_bit_width = QSpinBox()
        spin_hw_bit_width.setRange(20, 500)
        spin_hw_bit_width.setSuffix(" ms")
        spin_hw_bit_width.setValue(hw_bit_width_ms)
        spin_hw_bit_width.setToolTip(
            "Duration of each bit in milliseconds. "
            "Recommended: at least 2× camera frame interval (e.g. 80 ms at 25 fps)."
        )
        form.addRow("HW Bit Width:", spin_hw_bit_width)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("color: #334466;")
        form.addRow(separator)

        sw_label = QLabel("Software barcode parameters (used when hardware mode is OFF):")
        sw_label.setStyleSheet("color: #8fa6bf;")
        form.addRow(sw_label)

        # ── Software barcode parameters ──
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

        # ── Enable/disable software fields based on HW checkbox ──
        sw_widgets = [spin_bits, spin_start_hi, spin_start_lo, spin_bit, spin_interval,
                      spin_primary_pin, check_mirror_pin, spin_mirror_pin]

        def _toggle_sw_fields(hw_on):
            for w in sw_widgets:
                w.setEnabled(not hw_on)
            spin_hw_bit_width.setEnabled(hw_on)

        check_hw.toggled.connect(_toggle_sw_fields)
        _toggle_sw_fields(check_hw.isChecked())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        new_hw_enabled = bool(check_hw.isChecked())
        new_hw_bit_width = int(spin_hw_bit_width.value())

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

        if not new_hw_enabled and mirror_enabled and mirror_pin_val == primary_pin_val:
            self._on_error_occurred("Mirror pin must be different from the primary barcode output pin.")
            return

        # Apply hardware barcode settings
        if self.arduino_worker:
            self.arduino_worker.set_hw_barcode_enabled(new_hw_enabled)
            self.arduino_worker.set_hw_barcode_bit_width_ms(new_hw_bit_width)
        else:
            self.settings.setValue("hw_barcode_enabled", int(new_hw_enabled))
            self.settings.setValue("hw_barcode_bit_width_ms", new_hw_bit_width)

        # Apply software barcode settings
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

        if not new_hw_enabled:
            barcode_pins = [primary_pin_val]
            if mirror_enabled:
                barcode_pins.append(mirror_pin_val)
            self._apply_barcode_output_pins(barcode_pins, persist=True)

        if new_hw_enabled:
            self._on_status_update(
                f"Hardware barcode enabled: Timer1 ISR on D8/D9, "
                f"bit_width={new_hw_bit_width} ms, "
                f"word_time={new_hw_bit_width * 32 / 1000:.2f} s"
            )
        else:
            self._on_status_update(
                "Software barcode params updated: "
                f"bits={bits_val}, start={start_hi_val:.3f}/{start_lo_val:.3f}s, "
                f"bit={bit_val:.3f}s, gap={interval_val:.3f}s, "
                f"word={word_duration:.3f}s, cycle={cycle_duration:.3f}s, "
                f"pins={self._format_pin_list(barcode_pins)}"
            )

    def _setup_worker(self):
        """Initialize the camera worker thread and connect signals."""
        self.worker = CameraWorker()
        self.camera_stream_manager = CameraStreamManager(self)
        self.camera_stream_manager.primary_camera_info_provider = self._primary_camera_info
        self.camera_stream_manager.status_message.connect(self._on_status_update)
        self.camera_stream_manager.error_message.connect(self._on_error_occurred)

        # Connect worker signals to GUI slots
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.preview_packet_ready.connect(self._on_preview_packet_ready)
        self.worker.record_frame_packet_ready.connect(self._on_record_frame_packet_ready)
        self.worker.frame_metadata_ready.connect(self._on_frame_metadata_ready)
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
        self.worker.set_roi(dict(self.roi_rect) if isinstance(self.roi_rect, dict) else None)
        self._setup_live_detection_worker()

    def _setup_live_detection_worker(self):
        if self.live_inference_worker is not None:
            return
        self.live_inference_worker = LiveInferenceWorker(self)
        self.live_inference_worker.result_ready.connect(self._on_live_detection_result)
        self.live_inference_worker.status_changed.connect(self._on_live_detection_status_changed)
        self.live_inference_worker.error_occurred.connect(self._on_error_occurred)
        if self.worker is not None:
            self.worker.live_inference_packet_ready.connect(
                self._on_live_inference_packet_ready,
                Qt.ConnectionType.DirectConnection,
            )
            self.worker.live_inference_packet_ready.connect(
                self.live_inference_worker.submit_preview,
                Qt.ConnectionType.DirectConnection,
            )

    def _apply_pipeline_settings_to_worker(self):
        """Push preview and buffering preferences into the camera worker."""
        if not self.worker:
            return
        self.worker.set_preview_enabled(self.check_preview_enabled.isChecked())
        self.worker.set_preview_fps(self.spin_preview_fps.value())
        self.worker.set_preview_max_width(self.spin_preview_width.value())
        self.worker.set_frame_buffer_size(self.spin_frame_buffer.value())
        self.worker.set_metadata_stats_interval(self.spin_metadata_stats_interval.value())
        # Ship inference frames at a throttled, downscaled rate so the live
        # inference worker stays fed without the 60 fps acquisition thread
        # starving it (the worker only ever consumes the newest frame).
        if hasattr(self.worker, "set_live_inference_emit_fps"):
            self.worker.set_live_inference_emit_fps(35.0)
            self.worker.set_live_inference_emit_max_width(
                int(self.live_detection_panel.spin_inference_width.value())
                if self.live_detection_panel is not None
                else 960
            )

    def _get_max_record_seconds(self) -> int:
        """Return the configured recording limit in seconds, or 0 if unlimited/disabled."""
        if self.check_unlimited.currentText() != "Limited":
            return 0
        return (
            self.spin_hours.value() * 3600
            + self.spin_minutes.value() * 60
            + self.spin_seconds.value()
        )

    def _parse_duration_seconds(self, raw_value) -> int:
        """Accept either HH:MM:SS text or legacy raw-second values."""
        if raw_value is None:
            return 0
        try:
            if isinstance(raw_value, (int, float)):
                return max(0, int(float(raw_value)))
            text = str(raw_value).strip()
            if not text:
                return 0
            if re.fullmatch(r"\d+:\d{1,2}:\d{1,2}", text):
                hours_s, minutes_s, seconds_s = text.split(":")
                return max(0, (int(hours_s) * 3600) + (int(minutes_s) * 60) + int(seconds_s))
            return max(0, int(float(text)))
        except (TypeError, ValueError):
            return 0

    def _format_duration_input_hms(self, raw_value) -> str:
        """Normalize supported duration input into HH:MM:SS."""
        return self._format_duration_hms(self._parse_duration_seconds(raw_value))

    def _planner_duration_value(self, payload: Dict[str, str]) -> str:
        """Read planner duration from current or legacy column names."""
        return payload.get(PLANNER_DURATION_HEADER, payload.get(LEGACY_PLANNER_DURATION_HEADER, ""))

    def _set_recording_length_seconds(self, total_seconds: int) -> None:
        """Load a whole-second duration into the recording length controls."""
        total_seconds = max(0, int(total_seconds))
        self.spin_hours.setValue(total_seconds // 3600)
        self.spin_minutes.setValue((total_seconds % 3600) // 60)
        self.spin_seconds.setValue(total_seconds % 60)

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

    def _selected_planner_rows(self) -> List[int]:
        """Return the sorted set of selected planner rows."""
        if self.planner_table is None or self.planner_table.selectionModel() is None:
            return []
        return sorted({index.row() for index in self.planner_table.selectionModel().selectedRows()})

    def _planner_autosave_path(self) -> Path:
        """Return the JSON snapshot path used for planner persistence."""
        app_data_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        base_dir = Path(app_data_dir) if app_data_dir else Path.home() / ".pykaboo"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / "last_trial_plan.json"

    def _planner_snapshot(self) -> Dict[str, object]:
        """Serialize planner rows, columns, and selection for startup restore."""
        rows = []
        if self.planner_table is not None:
            rows = [self._planner_row_payload(row) for row in range(self.planner_table.rowCount())]
        return {
            "custom_columns": list(self.planner_custom_columns),
            "rows": rows,
            "recording_base_paths": {
                str(row): base_path
                for row in range(self.planner_table.rowCount() if self.planner_table is not None else 0)
                if (base_path := self._planner_row_recording_base_path(row))
            },
            "manual_pending_rows": [
                row
                for row in range(self.planner_table.rowCount() if self.planner_table is not None else 0)
                if self._planner_row_manual_pending(row)
            ],
            "manual_acquired_rows": [
                row
                for row in range(self.planner_table.rowCount() if self.planner_table is not None else 0)
                if self._planner_row_manual_acquired(row)
            ],
            "selected_rows": self._selected_planner_rows(),
            "active_row": self.active_planner_row,
            "next_trial_number": self.planner_next_trial_number,
        }

    def _save_planner_state_snapshot(self) -> None:
        """Persist the current planner as the default plan for the next launch."""
        if not self._planner_autosave_enabled or self._planner_state_loading or self.planner_table is None:
            return
        try:
            self._planner_autosave_path().write_text(
                json.dumps(self._planner_snapshot(), ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _restore_last_planner_state(self) -> bool:
        """Restore the last saved planner snapshot or fall back to the last CSV."""
        if self.planner_table is None:
            return False

        snapshot_path = self._planner_autosave_path()
        if snapshot_path.exists():
            try:
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except Exception:
                snapshot = None
            if self._apply_planner_snapshot(snapshot):
                return True

        last_csv_path = self._planner_last_csv_path()
        if last_csv_path:
            return self._load_planner_trials_from_csv(last_csv_path)
        return False

    def _apply_planner_snapshot(self, snapshot: object) -> bool:
        """Load a saved planner snapshot from disk."""
        if self.planner_table is None or not isinstance(snapshot, dict):
            return False

        raw_rows = snapshot.get("rows", [])
        if not isinstance(raw_rows, list):
            return False

        custom_columns = []
        for name in snapshot.get("custom_columns", []):
            value = str(name).strip()
            if value and value not in self.planner_default_columns and value not in custom_columns:
                custom_columns.append(value)

        rows: List[Dict[str, str]] = []
        for payload in raw_rows:
            if isinstance(payload, dict):
                rows.append(self._normalize_planner_seed(payload))

        selected_rows = []
        for row in snapshot.get("selected_rows", []):
            try:
                selected_rows.append(int(row))
            except (TypeError, ValueError):
                continue

        try:
            next_trial_number = max(1, int(snapshot.get("next_trial_number", 1)))
        except (TypeError, ValueError):
            next_trial_number = 1

        try:
            saved_active_row = int(snapshot.get("active_row", -1))
        except (TypeError, ValueError):
            saved_active_row = -1

        recording_base_paths = snapshot.get("recording_base_paths", {})
        if not isinstance(recording_base_paths, dict):
            recording_base_paths = {}

        manual_pending_rows = set()
        for row_value in snapshot.get("manual_pending_rows", []):
            try:
                manual_pending_rows.add(int(row_value))
            except (TypeError, ValueError):
                continue

        manual_acquired_rows = set()
        for row_value in snapshot.get("manual_acquired_rows", []):
            try:
                manual_acquired_rows.add(int(row_value))
            except (TypeError, ValueError):
                continue

        self._planner_state_loading = True
        try:
            self.planner_custom_columns = custom_columns
            self._refresh_planner_columns()
            self.planner_table.clearSelection()
            self.planner_table.setRowCount(0)
            self.planner_next_trial_number = 1
            for payload in rows:
                self._append_planner_trial(payload)
            self.planner_next_trial_number = max(self.planner_next_trial_number, next_trial_number)
            for row_text, base_path in recording_base_paths.items():
                try:
                    row = int(row_text)
                except (TypeError, ValueError):
                    continue
                if 0 <= row < self.planner_table.rowCount():
                    self._set_planner_row_recording_base_path(row, str(base_path or ""))
            for row in manual_pending_rows:
                if 0 <= row < self.planner_table.rowCount():
                    self._set_planner_row_manual_pending(row, True)
            for row in manual_acquired_rows:
                if 0 <= row < self.planner_table.rowCount():
                    self._set_planner_row_manual_acquired(row, True)

            preferred_row = None
            for row in selected_rows:
                if 0 <= row < self.planner_table.rowCount():
                    preferred_row = row
                    break
            if preferred_row is None and 0 <= saved_active_row < self.planner_table.rowCount():
                preferred_row = saved_active_row
            if preferred_row is None and self.planner_table.rowCount() > 0:
                preferred_row = 0

            self.active_planner_row = preferred_row
            if preferred_row is not None:
                self.planner_table.selectRow(preferred_row)
                self._load_planner_row_into_metadata(
                    preferred_row,
                    announce=False,
                    clear_filename_override=False,
                )
        finally:
            self._planner_state_loading = False

        self._sync_planner_recording_statuses()
        self._fit_planner_columns()
        self._update_planner_summary()
        return True

    def _sync_active_trial_duration_cell(self):
        """Mirror the current recording-length controls into the active planner row."""
        if self._syncing_planner_to_recording or self._syncing_recording_to_planner:
            return
        if self.planner_table is None:
            return
        row = self._active_planner_row_index()
        if row is None:
            return
        duration_seconds = self._get_max_record_seconds()
        duration_text = self._format_duration_input_hms(duration_seconds)
        headers = self._planner_headers()
        if PLANNER_DURATION_HEADER not in headers:
            return
        duration_column = headers.index(PLANNER_DURATION_HEADER)
        item = self.planner_table.item(row, duration_column)
        if item is not None and item.text().strip() == duration_text:
            return
        self.planner_table.blockSignals(True)
        self._set_planner_cell(row, PLANNER_DURATION_HEADER, duration_text)
        self.planner_table.blockSignals(False)
        self._update_planner_summary()

    def _recording_metadata_planner_values(self) -> Dict[str, str]:
        """Return recording-form fields that correspond to planner columns."""
        values = {
            "Trial": self.meta_trial.text().strip() if self.meta_trial is not None else "",
            "Arena": self.meta_arena.text().strip() if self.meta_arena is not None else "",
            "Animal ID": self.meta_animal_id.text().strip() if self.meta_animal_id is not None else "",
            "Session": self.meta_session.text().strip() if self.meta_session is not None else "",
            "Experiment": self.meta_experiment.text().strip() if hasattr(self, "meta_experiment") and self.meta_experiment is not None else "",
            "Condition": self.meta_condition.text().strip() if self.meta_condition is not None else "",
            "Comments": self.meta_notes.toPlainText().strip() if hasattr(self, "meta_notes") and self.meta_notes is not None else "",
        }
        for field_name, field_edit in self.custom_metadata_fields.items():
            if field_name in values:
                continue
            values[field_name] = field_edit.text().strip()
        return values

    def _sync_active_trial_metadata_cells(self):
        """Mirror recording-form metadata into the active planner row."""
        if self._syncing_planner_to_recording or self._syncing_recording_to_planner:
            return
        if self.planner_table is None:
            return
        row = self._active_planner_row_index()
        if row is None:
            return

        headers = self._planner_headers()
        values = self._recording_metadata_planner_values()
        changed = False
        self._syncing_recording_to_planner = True
        self.planner_table.blockSignals(True)
        try:
            for header, value in values.items():
                if header not in headers:
                    continue
                column = headers.index(header)
                item = self.planner_table.item(row, column)
                existing = item.text().strip() if item is not None else ""
                if existing == value:
                    continue
                self._set_planner_cell(row, header, value)
                changed = True
        finally:
            self.planner_table.blockSignals(False)
            self._syncing_recording_to_planner = False

        if changed:
            self.active_planner_row = row
            self._update_planner_summary()

    def _on_recording_metadata_controls_changed(self, *_args):
        """Keep the active planner row aligned with recording metadata edits."""
        self._sync_active_trial_metadata_cells()
        self._refresh_recording_session_summary()

    def _apply_recording_frame_limit(self):
        """Align the worker's exact-duration frame cap with the current UI setting.

        The recording length is enforced deterministically as a frame count on
        the acquisition thread (frames = duration x encode FPS), so the saved
        file is exactly the requested length regardless of GUI-thread load. The
        wall-clock timer below is only a safety net for a stalled camera.
        """
        if self.worker is not None:
            max_seconds = self._get_max_record_seconds()
            fps = float(
                getattr(self.worker, "recording_output_fps", None)
                or getattr(self.worker, "fps_target", None)
                or 0.0
            )
            if max_seconds > 0 and fps > 0:
                self.worker.set_recording_frame_limit(max(1, int(round(fps * max_seconds))))
            else:
                self.worker.set_recording_frame_limit(None)
        self._restart_recording_duration_timer()

    def _restart_recording_duration_timer(self):
        """Arm a *safety* wall-clock stop in case the camera stalls before the cap.

        The exact stop is the worker's frame-count cap; this timer only fires if
        frames stop arriving and the cap can never be reached, so it uses a
        generous margin and must not pre-empt a legitimately slow-but-working
        capture.
        """
        self.recording_duration_timer.stop()
        if not self.recording_start_time:
            return

        max_seconds = self._get_max_record_seconds()
        if max_seconds <= 0:
            return

        # Safety margin: double the target plus 10 s. A working capture finishes
        # via the frame cap well before this; a stalled one is bounded here.
        safety_seconds = float(max_seconds) * 2.0 + 10.0
        elapsed_seconds = max(
            0.0,
            float((datetime.now() - self.recording_start_time).total_seconds()),
        )
        remaining_ms = int(round((safety_seconds - elapsed_seconds) * 1000.0))
        if remaining_ms <= 0:
            QTimer.singleShot(0, self._on_recording_duration_timeout)
            return
        self.recording_duration_timer.start(remaining_ms)

    @Slot()
    def _on_recording_duration_timeout(self):
        """Stop the current recording when the requested wall-clock limit is reached."""
        if self.worker is None or not self.worker.is_recording:
            return
        duration_text = self._format_duration_hms(self._get_max_record_seconds())
        self._on_status_update(f"Reached recording duration target: {duration_text}")
        self._request_recording_stop("duration_limit")

    def _request_recording_stop(self, reason: str):
        """Request a recording stop while preserving the best available stop timestamp."""
        if self.worker is None or not self.worker.is_recording:
            return
        if self.recording_stop_requested_at is None:
            self.recording_stop_requested_at = time.time()
        self.recording_stop_reason = str(reason or "manual")
        if self.active_recording_timing_audit:
            self.active_recording_timing_audit["stop_reason"] = self.recording_stop_reason
            self.active_recording_timing_audit["stop_requested_wallclock"] = float(
                self.recording_stop_requested_at
            )
        self.recording_duration_timer.stop()
        self.worker.stop_recording()
        if self.camera_stream_manager is not None:
            self.camera_stream_manager.stop_recording_all()

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

        # Select the last used port; actual auto-connect happens after all saved
        # live-detection mappings are restored.
        self._select_saved_arduino_port()

        # Auxiliary boards (extra outputs / extra recorded inputs) are managed
        # separately so the primary worker above stays untouched.
        self._setup_auxiliary_arduinos()

    def _setup_auxiliary_arduinos(self):
        """Create the auxiliary-device manager and restore the saved roster."""
        self.aux_arduino_manager = ArduinoDeviceManager(self.settings)
        self.aux_arduino_manager.on_device_created = self._bind_aux_device_signals
        self.aux_arduino_manager.load()
        self._rebuild_aux_device_selector()

    def _bind_aux_device_signals(self, worker):
        """Wire one auxiliary worker's signals to the GUI (called per device)."""
        worker.connection_status.connect(self._on_aux_connection_status)
        worker.states_updated.connect(self._on_aux_states_updated)
        worker.error_occurred.connect(self._on_error_occurred)

    def _scan_cameras(self):
        """Scan for Basler, FLIR, and generic USB cameras."""
        self.combo_camera.clear()
        cameras = []
        scan_details: List[str] = []
        basler_scan_error = ""

        try:
            basler_cameras = discover_basler_cameras()
        except Exception as e:
            basler_cameras = []
            basler_scan_error = str(e)
            if hasattr(self, "status_bar"):
                self._on_status_update(f"Basler scan error: {basler_scan_error}")

        flir_cameras, reserved_usb_indices = discover_flir_cameras()
        usb_cameras = discover_usb_cameras(skip_indices=reserved_usb_indices)
        backend_diagnostics = get_camera_backend_diagnostics()
        scan_details.append(f"Basler {len(basler_cameras)}")
        scan_details.append(f"FLIR {len(flir_cameras)}")
        scan_details.append(f"USB {len(usb_cameras)}")

        for camera_info in basler_cameras + flir_cameras + usb_cameras:
            self.combo_camera.addItem(camera_info.get("label", "Camera"), camera_info)
            cameras.append(camera_info)

        pypylon_diag = backend_diagnostics.get("pypylon", "")
        if not basler_cameras and pypylon_diag and hasattr(self, "status_bar"):
            self._on_status_update(f"Basler unavailable: {pypylon_diag}")

        if not any(cam.get("backend") == "spinnaker" for cam in flir_cameras):
            pyspin_diag = backend_diagnostics.get("pyspin", "")
            if pyspin_diag and hasattr(self, "status_bar"):
                self._on_status_update(f"FLIR Spinnaker unavailable: {pyspin_diag}")
        else:
            pyspin_diag = ""

        detail_lines = [f"Scan: {', '.join(scan_details)}"]
        runtime_name = Path(sys.executable).name
        runtime_mode = "frozen" if getattr(sys, "frozen", False) else "python"
        detail_lines.append(f"Runtime: {runtime_name} ({runtime_mode})")
        if basler_scan_error:
            detail_lines.append(f"Basler scan error: {basler_scan_error}")
        elif pypylon_diag:
            detail_lines.append(f"Basler backend: {pypylon_diag}")
        elif not basler_cameras:
            detail_lines.append("Basler backend loaded but returned 0 devices.")
        if pyspin_diag:
            detail_lines.append(f"FLIR backend: {pyspin_diag}")
        usb_diag = backend_diagnostics.get("usb", "")
        if usb_diag:
            detail_lines.append(f"USB backend: {usb_diag}")
            if not usb_cameras and hasattr(self, "status_bar"):
                self._on_status_update(f"USB cameras: {usb_diag}")

        summary_text = f"{detail_lines[0]}."
        if not cameras:
            summary_text = f"No cameras detected. {detail_lines[0]}."
        self.label_camera_scan_diagnostics.setText(summary_text)
        self.label_camera_scan_diagnostics.setToolTip("\n".join(detail_lines))
        self._refresh_camera_sources_list(cameras)

        if not cameras:
            self.combo_camera.addItem("No cameras detected", None)
            return

        self._select_saved_camera()

    def _select_saved_camera(self) -> bool:
        if not hasattr(self, "combo_camera"):
            return False
        saved_settings = self._saved_camera_settings()
        best_index = -1
        best_score = -1
        for index in range(self.combo_camera.count()):
            data = self.combo_camera.itemData(index)
            score = saved_camera_match_score(data, saved_settings)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index >= 0:
            self.combo_camera.setCurrentIndex(best_index)
            return True
        return False

    def _camera_info_matches_saved_selection(self, camera_info: Optional[Dict]) -> bool:
        return camera_matches_saved_selection(camera_info, self._saved_camera_settings())

    def _saved_camera_settings(self) -> Dict[str, str]:
        return {
            "last_camera_type": str(self.settings.value("last_camera_type", "") or "").strip(),
            "last_camera_backend": str(self.settings.value("last_camera_backend", "") or "").strip(),
            "last_camera_index": str(self.settings.value("last_camera_index", "") or "").strip(),
            "last_camera_serial": str(self.settings.value("last_camera_serial", "") or "").strip(),
            "last_camera_video_index": str(self.settings.value("last_camera_video_index", "") or "").strip(),
            "last_camera_serial_port": str(self.settings.value("last_camera_serial_port", "") or "").strip(),
            "last_camera_label": str(self.settings.value("last_camera_label", "") or "").strip(),
        }

    def _select_saved_arduino_port(self) -> bool:
        if self.arduino_worker is None or not hasattr(self, "combo_arduino_port"):
            return False
        saved_port = str(
            self.arduino_worker.port_name
            or self.settings.value("arduino_port", "")
            or ""
        ).strip()
        if not saved_port:
            return False
        for index in range(self.combo_arduino_port.count()):
            item_text = self.combo_arduino_port.itemText(index)
            port_name = item_text.split(" - ")[0].strip()
            if port_name.upper() == saved_port.upper():
                self.combo_arduino_port.setCurrentIndex(index)
                return True
        return False

    def _auto_connect_startup_devices(self):
        if self._startup_autoconnect_done:
            return
        self._startup_autoconnect_done = True

        self._auto_connect_last_camera()
        self._auto_connect_last_arduino()
        self._auto_connect_aux_arduinos()

    def _auto_connect_last_camera(self):
        if self.is_camera_connected or self.worker is None:
            return
        saved_settings = self._saved_camera_settings()
        if not saved_camera_settings_available(saved_settings):
            return
        if not self._select_saved_camera():
            self._schedule_startup_camera_autoconnect_retry(rescan=True)
            return
        camera_info = self.combo_camera.currentData()
        if not self._camera_info_matches_saved_selection(camera_info):
            self._schedule_startup_camera_autoconnect_retry(rescan=True)
            return
        camera_name = self.combo_camera.currentText().strip() or "last camera"
        self._on_status_update(f"Auto-connecting camera: {camera_name}")
        self._on_connect_clicked()
        if not self.is_camera_connected:
            self._schedule_startup_camera_autoconnect_retry(rescan=False)

    def _schedule_startup_camera_autoconnect_retry(self, *, rescan: bool) -> None:
        if self.is_camera_connected:
            return
        if self._startup_camera_autoconnect_attempts >= STARTUP_CAMERA_AUTOCONNECT_MAX_ATTEMPTS:
            return
        self._startup_camera_autoconnect_attempts += 1

        def retry() -> None:
            if self.is_camera_connected:
                return
            if rescan:
                self._scan_cameras()
            self._auto_connect_last_camera()

        QTimer.singleShot(STARTUP_CAMERA_AUTOCONNECT_RETRY_MS, retry)

    def _auto_connect_last_arduino(self):
        if self.is_arduino_connected or self.arduino_worker is None:
            return
        if not self._select_saved_arduino_port():
            return
        port = self.combo_arduino_port.currentText().strip()
        if not port:
            return
        self._on_status_update(f"Auto-connecting Arduino: {port}")
        self._on_arduino_connect_clicked()

    def _auto_connect_aux_arduinos(self):
        """Connect any auxiliary boards that have a saved port."""
        if self.aux_arduino_manager is None:
            return
        for worker in self.aux_arduino_manager.devices():
            if worker.is_connected or not worker.port_name:
                continue
            self._on_status_update(f"Auto-connecting {worker.name}: {worker.port_name}")
            if self.aux_arduino_manager.connect_device(worker.device_id):
                self._refresh_aux_device_controls()

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
            field_edit.textChanged.connect(self._save_recording_form_state)
            field_edit.textChanged.connect(self._on_recording_metadata_controls_changed)
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
                    if 'session' in template and self.meta_session is not None:
                        self.meta_session.setText(template['session'])
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
        return ["Animal ID", "Session", "Trial", "Experiment", "Condition", "Arena", "Date", "(skip)"]

    def _filename_label_to_key(self, label: str) -> str:
        mapping = {
            "Animal ID": "animal_id",
            "Session": "session",
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
            "session": "Session",
            "trial": "Trial",
            "experiment": "Experiment",
            "condition": "Condition",
            "arena": "Arena",
            "date": "Date",
            "": "(skip)",
        }
        return mapping.get(key, "(skip)")

    def _default_filename_order(self) -> List[str]:
        return ["animal_id", "session", "experiment", "condition"]

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
        self._update_planner_summary()

    def _on_organize_recordings_toggled(self, checked: bool):
        """Persist folder-organization mode and refresh preview text."""
        self.settings.setValue("organize_recordings_by_session", 1 if checked else 0)
        self._set_folder_structure_controls_enabled(checked)
        self._update_folder_structure_preview()
        self._update_filename_preview()
        self._update_planner_summary()

    def _organize_recordings_enabled(self) -> bool:
        checkbox = getattr(self, "check_organize_session_folders", None)
        return bool(checkbox is not None and checkbox.isChecked())

    def _default_folder_order(self) -> List[str]:
        return ["animal_id", "session"]

    def _selected_folder_order(self) -> List[str]:
        boxes = getattr(self, "folder_order_boxes", None)
        if not boxes:
            return self._default_folder_order()
        order: List[str] = []
        for combo in boxes:
            key = self._filename_label_to_key(combo.currentText())
            if key:
                order.append(key)
        return order

    def _set_folder_order_controls(self):
        """Populate folder-structure combo boxes from persisted settings."""
        boxes = getattr(self, "folder_order_boxes", None)
        if not boxes:
            return
        defaults = self._default_folder_order()
        for index, combo in enumerate(boxes):
            default_key = defaults[index] if index < len(defaults) else ""
            key = str(self.settings.value(f"folder_part_{index + 1}", default_key))
            label = self._filename_key_to_label(key)
            combo.blockSignals(True)
            if label in [combo.itemText(i) for i in range(combo.count())]:
                combo.setCurrentText(label)
            else:
                combo.setCurrentText("(skip)")
            combo.blockSignals(False)
        self._update_folder_structure_preview()

    def _on_folder_order_changed(self, *_args):
        """Persist folder structure and refresh the live folder preview."""
        for index, combo in enumerate(self.folder_order_boxes, start=1):
            self.settings.setValue(f"folder_part_{index}", self._filename_label_to_key(combo.currentText()))
        self._update_folder_structure_preview()
        self._update_filename_preview()
        self._update_planner_summary()

    def _set_folder_structure_controls_enabled(self, enabled: bool):
        """Grey out the folder-structure controls when nesting is disabled."""
        group = getattr(self, "folder_structure_group", None)
        if group is not None:
            group.setEnabled(enabled)

    def _update_folder_structure_preview(self):
        """Refresh the Storage tab preview of the resulting folder hierarchy."""
        label = getattr(self, "label_folder_structure_preview", None)
        if label is None:
            return
        if not self._organize_recordings_enabled():
            label.setText("Recordings save directly in the save root (no nested folders).")
            return
        order = self._selected_folder_order()
        if not order:
            label.setText("No folder levels selected; recordings save directly in the save root.")
            return
        structure = " / ".join(self._filename_key_to_label(key) for key in order)
        values = self._metadata_token_values()
        example_parts = []
        for key in order:
            token = self._sanitize_filename_part(values.get(key, ""))
            example_parts.append(token or f"<{self._filename_key_to_label(key)}>")
        example = " / ".join(["Save root", *example_parts])
        label.setText(f"Structure: {structure}\nExample: {example}")

    def _organized_recording_folder_parts(self, values: Optional[Dict[str, str]] = None) -> List[str]:
        if not self._organize_recordings_enabled():
            return []
        values = values or self._metadata_token_values()
        parts: List[str] = []
        for key in self._selected_folder_order():
            token = self._sanitize_filename_part(values.get(key, ""))
            if token:
                parts.append(token)
        return parts

    def _recording_destination_folder(self, values: Optional[Dict[str, str]] = None) -> Path:
        root_text = self.edit_save_folder.text().strip() if hasattr(self, "edit_save_folder") and self.edit_save_folder is not None else ""
        root = Path(root_text or self.last_save_folder or ".")
        folder = root
        for part in self._organized_recording_folder_parts(values=values):
            folder /= part
        return folder

    def _recording_destination_preview(self, values: Optional[Dict[str, str]] = None) -> str:
        parts = self._organized_recording_folder_parts(values=values)
        if not parts:
            return "Save root"
        return " / ".join(parts)

    def _recording_output_preview_path(
        self,
        values: Optional[Dict[str, str]] = None,
        custom_override: Optional[str] = None,
    ) -> Path:
        current_path = str(self.current_recording_filepath or "").strip()
        if current_path and values is None and custom_override is None:
            return Path(current_path)

        folder = self._recording_destination_folder(values=values)
        base_name = self._compose_recording_basename(values=values, custom_override=custom_override).strip() or "recording"
        candidate = folder / base_name
        if self._organize_recordings_enabled():
            return candidate
        return self._get_unique_recording_path(folder, base_name)

    def _nearest_existing_folder(self, folder: Path) -> Optional[Path]:
        candidate = Path(folder)
        while True:
            if candidate.exists() and candidate.is_dir():
                return candidate
            parent = candidate.parent
            if parent == candidate:
                return None
            candidate = parent

    def _open_folder_in_explorer(self, folder: Path, missing_label: str = "Folder") -> bool:
        target_folder = Path(folder)
        existing_folder = self._nearest_existing_folder(target_folder)
        if existing_folder is None:
            self._on_error_occurred(f"Could not find an existing folder to open for: {target_folder}")
            return False

        try:
            if os.name == "nt":
                os.startfile(str(existing_folder))
            else:
                opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(existing_folder)))
                if not opened:
                    raise RuntimeError("desktop services rejected the folder path")
        except Exception as exc:
            self._on_error_occurred(f"Could not open folder in Explorer: {str(exc)}")
            return False

        if existing_folder == target_folder:
            self._on_status_update(f"Opened folder: {existing_folder}")
        else:
            self._on_status_update(
                f"{missing_label} does not exist yet; opened nearest existing parent: {existing_folder}"
            )
        return True

    def _planner_payload_token_values(self, payload: Dict[str, str]) -> Dict[str, str]:
        values = self._metadata_token_values()
        values.update(
            {
                "animal_id": str(payload.get("Animal ID", "") or "").strip(),
                "session": str(payload.get("Session", "") or "").strip(),
                "trial": str(payload.get("Trial", "") or "").strip(),
                "experiment": str(payload.get("Experiment", "") or "").strip(),
                "condition": str(payload.get("Condition", "") or "").strip(),
                "arena": str(payload.get("Arena", "") or "").strip(),
            }
        )
        return values

    def _planner_row_output_folder(self, row: int) -> Optional[Path]:
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return None
        payload = self._planner_row_payload(row)
        values = self._planner_payload_token_values(payload)
        return self._recording_destination_folder(values=values)

    def _open_recording_output_folder(self):
        folder = self._recording_output_preview_path().parent
        self._open_folder_in_explorer(folder, missing_label="Recording folder")

    def _open_selected_planner_output_folder(self, row: Optional[int] = None):
        if self.planner_table is None:
            return

        target_row = row
        if target_row is None:
            selected_rows = self.planner_table.selectionModel().selectedRows()
            if selected_rows:
                target_row = selected_rows[0].row()

        if target_row is None:
            self._on_error_occurred("Select a planner row to open its output folder.")
            return

        folder = self._planner_row_output_folder(target_row)
        if folder is None:
            self._on_error_occurred("Could not resolve the planner output folder.")
            return

        payload = self._planner_row_payload(target_row)
        trial_label = payload.get("Trial", "?") or "?"
        self._open_folder_in_explorer(folder, missing_label=f"Planner folder for trial {trial_label}")

    def _create_recording_folders(self):
        folder = self._recording_destination_folder()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._on_error_occurred(f"Could not create recording folders: {str(exc)}")
            return
        self._on_status_update(f"Recording folder ready: {folder}")
        self._update_filename_preview()

    def _metadata_token_values(self) -> Dict[str, str]:
        return {
            "animal_id": self.meta_animal_id.text().strip() if self.meta_animal_id else "",
            "session": self.meta_session.text().strip() if self.meta_session else "",
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
        if self.meta_session is not None:
            self.settings.setValue("recording_meta_session", self.meta_session.text().strip())
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
        if self.meta_session is not None:
            self.meta_session.setText(str(self.settings.value("recording_meta_session", "")))
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

    def _compose_generated_recording_basename(self, values: Optional[Dict[str, str]] = None) -> str:
        values = values or self._metadata_token_values()
        ordered_parts = []
        for key in self._selected_filename_order():
            token = self._sanitize_filename_part(values.get(key, ""))
            if token:
                ordered_parts.append(token)
        has_primary_identity = any(
            self._sanitize_filename_part(values.get(key, ""))
            for key in ("animal_id", "session", "experiment", "condition")
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

    def _compose_recording_basename(
        self,
        values: Optional[Dict[str, str]] = None,
        custom_override: Optional[str] = None,
    ) -> str:
        custom_override = self._sanitize_filename_part(
            custom_override if custom_override is not None else self._current_custom_filename_override()
        )
        if custom_override:
            return custom_override
        return self._compose_generated_recording_basename(values=values)

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

    def _clear_custom_filename_override(self):
        """Return filename control to metadata-driven generation."""
        self._custom_filename_override = ""
        self._set_filename_field_text("")

    def _update_filename_preview(self, *_args):
        """Refresh the generated filename preview and formula label."""
        active_path = str(self.current_recording_filepath or "").strip()
        generated_basename = self._compose_generated_recording_basename()
        basename = Path(active_path).name if active_path else self._compose_recording_basename()
        output_preview = active_path or str(self._recording_output_preview_path())
        if hasattr(self, "edit_filename") and self.edit_filename is not None:
            if not self.edit_filename.hasFocus():
                self._set_filename_field_text(basename)
        if hasattr(self, "edit_path_preview") and self.edit_path_preview is not None:
            self.edit_path_preview.setText(output_preview)
            self.edit_path_preview.setToolTip(output_preview)
        if hasattr(self, "label_filename_formula") and self.label_filename_formula is not None:
            custom_override = self._current_custom_filename_override()
            if custom_override:
                folder_hint = self._recording_destination_preview()
                self.label_filename_formula.setText(
                    f"Custom filename override\nFolder: {folder_hint}\nGenerated fallback: {generated_basename}\nPreview: {basename}"
                )
            else:
                readable = " / ".join(
                    self._filename_key_to_label(key)
                    for key in self._selected_filename_order()
                ) or "No filename parts selected"
                folder_hint = self._recording_destination_preview()
                self.label_filename_formula.setText(f"{readable}\nFolder: {folder_hint}\nPreview: {basename}")
        self._update_folder_structure_preview()
        self._save_recording_form_state()
        self._refresh_recording_session_summary()

    def _planner_status_style(self, status: str):
        status = self._normalize_planner_status(status)
        palette = {
            "Pending": ("#3c2510", "#ffd89c"),
            "Acquiring": ("#102b43", "#9dd9ff"),
            "Acquired": ("#123324", "#7ef0ac"),
        }
        return palette.get(status, ("#111827", "#dbe7f3"))

    def _normalize_planner_status(self, status: object) -> str:
        status_text = str(status or "").strip()
        if status_text in {"Pending", "Acquiring", "Acquired"}:
            return status_text
        return "Pending"

    def _set_planner_row_status(
        self,
        row: int,
        status: str,
        *,
        manual_pending: bool = False,
        manual_acquired: bool = False,
    ):
        """Write and tint the planner status cell."""
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return
        status = self._normalize_planner_status(status)
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
        if status == "Pending":
            item.setData(PLANNER_RECORDING_BASE_ROLE, "")
            item.setData(PLANNER_MANUAL_PENDING_ROLE, "1" if manual_pending else "")
            item.setData(PLANNER_MANUAL_ACQUIRED_ROLE, "")
        elif status == "Acquired":
            item.setData(PLANNER_MANUAL_PENDING_ROLE, "")
            item.setData(PLANNER_MANUAL_ACQUIRED_ROLE, "1" if manual_acquired else "")
        else:
            item.setData(PLANNER_MANUAL_PENDING_ROLE, "")
            item.setData(PLANNER_MANUAL_ACQUIRED_ROLE, "")

    def _set_planner_row_recording_base_path(self, row: int, base_path: str) -> None:
        """Attach the recording base path to the status cell without adding a visible column."""
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return
        headers = self._planner_headers()
        if "Status" not in headers:
            return
        item = self.planner_table.item(row, headers.index("Status"))
        if item is None:
            self._set_planner_cell(row, "Status", "Pending")
            item = self.planner_table.item(row, headers.index("Status"))
        if item is not None:
            item.setData(PLANNER_RECORDING_BASE_ROLE, str(base_path or "").strip())

    def _planner_row_recording_base_path(self, row: int) -> str:
        """Return the recorded base path stored on a planner row, if any."""
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return ""
        headers = self._planner_headers()
        if "Status" not in headers:
            return ""
        item = self.planner_table.item(row, headers.index("Status"))
        if item is None:
            return ""
        return str(item.data(PLANNER_RECORDING_BASE_ROLE) or "").strip()

    def _set_planner_row_manual_pending(self, row: int, enabled: bool) -> None:
        """Remember that a pending planner row was deliberately reset by the user."""
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return
        headers = self._planner_headers()
        if "Status" not in headers:
            return
        item = self.planner_table.item(row, headers.index("Status"))
        if item is None:
            self._set_planner_cell(row, "Status", "Pending")
            item = self.planner_table.item(row, headers.index("Status"))
        if item is not None:
            item.setData(PLANNER_MANUAL_PENDING_ROLE, "1" if enabled else "")
            if enabled:
                item.setData(PLANNER_MANUAL_ACQUIRED_ROLE, "")

    def _planner_row_manual_pending(self, row: int) -> bool:
        """Return True when auto-discovery should not flip a pending row back to acquired."""
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return False
        headers = self._planner_headers()
        if "Status" not in headers:
            return False
        item = self.planner_table.item(row, headers.index("Status"))
        if item is None:
            return False
        status = self._normalize_planner_status(item.text())
        return status == "Pending" and str(item.data(PLANNER_MANUAL_PENDING_ROLE) or "").strip() == "1"

    def _set_planner_row_manual_acquired(self, row: int, enabled: bool) -> None:
        """Remember that an acquired planner row was deliberately set by the user."""
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return
        headers = self._planner_headers()
        if "Status" not in headers:
            return
        item = self.planner_table.item(row, headers.index("Status"))
        if item is None:
            self._set_planner_cell(row, "Status", "Acquired")
            item = self.planner_table.item(row, headers.index("Status"))
        if item is not None:
            item.setData(PLANNER_MANUAL_ACQUIRED_ROLE, "1" if enabled else "")
            if enabled:
                item.setData(PLANNER_MANUAL_PENDING_ROLE, "")

    def _planner_row_manual_acquired(self, row: int) -> bool:
        """Return True when auto-discovery should not flip an acquired row back to pending."""
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return False
        headers = self._planner_headers()
        if "Status" not in headers:
            return False
        item = self.planner_table.item(row, headers.index("Status"))
        if item is None:
            return False
        status = self._normalize_planner_status(item.text())
        return status == "Acquired" and str(item.data(PLANNER_MANUAL_ACQUIRED_ROLE) or "").strip() == "1"

    def _find_planner_row_for_current_session(self) -> Optional[int]:
        """Resolve the planner row associated with the current metadata selection."""
        if self.planner_table is None:
            return None
        selected = self.planner_table.selectionModel().selectedRows()
        if selected:
            return selected[0].row()

        current_trial = self.meta_trial.text().strip() if self.meta_trial else ""
        current_animal = self.meta_animal_id.text().strip() if self.meta_animal_id else ""
        current_session = self.meta_session.text().strip() if self.meta_session else ""
        for row in range(self.planner_table.rowCount()):
            payload = self._planner_row_payload(row)
            if current_trial and payload.get("Trial", "").strip() == current_trial:
                if (not current_animal or payload.get("Animal ID", "").strip() == current_animal) and (
                    not current_session or payload.get("Session", "").strip() == current_session
                ):
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

    def _next_pending_planner_row_after(self, current_row: int) -> Optional[int]:
        """Return the next pending row below current_row without wrapping to the top."""
        if self.planner_table is None:
            return None
        start_row = max(0, int(current_row) + 1)
        for row in range(start_row, self.planner_table.rowCount()):
            status = self._normalize_planner_status(
                self._planner_row_payload(row).get("Status", "Pending")
            )
            if status == "Pending":
                return row
        return None

    def _planner_row_for_recording_start(self) -> Optional[int]:
        """Choose the row that should drive the next recording."""
        if self.planner_table is None or self.planner_table.rowCount() == 0:
            return None

        self._sync_planner_recording_statuses()
        selected_rows = self.planner_table.selectionModel().selectedRows()
        selected_row = selected_rows[0].row() if selected_rows else None
        if selected_row is not None:
            selected_status = self._normalize_planner_status(
                self._planner_row_payload(selected_row).get("Status", "Pending")
            )
            if selected_status == "Pending":
                return selected_row
            if selected_status == "Acquired":
                next_row = self._next_pending_planner_row_after(selected_row)
                if next_row is not None:
                    self.planner_table.selectRow(next_row)
                    return next_row
                self._on_error_occurred("Selected trial is already acquired. Mark it Pending before re-recording.")
                return None

        if (
            self.active_planner_row is not None
            and 0 <= self.active_planner_row < self.planner_table.rowCount()
        ):
            active_status = self._normalize_planner_status(
                self._planner_row_payload(self.active_planner_row).get("Status", "Pending")
            )
            if active_status == "Pending":
                return self.active_planner_row

        return self._next_pending_planner_row_after(-1)

    def _advance_to_next_planner_trial(self):
        """Select the next pending trial after the current acquisition completes."""
        if self.planner_table is None or self.planner_table.rowCount() == 0:
            return

        current_row = self.active_planner_row
        if current_row is None:
            current_row = self._find_planner_row_for_current_session()
        if current_row is None:
            return

        row = self._next_pending_planner_row_after(current_row)
        if row is None:
            self._on_status_update("No pending planner trial remains below the completed row.")
            return
        self.planner_table.selectRow(row)
        self._load_planner_row_into_metadata(
            row,
            announce=True,
            clear_filename_override=True,
        )

    def _on_planner_item_changed(self, item: QTableWidgetItem):
        """React to planner table edits."""
        if item is None:
            return
        headers = self._planner_headers()
        if item.column() < 0 or item.column() >= len(headers):
            return
        header = headers[item.column()]
        if header == "Status":
            status = self._normalize_planner_status(item.text().strip() or "Pending")
            self._set_planner_row_status(
                item.row(),
                status,
                manual_pending=status == "Pending",
                manual_acquired=status == "Acquired",
            )
        elif header == PLANNER_DURATION_HEADER and self.planner_table is not None:
            normalized_duration = self._format_duration_input_hms(item.text())
            if item.text().strip() != normalized_duration:
                self.planner_table.blockSignals(True)
                item.setText(normalized_duration)
                self.planner_table.blockSignals(False)
        if (
            not self._syncing_recording_to_planner
            and self._active_planner_row_index() == item.row()
        ):
            self._load_planner_row_into_metadata(
                item.row(),
                announce=False,
                apply_duration=True,
                clear_filename_override=True,
            )
        self._update_planner_summary()

    def _planner_headers(self) -> List[str]:
        return self.planner_default_columns + self.planner_custom_columns

    def _normalize_planner_header_name(self, header: str) -> str:
        """Map legacy planner headers onto the current schema."""
        return PLANNER_DURATION_HEADER if header == LEGACY_PLANNER_DURATION_HEADER else header

    def _normalize_planner_seed(self, seed: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Normalize imported planner data into the current header/value format."""
        normalized: Dict[str, str] = {}
        for header, value in (seed or {}).items():
            normalized[self._normalize_planner_header_name(str(header))] = "" if value is None else str(value)
        if PLANNER_DURATION_HEADER in normalized:
            normalized[PLANNER_DURATION_HEADER] = self._format_duration_input_hms(normalized[PLANNER_DURATION_HEADER])
        return normalized

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

    def _planner_default_duration_text(self) -> str:
        """Return the current recording limit formatted for planner rows."""
        return self._format_duration_input_hms(self._planner_default_duration_seconds())

    def _planner_status_totals(self) -> Dict[str, int]:
        """Return planner row counts grouped by acquisition state."""
        totals = {"total": 0, "Pending": 0, "Acquiring": 0, "Acquired": 0}
        if self.planner_table is None:
            return totals
        totals["total"] = self.planner_table.rowCount()
        for row in range(self.planner_table.rowCount()):
            status = self._normalize_planner_status(self._planner_row_payload(row).get("Status", "Pending"))
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

    def _planner_status_tone(self, status: str) -> str:
        return {
            "Pending": "warning",
            "Acquiring": "accent",
            "Acquired": "success",
        }.get(status, "default")

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
            "Session": self.meta_session.text().strip() if self.meta_session is not None else "",
            "Experiment": self.meta_experiment.text().strip() if hasattr(self, "meta_experiment") and self.meta_experiment is not None else "",
            "Condition": self.meta_condition.text().strip() if self.meta_condition is not None else "",
            "Arena": self.meta_arena.text().strip() if self.meta_arena is not None else "",
        }

    def _update_active_trial_header(self):
        """Refresh the Active Session header chip with the current trial and filename."""
        if self.label_recording_session_header is None:
            return

        payload = self._current_session_payload()
        status = payload.get("Status", "Pending").strip() or "Pending"
        trial = payload.get("Trial", "").strip()
        filename = ""
        if self.current_recording_filepath:
            filename = Path(str(self.current_recording_filepath)).name
        elif hasattr(self, "edit_filename") and self.edit_filename is not None:
            filename = self.edit_filename.text().strip()

        trial_text = f"Trial {trial}" if trial else "No trial"
        filename = filename or "No file"
        display_filename = filename if len(filename) <= 42 else f"{filename[:39]}..."
        self._set_status_chip(
            self.label_recording_session_header,
            f"{trial_text} | {display_filename}",
            self._planner_status_tone(status),
        )
        self.label_recording_session_header.setToolTip(f"{status} | {trial_text} | {filename}")

    def _refresh_recording_session_summary(self):
        """Keep the live-view session chip aligned with the active planner row."""
        # The recording length controls and the top-bar chip are built together
        # with the centre workspace; bail out if an early planner/metadata signal
        # fires before they exist (restores the pre-redesign guard).
        if getattr(self, "check_unlimited", None) is None:
            return
        payload = self._current_session_payload()
        trial = payload.get("Trial", "").strip() or "No trial"
        animal = payload.get("Animal ID", "").strip() or "No subject"
        session = payload.get("Session", "").strip() or "No session"
        status = payload.get("Status", "Pending").strip() or "Pending"
        experiment = payload.get("Experiment", "").strip() or "No experiment"
        condition = payload.get("Condition", "").strip() or "No condition"
        arena = payload.get("Arena", "").strip() or "No arena"
        filename = self.edit_filename.text().strip() if hasattr(self, "edit_filename") and self.edit_filename is not None else ""
        max_length_seconds = self._get_max_record_seconds()
        max_length_text = "Unlimited" if max_length_seconds <= 0 else self._format_duration_hms(max_length_seconds)

        # Compact top-bar chip: trial / subject / session, with the full context
        # in the tooltip (this replaces the old, space-hungry "Active Session"
        # strip that used to live inside the Recording panel).
        if self.live_header_session is not None:
            has_trial = payload.get("Trial", "").strip() != ""
            chip_text = f"T{trial} · {animal} · {session}" if has_trial else "No session"
            self._set_status_chip(
                self.live_header_session, chip_text, self._planner_status_tone(status)
            )
            tooltip = (
                f"{status}  |  Trial {trial}  |  {animal}  |  Session {session}\n"
                f"{experiment}  |  {condition}  |  {arena}  |  Max {max_length_text}"
            )
            if filename:
                tooltip += f"\nNext file: {filename}"
            self.live_header_session.setToolTip(tooltip)

        # Legacy strip labels are gone, but stay defensive in case other code
        # paths still reference them.
        if self.label_recording_plan_summary is not None:
            self.label_recording_plan_summary.setText(f"{status}  |  Trial {trial}  |  {animal}  |  Session {session}")
        if self.label_recording_plan_details is not None:
            details = f"{experiment}  |  {condition}  |  {arena}  |  Max {max_length_text}"
            if filename:
                details += f"\nNext file: {filename}"
            self.label_recording_plan_details.setText(details)
        self._update_active_trial_header()

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
            elif header in ("Start Delay (s)", PLANNER_DURATION_HEADER):
                width = 120
            self.planner_table.setColumnWidth(index, width)
            header_item = self.planner_table.horizontalHeaderItem(index)
            if header_item is not None:
                header_item.setToolTip(header)
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

    def _apply_planner_payload_to_row(
        self,
        row: int,
        payload: Dict[str, str],
        preserve_trial: bool = False,
        preserve_status: bool = False,
    ) -> None:
        """Write a payload onto an existing planner row."""
        if self.planner_table is None:
            return
        if row < 0 or row >= self.planner_table.rowCount():
            return

        normalized = self._normalize_planner_seed(payload)
        existing = self._planner_row_payload(row)
        status_value = existing.get("Status", "Pending") if preserve_status else normalized.get("Status", "Pending")
        status_value = self._normalize_planner_status(status_value)
        trial_value = existing.get("Trial", "") if preserve_trial else normalized.get("Trial", "")

        self.planner_table.blockSignals(True)
        try:
            for header in self._planner_headers():
                if header == "Status":
                    value = status_value
                elif header == "Trial":
                    value = trial_value
                else:
                    value = normalized.get(header, "")
                self._set_planner_cell(row, header, value)
        finally:
            self.planner_table.blockSignals(False)
        self._set_planner_row_status(row, status_value or "Pending")
        if status_value == "Acquired" and normalized.get("_recording_base_path"):
            self._set_planner_row_recording_base_path(row, str(normalized.get("_recording_base_path", "")))

    def _insert_planner_trial(self, row: int, seed: Optional[Dict[str, str]] = None):
        """Insert one trial row into the planner table."""
        if self.planner_table is None:
            return
        seed = self._normalize_planner_seed(seed)
        row = max(0, min(int(row), self.planner_table.rowCount()))
        self.planner_table.insertRow(row)

        trial_value = str(seed.get("Trial", self.planner_next_trial_number))
        status_value = self._normalize_planner_status(seed.get("Status", "Pending"))
        defaults = {
            "Status": status_value,
            "Trial": trial_value,
            "Arena": str(seed.get("Arena", self.meta_arena.text().strip() if self.meta_arena else "Arena 1")),
            "Animal ID": str(seed.get("Animal ID", self.meta_animal_id.text().strip())),
            "Session": str(seed.get("Session", self.meta_session.text().strip() if self.meta_session else "")),
            "Experiment": str(seed.get("Experiment", self.meta_experiment.text().strip())),
            "Condition": str(seed.get("Condition", self.meta_condition.text().strip() if self.meta_condition else "")),
            "Start Delay (s)": str(seed.get("Start Delay (s)", "0")),
            PLANNER_DURATION_HEADER: str(seed.get(PLANNER_DURATION_HEADER, self._planner_default_duration_text())),
            "Comments": str(seed.get("Comments", "")),
        }
        self.planner_table.blockSignals(True)
        for header in self._planner_headers():
            value = status_value if header == "Status" else seed.get(header, defaults.get(header, ""))
            self._set_planner_cell(row, header, value)
        self.planner_table.blockSignals(False)
        self._set_planner_row_status(row, defaults["Status"])
        if seed.get("_recording_base_path"):
            self._set_planner_row_recording_base_path(row, str(seed.get("_recording_base_path", "")))

        try:
            self.planner_next_trial_number = max(self.planner_next_trial_number, int(trial_value) + 1)
        except Exception:
            self.planner_next_trial_number += 1

    def _append_planner_trial(self, seed: Optional[Dict[str, str]] = None):
        """Append one trial row to the planner table."""
        if self.planner_table is None:
            return
        self._insert_planner_trial(self.planner_table.rowCount(), seed)

    def _planner_row_payload(self, row: int) -> Dict[str, str]:
        """Return one planner row as a dict."""
        payload = {}
        if self.planner_table is None:
            return payload
        for column, header in enumerate(self._planner_headers()):
            item = self.planner_table.item(row, column)
            value = item.text().strip() if item else ""
            if header == PLANNER_DURATION_HEADER:
                value = self._format_duration_input_hms(value)
            payload[header] = value
        return payload

    def _add_one_planner_trial(self):
        """Append a single trial row and select it for immediate editing.

        This is the everyday one-click path; bulk creation lives behind the
        Plan menu so the common action stays instant and dialog-free.
        """
        self._append_planner_trial()
        self._renumber_planner_trials()
        self._fit_planner_columns()
        self._update_planner_summary()
        if self.planner_table is not None and self.planner_table.rowCount() > 0:
            self.planner_table.selectRow(self.planner_table.rowCount() - 1)

    def _add_planner_trials(self):
        """Append one or more trials to the planner."""
        from PySide6.QtWidgets import QInputDialog

        count, ok = QInputDialog.getInt(self, "Add Trials", "Number of trials:", 1, 1, 500, 1)
        if not ok:
            return
        for _ in range(int(count)):
            self._append_planner_trial()
        self._renumber_planner_trials()
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

    def _renumber_planner_trials(self) -> None:
        """Force the Trial column to be sequential 1..N in display order.

        Why: users want duplicates and manual reorders to always produce a clean
        1,2,3... sequence so the next row after four existing trials is "5",
        independent of whatever Trial label was on the copied row.
        """
        if self.planner_table is None:
            return
        headers = self._planner_headers()
        if "Trial" not in headers:
            return
        trial_col = headers.index("Trial")
        self.planner_table.blockSignals(True)
        try:
            for row in range(self.planner_table.rowCount()):
                expected = str(row + 1)
                item = self.planner_table.item(row, trial_col)
                if item is None:
                    self._set_planner_cell(row, "Trial", expected)
                elif item.text().strip() != expected:
                    item.setText(expected)
        finally:
            self.planner_table.blockSignals(False)
        self.planner_next_trial_number = self.planner_table.rowCount() + 1

    def _remove_selected_planner_trials(self):
        """Remove the selected planner rows."""
        if self.planner_table is None:
            return
        rows = sorted(self._selected_planner_rows(), reverse=True)
        if not rows:
            return
        if self.active_planner_row in rows:
            self.active_planner_row = None
        for row in rows:
            self.planner_table.removeRow(row)
        self._renumber_planner_trials()
        self._update_planner_summary()

    def _mark_selected_planner_trials_pending(self):
        """Reset acquired planner rows back to Pending from the context menu."""
        if self.planner_table is None:
            return
        rows = [
            row
            for row in self._selected_planner_rows()
            if (self._planner_row_payload(row).get("Status", "Pending").strip() or "Pending") == "Acquired"
        ]
        if not rows:
            return
        for row in rows:
            self._set_planner_row_status(row, "Pending", manual_pending=True)
        self._update_planner_summary()
        self._on_status_update(f"Marked {len(rows)} planner trial(s) as pending.")

    def _mark_selected_planner_trials_acquired(self):
        """Mark selected planner rows as Acquired from the context menu."""
        if self.planner_table is None:
            return
        rows = [
            row
            for row in self._selected_planner_rows()
            if self._normalize_planner_status(
                self._planner_row_payload(row).get("Status", "Pending")
            ) != "Acquired"
        ]
        if not rows:
            return
        for row in rows:
            self._set_planner_row_status(row, "Acquired", manual_acquired=True)
        self._update_planner_summary()
        self._on_status_update(f"Marked {len(rows)} planner trial(s) as acquired.")

    def _duplicate_selected_planner_trials(self):
        """Duplicate selected planner rows below the current selection."""
        if self.planner_table is None:
            return
        rows = self._selected_planner_rows()
        if not rows:
            self._on_error_occurred("Select one or more trial rows to duplicate.")
            return

        insert_row = rows[-1] + 1
        for offset, row in enumerate(rows):
            payload = self._planner_row_payload(row)
            payload["Status"] = "Pending"
            payload["Trial"] = ""  # renumber will assign the next sequential value
            self._insert_planner_trial(insert_row + offset, payload)

        self._renumber_planner_trials()
        self.planner_table.selectRow(insert_row)
        self._fit_planner_columns()
        self._update_planner_summary()
        self._on_status_update(f"Duplicated {len(rows)} planner trial(s).")

    def _planner_clipboard_payload(self) -> Optional[Dict[str, object]]:
        """Read planner rows from the clipboard when available."""
        text = QGuiApplication.clipboard().text().strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or payload.get("type") != "pykaboo/planner-rows":
            return None
        return payload

    def _copy_selected_planner_trials(self):
        """Copy selected planner rows to the clipboard as structured JSON."""
        if self.planner_table is None:
            return
        if self.planner_table.state() == QAbstractItemView.EditingState:
            return
        rows = self._selected_planner_rows()
        if not rows:
            self._on_error_occurred("Select one or more trial rows to copy.")
            return

        payload = {
            "type": "pykaboo/planner-rows",
            "custom_columns": list(self.planner_custom_columns),
            "rows": [self._planner_row_payload(row) for row in rows],
        }
        QGuiApplication.clipboard().setText(json.dumps(payload, ensure_ascii=True))
        self._on_status_update(f"Copied {len(rows)} planner trial(s) to the clipboard.")

    def _planner_current_paste_header(self) -> Optional[str]:
        """Return the planner header for the currently focused column."""
        if self.planner_table is None:
            return None
        headers = self._planner_headers()
        column = self.planner_table.currentColumn()
        if 0 <= column < len(headers):
            return headers[column]
        return None

    def _paste_single_planner_value(self, raw_text: str) -> bool:
        """Paste one plain-text value into the current planner column across selected rows."""
        if self.planner_table is None:
            return False

        text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text or "\n" in text or "\t" in text:
            return False

        target_rows = self._selected_planner_rows()
        if not target_rows:
            self._on_error_occurred("Select the target planner row before pasting.")
            return True

        header = self._planner_current_paste_header()
        if not header:
            self._on_error_occurred("Click the planner column you want to paste into, then paste again.")
            return True

        normalized_value = text
        if header == PLANNER_DURATION_HEADER:
            normalized_value = self._format_duration_input_hms(text)

        if header == "Status":
            for target_row in target_rows:
                status = self._normalize_planner_status(normalized_value or "Pending")
                self._set_planner_row_status(
                    target_row,
                    status,
                    manual_pending=status == "Pending",
                    manual_acquired=status == "Acquired",
                )
        else:
            self.planner_table.blockSignals(True)
            try:
                for target_row in target_rows:
                    self._set_planner_cell(target_row, header, normalized_value)
            finally:
                self.planner_table.blockSignals(False)

        if header == "Trial":
            try:
                self.planner_next_trial_number = max(self.planner_next_trial_number, int(normalized_value) + 1)
            except Exception:
                pass

        self.planner_table.selectRow(target_rows[0])
        self._fit_planner_columns()
        self._update_planner_summary()
        self._on_status_update(f"Pasted {header} onto {len(target_rows)} planner row(s).")
        return True

    def _paste_selected_planner_trials(self):
        """Paste copied planner content onto selected target rows."""
        if self.planner_table is None:
            return
        if self.planner_table.state() == QAbstractItemView.EditingState:
            return

        clipboard_text = QGuiApplication.clipboard().text()
        clipboard_payload = self._planner_clipboard_payload()
        if clipboard_payload is None:
            if self._paste_single_planner_value(clipboard_text):
                return
            self._on_error_occurred("Clipboard does not contain copied PyKaboo planner rows or a single-cell value.")
            return

        source_rows = [
            self._normalize_planner_seed(payload)
            for payload in clipboard_payload.get("rows", [])
            if isinstance(payload, dict)
        ]
        if not source_rows:
            self._on_error_occurred("Clipboard planner payload is empty.")
            return

        new_custom_columns = []
        for name in clipboard_payload.get("custom_columns", []):
            value = str(name).strip()
            if value and value not in self.planner_default_columns and value not in self.planner_custom_columns and value not in new_custom_columns:
                new_custom_columns.append(value)
        if new_custom_columns:
            self.planner_custom_columns.extend(new_custom_columns)
            self._refresh_planner_columns()

        target_rows = self._selected_planner_rows()
        if not target_rows:
            self._on_error_occurred("Select the target planner row before pasting.")
            return
        if len(source_rows) != 1 and len(source_rows) != len(target_rows):
            self._on_error_occurred("Paste expects one copied row or the same number of copied and selected rows.")
            return

        self.planner_table.blockSignals(True)
        try:
            for index, target_row in enumerate(target_rows):
                source_payload = dict(source_rows[0] if len(source_rows) == 1 else source_rows[index])
                source_payload["Status"] = "Pending"
                source_payload.pop("_recording_base_path", None)
                self._apply_planner_payload_to_row(
                    target_row,
                    source_payload,
                    preserve_trial=True,
                    preserve_status=False,
                )
        finally:
            self.planner_table.blockSignals(False)

        self.planner_table.selectRow(target_rows[0])
        self._fit_planner_columns()
        self._update_planner_summary()
        self._on_status_update(f"Pasted planner content onto {len(target_rows)} row(s).")

    def _move_selected_planner_trials(self, direction: int):
        """Move one contiguous planner selection up or down by one row."""
        if self.planner_table is None:
            return
        rows = self._selected_planner_rows()
        if not rows:
            self._on_error_occurred("Select one or more trial rows to move.")
            return
        if any((left + 1) != right for left, right in zip(rows, rows[1:])):
            self._on_error_occurred("Move Up and Move Down require a contiguous row selection.")
            return

        start_row = rows[0]
        end_row = rows[-1]
        if direction < 0 and start_row == 0:
            return
        if direction > 0 and end_row >= self.planner_table.rowCount() - 1:
            return

        payloads = []
        for row in range(self.planner_table.rowCount()):
            payload = self._planner_row_payload(row)
            recording_base_path = self._planner_row_recording_base_path(row)
            if recording_base_path:
                payload["_recording_base_path"] = recording_base_path
            payloads.append(payload)
        block = payloads[start_row:end_row + 1]
        if direction < 0:
            payloads = payloads[:start_row - 1] + block + [payloads[start_row - 1]] + payloads[end_row + 1:]
            next_row = start_row - 1
        else:
            payloads = payloads[:start_row] + [payloads[end_row + 1]] + block + payloads[end_row + 2:]
            next_row = start_row + 1

        self._planner_state_loading = True
        try:
            self.planner_table.clearSelection()
            self.planner_table.setRowCount(0)
            self.planner_next_trial_number = 1
            for payload in payloads:
                self._append_planner_trial(payload)
            self._renumber_planner_trials()
            self.active_planner_row = next_row
            self.planner_table.selectRow(next_row)
        finally:
            self._planner_state_loading = False

        self._fit_planner_columns()
        self._update_planner_summary()

    def _show_planner_context_menu(self, position):
        """Open the planner row context menu."""
        if self.planner_table is None:
            return
        self._sync_planner_recording_statuses()

        index = self.planner_table.indexAt(position)
        target_row = index.row() if index.isValid() else None
        if index.isValid() and index.row() not in self._selected_planner_rows():
            self.planner_table.selectRow(index.row())

        selected_rows = self._selected_planner_rows()
        acquired_rows = [
            row
            for row in selected_rows
            if self._normalize_planner_status(
                self._planner_row_payload(row).get("Status", "Pending")
            ) == "Acquired"
        ]
        not_acquired_rows = [
            row
            for row in selected_rows
            if self._normalize_planner_status(
                self._planner_row_payload(row).get("Status", "Pending")
            ) != "Acquired"
        ]

        menu = QMenu(self)
        action_duplicate = menu.addAction("Duplicate")
        action_copy = menu.addAction("Copy")
        action_paste = menu.addAction("Paste")
        action_move_up = menu.addAction("Move Up")
        action_move_down = menu.addAction("Move Down")
        action_mark_pending = menu.addAction("Mark as Pending")
        action_mark_pending.setEnabled(bool(acquired_rows))
        action_mark_acquired = menu.addAction("Mark as Acquired")
        action_mark_acquired.setEnabled(bool(not_acquired_rows))
        menu.addSeparator()
        action_use_selected = menu.addAction("Use Selected")
        action_open_output_folder = menu.addAction("Open Output Folder")
        action_remove = menu.addAction("Remove")

        if target_row is None and self._selected_planner_rows():
            target_row = self._selected_planner_rows()[0]
        action_open_output_folder.setEnabled(target_row is not None)

        chosen = menu.exec(self.planner_table.viewport().mapToGlobal(position))
        if chosen == action_duplicate:
            self._duplicate_selected_planner_trials()
        elif chosen == action_copy:
            self._copy_selected_planner_trials()
        elif chosen == action_paste:
            self._paste_selected_planner_trials()
        elif chosen == action_move_up:
            self._move_selected_planner_trials(-1)
        elif chosen == action_move_down:
            self._move_selected_planner_trials(1)
        elif chosen == action_mark_pending:
            self._mark_selected_planner_trials_pending()
        elif chosen == action_mark_acquired:
            self._mark_selected_planner_trials_acquired()
        elif chosen == action_use_selected:
            self._apply_selected_planner_trial()
        elif chosen == action_open_output_folder:
            self._open_selected_planner_output_folder(target_row)
        elif chosen == action_remove:
            self._remove_selected_planner_trials()

    def _planner_last_csv_path(self) -> str:
        return str(self.settings.value("planner_last_csv_path", "") or "").strip()

    def _remember_planner_csv_path(self, filepath: str):
        resolved = str(Path(filepath))
        self.settings.setValue("planner_last_csv_path", resolved)
        self._update_planner_load_last_button_state()

    def _update_planner_load_last_button_state(self):
        action = getattr(self, "action_planner_load_last", None)
        if action is None:
            return

        raw_path = self._planner_last_csv_path()
        if not raw_path:
            action.setEnabled(False)
            action.setToolTip("No planner CSV has been imported or exported yet.")
            return

        path = Path(raw_path)
        action.setEnabled(path.exists() and path.is_file())
        action.setToolTip(str(path))

    def _load_planner_trials_from_csv(self, filepath: str) -> bool:
        """Load planner rows from a CSV file path."""
        if self.planner_table is None:
            return False

        import csv

        path = Path(filepath)
        if not path.exists() or not path.is_file():
            self._on_error_occurred(f"Planner CSV not found: {path}")
            self._update_planner_load_last_button_state()
            return False

        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        if not fieldnames:
            self._on_error_occurred("Selected CSV has no header row.")
            return False

        normalized_fieldnames = [self._normalize_planner_header_name(field) for field in fieldnames]
        extras = []
        for field in normalized_fieldnames:
            if field not in self.planner_default_columns and field not in extras:
                extras.append(field)
        self.planner_custom_columns = extras
        self._refresh_planner_columns()
        self.planner_table.setRowCount(0)
        self.planner_next_trial_number = 1
        for row in rows:
            normalized_row = {}
            for key, value in row.items():
                normalized_row[self._normalize_planner_header_name(key)] = value
            self._append_planner_trial({key: normalized_row.get(key, "") for key in self._planner_headers()})
        self._renumber_planner_trials()
        if self.planner_table.rowCount() > 0:
            self.planner_table.selectRow(0)
        self._fit_planner_columns()
        self._remember_planner_csv_path(str(path))
        self._on_status_update(f"Imported planner CSV: {path.name}")
        self._update_planner_summary()
        return True

    def _import_planner_trials(self):
        """Load planner rows from a CSV file."""
        if self.planner_table is None:
            return

        filepath, _ = QFileDialog.getOpenFileName(self, "Import Trial Plan", self.last_save_folder, "CSV Files (*.csv)")
        if not filepath:
            return
        self._load_planner_trials_from_csv(filepath)

    def _load_last_planner_trials(self):
        """Reload the most recent planner CSV used by this app."""
        last_path = self._planner_last_csv_path()
        if not last_path:
            self._on_error_occurred("No previous planner CSV is saved yet.")
            self._update_planner_load_last_button_state()
            return
        self._load_planner_trials_from_csv(last_path)

    def _export_planner_trials(self):
        """Export the planner table to CSV."""
        if self.planner_table is None:
            return
        import csv

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Trial Plan",
            str(Path(self.last_save_folder) / "pykaboo_trial_plan.csv"),
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
        self._remember_planner_csv_path(filepath)
        self._on_status_update(f"Planner exported: {Path(filepath).name}")

    def _ensure_custom_metadata_field(self, field_name: str) -> QLineEdit:
        """Return an existing custom metadata field or create one in the form."""
        if field_name in self.custom_metadata_fields:
            return self.custom_metadata_fields[field_name]
        field_edit = QLineEdit()
        field_edit.setPlaceholderText(f"Enter {field_name}...")
        field_edit.textChanged.connect(self._update_filename_preview)
        field_edit.textChanged.connect(self._save_recording_form_state)
        field_edit.textChanged.connect(self._on_recording_metadata_controls_changed)
        self.metadata_layout.addRow(f"{field_name}:", field_edit)
        self.custom_metadata_fields[field_name] = field_edit
        return field_edit

    def _fit_planner_columns(self):
        """Resize planner columns for readability."""
        if self.planner_table is None:
            return
        headers = self._planner_headers()
        if not headers:
            return

        header_view = self.planner_table.horizontalHeader()
        viewport_width = max(0, self.planner_table.viewport().width())
        font_metrics = self.planner_table.fontMetrics()
        widths = []
        minimums = []

        for index, header in enumerate(headers):
            minimum_width, target_width = self._planner_column_width_bounds(header)
            content_width = max(
                self.planner_table.sizeHintForColumn(index) + 18,
                font_metrics.horizontalAdvance(header) + 28,
            )
            width = max(minimum_width, min(target_width, content_width))
            widths.append(width)
            minimums.append(minimum_width)

        total_width = sum(widths)
        minimum_total_width = sum(minimums)
        if viewport_width > 0 and minimum_total_width > viewport_width:
            header_view.setSectionResizeMode(QHeaderView.Stretch)
            header_view.setStretchLastSection(False)
            return

        header_view.setSectionResizeMode(QHeaderView.Interactive)
        if viewport_width > 0 and total_width > viewport_width:
            deficit = total_width - viewport_width
            for header_name in ("Comments", "Experiment", "Condition", "Session", "Animal ID", "Arena", "Trial", "Status"):
                if deficit <= 0 or header_name not in headers:
                    continue
                index = headers.index(header_name)
                reducible = widths[index] - minimums[index]
                if reducible <= 0:
                    continue
                reduction = min(reducible, deficit)
                widths[index] -= reduction
                deficit -= reduction
            for index, current_width in enumerate(widths):
                if deficit <= 0:
                    break
                reducible = current_width - minimums[index]
                if reducible <= 0:
                    continue
                reduction = min(reducible, deficit)
                widths[index] -= reduction
                deficit -= reduction
        elif viewport_width > total_width:
            stretch_headers = [
                header_name
                for header_name in ("Session", "Experiment", "Condition", "Comments")
                if header_name in headers
            ]
            stretch_headers.extend(
                header_name
                for header_name in headers
                if header_name not in stretch_headers and header_name not in self.planner_default_columns
            )
            stretch_indexes = [headers.index(header_name) for header_name in stretch_headers]
            if stretch_indexes:
                extra = viewport_width - total_width
                share, remainder = divmod(extra, len(stretch_indexes))
                for offset, index in enumerate(stretch_indexes):
                    widths[index] += share + (1 if offset < remainder else 0)

        for index, width in enumerate(widths):
            self.planner_table.setColumnWidth(index, max(minimums[index], width))
        header_view.setStretchLastSection(False)

    def _planner_column_width_bounds(self, header: str):
        """Return the minimum and preferred widths for one planner column."""
        if header == "Status":
            return 72, 92
        if header == "Trial":
            return 52, 70
        if header == "Arena":
            return 68, 92
        if header == "Animal ID":
            return 82, 118
        if header in ("Session", "Experiment", "Condition"):
            return 96, 140
        if header in ("Start Delay (s)", PLANNER_DURATION_HEADER):
            return 96, 122
        if header == "Comments":
            return 110, 180
        return 96, 136

    def _schedule_planner_column_fit(self):
        """Queue a planner column fit once the dock/table has its final size."""
        if self.planner_table is None or self._planner_fit_pending:
            return
        self._planner_fit_pending = True
        QTimer.singleShot(0, self._run_planner_column_fit)

    def _run_planner_column_fit(self):
        """Execute a pending planner fit request."""
        self._planner_fit_pending = False
        self._fit_planner_columns()

    def _toggle_planner_detach(self):
        """Detach the planner into a larger floating window or reattach it."""
        if self.planner_panel_widget is None or self.planner_host_layout is None:
            return
        if self.planner_detached:
            self._reattach_planner_panel()
            return

        self.planner_dialog = QDialog(self)
        self.planner_dialog.setWindowTitle(f"{APP_NAME} Planner")
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
        self._update_side_panel_bounds()
        self._schedule_planner_column_fit()

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
        self._set_button_icon(self.btn_planner_detach, "detach", "#ffb35d", "ghostButton")
        self.planner_reattaching = False
        self._update_side_panel_bounds()
        self._schedule_planner_column_fit()

    def _load_planner_row_into_metadata(
        self,
        row: int,
        announce: bool = False,
        apply_duration: bool = True,
        clear_filename_override: bool = False,
    ):
        """Copy one planner row into the hidden metadata/session fields."""
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return

        payload = self._planner_row_payload(row)
        animal_id = payload.get("Animal ID", "")
        session = payload.get("Session", "")
        experiment = payload.get("Experiment", "")
        comments = payload.get("Comments", "")
        condition = payload.get("Condition", "")

        self.active_planner_row = row
        self._syncing_planner_to_recording = True
        try:
            if clear_filename_override:
                self._clear_custom_filename_override()
            self.meta_animal_id.setText(animal_id)
            self.meta_session.setText(session)
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
                    duration_seconds = self._parse_duration_seconds(self._planner_duration_value(payload))
                    if duration_seconds > 0:
                        self.check_unlimited.setCurrentText("Limited")
                        self._set_recording_length_seconds(duration_seconds)
                    else:
                        self.check_unlimited.setCurrentText("Unlimited")
                except Exception:
                    pass
        finally:
            self._syncing_planner_to_recording = False
        self._update_filename_preview()

        if self.label_session_summary is not None:
            self.label_session_summary.setText(
                f"Trial {payload.get('Trial', '?')}  |  {payload.get('Animal ID', '').strip() or 'No subject'}  |  "
                f"Session {payload.get('Session', '').strip() or '-'}"
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
            self._load_planner_row_into_metadata(
                selected_rows[0].row(),
                announce=False,
                clear_filename_override=True,
            )
        self._update_planner_summary()

    def _apply_selected_planner_trial(self):
        """Load the selected planner row into the active metadata/session form."""
        if self.planner_table is None:
            return
        selected_rows = self.planner_table.selectionModel().selectedRows()
        if not selected_rows:
            self._on_error_occurred("Select a trial row to load into the session form.")
            return
        self._load_planner_row_into_metadata(
            selected_rows[0].row(),
            announce=True,
            clear_filename_override=True,
        )

    def _update_planner_summary(self):
        """Refresh the footer summary for the planner dock."""
        if self.planner_table is None:
            return
        self._sync_planner_recording_statuses()
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
            self._save_planner_state_snapshot()
            return
        payload = self._planner_row_payload(selected_rows[0].row())
        self.label_planner_summary.setText(
            f"{payload.get('Status', 'Pending')}  |  Trial {payload.get('Trial', '?')}  |  "
            f"{payload.get('Animal ID', 'No subject')}  |  Session {payload.get('Session', '-') or '-'}  |  "
            f"{payload.get('Experiment', 'No experiment')}"
        )
        self._refresh_recording_session_summary()
        self._save_planner_state_snapshot()

    def _load_ui_settings(self):
        """Load saved UI settings."""
        self.spin_fps.blockSignals(True)
        self.spin_fps.setValue(float(self.settings.value('camera_fps', 60.0)))
        self.spin_fps.blockSignals(False)

        self.spin_exposure.blockSignals(True)
        self.spin_exposure.setValue(float(self.settings.value('exposure_ms', 10.0)))
        self.spin_exposure.blockSignals(False)

        self.spin_width.blockSignals(True)
        self.spin_width.setValue(int(self.settings.value('camera_width', 1920)))
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
        self.spin_preview_fps.setValue(float(self.settings.value('preview_fps', 30.0)))
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
        self.roi_rect = self._load_saved_camera_roi()
        self._sync_camera_roi_ui_state()

        self._load_line_label_settings()
        self._load_behavior_panel_settings()
        self._load_recording_form_state()
        self._restore_last_planner_state()
        self._planner_autosave_enabled = True
        self._save_planner_state_snapshot()
        self._set_filename_order_controls()
        self._load_user_flag_settings()
        if self.check_organize_session_folders is not None:
            self.check_organize_session_folders.blockSignals(True)
            self.check_organize_session_folders.setChecked(
                int(self.settings.value("organize_recordings_by_session", 0)) == 1
            )
            self.check_organize_session_folders.blockSignals(False)
        self._set_folder_order_controls()
        self._set_folder_structure_controls_enabled(self._organize_recordings_enabled())
        self._update_planner_summary()

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

    def _normalize_user_flag_config(self, raw_config: Dict[str, object] | None, fallback_index: int = 1) -> Dict[str, object]:
        """Normalize one user flag entry to the internal shape."""
        data = dict(raw_config or {})
        default_label = "User Flag" if int(fallback_index) <= 1 else f"User Flag {fallback_index}"
        label = str(data.get("label", "") or "").strip() or default_label
        shortcut = str(data.get("shortcut", "") or "").strip()
        output_id = str(data.get("output_id", "") or "").strip().upper()
        if output_id == "NONE":
            output_id = ""
        try:
            pulse_ms = max(5, int(data.get("pulse_ms", 100) or 100))
        except (TypeError, ValueError):
            pulse_ms = 100
        flag_id = str(data.get("flag_id", "") or "").strip() or f"flag-{uuid.uuid4().hex[:8]}"
        return {
            "flag_id": flag_id,
            "label": label,
            "shortcut": shortcut,
            "output_id": output_id,
            "pulse_ms": pulse_ms,
        }

    def _current_user_flag_configs(self) -> List[Dict[str, object]]:
        """Return the currently configured user flags."""
        return [dict(config) for config in self.user_flag_configs]

    def _current_user_flag_config(self) -> Dict[str, object]:
        """Return the first configured user flag for legacy callers."""
        configs = self._current_user_flag_configs()
        if configs:
            return configs[0]
        return self._normalize_user_flag_config({}, fallback_index=1)

    def _load_user_flag_settings(self):
        """Restore user-flag settings from QSettings, including legacy single-flag entries."""
        raw_json = str(self.settings.value("user_flags_json", "") or "").strip()
        configs: List[Dict[str, object]] = []
        if raw_json:
            try:
                parsed = json.loads(raw_json)
            except Exception:
                parsed = []
            if isinstance(parsed, list):
                configs = [
                    self._normalize_user_flag_config(item, fallback_index=index + 1)
                    for index, item in enumerate(parsed)
                    if isinstance(item, dict)
                ]
        if not configs:
            legacy = self._normalize_user_flag_config(
                {
                    "label": str(self.settings.value("user_flag_label", "User Flag") or "User Flag"),
                    "shortcut": str(self.settings.value("user_flag_shortcut", "") or ""),
                    "output_id": str(self.settings.value("user_flag_output", "None") or "None"),
                    "pulse_ms": int(self.settings.value("user_flag_pulse_ms", 100) or 100),
                },
                fallback_index=1,
            )
            if str(legacy.get("shortcut", "") or "").strip():
                configs = [legacy]
        self.user_flag_configs = configs
        self._refresh_user_flag_summary()
        self._refresh_user_flag_shortcuts()
        self._update_user_flag_pin_summary()

    def _persist_user_flag_settings(self, sync: bool = False):
        """Persist all configured user flags."""
        configs = [self._normalize_user_flag_config(config, fallback_index=index + 1) for index, config in enumerate(self.user_flag_configs)]
        self.user_flag_configs = configs
        self.settings.setValue("user_flags_json", json.dumps(configs))
        legacy = configs[0] if configs else self._normalize_user_flag_config({}, fallback_index=1)
        self.settings.setValue("user_flag_label", str(legacy["label"]))
        self.settings.setValue("user_flag_shortcut", str(legacy["shortcut"]))
        self.settings.setValue("user_flag_output", str(legacy["output_id"] or "None"))
        self.settings.setValue("user_flag_pulse_ms", int(legacy["pulse_ms"]))
        if sync:
            self.settings.sync()

    @Slot()
    def _on_user_flag_settings_changed(self):
        """Save user-flag settings and refresh the bound shortcuts."""
        self._persist_user_flag_settings(sync=False)
        self._refresh_user_flag_summary()
        self._refresh_user_flag_shortcuts()
        self._update_user_flag_pin_summary()
        self._populate_user_flag_table()

    def _refresh_user_flag_shortcuts(self):
        """Bind one window shortcut per configured user flag."""
        for shortcut in self.user_flag_shortcut_bindings:
            try:
                shortcut.activated.disconnect()
            except Exception:
                pass
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self.user_flag_shortcut_bindings = []

        occupied_shortcuts = {
            str(config.get("shortcut", "") or "").strip()
            for config in self.user_flag_configs
            if str(config.get("shortcut", "") or "").strip()
        }
        for config in self.user_flag_configs:
            shortcut_text = str(config.get("shortcut", "") or "").strip()
            if not shortcut_text:
                continue
            sequence = QKeySequence.fromString(shortcut_text, QKeySequence.PortableText)
            if sequence.isEmpty():
                continue
            binding = QShortcut(sequence, self)
            binding.setContext(Qt.WindowShortcut)
            flag_id = str(config.get("flag_id", "") or "").strip()
            binding.activated.connect(lambda flag_id=flag_id: self._trigger_user_flag(flag_id))
            binding.setEnabled(True)
            self.user_flag_shortcut_bindings.append(binding)

        record_space_text = QKeySequence(Qt.Key_Space).toString(QKeySequence.PortableText)
        self.space_record_shortcut.setEnabled(record_space_text not in occupied_shortcuts)

    def _refresh_user_flag_summary(self):
        """Refresh the compact summary shown on the settings page."""
        if self.label_user_flag_summary is not None:
            self.label_user_flag_summary.setText(f"Flags: {len(self.user_flag_configs)}")
        if self.label_user_flag_details is None:
            return
        if not self.user_flag_configs:
            self.label_user_flag_details.setText("No user flags configured")
            return
        lines = []
        for config in self.user_flag_configs[:4]:
            output_id = str(config.get("output_id", "") or "").strip().upper() or "No TTL"
            lines.append(
                f"{config.get('label', 'User Flag')} [{config.get('shortcut', '')}] -> {output_id} ({int(config.get('pulse_ms', 100) or 100)} ms)"
            )
        if len(self.user_flag_configs) > 4:
            lines.append(f"+ {len(self.user_flag_configs) - 4} more")
        self.label_user_flag_details.setText("\n".join(lines))

    def _update_user_flag_pin_summary(self):
        """Show which pins are involved across all configured user-flag outputs."""
        if self.label_user_flag_pin_summary is None:
            return
        used_outputs = []
        for config in self.user_flag_configs:
            output_id = str(config.get("output_id", "") or "").strip().upper()
            if output_id and output_id not in used_outputs:
                used_outputs.append(output_id)
        if not used_outputs:
            self.label_user_flag_pin_summary.setText("Pins: no TTL outputs selected")
            return
        fragments = []
        for output_id in used_outputs:
            pins = [int(pin) for pin in self.live_output_mapping.get(output_id, [])]
            if pins:
                fragments.append(f"{output_id}: {', '.join(str(pin) for pin in pins)}")
            else:
                fragments.append(f"{output_id}: not mapped")
        self.label_user_flag_pin_summary.setText("Pins: " + " | ".join(fragments))
        self._populate_user_flag_table()

    def _populate_user_flag_table(self):
        """Refresh the flag manager table with the latest configured entries."""
        if self.user_flag_table is None:
            return
        self.user_flag_table.setRowCount(len(self.user_flag_configs))
        for row, config in enumerate(self.user_flag_configs):
            output_id = str(config.get("output_id", "") or "").strip().upper()
            pins = [int(pin) for pin in self.live_output_mapping.get(output_id, [])] if output_id else []
            values = [
                str(config.get("label", "User Flag") or "User Flag"),
                str(config.get("shortcut", "") or ""),
                output_id or "None",
                f"{int(config.get('pulse_ms', 100) or 100)} ms",
                ", ".join(str(pin) for pin in pins) if pins else ("not mapped" if output_id else ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, str(config.get("flag_id", "") or ""))
                self.user_flag_table.setItem(row, column, item)
        self.user_flag_table.setVisible(bool(self.user_flag_configs))
        if self.btn_edit_user_flag is not None:
            self.btn_edit_user_flag.setEnabled(bool(self.user_flag_configs))
        if self.btn_remove_user_flag is not None:
            self.btn_remove_user_flag.setEnabled(bool(self.user_flag_configs))

    def _selected_user_flag_id(self) -> str:
        """Return the selected flag id from the manager table."""
        if self.user_flag_table is None:
            return ""
        row = int(self.user_flag_table.currentRow())
        if row < 0:
            return ""
        item = self.user_flag_table.item(row, 0)
        if item is None:
            return ""
        return str(item.data(Qt.UserRole) or "").strip()

    def _find_user_flag_config(self, flag_id: str) -> Dict[str, object] | None:
        """Look up one user flag config by id."""
        target = str(flag_id or "").strip()
        if not target:
            return None
        for config in self.user_flag_configs:
            if str(config.get("flag_id", "") or "").strip() == target:
                return dict(config)
        return None

    def _show_user_flag_dialog(self):
        """Open the manager dialog for creating and editing user flags."""
        if self.user_flag_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("User Flag Manager")
            dialog.resize(760, 340)
            layout = QVBoxLayout(dialog)
            hint = QLabel("Configure multiple manual markers. Each flag gets its own keyboard shortcut and optional TTL pulse.")
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #8fa6bf;")
            layout.addWidget(hint)

            table = QTableWidget(0, 5)
            table.setHorizontalHeaderLabels(["Label", "Shortcut", "Output", "Pulse", "Pins"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
            table.verticalHeader().setVisible(False)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSelectionMode(QAbstractItemView.SingleSelection)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.cellDoubleClicked.connect(lambda *_: self._edit_selected_user_flag())
            layout.addWidget(table, 1)
            self.user_flag_table = table

            button_row = QHBoxLayout()
            self.btn_add_user_flag = QPushButton("Add Flag")
            self.btn_add_user_flag.clicked.connect(self._add_user_flag)
            button_row.addWidget(self.btn_add_user_flag)
            self.btn_edit_user_flag = QPushButton("Edit Flag")
            self.btn_edit_user_flag.clicked.connect(self._edit_selected_user_flag)
            button_row.addWidget(self.btn_edit_user_flag)
            self.btn_remove_user_flag = QPushButton("Remove Flag")
            self.btn_remove_user_flag.clicked.connect(self._remove_selected_user_flag)
            button_row.addWidget(self.btn_remove_user_flag)
            button_row.addStretch()
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            button_row.addWidget(close_button)
            layout.addLayout(button_row)
            self.user_flag_dialog = dialog
        self._populate_user_flag_table()
        self.user_flag_dialog.show()
        self.user_flag_dialog.raise_()
        self.user_flag_dialog.activateWindow()

    def _edit_user_flag_dialog(self, initial: Dict[str, object] | None = None) -> Dict[str, object] | None:
        """Prompt for one user flag entry and return the normalized config on success."""
        dialog = QDialog(self)
        dialog.setWindowTitle("User Flag")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        layout.addLayout(form)

        current = self._normalize_user_flag_config(initial or {}, fallback_index=len(self.user_flag_configs) + 1)
        edit_label = QLineEdit(str(current.get("label", "") or ""))
        form.addRow("Label:", edit_label)

        edit_shortcut = QKeySequenceEdit()
        edit_shortcut.setKeySequence(QKeySequence.fromString(str(current.get("shortcut", "") or ""), QKeySequence.PortableText))
        form.addRow("Shortcut:", edit_shortcut)

        combo_output = QComboBox()
        combo_output.addItem("None")
        combo_output.addItems([f"DO{i}" for i in range(1, 9)])
        combo_output.setCurrentText(str(current.get("output_id", "") or "").strip().upper() or "None")
        form.addRow("TTL Output:", combo_output)

        spin_pulse = QSpinBox()
        spin_pulse.setRange(5, 5000)
        spin_pulse.setSingleStep(5)
        spin_pulse.setSuffix(" ms")
        spin_pulse.setValue(int(current.get("pulse_ms", 100) or 100))
        form.addRow("Pulse Width:", spin_pulse)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None

        config = self._normalize_user_flag_config(
            {
                "flag_id": current.get("flag_id", ""),
                "label": edit_label.text().strip(),
                "shortcut": edit_shortcut.keySequence().toString(QKeySequence.PortableText).strip(),
                "output_id": combo_output.currentText(),
                "pulse_ms": int(spin_pulse.value()),
            },
            fallback_index=len(self.user_flag_configs) + 1,
        )
        shortcut_text = str(config.get("shortcut", "") or "").strip()
        if not shortcut_text:
            QMessageBox.warning(self, "User Flag", "Each user flag needs a shortcut.")
            return None
        occupied = {
            str(entry.get("shortcut", "") or "").strip().lower(): str(entry.get("flag_id", "") or "").strip()
            for entry in self.user_flag_configs
        }
        existing_flag_id = occupied.get(shortcut_text.lower(), "")
        if existing_flag_id and existing_flag_id != str(config.get("flag_id", "") or "").strip():
            QMessageBox.warning(self, "User Flag", f"The shortcut '{shortcut_text}' is already assigned to another flag.")
            return None
        return config

    def _add_user_flag(self):
        """Create a new user flag entry."""
        config = self._edit_user_flag_dialog()
        if config is None:
            return
        self.user_flag_configs.append(config)
        self._on_user_flag_settings_changed()

    def _edit_selected_user_flag(self):
        """Edit the selected flag entry."""
        flag_id = self._selected_user_flag_id()
        if not flag_id:
            return
        original = self._find_user_flag_config(flag_id)
        if original is None:
            return
        updated = self._edit_user_flag_dialog(original)
        if updated is None:
            return
        for index, config in enumerate(self.user_flag_configs):
            if str(config.get("flag_id", "") or "").strip() == flag_id:
                self.user_flag_configs[index] = updated
                break
        self._on_user_flag_settings_changed()

    def _remove_selected_user_flag(self):
        """Remove the selected flag entry."""
        flag_id = self._selected_user_flag_id()
        if not flag_id:
            return
        self.user_flag_configs = [
            dict(config)
            for config in self.user_flag_configs
            if str(config.get("flag_id", "") or "").strip() != flag_id
        ]
        self._on_user_flag_settings_changed()

    def _persist_settings_snapshot(self, sync: bool = True):
        """Write the current general/session settings back to QSettings."""
        self.last_save_folder = self.edit_save_folder.text().strip() or self.last_save_folder
        self._save_ui_setting('camera_fps', float(self.spin_fps.value()))
        self._save_ui_setting('exposure_ms', float(self.spin_exposure.value()))
        self._save_ui_setting('camera_width', int(self.spin_width.value()))
        self._save_ui_setting('camera_height', int(self.spin_height.value()))
        self._save_ui_setting('encoder_index', int(self.combo_encoder.currentIndex()))
        self._save_ui_setting('image_format', self.combo_image_format.currentText())
        self._save_ui_setting('preview_enabled', 1 if self.check_preview_enabled.isChecked() else 0)
        self._save_ui_setting('preview_fps', float(self.spin_preview_fps.value()))
        self._save_ui_setting('preview_width', int(self.spin_preview_width.value()))
        self._save_ui_setting('frame_buffer_size', int(self.spin_frame_buffer.value()))
        self._save_ui_setting('metadata_stats_interval', int(self.spin_metadata_stats_interval.value()))
        self._save_ui_setting('max_hours', int(self.spin_hours.value()))
        self._save_ui_setting('max_minutes', int(self.spin_minutes.value()))
        self._save_ui_setting('max_seconds', int(self.spin_seconds.value()))
        self._save_ui_setting('max_unlimited', 1 if self.check_unlimited.currentText() == "Unlimited" else 0)
        self._save_ui_setting('last_save_folder', self.last_save_folder)
        self.settings.setValue(
            "camera_roi_json",
            json.dumps(self.roi_rect) if isinstance(self.roi_rect, dict) else "",
        )

        for index, combo in enumerate(self.filename_order_boxes, start=1):
            self.settings.setValue(f"filename_part_{index}", self._filename_label_to_key(combo.currentText()))

        for index, combo in enumerate(self.folder_order_boxes, start=1):
            self.settings.setValue(f"folder_part_{index}", self._filename_label_to_key(combo.currentText()))

        if self.check_organize_session_folders is not None:
            self.settings.setValue(
                "organize_recordings_by_session",
                1 if self.check_organize_session_folders.isChecked() else 0,
            )

        self._persist_user_flag_settings(sync=False)
        self._save_recording_form_state()
        if sync:
            self.settings.sync()

    def _save_current_settings_snapshot(self):
        """Explicitly save the current UI settings immediately."""
        self._persist_settings_snapshot(sync=True)
        self._on_status_update("Current settings saved")

    def _set_current_settings_as_default(self):
        """Save the current setup as the next-launch default."""
        self._persist_settings_snapshot(sync=True)
        self.default_fps = float(self.spin_fps.value())
        self.default_width = int(self.spin_width.value())
        self.default_height = int(self.spin_height.value())
        self.default_image_format = self.combo_image_format.currentText()
        self._save_metadata_template()
        self._on_status_update("Current settings saved as defaults")

    # ── Preset save / load ────────────────────────────────────────────

    def _collect_preset(self) -> Dict[str, object]:
        """Gather all current UI settings into a serialisable dict."""
        # Camera / recording
        camera = {
            "fps": float(self.spin_fps.value()),
            "exposure_ms": float(self.spin_exposure.value()),
            "width": int(self.spin_width.value()),
            "height": int(self.spin_height.value()),
            "image_format": self.combo_image_format.currentText(),
            "encoder_index": int(self.combo_encoder.currentIndex()),
            "preview_enabled": self.check_preview_enabled.isChecked(),
            "preview_fps": float(self.spin_preview_fps.value()),
            "preview_width": int(self.spin_preview_width.value()),
            "frame_buffer_size": int(self.spin_frame_buffer.value()),
            "metadata_stats_interval": int(self.spin_metadata_stats_interval.value()),
            "roi_rect": self.roi_rect if isinstance(self.roi_rect, dict) else None,
            "max_hours": int(self.spin_hours.value()),
            "max_minutes": int(self.spin_minutes.value()),
            "max_seconds": int(self.spin_seconds.value()),
            "max_unlimited": self.check_unlimited.currentText() == "Unlimited",
            "save_folder": (self.edit_save_folder.text().strip()
                            if hasattr(self, "edit_save_folder") and self.edit_save_folder else
                            self.last_save_folder),
            "filename_parts": self._selected_filename_order(),
            "folder_parts": self._selected_folder_order(),
            "organize_by_session": (
                self.check_organize_session_folders.isChecked()
                if self.check_organize_session_folders is not None else False
            ),
        }

        # Behavior pin / role / label / enabled
        behavior_pins: Dict[str, Dict] = {}
        for key in self.BEHAVIOR_PIN_KEYS:
            pin_edit = self.behavior_pin_edits.get(key)
            role_box = self.behavior_role_boxes.get(key)
            label_edit = self.signal_label_edits.get(key)
            enabled_check = self.signal_enabled_checks.get(key)
            behavior_pins[key] = {
                "pins": pin_edit.text() if pin_edit else "",
                "role": role_box.currentText() if role_box else "Output",
                "label": label_edit.text() if label_edit else key,
                "enabled": enabled_check.isChecked() if enabled_check else True,
            }

        sync_params = {
            "period_s": float(self.settings.value("sync_period_s", 1.0)),
            "pulse_s": float(self.settings.value("sync_pulse_s", 0.05)),
        }

        # Live detection (model config + ROIs + rules + output mapping)
        live: Dict[str, object] = {}
        if self.live_detection_panel is not None:
            live = dict(self.live_detection_panel.detection_config())
        live["rois"] = [roi.to_dict() for roi in self.live_rois.values()]
        live["rules"] = [rule.to_dict() for rule in self.live_rules]
        live["output_mapping"] = dict(self.live_output_mapping)

        return {
            "version": 1,
            "camera": camera,
            "behavior_pins": behavior_pins,
            "sync": sync_params,
            "live_detection": live,
            "user_flags": self._current_user_flag_configs(),
            "user_flag": self._current_user_flag_config(),
            "planner": self._planner_snapshot(),
        }

    def _apply_preset(self, data: Dict[str, object]) -> None:
        """Restore UI from a preset dict (as written by _collect_preset)."""
        if not isinstance(data, dict):
            return

        cam = data.get("camera", {})
        if isinstance(cam, dict):
            if "fps" in cam:
                self.spin_fps.blockSignals(True)
                self.spin_fps.setValue(float(cam["fps"]))
                self.spin_fps.blockSignals(False)
            if "exposure_ms" in cam:
                self.spin_exposure.blockSignals(True)
                self.spin_exposure.setValue(float(cam["exposure_ms"]))
                self.spin_exposure.blockSignals(False)
            if "width" in cam:
                self.spin_width.blockSignals(True)
                self.spin_width.setValue(int(cam["width"]))
                self.spin_width.blockSignals(False)
            if "height" in cam:
                self.spin_height.blockSignals(True)
                self.spin_height.setValue(int(cam["height"]))
                self.spin_height.blockSignals(False)
            if "image_format" in cam and cam["image_format"] in ("Mono8", "BGR8"):
                self.combo_image_format.setCurrentText(cam["image_format"])
            if "encoder_index" in cam:
                idx = int(cam["encoder_index"])
                if 0 <= idx < self.combo_encoder.count():
                    self.combo_encoder.setCurrentIndex(idx)
            if "preview_enabled" in cam:
                self.check_preview_enabled.blockSignals(True)
                self.check_preview_enabled.setChecked(bool(cam["preview_enabled"]))
                self.check_preview_enabled.blockSignals(False)
            if "preview_fps" in cam:
                self.spin_preview_fps.blockSignals(True)
                self.spin_preview_fps.setValue(float(cam["preview_fps"]))
                self.spin_preview_fps.blockSignals(False)
            if "preview_width" in cam:
                self.spin_preview_width.blockSignals(True)
                self.spin_preview_width.setValue(int(cam["preview_width"]))
                self.spin_preview_width.blockSignals(False)
            if "frame_buffer_size" in cam:
                self.spin_frame_buffer.blockSignals(True)
                self.spin_frame_buffer.setValue(int(cam["frame_buffer_size"]))
                self.spin_frame_buffer.blockSignals(False)
            if "metadata_stats_interval" in cam:
                self.spin_metadata_stats_interval.blockSignals(True)
                self.spin_metadata_stats_interval.setValue(int(cam["metadata_stats_interval"]))
                self.spin_metadata_stats_interval.blockSignals(False)
            if cam.get("roi_rect") is not None:
                self.roi_rect = cam["roi_rect"]
                self._sync_camera_roi_ui_state()
            if "max_hours" in cam:
                self.spin_hours.setValue(int(cam["max_hours"]))
            if "max_minutes" in cam:
                self.spin_minutes.setValue(int(cam["max_minutes"]))
            if "max_seconds" in cam:
                self.spin_seconds.setValue(int(cam["max_seconds"]))
            if "max_unlimited" in cam:
                self.check_unlimited.setCurrentIndex(1 if cam["max_unlimited"] else 0)
            if "save_folder" in cam and cam["save_folder"]:
                folder = str(cam["save_folder"])
                self.last_save_folder = folder
                if hasattr(self, "edit_save_folder") and self.edit_save_folder:
                    self.edit_save_folder.setText(folder)
                if hasattr(self, "label_file_save_folder") and self.label_file_save_folder:
                    self.label_file_save_folder.setText(folder)
            if "filename_parts" in cam and isinstance(cam["filename_parts"], list):
                for index, combo in enumerate(self.filename_order_boxes):
                    if index < len(cam["filename_parts"]):
                        label = self._filename_key_to_label(cam["filename_parts"][index])
                        if label:
                            combo.blockSignals(True)
                            combo.setCurrentText(label)
                            combo.blockSignals(False)
            if "folder_parts" in cam and isinstance(cam["folder_parts"], list):
                for index, combo in enumerate(self.folder_order_boxes):
                    if index < len(cam["folder_parts"]):
                        label = self._filename_key_to_label(cam["folder_parts"][index])
                    else:
                        label = "(skip)"
                    if label:
                        combo.blockSignals(True)
                        combo.setCurrentText(label)
                        combo.blockSignals(False)
            if "organize_by_session" in cam and self.check_organize_session_folders is not None:
                self.check_organize_session_folders.blockSignals(True)
                self.check_organize_session_folders.setChecked(bool(cam["organize_by_session"]))
                self.check_organize_session_folders.blockSignals(False)
                self._set_folder_structure_controls_enabled(self._organize_recordings_enabled())

        user_flags = data.get("user_flags", None)
        if isinstance(user_flags, list):
            self.user_flag_configs = [
                self._normalize_user_flag_config(item, fallback_index=index + 1)
                for index, item in enumerate(user_flags)
                if isinstance(item, dict)
            ]
            self._on_user_flag_settings_changed()
        else:
            user_flag = data.get("user_flag", {})
            if isinstance(user_flag, dict):
                self.user_flag_configs = [self._normalize_user_flag_config(user_flag, fallback_index=1)]
                self._on_user_flag_settings_changed()

        # Behavior pins
        bp = data.get("behavior_pins", {})
        if isinstance(bp, dict):
            for key, entry in bp.items():
                if not isinstance(entry, dict) or key not in self.BEHAVIOR_PIN_KEYS:
                    continue
                pin_edit = self.behavior_pin_edits.get(key)
                role_box = self.behavior_role_boxes.get(key)
                label_edit = self.signal_label_edits.get(key)
                enabled_check = self.signal_enabled_checks.get(key)
                if pin_edit and "pins" in entry:
                    pin_edit.setText(str(entry["pins"]))
                if role_box and "role" in entry:
                    role_box.blockSignals(True)
                    role_box.setCurrentText(str(entry["role"]))
                    role_box.blockSignals(False)
                if label_edit and "label" in entry:
                    label_edit.setText(str(entry["label"]))
                if enabled_check and "enabled" in entry:
                    enabled_check.setChecked(bool(entry["enabled"]))
            self._apply_behavior_pin_configuration(persist=True)

        # Sync params
        sync = data.get("sync", {})
        if isinstance(sync, dict):
            if "period_s" in sync:
                self.settings.setValue("sync_period_s", float(sync["period_s"]))
            if "pulse_s" in sync:
                self.settings.setValue("sync_pulse_s", float(sync["pulse_s"]))

        # Live detection
        live = data.get("live_detection", {})
        if isinstance(live, dict) and self.live_detection_panel is not None:
            model_index = self.live_detection_panel.combo_model_key.findData(live.get("model_key", ""))
            if model_index >= 0:
                self.live_detection_panel.combo_model_key.setCurrentIndex(model_index)
            if "checkpoint_path" in live:
                self.live_detection_panel.edit_checkpoint.setText(str(live["checkpoint_path"] or ""))
            if "keypoint_source" in live and hasattr(self.live_detection_panel, "combo_keypoint_source"):
                source_index = self.live_detection_panel.combo_keypoint_source.findData(live["keypoint_source"])
                if source_index >= 0:
                    self.live_detection_panel.combo_keypoint_source.setCurrentIndex(source_index)
            if "closed_loop_fast" in live and hasattr(self.live_detection_panel, "check_closed_loop_fast"):
                self.live_detection_panel.check_closed_loop_fast.setChecked(bool(live["closed_loop_fast"]))
            if "pose_checkpoint_path" in live:
                self.live_detection_panel.edit_pose_checkpoint.setText(str(live["pose_checkpoint_path"] or ""))
            if "pose_threshold" in live:
                self.live_detection_panel.spin_pose_threshold.setValue(float(live["pose_threshold"]))
            if "min_pose_keypoints" in live:
                self.live_detection_panel.spin_min_pose_kp.setValue(int(live["min_pose_keypoints"]))
            if "clean_masks" in live and hasattr(self.live_detection_panel, "check_clean_masks"):
                self.live_detection_panel.check_clean_masks.setChecked(bool(live["clean_masks"]))
            if "threshold" in live:
                self.live_detection_panel.spin_threshold.setValue(float(live["threshold"]))
            if "selected_class_ids" in live:
                ids = live["selected_class_ids"]
                self.live_detection_panel.edit_selected_classes.setText(
                    ",".join(str(v) for v in ids) if isinstance(ids, list) else str(ids)
                )
            if "identity_mode" in live:
                iidx = self.live_detection_panel.combo_identity_mode.findData(live["identity_mode"])
                if iidx >= 0:
                    self.live_detection_panel.combo_identity_mode.setCurrentIndex(iidx)
            if "expected_mouse_count" in live:
                self.live_detection_panel.spin_expected_mice.setValue(int(live["expected_mouse_count"]))
            if "inference_max_width" in live:
                self.live_detection_panel.spin_inference_width.setValue(int(live["inference_max_width"]))
            if "acceleration_mode" in live:
                acceleration_index = self.live_detection_panel.combo_acceleration_mode.findData(live["acceleration_mode"])
                if acceleration_index >= 0:
                    self.live_detection_panel.combo_acceleration_mode.setCurrentIndex(acceleration_index)
            self.live_detection_panel.set_overlay_options(
                show_masks=bool(live.get("show_masks", True)),
                show_boxes=bool(live.get("show_boxes", True)),
                show_keypoints=bool(live.get("show_keypoints", True)),
                save_overlay_video=bool(live.get("save_overlay_video", False)),
                save_tracking_csv=bool(live.get("save_tracking_csv", False)),
                save_masks_coco=bool(live.get("save_masks_coco", False)),
                mask_opacity=float(live.get("mask_opacity", 0.18)),
            )
            # ROIs
            self.live_rois = {}
            for entry in live.get("rois", []):
                try:
                    roi = BehaviorROI.from_dict(entry)
                    self.live_rois[roi.name] = roi
                except Exception:
                    continue
            # Rules
            self.live_rules = []
            for entry in live.get("rules", []):
                try:
                    self.live_rules.append(LiveTriggerRule.from_dict(entry))
                except Exception:
                    continue
            # Output mapping
            if "output_mapping" in live:
                self.live_output_mapping = self._normalize_live_output_mapping(live["output_mapping"])
            self.live_rule_engine.set_rois(self.live_rois)
            self.live_rule_engine.set_rules(self.live_rules)
            self.live_detection_panel.set_output_mapping(self.live_output_mapping)
            self._refresh_live_panel_state()
            self._update_user_flag_pin_summary()
            self._persist_live_detection_settings()

        # Planner
        planner_snapshot = data.get("planner")
        if planner_snapshot:
            self._apply_planner_snapshot(planner_snapshot)

        self._persist_settings_snapshot(sync=True)
        self._update_filename_preview()

    def _save_preset_to_file(self) -> None:
        """Export all settings to a user-chosen JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", self.last_save_folder, "PyKaboo Preset (*.pkpreset *.json)"
        )
        if not path:
            return
        try:
            preset = self._collect_preset()
            Path(path).write_text(json.dumps(preset, indent=2, ensure_ascii=False), encoding="utf-8")
            self._on_status_update(f"Preset saved: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Preset Failed", str(exc))

    def _load_preset_from_file(self) -> None:
        """Import settings from a user-chosen JSON preset file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Preset", self.last_save_folder, "PyKaboo Preset (*.pkpreset *.json)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Load Preset Failed", f"Could not read preset:\n{exc}")
            return
        if not isinstance(data, dict):
            QMessageBox.critical(self, "Load Preset Failed", "Not a valid preset file.")
            return
        self._apply_preset(data)
        self._on_status_update(f"Preset loaded: {Path(path).name}")

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
        self._refresh_behavior_panel_visibility()

    def _save_ui_setting(self, key: str, value):
        """Persist a UI setting."""
        self.settings.setValue(key, value)

    def _load_live_detection_settings(self):
        """Restore live-detection model, ROI, rule, and DO mapping settings."""
        if self.live_detection_panel is None:
            return

        # Bundled defaults (pykaboo/models/): a fresh install with no saved path
        # points at the shipped seg + pose checkpoints automatically.
        from default_models import default_pose_checkpoint, default_seg_checkpoint

        seg_default = default_seg_checkpoint()
        pose_default = default_pose_checkpoint()

        config_payload = {
            "model_key": str(self.settings.value("live_model_key", "rfdetr-seg-medium")),
            "checkpoint_path": str(self.settings.value("live_checkpoint_path", seg_default) or seg_default),
            "keypoint_source": str(self.settings.value("live_keypoint_source", "yolo_pose") or "yolo_pose"),
            "closed_loop_fast": int(self.settings.value("live_closed_loop_fast", 1)) == 1,
            "pose_checkpoint_path": str(self.settings.value("live_pose_checkpoint_path", pose_default) or pose_default),
            "pose_threshold": float(self.settings.value("live_pose_threshold", 0.25)),
            "min_pose_keypoints": int(self.settings.value("live_min_pose_keypoints", 0)),
            "clean_masks": int(self.settings.value("live_clean_masks", 1)) == 1,
            "threshold": float(self.settings.value("live_threshold", 0.35)),
            "selected_class_ids": self._parse_int_csv(self.settings.value("live_selected_classes", "0")),
            "identity_mode": str(self.settings.value("live_identity_mode", "tracker")),
            "expected_mouse_count": int(self.settings.value("live_expected_mouse_count", 1)),
            "inference_max_width": int(self.settings.value("live_inference_max_width", 960)),
            "acceleration_mode": str(self.settings.value("live_acceleration_mode", "balanced") or "balanced"),
            "show_masks": int(self.settings.value("live_show_masks", 1)) == 1,
            "show_boxes": int(self.settings.value("live_show_boxes", 1)) == 1,
            "show_keypoints": int(self.settings.value("live_show_keypoints", 1)) == 1,
            "mask_opacity": float(self.settings.value("live_mask_opacity", 0.18)),
            "save_overlay_video": int(self.settings.value("live_save_overlay_video", 0)) == 1,
            "save_tracking_csv": int(self.settings.value("live_save_tracking_csv", 0)) == 1,
            "save_masks_coco": int(self.settings.value("live_save_masks_coco", 0)) == 1,
        }

        model_index = self.live_detection_panel.combo_model_key.findData(config_payload["model_key"])
        if model_index >= 0:
            self.live_detection_panel.combo_model_key.setCurrentIndex(model_index)
        self.live_detection_panel.edit_checkpoint.setText(config_payload["checkpoint_path"])
        if hasattr(self.live_detection_panel, "combo_keypoint_source"):
            source_index = self.live_detection_panel.combo_keypoint_source.findData(config_payload["keypoint_source"])
            if source_index >= 0:
                self.live_detection_panel.combo_keypoint_source.setCurrentIndex(source_index)
        if hasattr(self.live_detection_panel, "check_closed_loop_fast"):
            self.live_detection_panel.check_closed_loop_fast.setChecked(config_payload["closed_loop_fast"])
        self.live_detection_panel.edit_pose_checkpoint.setText(config_payload["pose_checkpoint_path"])
        self.live_detection_panel.spin_pose_threshold.setValue(config_payload["pose_threshold"])
        self.live_detection_panel.spin_min_pose_kp.setValue(config_payload["min_pose_keypoints"])
        if hasattr(self.live_detection_panel, "check_clean_masks"):
            self.live_detection_panel.check_clean_masks.setChecked(config_payload["clean_masks"])
        self.live_detection_panel.spin_threshold.setValue(config_payload["threshold"])
        self.live_detection_panel.edit_selected_classes.setText(
            ",".join(str(value) for value in config_payload["selected_class_ids"])
        )
        identity_index = self.live_detection_panel.combo_identity_mode.findData(config_payload["identity_mode"])
        if identity_index >= 0:
            self.live_detection_panel.combo_identity_mode.setCurrentIndex(identity_index)
        self.live_detection_panel.spin_expected_mice.setValue(config_payload["expected_mouse_count"])
        self.live_detection_panel.spin_inference_width.setValue(config_payload["inference_max_width"])
        acceleration_index = self.live_detection_panel.combo_acceleration_mode.findData(config_payload["acceleration_mode"])
        if acceleration_index >= 0:
            self.live_detection_panel.combo_acceleration_mode.setCurrentIndex(acceleration_index)
        self.live_detection_panel.set_overlay_options(
            show_masks=config_payload["show_masks"],
            show_boxes=config_payload["show_boxes"],
            save_overlay_video=config_payload["save_overlay_video"],
            show_keypoints=config_payload["show_keypoints"],
            save_tracking_csv=config_payload["save_tracking_csv"],
            save_masks_coco=config_payload["save_masks_coco"],
            mask_opacity=config_payload["mask_opacity"],
        )

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
        self.settings.setValue("live_keypoint_source", config.get("keypoint_source", "yolo_pose"))
        self.settings.setValue("live_closed_loop_fast", 1 if config.get("closed_loop_fast", True) else 0)
        self.settings.setValue("live_threshold", float(config["threshold"]))
        self.settings.setValue(
            "live_selected_classes",
            ",".join(str(value) for value in config["selected_class_ids"]),
        )
        self.settings.setValue("live_identity_mode", config["identity_mode"])
        self.settings.setValue("live_expected_mouse_count", int(config["expected_mouse_count"]))
        self.settings.setValue("live_inference_max_width", int(config.get("inference_max_width", 960)))
        self.settings.setValue("live_acceleration_mode", str(config.get("acceleration_mode", "balanced") or "balanced"))
        self.settings.setValue("live_show_masks", 1 if config.get("show_masks", True) else 0)
        self.settings.setValue("live_show_boxes", 1 if config.get("show_boxes", True) else 0)
        self.settings.setValue("live_show_keypoints", 1 if config.get("show_keypoints", True) else 0)
        self.settings.setValue("live_mask_opacity", float(config.get("mask_opacity", 0.18)))
        self.settings.setValue("live_save_overlay_video", 1 if config.get("save_overlay_video", False) else 0)
        self.settings.setValue("live_save_tracking_csv", 1 if config.get("save_tracking_csv", False) else 0)
        self.settings.setValue("live_save_masks_coco", 1 if config.get("save_masks_coco", False) else 0)
        self.settings.setValue("live_pose_checkpoint_path", config.get("pose_checkpoint_path", "") or "")
        self.settings.setValue("live_pose_threshold", float(config.get("pose_threshold", 0.25)))
        self.settings.setValue("live_min_pose_keypoints", int(config.get("min_pose_keypoints", 0)))
        self.settings.setValue("live_clean_masks", 1 if config.get("clean_masks", True) else 0)
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

    def _on_live_overlay_options_changed(self):
        self._persist_live_detection_settings()
        if self.worker is None or not bool(getattr(self.worker, "is_recording", False)):
            return
        config = self.live_detection_panel.detection_config() if self.live_detection_panel else {}
        save_overlay = bool(config.get("save_overlay_video", False))
        if save_overlay and not self.live_overlay_video_enabled:
            base_path = self.current_recording_filepath or getattr(self.worker, "recording_filename", "")
            if base_path:
                self._start_live_overlay_video_recording(str(base_path))
        elif not save_overlay and self.live_overlay_video_enabled:
            self._stop_live_overlay_video_recording()

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
        keypoint_source = str(config.get("keypoint_source", "yolo_pose") or "yolo_pose")
        closed_loop_fast = bool(config.get("closed_loop_fast", True))
        full_masks_requested = bool(
            config.get("show_masks", True)
            or config.get("save_masks_coco", False)
            or config.get("save_overlay_video", False)
        )
        return LiveInferenceConfig(
            model_key=str(config.get("model_key", "rfdetr-seg-medium")),
            checkpoint_path=str(config.get("checkpoint_path", "") or ""),
            threshold=float(config.get("threshold", 0.35)),
            selected_class_ids=list(config.get("selected_class_ids", [])),
            identity_mode=str(config.get("identity_mode", "tracker")),
            expected_mouse_count=max(1, int(config.get("expected_mouse_count", 1))),
            inference_max_width=max(0, int(config.get("inference_max_width", 960))),
            keypoint_source=keypoint_source,
            pose_checkpoint_path=str(config.get("pose_checkpoint_path", "") or ""),
            pose_threshold=float(config.get("pose_threshold", 0.25)),
            min_pose_keypoints=max(0, int(config.get("min_pose_keypoints", 0))),
            clean_masks=bool(config.get("clean_masks", True)),
            acceleration_mode=str(config.get("acceleration_mode", "balanced") or "balanced"),
            tracking_mode=bool(self.live_tracking_mode_active),
            output_masks=bool(not (keypoint_source == "mask_geometry" and closed_loop_fast) and full_masks_requested),
        )

    @Slot(object)
    def _on_preview_packet_ready(self, packet: object):
        if not isinstance(packet, PreviewFramePacket):
            return
        self._update_camera_line_plot_from_metadata(packet.metadata)
        self.live_preview_packet = packet
        self.live_preview_frame_index = int(packet.frame_index)
        self.live_preview_timestamp_s = float(packet.timestamp_s)

    @Slot(object)
    def _on_live_inference_packet_ready(self, packet: object):
        if not isinstance(packet, PreviewFramePacket):
            return
        frame_id = int(packet.frame_index)
        self.live_inference_frame_cache[frame_id] = packet
        while len(self.live_inference_frame_cache) > 128:
            oldest_key = min(self.live_inference_frame_cache)
            self.live_inference_frame_cache.pop(oldest_key, None)

    @Slot(object)
    def _on_record_frame_packet_ready(self, packet: object):
        if not isinstance(packet, PreviewFramePacket):
            return
        self._record_live_overlay_frame(packet)

    @Slot(dict)
    def _on_frame_metadata_ready(self, metadata: dict):
        self._update_camera_line_plot_from_metadata(metadata)

    @Slot(bool)
    @Slot(bool)
    def _on_tracking_mode_toggled(self, enabled: bool):
        """One-click tracking: parallel mask + pose inference with exports armed."""
        if self.live_detection_panel is None or self.live_inference_worker is None:
            return

        if enabled:
            config = self.live_detection_panel.detection_config()
            if not str(config.get("checkpoint_path", "")).strip():
                self._on_error_occurred(
                    "Tracking mode needs a mask checkpoint. Set it in the Live Detection "
                    "panel (Checkpoint), then toggle Tracking again."
                )
                self._set_tracking_button_checked(False)
                return
            keypoint_source = str(config.get("keypoint_source", "yolo_pose") or "yolo_pose")
            if keypoint_source == "yolo_pose" and not str(config.get("pose_checkpoint_path", "")).strip():
                self._on_error_occurred(
                    "Tracking mode needs a pose checkpoint. Set it in the Live Detection "
                    "panel (Pose checkpoint), then toggle Tracking again."
                )
                self._set_tracking_button_checked(False)
                return

            self.live_tracking_mode_active = True
            # YOLO-pose tracking keeps the historical behavior of arming COCO
            # mask export. Mask-geometry tracking preserves the user's mask
            # export choice so closed-loop mode can avoid full-frame masks.
            closed_loop_fast = bool(config.get("closed_loop_fast", True))
            fast_mask_geometry = keypoint_source == "mask_geometry" and closed_loop_fast
            save_masks_coco = False if fast_mask_geometry else (
                bool(config.get("save_masks_coco", False))
                if keypoint_source == "mask_geometry"
                else True
            )
            self.live_detection_panel.set_overlay_options(
                show_masks=False if fast_mask_geometry else bool(config.get("show_masks", True)),
                show_boxes=bool(config.get("show_boxes", True)),
                save_overlay_video=False if fast_mask_geometry else bool(config.get("save_overlay_video", False)),
                show_keypoints=True,
                save_tracking_csv=True,
                save_masks_coco=save_masks_coco,
                mask_opacity=float(config.get("mask_opacity", 0.18)),
            )
            if not self.live_detection_enabled:
                self.live_detection_panel.set_detection_running(True)
                self._on_live_detection_toggled(True)
            else:
                # Inference already running: push the parallel-pipeline config.
                self.live_inference_worker.start_inference(self._build_live_inference_config())
            self._on_status_update(
                "Tracking mode ON: masks tracked with selected keypoint source; COCO and DLC exports armed."
            )
        else:
            self.live_tracking_mode_active = False
            if self.live_detection_enabled:
                self.live_detection_panel.set_detection_running(False)
                self._on_live_detection_toggled(False)
            self._on_status_update("Tracking mode off.")

    def _set_tracking_button_checked(self, checked: bool):
        if self.btn_tracking_mode is None:
            return
        self.btn_tracking_mode.blockSignals(True)
        self.btn_tracking_mode.setChecked(bool(checked))
        self.btn_tracking_mode.blockSignals(False)

    def _on_live_detection_toggled(self, enabled: bool):
        if self.live_detection_panel is None or self.live_inference_worker is None:
            return

        if enabled:
            if not self.check_preview_enabled.isChecked():
                self._on_error_occurred("Enable preview before starting live detection.")
                self.live_detection_panel.set_detection_running(False)
                return
            self.live_detection_enabled = True
            self.live_detection_last_result = None
            self.live_detection_result_history.clear()
            self.live_inference_frame_cache.clear()
            self.live_detection_results_by_frame.clear()
            self.live_overlay_pending_packets.clear()
            self.live_overlay_last_written_frame_index = None
            self.live_synced_overlay_active = False
            self.live_synced_overlay_last_update_s = 0.0
            self.live_level_output_states = {f"DO{i}": False for i in range(1, 9)}
            self.live_rule_engine.clear_runtime_state()
            if self.worker is not None:
                config = self.live_detection_panel.detection_config()
                self.worker.set_live_inference_emit_max_width(int(config.get("inference_max_width", 960)))
                self.worker.set_live_inference_packets_enabled(True)
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
            self.live_rule_timer.start()
            self._sync_live_circle_roi_items()
            return

        self.live_detection_enabled = False
        if self.live_tracking_mode_active:
            self.live_tracking_mode_active = False
            self._set_tracking_button_checked(False)
        self.live_rule_timer.stop()
        self.live_detection_last_result = None
        self.live_detection_result_history.clear()
        self.live_inference_frame_cache.clear()
        self.live_detection_results_by_frame.clear()
        self.live_overlay_pending_packets.clear()
        self.live_overlay_last_written_frame_index = None
        self.live_synced_overlay_active = False
        self.live_synced_overlay_last_update_s = 0.0
        self.live_active_rule_ids = []
        self.live_output_states = {f"DO{i}": False for i in range(1, 9)}
        self.live_level_output_states = {f"DO{i}": False for i in range(1, 9)}
        if self.worker is not None:
            self.worker.set_live_inference_packets_enabled(False)
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
        if not self.live_detection_enabled or not isinstance(result, LiveDetectionResult):
            return
        self.live_detection_last_result = result
        self.live_detection_result_history.append(result)
        self.live_detection_results_by_frame[int(result.frame_index)] = result
        while len(self.live_detection_results_by_frame) > 256:
            oldest_key = min(self.live_detection_results_by_frame)
            self.live_detection_results_by_frame.pop(oldest_key, None)
        self._apply_live_rule_evaluation(
            result,
            now_ms=self._live_rule_now_ms(),
            record_export=True,
        )
        if self.live_detection_panel is not None:
            self.live_detection_panel.set_status(format_live_detection_status(result))
        self._display_synced_live_detection_result(result)
        try:
            self._flush_live_overlay_frame_for_result(result)
        except Exception as exc:
            self._on_error_occurred(f"Overlay video sync error: {exc}")

    def _live_detection_overlay_visible(self) -> bool:
        if not self.live_detection_enabled or self.live_detection_panel is None:
            return False
        options = self.live_detection_panel.overlay_options()
        return bool(
            options.get("show_masks", True)
            or options.get("show_boxes", True)
            or options.get("show_keypoints", True)
        )

    def _display_synced_live_detection_result(self, result: LiveDetectionResult) -> None:
        """Mark the latest inference result fresh for the preview overlay.

        The live preview (``_on_frame_ready``) now owns all on-screen rendering
        and paints the most recent result on every preview frame at the full
        preview rate. We only record the update time here so the carry-forward
        retention can keep the overlay visible between inference results.
        """
        self.live_synced_overlay_active = False
        self.live_synced_overlay_last_update_s = time.time()

    def _live_rule_now_ms(self) -> int:
        return int(round(time.time() * 1000.0))

    def _apply_live_rule_evaluation(
        self,
        result: Optional[LiveDetectionResult],
        *,
        now_ms: Optional[int] = None,
        record_export: bool = False,
    ) -> None:
        evaluation = self.live_rule_engine.evaluate(
            result,
            int(now_ms if now_ms is not None else self._live_rule_now_ms()),
        )
        next_active_rule_ids = list(evaluation.active_rule_ids)
        next_output_states = {
            f"DO{i}": bool(evaluation.output_states.get(f"DO{i}", False))
            for i in range(1, 9)
        }
        next_level_output_states = {
            f"DO{i}": bool(evaluation.level_output_states.get(f"DO{i}", False))
            for i in range(1, 9)
        }
        rules_changed = next_active_rule_ids != self.live_active_rule_ids
        outputs_changed = next_output_states != self.live_output_states
        level_outputs_changed = next_level_output_states != self.live_level_output_states
        self.live_active_rule_ids = next_active_rule_ids
        self.live_output_states = next_output_states
        self.live_level_output_states = next_level_output_states
        if record_export and result is not None:
            self._record_live_detection_export(result, self.live_output_states, self.live_active_rule_ids)
        if self.arduino_worker is not None and self.is_arduino_connected:
            if level_outputs_changed:
                for output_id, state in next_level_output_states.items():
                    self.arduino_worker.set_live_output_level(output_id, bool(state))
            for output_id, duration_ms, pulse_count, pulse_frequency_hz in evaluation.triggered_pulses:
                self.arduino_worker.start_live_output_pulse_train(
                    output_id,
                    int(duration_ms),
                    pulse_count=int(pulse_count),
                    pulse_frequency_hz=float(pulse_frequency_hz),
                )
        if rules_changed or outputs_changed or level_outputs_changed or bool(evaluation.triggered_pulses):
            self._refresh_live_panel_state()

    @Slot()
    def _on_live_rule_timer_timeout(self):
        if not self.live_detection_enabled:
            return
        self._apply_live_rule_evaluation(self.live_detection_last_result)

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
        self._update_user_flag_pin_summary()
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

    def _reset_live_rule_runtime_outputs(self):
        self.live_active_rule_ids = []
        self.live_output_states = {f"DO{i}": False for i in range(1, 9)}
        self.live_level_output_states = {f"DO{i}": False for i in range(1, 9)}
        self.live_rule_engine.clear_runtime_state()
        if self.arduino_worker is not None and self.is_arduino_connected:
            self.arduino_worker.clear_live_outputs()

    @Slot(str)
    def _edit_live_rule(self, rule_id: str):
        target_id = str(rule_id or "").strip()
        rule_index = next(
            (index for index, rule in enumerate(self.live_rules) if rule.rule_id == target_id),
            -1,
        )
        if rule_index < 0:
            return
        rule = self.live_rules[rule_index]

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Trigger Rule")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        layout.addLayout(form)

        def set_combo_data(combo: QComboBox, value: str) -> None:
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)

        combo_type = QComboBox()
        combo_type.addItem("ROI occupancy", "roi_occupancy")
        combo_type.addItem("Mouse center distance", "mouse_proximity")
        combo_type.addItem("Mask edge touch", "mask_contact")
        set_combo_data(combo_type, str(rule.rule_type or "roi_occupancy"))
        form.addRow("Rule type:", combo_type)

        spin_mouse_id = QSpinBox()
        spin_mouse_id.setRange(1, 8)
        spin_mouse_id.setValue(max(1, int(rule.mouse_id)))
        label_mouse_id = QLabel("Mouse:")
        form.addRow(label_mouse_id, spin_mouse_id)

        combo_roi = QComboBox()
        roi_names = list(self.live_rois.keys())
        if rule.roi_name and rule.roi_name not in roi_names:
            roi_names.append(rule.roi_name)
        combo_roi.addItems(roi_names)
        if rule.roi_name:
            index = combo_roi.findText(rule.roi_name)
            if index >= 0:
                combo_roi.setCurrentIndex(index)
        label_roi = QLabel("ROI:")
        form.addRow(label_roi, combo_roi)

        spin_peer_id = QSpinBox()
        spin_peer_id.setRange(1, 8)
        spin_peer_id.setValue(max(1, int(rule.peer_mouse_id)))
        label_peer_id = QLabel("Mouse B:")
        form.addRow(label_peer_id, spin_peer_id)

        spin_distance = QDoubleSpinBox()
        spin_distance.setRange(0.0, 100000.0)
        spin_distance.setDecimals(2)
        spin_distance.setSingleStep(1.0)
        spin_distance.setValue(max(0.0, float(rule.distance_px)))
        label_distance = QLabel("Distance px:")
        form.addRow(label_distance, spin_distance)

        combo_output = QComboBox()
        for output_index in range(1, 9):
            combo_output.addItem(f"DO{output_index}", f"DO{output_index}")
        set_combo_data(combo_output, normalize_output_id(rule.output_id))
        form.addRow("Output:", combo_output)

        combo_mode = QComboBox()
        combo_mode.addItem("Gate", "gate")
        combo_mode.addItem("Level", "level")
        combo_mode.addItem("Pulse", "pulse")
        set_combo_data(combo_mode, str(rule.mode or "gate"))
        form.addRow("Mode:", combo_mode)

        spin_duration = QSpinBox()
        spin_duration.setRange(1, 600000)
        spin_duration.setValue(max(1, int(rule.duration_ms)))
        label_duration = QLabel("Pulse ms:")
        form.addRow(label_duration, spin_duration)

        spin_pulse_count = QSpinBox()
        spin_pulse_count.setRange(1, 10000)
        spin_pulse_count.setValue(max(1, int(rule.pulse_count)))
        label_pulse_count = QLabel("Pulse count:")
        form.addRow(label_pulse_count, spin_pulse_count)

        spin_frequency = QDoubleSpinBox()
        spin_frequency.setRange(0.001, 1000.0)
        spin_frequency.setDecimals(3)
        spin_frequency.setSingleStep(1.0)
        spin_frequency.setSuffix(" Hz")
        spin_frequency.setValue(max(0.001, float(rule.pulse_frequency_hz)))
        label_frequency = QLabel("Frequency:")
        form.addRow(label_frequency, spin_frequency)

        combo_activation = QComboBox()
        combo_activation.addItem("At entry", "entry")
        combo_activation.addItem("At exit", "exit")
        combo_activation.addItem("Continuous", "continuous")
        set_combo_data(combo_activation, normalize_activation_pattern(rule.activation_pattern))
        label_activation = QLabel("Activation:")
        form.addRow(label_activation, combo_activation)

        spin_inter_train_interval = QSpinBox()
        spin_inter_train_interval.setRange(0, 600000)
        spin_inter_train_interval.setSuffix(" ms")
        spin_inter_train_interval.setValue(max(0, int(rule.inter_train_interval_ms)))
        label_inter_train_interval = QLabel("Inter-train interval:")
        form.addRow(label_inter_train_interval, spin_inter_train_interval)

        def update_condition_controls() -> None:
            rule_type = str(combo_type.currentData() or "roi_occupancy")
            is_roi_rule = rule_type == "roi_occupancy"
            uses_peer_mouse = rule_type in {"mouse_proximity", "mask_contact"}
            uses_distance = rule_type == "mouse_proximity"
            for widget in (label_roi, combo_roi):
                widget.setVisible(is_roi_rule)
            for widget in (label_peer_id, spin_peer_id):
                widget.setVisible(uses_peer_mouse)
            for widget in (label_distance, spin_distance):
                widget.setVisible(uses_distance)
            label_mouse_id.setText("ROI mouse:" if is_roi_rule else "Mouse A:")

        def update_pulse_controls() -> None:
            is_pulse = str(combo_mode.currentData() or "gate") == "pulse"
            continuous = str(combo_activation.currentData() or "entry") == "continuous"
            for widget in (
                label_duration,
                spin_duration,
                label_pulse_count,
                spin_pulse_count,
                label_frequency,
                spin_frequency,
                label_activation,
                combo_activation,
            ):
                widget.setVisible(is_pulse)
            for widget in (label_inter_train_interval, spin_inter_train_interval):
                widget.setVisible(is_pulse and continuous)

        combo_type.currentIndexChanged.connect(lambda _index: update_condition_controls())
        combo_mode.currentIndexChanged.connect(lambda _index: update_pulse_controls())
        combo_activation.currentIndexChanged.connect(lambda _index: update_pulse_controls())
        update_condition_controls()
        update_pulse_controls()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        rule_type = str(combo_type.currentData() or "roi_occupancy")
        roi_name = str(combo_roi.currentText() or "").strip()
        if rule_type == "roi_occupancy" and roi_name not in self.live_rois:
            QMessageBox.warning(self, "Edit Rule", "Select an existing ROI for an ROI occupancy rule.")
            return

        updated_rule = LiveTriggerRule(
            rule_id=rule.rule_id,
            rule_type=rule_type,
            output_id=normalize_output_id(combo_output.currentText()),
            mode=str(combo_mode.currentData() or "gate"),
            duration_ms=max(1, int(spin_duration.value())),
            pulse_count=max(1, int(spin_pulse_count.value())),
            pulse_frequency_hz=max(0.001, float(spin_frequency.value())),
            inter_train_interval_ms=max(0, int(spin_inter_train_interval.value())),
            activation_pattern=str(combo_activation.currentData() or "entry"),
            mouse_id=max(1, int(spin_mouse_id.value())),
            peer_mouse_id=max(1, int(spin_peer_id.value())),
            roi_name=roi_name if rule_type == "roi_occupancy" else "",
            distance_px=max(0.0, float(spin_distance.value())) if rule_type == "mouse_proximity" else 0.0,
        )

        self.live_rules[rule_index] = updated_rule
        self._reset_live_rule_runtime_outputs()
        self.live_rule_engine.set_rules(self.live_rules)
        self._refresh_live_panel_state()
        self._persist_live_detection_settings()
        self._on_status_update(f"Updated rule: {build_rule_label(updated_rule)}")

    @Slot(str)
    def _test_live_rule(self, rule_id: str):
        target_id = str(rule_id or "").strip()
        rule = next((entry for entry in self.live_rules if entry.rule_id == target_id), None)
        if rule is None:
            return
        if self.arduino_worker is None or not self.is_arduino_connected:
            self._on_error_occurred("Connect Arduino before testing a live trigger rule.")
            return

        output_id = normalize_output_id(rule.output_id)
        pins = self.live_output_mapping.get(output_id, [])
        if not pins:
            self._on_error_occurred(f"{output_id} has no mapped Arduino pin. Add a DO mapping before testing.")
            return

        try:
            self.arduino_worker.configure_live_output_mapping(self.live_output_mapping)
        except Exception as exc:
            self._on_error_occurred(str(exc))
            return

        mode = str(rule.mode or "gate").strip().lower()
        duration_ms = max(1, int(rule.duration_ms))
        if mode == "pulse":
            pulse_count = max(1, int(rule.pulse_count))
            pulse_frequency_hz = max(0.001, float(rule.pulse_frequency_hz))
        else:
            pulse_count = 1
            pulse_frequency_hz = 1.0

        self.arduino_worker.start_live_output_pulse_train(
            output_id,
            duration_ms,
            pulse_count=pulse_count,
            pulse_frequency_hz=pulse_frequency_hz,
        )
        self._on_status_update(f"Manual rule test fired on {output_id}: {build_rule_label(rule)}")

    @Slot(str)
    def _remove_live_rule(self, rule_id: str):
        self.live_rules = [rule for rule in self.live_rules if rule.rule_id != rule_id]
        self._reset_live_rule_runtime_outputs()
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
        self._sync_live_circle_roi_items()

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

        fallback_selectors = self._camera_line_backend_fallback_selectors()

        entries: List[Dict[str, object]] = []
        if live_capabilities:
            for index, capability in enumerate(live_capabilities[:4], start=1):
                selector = str(capability.get("selector", f"Line{index}"))
                saved = saved_defaults.get(selector, {})
                entries.append({
                    "selector": selector,
                    "display_name": self._format_camera_line_selector(selector, fallback_line_number=index),
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
            selector = fallback_selectors[index - 1] if index <= len(fallback_selectors) else f"Line{index}"
            entries.append({
                "selector": selector,
                "display_name": self._format_camera_line_selector(selector, fallback_line_number=index),
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
        self._refresh_behavior_panel_visibility()
        self._rebuild_monitor_visuals(reset_plot=False)

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
            label = combo.currentText() if combo is not None else self._camera_line_label_text(line)
            suffix = self._line_label_suffix(label)
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
        """Rename selected camera-line status columns and drop unlabeled raw lines."""
        if df is None or df.empty:
            return df

        df = df.copy()
        label_map = self._get_line_label_map()
        drop_columns = []
        keep_line_columns = set()

        for line_number in range(1, 5):
            key = f"line{line_number}_status"
            suffix = label_map.get(key, "")
            if not suffix:
                continue

            internal_renamed = f"{key}_{suffix}"
            exported = self._camera_line_labeled_export_column(line_number, suffix)
            keep_line_columns.add(exported)
            source_columns = [
                column
                for column in (exported, internal_renamed, key)
                if column in df.columns
            ]
            if not source_columns:
                continue

            primary = source_columns[0]
            if exported not in df.columns:
                df = df.rename(columns={primary: exported})
            elif primary != exported:
                try:
                    df[exported] = df[exported].where(df[exported].notna(), df[primary])
                except Exception:
                    pass

            for column in source_columns:
                if column != exported:
                    drop_columns.append(column)

        for line_number in range(1, 5):
            raw_column = f"line{line_number}_status"
            if raw_column in label_map:
                continue
            if raw_column in df.columns:
                drop_columns.append(raw_column)
            for column in list(df.columns):
                if column in keep_line_columns:
                    continue
                if re.match(rf"^{re.escape(raw_column)}_.+$", str(column)):
                    drop_columns.append(column)

        if "line_status_all" in df.columns:
            drop_columns.append("line_status_all")

        if drop_columns:
            df = df.drop(columns=[column for column in drop_columns if column in df.columns])
        return df

    def _drop_unselected_signal_export_columns(self, df):
        """Remove disabled and low-level raw signal columns from exported CSVs."""
        if df is None or df.empty:
            return df

        definitions = self._signal_export_definitions()
        active_keys = self._active_signal_keys()
        active_ttl_keys = self._active_signal_keys(group="ttl")
        active_behavior_keys = self._active_signal_keys(group="behavior")
        keep_columns = set()

        for key in active_keys:
            definition = definitions.get(key, {})
            keep_columns.add(str(definition.get("state_column", "")).strip())
            keep_columns.add(str(definition.get("count_column", "")).strip())

        drop_candidates = {
            "gate",
            "gate_state",
            "gate_count",
            "gate_ttl",
            "sync",
            "sync_state",
            "sync_count",
            "sync_1hz_ttl",
            "sync_10hz_ttl",
            "barcode",
            "barcode_state",
            "barcode_count",
            "barcode0",
            "barcode1",
            "barcode0_state",
            "barcode1_state",
            "barcode0_count",
            "barcode1_count",
            "barcode_pin0_ttl",
            "barcode_pin1_ttl",
            "lever",
            "lever_state",
            "lever_count",
            "lever_ttl",
            "cue",
            "cue_state",
            "cue_count",
            "cue_ttl",
            "reward",
            "reward_state",
            "reward_count",
            "reward_ttl",
            "iti",
            "iti_state",
            "iti_count",
            "iti_ttl",
        }
        drop_candidates.update({f"do{index}_ttl" for index in range(1, 9)})
        drop_candidates.update({f"live_do{index}_state" for index in range(1, 9)})

        for key, definition in definitions.items():
            drop_candidates.add(str(definition.get("state_column", "")).strip())
            drop_candidates.add(str(definition.get("count_column", "")).strip())
            state_key = self._state_key_for_display(key)
            drop_candidates.add(state_key)
            drop_candidates.add(f"{state_key}_state")
            drop_candidates.add(f"{state_key}_count")

        removable = [
            column
            for column in drop_candidates
            if column and column in df.columns and column not in keep_columns
        ]
        if removable:
            df = df.drop(columns=removable)

        if not active_ttl_keys:
            ttl_columns = [column for column in ("ttl_state", "ttl_state_vector") if column in df.columns]
            if ttl_columns:
                df = df.drop(columns=ttl_columns)
        if not active_behavior_keys:
            behavior_columns = [column for column in ("behavior_state", "behavior_state_vector") if column in df.columns]
            if behavior_columns:
                df = df.drop(columns=behavior_columns)

        return df

    def _live_roi_metadata_payload(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for roi in self.live_rois.values():
            rows.append(
                {
                    "name": roi.name,
                    "roi_type": roi.roi_type,
                    "properties": roi_geometry_properties(roi),
                    "properties_text": format_roi_properties(roi),
                    "raw_data": roi.data,
                    "color": list(roi.color),
                }
            )
        return rows

    def _collect_metadata(self):
        """Collect all metadata fields."""
        live_roi_metadata = self._live_roi_metadata_payload()
        user_flag_configs = self._current_user_flag_configs()
        user_flag_config = user_flag_configs[0] if user_flag_configs else self._current_user_flag_config()
        user_flag_enabled = any(bool(str(config.get("shortcut", "") or "").strip()) for config in user_flag_configs)
        self.metadata = {
            'animal_id': self.meta_animal_id.text(),
            'session': self.meta_session.text() if self.meta_session is not None else "",
            'trial': self.meta_trial.text(),
            'experiment': self.meta_experiment.text(),
            'condition': self.meta_condition.text(),
            'arena': self.meta_arena.text(),
            'date': self.meta_date.text(),
            'notes': self.meta_notes.toPlainText(),
            'timestamp': datetime.now().isoformat(),
            'filename_preview': self._compose_recording_basename(),
            'filename_order': self._selected_filename_order(),
            'organize_recordings_by_session': self._organize_recordings_enabled(),
            'recording_root_folder': self.edit_save_folder.text() if hasattr(self, "edit_save_folder") and self.edit_save_folder is not None else "",
            'recording_output_folder': str(self._recording_destination_folder()),
            'live_roi_count': len(live_roi_metadata),
            'live_rois': live_roi_metadata,
            'user_flag_count': len(self.user_flag_events),
        }
        if user_flag_enabled:
            self.metadata['user_flags'] = user_flag_configs
            self.metadata['user_flag'] = user_flag_config

        # Add custom fields
        for field_name, field_edit in self.custom_metadata_fields.items():
            self.metadata[field_name] = field_edit.text()

        # Add barcode mode info
        if self.arduino_worker:
            hw_enabled = self.arduino_worker.get_hw_barcode_enabled()
            self.metadata['barcode_mode'] = 'hardware_timer1' if hw_enabled else 'software'
            if hw_enabled:
                self.metadata['barcode_hw_bit_width_ms'] = self.arduino_worker.get_hw_barcode_bit_width_ms()
                self.metadata['barcode_hw_pins'] = {'sync': 8, 'data': 9}
            else:
                params = self.arduino_worker.get_barcode_parameters()
                self.metadata['barcode_sw_params'] = params

        return self.metadata

    def _finalize_recording_timing_audit(self) -> Dict[str, object]:
        """Build one timing-audit payload for the just-finished recording."""
        audit = dict(self.active_recording_timing_audit or {})

        requested_duration_s = 0
        try:
            requested_duration_s = max(0, int(audit.get("requested_duration_seconds", 0) or 0))
        except (TypeError, ValueError):
            requested_duration_s = 0

        recorded_frames = int(self.last_frame_drop_stats.get("recorded_frames", 0) or 0)
        if recorded_frames <= 0 and self.worker is not None:
            recorded_frames = int(getattr(self.worker, "frame_counter", 0) or 0)

        encoded_fps = None
        if self.worker is not None:
            try:
                encoded_fps_value = float(
                    getattr(self.worker, "last_recording_output_fps", 0.0)
                    or getattr(self.worker, "recording_output_fps", 0.0)
                    or 0.0
                )
            except (TypeError, ValueError):
                encoded_fps_value = 0.0
            if encoded_fps_value > 0.0:
                encoded_fps = encoded_fps_value

        encoded_video_duration_s = encoded_video_duration_seconds(recorded_frames, encoded_fps)

        capture_duration_s = capture_duration_seconds(
            self.recording_first_frame_wallclock,
            self.recording_last_frame_wallclock or self.recording_stop_requested_at,
        )
        measured_fps = measured_capture_fps(recorded_frames, capture_duration_s)
        fps_delta_pct = percent_delta(encoded_fps, measured_fps)

        duration_delta_s = None
        if requested_duration_s > 0 and capture_duration_s is not None:
            duration_delta_s = float(capture_duration_s) - float(requested_duration_s)

        audio_duration_s = None
        audio_duration_reference_s = encoded_video_duration_s
        audio_duration_reference_type = "encoded_video_duration"
        if audio_duration_reference_s is None:
            audio_duration_reference_s = capture_duration_s
            audio_duration_reference_type = "capture_duration"
        audio_duration_delta_s = None
        if self.last_audio_recording_metadata:
            try:
                audio_duration_value = float(
                    self.last_audio_recording_metadata.get("duration_seconds", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                audio_duration_value = 0.0
            if audio_duration_value > 0.0:
                audio_duration_s = audio_duration_value
                if audio_duration_reference_s is not None:
                    audio_duration_delta_s = audio_duration_s - float(audio_duration_reference_s)

        warnings = build_recording_timing_warnings(
            requested_duration_s=requested_duration_s if requested_duration_s > 0 else None,
            capture_duration_s=capture_duration_s,
            encoded_fps=encoded_fps,
            measured_fps=measured_fps,
            audio_duration_s=audio_duration_s,
            encoded_video_duration_s=encoded_video_duration_s,
        )

        audit.update(
            {
                "recorded_frames": int(recorded_frames),
                "video_first_frame_wallclock": self.recording_first_frame_wallclock,
                "video_last_frame_wallclock": self.recording_last_frame_wallclock,
                "capture_duration_seconds": capture_duration_s,
                "capture_duration_delta_seconds": duration_delta_s,
                "encoded_output_fps": encoded_fps,
                "encoded_video_duration_seconds": encoded_video_duration_s,
                "measured_capture_fps": measured_fps,
                "encoded_vs_measured_fps_delta_pct": fps_delta_pct,
                "audio_duration_seconds": audio_duration_s,
                "audio_duration_reference_seconds": audio_duration_reference_s,
                "audio_duration_reference_type": audio_duration_reference_type,
                "audio_duration_delta_seconds": audio_duration_delta_s,
                "warnings": warnings,
            }
        )
        return audit

    def _save_metadata_to_file(self, folder: str):
        """Save metadata to JSON file."""
        self._collect_metadata()
        metadata_file = Path(folder) / f"{self.edit_filename.text()}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def _save_recording_json_metadata(self, base_path: str):
        roi_metadata = self._live_roi_metadata_payload()
        metadata = dict(self.metadata) if self.metadata else self._collect_metadata()
        metadata["live_roi_count"] = len(roi_metadata)
        metadata["live_rois"] = roi_metadata
        metadata["recording_output_folder"] = str(Path(base_path).parent)
        metadata["recording_base_path"] = str(base_path)
        if self.last_recording_timing_audit:
            metadata["recording_timing_audit"] = dict(self.last_recording_timing_audit)
        if self.last_audio_recording_metadata:
            metadata["audio_recording"] = dict(self.last_audio_recording_metadata)
        self.metadata = metadata
        metadata_file = Path(f"{base_path}_metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=4)

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
        timing_audit = dict(self.last_recording_timing_audit)
        audio_metadata = dict(self.last_audio_recording_metadata)
        roi_metadata = self._live_roi_metadata_payload()
        metadata["live_roi_count"] = len(roi_metadata)
        metadata["live_rois"] = roi_metadata
        self.metadata = metadata

        try:
            lines = [
                f"{APP_NAME} Metadata Summary",
                f"generated_at: {datetime.now().isoformat()}",
                f"recording_base_path: {base_path}",
                f"overlay_video_path: {base_path}_overlay.mp4" if Path(f"{base_path}_overlay.mp4").exists() else "overlay_video_path:",
                "",
                "Session Metadata",
            ]

            for key, value in metadata.items():
                if key in {"live_roi_count", "live_rois"}:
                    continue
                value_text = "" if value is None else str(value)
                value_lines = value_text.splitlines() or [""]
                if len(value_lines) == 1:
                    lines.append(f"{key}: {value_lines[0]}")
                else:
                    lines.append(f"{key}:")
                    for value_line in value_lines:
                        lines.append(f"  {value_line}")

            lines.extend(["", "Live Behavioural ROIs", f"count: {len(roi_metadata)}"])
            if not roi_metadata:
                lines.append("none")
            else:
                for entry in roi_metadata:
                    name = str(entry.get("name", "ROI"))
                    roi_type = str(entry.get("roi_type", "unknown"))
                    properties_text = str(entry.get("properties_text", ""))
                    lines.append(f"{name}: {roi_type} | {properties_text}")

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

            lines.extend(["", "Recording Timing Audit"])
            if timing_audit:
                warnings = list(timing_audit.get("warnings", []) or [])
                for key, value in timing_audit.items():
                    if key == "warnings":
                        continue
                    lines.append(f"{key}: {value}")
                if warnings:
                    for warning in warnings:
                        lines.append(f"warning: {warning}")
            else:
                lines.append("not available")

            lines.extend(["", "Ultrasound Audio"])
            if audio_metadata:
                streams = audio_metadata.get("streams")
                for key, value in audio_metadata.items():
                    if key == "streams":
                        continue
                    lines.append(f"{key}: {value}")
                if isinstance(streams, list) and streams:
                    lines.append(f"stream_files: {len(streams)}")
                    for stream in streams:
                        if not isinstance(stream, dict):
                            continue
                        name = Path(str(stream.get("path", ""))).name or "?"
                        dur = float(stream.get("duration_seconds", 0.0) or 0.0)
                        sr = int(stream.get("samplerate", 0) or 0)
                        lines.append(f"  {name}: {dur:.2f}s · {sr} Hz")
            else:
                lines.append("not available")

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
                self.connected_camera_info = dict(camera_info)
                self._startup_camera_autoconnect_attempts = 0
                self.btn_connect.setText("Disconnect Camera")
                self._set_button_icon(self.btn_connect, "record", "#ffffff", "dangerButton")
                self.btn_record.setEnabled(True)
                for key, value in saved_camera_settings_from_info(camera_info).items():
                    self._save_ui_setting(key, value)
                camera_name = self.combo_camera.currentText().strip() or "Camera"
                self.label_camera_source_hint.setText(f"Connected: {camera_name}")
                self.label_recording_camera_hint.setText(f"Ready to record from {camera_name}.")
                self._update_live_header(
                    status_text=f"{camera_info.get('type', 'camera').upper()} online",
                    roi_text=self._current_camera_roi_text(),
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
                self._sync_connected_camera_line_labels()
                self.worker.set_roi(dict(self.roi_rect) if isinstance(self.roi_rect, dict) else None)

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
        self.connected_camera_info = None
        self._refresh_camera_line_selector_display_names([])
        self.btn_connect.setText("Connect Camera")
        self._set_button_icon(self.btn_connect, "play", "#eef6ff")
        self.btn_record.setEnabled(False)
        self.label_camera_source_hint.setText("No source connected")
        self.label_recording_camera_hint.setText("Camera source is managed from the left Camera panel.")

        self.roi_draw_mode = False
        self._remove_camera_roi_item()
        self._show_live_placeholder("Camera Disconnected", "Reconnect a Basler, FLIR, or USB source")
        self._update_live_header(
            status_text="No camera connected",
            resolution_text="-- x --",
            badge_text="Offline",
            badge_tone="warning",
        )
        self._refresh_behavior_panel_visibility()
        self._rebuild_monitor_visuals(reset_plot=False)
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
                planner_row = self._planner_row_for_recording_start()
                if planner_row is None:
                    return
                if 0 <= planner_row < self.planner_table.rowCount():
                    # The chosen planner row defines the trial that is about to be recorded.
                    self._load_planner_row_into_metadata(
                        planner_row,
                        announce=False,
                        apply_duration=True,
                        clear_filename_override=True,
                    )

            filename = self._compose_recording_basename()

            if not filename:
                self._on_error_occurred("Please enter metadata or filename")
                return

            try:
                resolved_path = self._resolve_recording_output_path(filename)
            except Exception as exc:
                self._on_error_occurred(f"Could not prepare recording folder: {str(exc)}")
                return
            if resolved_path is None:
                return
            filepath = str(resolved_path)
            self.current_recording_filepath = filepath
            self.edit_filename.setText(Path(filepath).name)
            self._update_filename_preview()
            self.active_planner_row = self._find_planner_row_for_current_session()
            self._sync_active_trial_status("Acquiring")
            if self.active_planner_row is not None:
                self._set_planner_row_recording_base_path(self.active_planner_row, filepath)

            self.last_audio_recording_metadata = {}
            self.last_recording_timing_audit = {}
            self.active_recording_timing_audit = {}
            self.recording_first_frame_wallclock = None
            self.recording_last_frame_wallclock = None
            self.recording_stop_requested_at = None
            self.recording_stop_reason = ""
            self.recording_start_anchor_locked = False
            self._audio_video_start_marked = False
            self.user_flag_events = []

            self._collect_metadata()
            self._save_recording_json_metadata(filepath)

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
            self._reset_live_recording_exports()
            self._reset_frame_drop_display(recording_active=True)

            requested_duration_s = self._get_max_record_seconds()
            self.active_recording_timing_audit = {
                "requested_duration_seconds": int(requested_duration_s),
                "audio_enabled": bool(self.audio_panel is not None and self.audio_panel.is_enabled()),
                "arduino_connected": bool(self.is_arduino_connected),
                "start_sequence": ["audio_prepare", "arduino_start", "video_start"],
            }

            audio_armed = False
            audio_prepare_completed_wallclock = None
            if self.audio_panel is not None and self.audio_panel.is_enabled():
                wav_path = f"{filepath}.wav"
                audio_prepare_started_wallclock = time.time()
                audio_prepare_started_perf = time.perf_counter()
                audio_armed = self.audio_panel.prepare_for_recording(wav_path)
                audio_prepare_completed_wallclock = time.time()
                self.active_recording_timing_audit["audio_prepare_started_wallclock"] = float(
                    audio_prepare_started_wallclock
                )
                self.active_recording_timing_audit["audio_prepare_call_ms"] = round(
                    (time.perf_counter() - audio_prepare_started_perf) * 1000.0,
                    3,
                )
                self.active_recording_timing_audit["audio_prepare_completed_wallclock"] = float(
                    audio_prepare_completed_wallclock
                )
                self.active_recording_timing_audit["audio_armed"] = bool(audio_armed)
                if not audio_armed:
                    self._on_status_update(
                        "Warning: ultrasound audio could not be armed; video will continue without WAV."
                    )

            arduino_start_requested_wallclock = None
            if self.is_arduino_connected:
                self._apply_behavior_pin_configuration(persist=True)
                arduino_start_requested_wallclock = time.time()
                arduino_start_started_perf = time.perf_counter()
                arduino_started = self.arduino_worker.start_recording()
                self.active_recording_timing_audit["arduino_start_requested_wallclock"] = float(
                    arduino_start_requested_wallclock
                )
                self.active_recording_timing_audit["arduino_start_call_ms"] = round(
                    (time.perf_counter() - arduino_start_started_perf) * 1000.0,
                    3,
                )
                self.active_recording_timing_audit["arduino_started"] = bool(arduino_started)
                # Record the Arduino's own generation_start_time — this is the
                # wall-clock instant the gate pin (the NPX trigger) is driven
                # high. Treating this as the master clock lets NPX, audio, and
                # video be aligned post-hoc to a single reference instant.
                try:
                    gate_edge = float(getattr(self.arduino_worker, "generation_start_time", 0.0) or 0.0)
                except Exception:
                    gate_edge = 0.0
                if gate_edge > 0.0:
                    self.active_recording_timing_audit["gate_edge_wallclock"] = gate_edge
                if audio_prepare_completed_wallclock is not None:
                    self.active_recording_timing_audit["audio_prepared_before_arduino"] = bool(
                        audio_prepare_completed_wallclock <= arduino_start_requested_wallclock
                    )

                if not arduino_started:
                    self._on_status_update("Warning: Arduino TTLs failed to start; recording will continue.")
                    self._set_ttl_status("START FAILED", "warning")
                    self._set_behavior_status("IDLE", "default")
                else:
                    # Reset and clear plot for new recording
                    self._reset_ttl_plot()

                    self._set_ttl_status("RECORDING", "danger")
                    self._set_behavior_status("ARMED", "accent")

            # Auxiliary boards record per-frame inputs independently of the
            # primary board (they may be connected even when it is not).
            if self.aux_arduino_manager is not None:
                self.aux_arduino_manager.start_recording()

            # Exact-duration recording is enforced by a frame-count cap on the
            # acquisition thread (set here, applied inside start_recording), not
            # by a GUI-thread timer which can fire late under load. 0 = unlimited.
            requested_record_seconds = self._get_max_record_seconds()
            self.worker.set_recording_duration_limit(
                requested_record_seconds if requested_record_seconds > 0 else None
            )

            video_start_requested_wallclock = time.time()
            video_start_started_perf = time.perf_counter()
            video_started = self.worker.start_recording(filepath)
            self.active_recording_timing_audit["video_start_requested_wallclock"] = float(
                video_start_requested_wallclock
            )
            self.active_recording_timing_audit["video_start_call_ms"] = round(
                (time.perf_counter() - video_start_started_perf) * 1000.0,
                3,
            )
            self.active_recording_timing_audit["video_started"] = bool(video_started)
            if arduino_start_requested_wallclock is not None:
                self.active_recording_timing_audit["host_gap_arduino_to_video_start_ms"] = round(
                    (video_start_requested_wallclock - arduino_start_requested_wallclock) * 1000.0,
                    3,
                )
            if audio_prepare_completed_wallclock is not None:
                self.active_recording_timing_audit["audio_prepared_before_video"] = bool(
                    audio_prepare_completed_wallclock <= video_start_requested_wallclock
                )

            if not video_started:
                if audio_armed and self.audio_panel is not None:
                    self.audio_panel.finalize_recording()
                self._reset_frame_drop_display()
                if self.is_arduino_connected:
                    self._stop_arduino_generation()
                self._sync_active_trial_status("Pending")
                self.active_recording_timing_audit = {}
                self.current_recording_filepath = None
                return

            if self.camera_stream_manager is not None:
                aux_started = self.camera_stream_manager.start_recording_all(
                    filepath,
                    requested_record_seconds if requested_record_seconds > 0 else None,
                )
                if aux_started:
                    self.active_recording_timing_audit["aux_streams_started"] = len(aux_started)

            if self._should_save_live_overlay_video():
                self._start_live_overlay_video_recording(filepath)

            self.recording_start_time = datetime.now()
            self.recording_timer.start(1000)
            self._apply_recording_frame_limit()

            self.btn_record.setText("Stop Recording")
            self._set_button_icon(self.btn_record, "record", "#ffffff", "dangerButton")
            self.label_recording.setText("Recording")
            self.label_recording.setStyleSheet(
                "QLabel { color: #ffb3b3; font-weight: 700;"
                " background-color: #3a1717; border: 1px solid #7b2323; }"
            )
            self._update_live_header(badge_text="REC", badge_tone="danger")
            self._show_recording_overlay(True)

            self.btn_connect.setEnabled(False)
            self.edit_filename.setEnabled(False)
            if self.is_arduino_connected:
                self.btn_test_ttl.setEnabled(False)
        else:
            self._request_recording_stop("manual")

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
                QKeySequenceEdit,
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
    def _on_user_flag_shortcut(self):
        """Legacy single-shortcut entry point; trigger the first configured flag."""
        self._trigger_user_flag()

    def _trigger_user_flag(self, flag_id: str | None = None):
        """Record one user-defined manual event and optionally pulse a DO line."""
        if self._focused_widget_blocks_space_record():
            return
        config = self._find_user_flag_config(flag_id or "") if flag_id else None
        if config is None:
            config = self._current_user_flag_config()
        shortcut_text = str(config.get("shortcut", "") or "").strip()
        if not shortcut_text:
            return

        label = str(config.get("label", "User Flag") or "User Flag").strip() or "User Flag"
        output_id = str(config.get("output_id", "") or "").strip().upper()
        pulse_ms = int(config.get("pulse_ms", 100) or 100)
        recording_active = bool(self.worker is not None and getattr(self.worker, "is_recording", False))
        arduino_active = bool(self.is_arduino_connected and self.arduino_worker is not None)
        preview_active = bool(self.is_camera_connected and self.last_frame_size is not None)
        if not recording_active and not arduino_active and not preview_active:
            return

        timestamp = time.time()
        self.user_flag_preview_text = f"FLAG: {label}"
        self.user_flag_preview_until_s = time.monotonic() + max(0.9, (float(pulse_ms) / 1000.0) + 0.35)
        if recording_active:
            self.user_flag_events.append(
                {
                    "timestamp_software": timestamp,
                    "flag_id": str(config.get("flag_id", "") or "").strip(),
                    "label": label,
                    "shortcut": shortcut_text,
                    "output_id": output_id,
                    "pulse_ms": pulse_ms,
                    "count": len(self.user_flag_events) + 1,
                }
            )

        ttl_message = "metadata only"
        if output_id and arduino_active:
            mapped_pins = [int(pin) for pin in self.live_output_mapping.get(output_id, [])]
            if mapped_pins:
                self.arduino_worker.start_live_output_pulse(output_id, pulse_ms)
                ttl_message = f"{output_id} pulse {pulse_ms} ms"
            else:
                ttl_message = f"{output_id} not mapped"
        elif output_id:
            ttl_message = f"{output_id} unavailable"

        if recording_active:
            self._on_status_update(f"User flag '{label}' marked ({ttl_message})")
        elif preview_active and not arduino_active:
            self._on_status_update(f"User flag '{label}' previewed")
        elif output_id and arduino_active:
            self._on_status_update(f"User flag '{label}' pulsed ({ttl_message})")

    @Slot()
    def _on_recording_stopped(self):
        """Handle recording stopped signal."""
        self.btn_record.setText("Start Recording")
        self._set_button_icon(self.btn_record, "record", "#07260e", "successButton")
        self.label_recording.setText("Not Recording")
        self.label_recording.setStyleSheet("")
        self.label_recording_time.setText("00:00:00")
        self.recording_timer.stop()
        self.recording_duration_timer.stop()
        self._show_recording_overlay(False)
        # The primary worker may have self-stopped at its exact frame cap. Give
        # the auxiliary streams a short grace period to reach their OWN frame
        # caps (so every file gets its full duration), then force-stop any
        # straggler. Force-stopping immediately here would cut aux a few frames
        # short of their target length.
        if self.camera_stream_manager is not None and self.camera_stream_manager.any_recording():
            manager = self.camera_stream_manager
            QTimer.singleShot(2000, lambda: manager.stop_recording_all() if manager.any_recording() else None)
        self.recording_start_time = None
        self.recording_start_anchor_locked = False
        if getattr(self, "recording_progress_bar", None) is not None:
            self.recording_progress_bar.hide()
            self.recording_progress_bar.setValue(0)
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

        video_stop_wallclock = (
            self.recording_last_frame_wallclock
            or self.recording_stop_requested_at
            or time.time()
        )
        recorded_frames = 0
        encoded_video_duration_s = None
        if self.worker is not None:
            try:
                recorded_frames = int(getattr(self.worker, "frame_counter", 0) or 0)
            except Exception:
                recorded_frames = 0
            try:
                encoded_fps_value = float(
                    getattr(self.worker, "last_recording_output_fps", 0.0)
                    or getattr(self.worker, "recording_output_fps", 0.0)
                    or 0.0
                )
            except (TypeError, ValueError):
                encoded_fps_value = 0.0
            if encoded_fps_value > 0.0:
                encoded_video_duration_s = encoded_video_duration_seconds(
                    recorded_frames,
                    encoded_fps_value,
                )

        # Stop audio first so the video-stop timestamp is captured tightly.
        audio_metadata: Dict[str, object] = {}
        if self.audio_panel is not None and self.audio_panel.is_recording():
            if self.recording_first_frame_wallclock is not None and not self._audio_video_start_marked:
                self.audio_panel.notify_video_started(self.recording_first_frame_wallclock)
                self._audio_video_start_marked = True
            self.audio_panel.notify_video_stopped(video_stop_wallclock)
            try:
                audio_metadata = self.audio_panel.finalize_recording(
                    target_duration_seconds=encoded_video_duration_s,
                ) or {}
            except Exception as exc:
                self._on_status_update(f"Audio finalize error: {exc}")
        self.last_audio_recording_metadata = dict(audio_metadata) if audio_metadata else {}
        self.last_recording_timing_audit = self._finalize_recording_timing_audit()

        if filepath:
            self._stop_live_overlay_video_recording()
            self._save_recording_json_metadata(filepath)
            self._save_recording_text_metadata(filepath)
            self._save_recording_frame_csv_outputs(filepath)
        else:
            self._stop_live_overlay_video_recording()

        # Auxiliary boards: their per-frame history was merged into the frame CSV
        # above (when a filepath exists), so finalize and clear it now. This runs
        # regardless of the primary board's connection state.
        if self.aux_arduino_manager is not None:
            self.aux_arduino_manager.stop_recording()
            self.aux_arduino_manager.clear_history()

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

        frames_written = int(recorded_frames)
        if frames_written <= 0 and self.worker is not None:
            try:
                frames_written = int(getattr(self.worker, "frame_counter", 0) or 0)
            except Exception:
                frames_written = 0
        has_first_frame = self.recording_first_frame_wallclock is not None
        if frames_written > 0 and has_first_frame:
            if self.active_planner_row is not None and filepath:
                self._set_planner_row_recording_base_path(self.active_planner_row, filepath)
            self._sync_active_trial_status("Acquired")
            self._advance_to_next_planner_trial()
        else:
            # Recording was aborted before any frame was captured — keep the row
            # pending so the operator can re-run it instead of silently marking it done.
            self._sync_active_trial_status("Pending")
            self.active_planner_row = None
        self._update_active_trial_header()
        self.current_recording_filepath = None
        self.active_recording_timing_audit = {}
        self.recording_first_frame_wallclock = None
        self.recording_last_frame_wallclock = None
        self.recording_stop_requested_at = None
        self.recording_stop_reason = ""
        self._audio_video_start_marked = False
        self._update_filename_preview()

    def _update_recording_time(self):
        """Update recording time display and the live progress strip."""
        if self.recording_start_time:
            elapsed_seconds = self._current_recording_elapsed_seconds()
            elapsed_text = self._format_duration_hms(elapsed_seconds)
            remaining_seconds = self._current_recording_remaining_seconds()
            if remaining_seconds is None:
                self.label_recording_time.setText(elapsed_text)
            else:
                remaining_text = self._format_duration_hms(remaining_seconds)
                self.label_recording_time.setText(f"{elapsed_text} | {remaining_text} left")
            self._update_recording_progress_bar(elapsed_seconds)
            self._update_recording_overlay(elapsed_seconds)

    def _update_recording_progress_bar(self, elapsed_seconds: int):
        bar = getattr(self, "recording_progress_bar", None)
        if bar is None:
            return
        max_seconds = self._get_max_record_seconds()
        recording = bool(self.worker is not None and getattr(self.worker, "is_recording", False))
        if not recording or max_seconds <= 0:
            bar.hide()
            bar.setValue(0)
            return
        bar.show()
        fraction = max(0.0, min(1.0, elapsed_seconds / float(max_seconds)))
        bar.setValue(int(round(fraction * 1000)))

    def _recording_output_paths(self, base_path: Path) -> List[Path]:
        base_text = str(base_path)
        return [
            Path(f"{base_text}.mp4"),
            Path(f"{base_text}_metadata.csv"),
            Path(f"{base_text}_metadata.json"),
            Path(f"{base_text}_metadata.txt"),
            Path(f"{base_text}_ttl_states.csv"),
            Path(f"{base_text}_ttl_counts.csv"),
            Path(f"{base_text}_behavior_summary.csv"),
            Path(f"{base_text}_live_detections.csv"),
            Path(f"{base_text}_overlay.mp4"),
            Path(f"{base_text}.wav"),
        ]

    def _delete_recording_outputs(self, base_path: Path):
        """Delete the known output files for one recording base path."""
        for output_path in self._recording_output_paths(base_path):
            if output_path.exists():
                output_path.unlink()

    def _confirm_recording_overwrite(self, base_path: Path) -> bool:
        """Ask whether an existing organized-session recording should be overwritten."""
        reply = QMessageBox.question(
            self,
            "Overwrite Recording?",
            (
                "A recording with this animal/session/trial filename already exists.\n\n"
                f"{base_path.name}\n"
                f"Folder: {base_path.parent}\n\n"
                "Overwrite the existing recording files?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return False
        self._delete_recording_outputs(base_path)
        return True

    def _resolve_recording_output_path(self, base_name: str) -> Optional[Path]:
        """Resolve the base path for a new recording, including folder organization rules."""
        folder = self._recording_destination_folder()
        folder.mkdir(parents=True, exist_ok=True)
        base_name = base_name.strip() or "recording"
        candidate = folder / base_name
        if self._organize_recordings_enabled():
            if self._recording_files_exist(candidate) and not self._confirm_recording_overwrite(candidate):
                return None
            return candidate
        return self._get_unique_recording_path(folder, base_name)

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
        return any(path.exists() for path in self._recording_output_paths(base_path))

    def _planner_payload_recording_base_path(self, payload: Dict[str, str]) -> Path:
        """Resolve the generated recording base path for one planner row."""
        values = self._planner_payload_token_values(payload)
        folder = self._recording_destination_folder(values=values)
        base_name = self._compose_recording_basename(values=values, custom_override="")
        return folder / (base_name.strip() or "recording")

    def _recording_metadata_matches_planner_payload(self, metadata: Dict[str, object], payload: Dict[str, str]) -> bool:
        """Return False when saved recording metadata contradicts the planner row."""
        pairs = (
            ("Animal ID", "animal_id"),
            ("Session", "session"),
            ("Trial", "trial"),
            ("Experiment", "experiment"),
            ("Condition", "condition"),
            ("Arena", "arena"),
        )
        for planner_key, metadata_key in pairs:
            expected = str(payload.get(planner_key, "") or "").strip()
            actual = str(metadata.get(metadata_key, "") or "").strip()
            if expected and actual and expected != actual:
                return False
        return True

    def _recording_json_metadata(self, base_path: Path) -> Dict[str, object]:
        metadata_path = Path(f"{base_path}_metadata.json")
        if not metadata_path.exists():
            return {}
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _recording_csv_first_row(self, base_path: Path) -> Dict[str, object]:
        metadata_path = Path(f"{base_path}_metadata.csv")
        try:
            if not metadata_path.exists() or metadata_path.stat().st_size <= 0:
                return {}
            import csv

            with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                return dict(next(reader, {}) or {})
        except Exception:
            return {}

    def _recording_csv_has_frames(self, base_path: Path) -> bool:
        metadata_path = Path(f"{base_path}_metadata.csv")
        try:
            if not metadata_path.exists() or metadata_path.stat().st_size <= 0:
                return False
            import csv

            with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                return next(reader, None) is not None
        except Exception:
            return False

    def _recording_json_recorded_frames(self, base_path: Path) -> int:
        metadata = self._recording_json_metadata(base_path)
        audit = metadata.get("recording_timing_audit", {})
        if not isinstance(audit, dict):
            return 0
        try:
            return max(0, int(audit.get("recorded_frames", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _recording_text_recorded_frames(self, base_path: Path) -> int:
        metadata_path = Path(f"{base_path}_metadata.txt")
        try:
            if not metadata_path.exists() or metadata_path.stat().st_size <= 0:
                return 0
            for line in metadata_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("recorded_frames:"):
                    return max(0, int(line.split(":", 1)[1].strip() or "0"))
        except Exception:
            return 0
        return 0

    def _recording_base_has_frames(self, base_path: Path) -> bool:
        """Return True only for a base path that looks like a real acquisition."""
        if self._recording_csv_has_frames(base_path):
            return True
        if self._recording_json_recorded_frames(base_path) > 0:
            return True
        if self._recording_text_recorded_frames(base_path) > 0:
            return True
        video_path = Path(f"{base_path}.mp4")
        try:
            return video_path.exists() and video_path.stat().st_size > 4096
        except Exception:
            return False

    def _recording_base_matches_planner_payload(self, base_path: Path, payload: Dict[str, str]) -> bool:
        """Check saved metadata against a planner row before accepting an output file."""
        json_metadata = self._recording_json_metadata(base_path)
        if json_metadata and not self._recording_metadata_matches_planner_payload(json_metadata, payload):
            return False

        csv_row = self._recording_csv_first_row(base_path)
        if csv_row and not self._recording_metadata_matches_planner_payload(csv_row, payload):
            return False

        return True

    def _planner_recording_base_candidates(self, payload: Dict[str, str]) -> List[Path]:
        """Return exact and numbered output bases that could belong to a planner row."""
        expected = self._planner_payload_recording_base_path(payload)
        candidates = [expected]
        folder = expected.parent
        stem = expected.name
        suffixes = (
            ".mp4",
            "_metadata.csv",
            "_metadata.json",
            "_metadata.txt",
        )
        try:
            if folder.exists() and folder.is_dir():
                for path in folder.iterdir():
                    name = path.name
                    for suffix in suffixes:
                        if not name.endswith(suffix):
                            continue
                        base_name = name[: -len(suffix)]
                        if base_name == stem or base_name.startswith(f"{stem}_"):
                            candidates.append(folder / base_name)
        except Exception:
            pass

        unique: List[Path] = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _planner_row_has_recording(self, row: int) -> bool:
        """Return True when the planner row has a matching completed recording on disk."""
        if self.planner_table is None or row < 0 or row >= self.planner_table.rowCount():
            return False
        payload = self._planner_row_payload(row)
        candidates: List[Path] = []
        stored_base = self._planner_row_recording_base_path(row)
        if stored_base:
            candidates.append(Path(stored_base))
        if row == self.active_planner_row and self.current_recording_filepath:
            candidates.append(Path(str(self.current_recording_filepath)))
        candidates.extend(self._planner_recording_base_candidates(payload))

        seen = set()
        for base_path in candidates:
            key = str(base_path)
            if key in seen:
                continue
            seen.add(key)
            if not self._recording_base_has_frames(base_path):
                continue
            if self._recording_base_matches_planner_payload(base_path, payload):
                self._set_planner_row_recording_base_path(row, str(base_path))
                return True
        return False

    def _sync_planner_recording_statuses(self) -> bool:
        """Reconcile planner Status cells with actual recording files."""
        if self.planner_table is None or self._syncing_planner_recording_statuses:
            return False

        recording_active = bool(self.worker is not None and getattr(self.worker, "is_recording", False))
        changed = False
        self._syncing_planner_recording_statuses = True
        try:
            for row in range(self.planner_table.rowCount()):
                payload = self._planner_row_payload(row)
                current_status = self._normalize_planner_status(payload.get("Status", "Pending"))
                active_row = row == self.active_planner_row
                recording_starting = active_row and bool(self.current_recording_filepath)
                if current_status == "Acquiring" and (recording_active or recording_starting):
                    continue
                if current_status == "Pending" and self._planner_row_manual_pending(row):
                    continue
                if current_status == "Acquired" and self._planner_row_manual_acquired(row):
                    continue

                has_recording = self._planner_row_has_recording(row)
                next_status = "Acquired" if has_recording else "Pending"
                if current_status != next_status:
                    self._set_planner_row_status(row, next_status)
                    changed = True
        finally:
            self._syncing_planner_recording_statuses = False

        return changed

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
            self._update_filename_preview()
            self._update_planner_summary()

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
        allow_paused_spinnaker_write = bool(
            self.worker
            and self.worker.is_spinnaker_camera()
            and node_name in {"OffsetX", "OffsetY"}
        )
        if self.worker and not self.worker._node_is_writable(node) and not allow_paused_spinnaker_write:
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
            return None
        try:
            if node_name in {"OffsetX", "OffsetY"}:
                applied = self.worker.set_camera_offset(node_name, int(value))
            else:
                applied = self.worker._write_numeric_node(node_name, value, integer=True)
        except Exception as exc:
            self._on_error_occurred(f"Failed to set {node_name}: {exc}")
            return None
        if applied is None:
            self._on_error_occurred(f"Failed to set {node_name}: unsupported by camera")
            return None
        return int(applied)

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
        applied = self._set_camera_int_node("OffsetX", value)
        if applied is not None and int(applied) != int(value):
            self._sync_offset_controls(int(applied), self.slider_offset_x, self.spin_offset_x)
        self._save_ui_setting('offset_x', int(applied if applied is not None else value))

    def _on_offset_y_changed(self, value: int):
        self._sync_offset_controls(value, self.slider_offset_y, self.spin_offset_y)
        applied = self._set_camera_int_node("OffsetY", value)
        if applied is not None and int(applied) != int(value):
            self._sync_offset_controls(int(applied), self.slider_offset_y, self.spin_offset_y)
        self._save_ui_setting('offset_y', int(applied if applied is not None else value))

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

        if not self.roi_draw_mode:
            if self.roi_item is None:
                geometry = self._camera_roi_geometry_for_frame()
                if geometry is None:
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
                    self.live_image_view.getView().addItem(self.roi_item)
                else:
                    roi_x, roi_y, roi_w, roi_h = geometry
                    self._sync_camera_roi_overlay()
                    if self.roi_item is None:
                        return
                    self.roi_item.setPos([roi_x, roi_y], finish=False)
                    self.roi_item.setSize([roi_w, roi_h], finish=False)
            self.roi_draw_mode = True
            self.btn_draw_roi.setText("Apply ROI")
            self.btn_draw_roi.setStyleSheet("QPushButton { background-color: #f59e0b; color: white; font-weight: bold; }")
            self.roi_item.setPen(pg.mkPen("#f59e0b", width=2))
            self._update_live_header(roi_text="Editing ROI")
            return

        if self.roi_item is not None:
            pos = self.roi_item.pos()
            size = self.roi_item.size()
            # Scale from preview pixel space to camera resolution
            sx, sy = self._roi_preview_to_camera_scale()
            self.roi_rect = {
                "x": max(0, int(round(pos.x() * sx))),
                "y": max(0, int(round(pos.y() * sy))),
                "w": max(1, int(round(size.x() * sx))),
                "h": max(1, int(round(size.y() * sy))),
            }
            if self.worker:
                self.worker.set_roi(self.roi_rect)
            self._persist_camera_roi_setting()
            self.roi_item.setPen(pg.mkPen("#22c55e", width=2))
            self._update_live_header(roi_text=f"ROI {self.roi_rect['w']} x {self.roi_rect['h']}")

        self.roi_draw_mode = False
        self.btn_draw_roi.setText("Edit ROI")
        self.btn_draw_roi.setStyleSheet("")

    def _clear_roi(self):
        """Clear ROI and reset cropping."""
        self.roi_rect = None
        self.roi_draw_mode = False
        self._remove_camera_roi_item()
        self._sync_camera_roi_ui_state()
        if self.worker:
            self.worker.set_roi(None)
        self._persist_camera_roi_setting()

    def _live_view_box(self):
        if self.live_image_view is None:
            return None
        view = self.live_image_view.getView()
        if hasattr(view, "mapSceneToView"):
            return view
        if hasattr(view, "getViewBox"):
            return view.getViewBox()
        return getattr(view, "vb", None)

    def _live_roi_graphics_parent(self):
        if self.live_image_view is None:
            return None
        view = self.live_image_view.getView()
        return view if hasattr(view, "addItem") else self._live_view_box()

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
        self._sync_live_circle_roi_items()
        self._on_status_update(
            f"Drawing {self.live_roi_draw_mode or 'roi'} '{self.live_roi_drawing_name}'. "
            "Use left-clicks on the live view; right-click closes polygons."
        )

    @Slot(str)
    def _center_live_circle_roi(self, roi_name: str):
        if self.live_image_view is None or self.last_frame_size is None:
            self._on_error_occurred("Preview a live frame before centering a circle ROI.")
            return

        name = str(roi_name or "").strip()
        if not name and self.live_detection_panel is not None:
            name = self.live_detection_panel.current_roi_name()
        name = name or f"ROI {len(self.live_rois) + 1}"

        width, height = self.last_frame_size
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        radius = max(8.0, min(float(width), float(height)) * 0.15)
        color = self._live_roi_color(len(self.live_rois))

        existing_roi = self.live_rois.get(name)
        if existing_roi is not None and existing_roi.roi_type != "circle":
            edit_name = (
                self.live_detection_panel.current_roi_name()
                if self.live_detection_panel is not None
                else ""
            )
            if edit_name and edit_name != name and edit_name not in self.live_rois:
                name = edit_name
                existing_roi = None
                color = self._live_roi_color(len(self.live_rois))
            else:
                self._on_error_occurred("Select a circle ROI, or enter a new circle ROI name.")
                return

        if existing_roi is not None:
            color = existing_roi.color
            if existing_roi.data:
                try:
                    radius = max(1.0, float(existing_roi.data[0][2]))
                except Exception:
                    pass

        self.live_rois[name] = BehaviorROI(
            name=name,
            roi_type="circle",
            data=[(cx, cy, radius)],
            color=color,
        )
        self.live_roi_draw_mode = ""
        self.live_roi_draw_points = []
        self.live_roi_circle_center = None
        self.live_rule_engine.set_rois(self.live_rois)
        self._refresh_live_panel_state()
        self._persist_live_detection_settings()
        self._on_status_update(f"Centered circle ROI '{name}' at the field-of-view center.")

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

    def _roi_geometry_spinbox(
        self,
        value: float,
        minimum: float = -1000000.0,
        maximum: float = 1000000.0,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        spin.setValue(float(value))
        return spin

    def _parse_polygon_points_text(self, raw_text: str) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for line in str(raw_text or "").splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            parts = [part for part in re.split(r"[,\s]+", cleaned) if part]
            if len(parts) < 2:
                raise ValueError("Each polygon point must contain x and y values.")
            points.append((float(parts[0]), float(parts[1])))
        if len(points) < 3:
            raise ValueError("Polygon ROIs need at least three points.")
        return points

    @Slot(str)
    def _edit_live_roi(self, roi_name: str):
        old_name = str(roi_name or "").strip()
        roi = self.live_rois.get(old_name)
        if roi is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit ROI - {old_name}")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        layout.addLayout(form)

        edit_name = QLineEdit(roi.name)
        form.addRow("Name:", edit_name)

        properties = roi_geometry_properties(roi)
        controls: dict[str, object] = {}
        if roi.roi_type == "circle":
            controls["center_x"] = self._roi_geometry_spinbox(float(properties.get("center_x", 0.0)))
            controls["center_y"] = self._roi_geometry_spinbox(float(properties.get("center_y", 0.0)))
            controls["diameter"] = self._roi_geometry_spinbox(float(properties.get("diameter", 2.0)), 2.0)
            form.addRow("Center X:", controls["center_x"])
            form.addRow("Center Y:", controls["center_y"])
            form.addRow("Diameter:", controls["diameter"])
        elif roi.roi_type == "rectangle":
            controls["x"] = self._roi_geometry_spinbox(float(properties.get("x", 0.0)))
            controls["y"] = self._roi_geometry_spinbox(float(properties.get("y", 0.0)))
            controls["width"] = self._roi_geometry_spinbox(float(properties.get("width", 1.0)), 1.0)
            controls["height"] = self._roi_geometry_spinbox(float(properties.get("height", 1.0)), 1.0)
            form.addRow("X:", controls["x"])
            form.addRow("Y:", controls["y"])
            form.addRow("Width (w):", controls["width"])
            form.addRow("Length/height (l):", controls["height"])
        elif roi.roi_type == "polygon":
            points = properties.get("points", [])
            points_edit = QTextEdit()
            points_edit.setAcceptRichText(False)
            points_edit.setMinimumHeight(140)
            points_edit.setPlaceholderText("One point per line: x, y")
            points_edit.setPlainText(
                "\n".join(f"{float(px):.2f}, {float(py):.2f}" for px, py in points)
            )
            controls["points"] = points_edit
            form.addRow("Points:", points_edit)
        else:
            QMessageBox.warning(self, "Edit ROI", f"Unsupported ROI type: {roi.roi_type}")
            return

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        new_name = edit_name.text().strip() or old_name
        if new_name != old_name and new_name in self.live_rois:
            QMessageBox.warning(self, "Edit ROI", f"An ROI named '{new_name}' already exists.")
            return

        try:
            if roi.roi_type == "circle":
                cx = float(controls["center_x"].value())
                cy = float(controls["center_y"].value())
                diameter = max(2.0, float(controls["diameter"].value()))
                data = [(cx, cy, diameter / 2.0)]
            elif roi.roi_type == "rectangle":
                x = float(controls["x"].value())
                y = float(controls["y"].value())
                width = max(1.0, float(controls["width"].value()))
                height = max(1.0, float(controls["height"].value()))
                data = [(x, y, x + width, y + height)]
            else:
                data = self._parse_polygon_points_text(controls["points"].toPlainText())
        except Exception as exc:
            QMessageBox.warning(self, "Edit ROI", str(exc))
            return

        if new_name != old_name:
            self.live_rois.pop(old_name, None)
            self._remove_live_circle_roi_item(old_name)
            for rule in self.live_rules:
                if rule.rule_type == "roi_occupancy" and rule.roi_name == old_name:
                    rule.roi_name = new_name

        self.live_rois[new_name] = BehaviorROI(
            name=new_name,
            roi_type=roi.roi_type,
            data=data,
            color=roi.color,
        )
        if self.live_detection_panel is not None:
            self.live_detection_panel.edit_roi_name.setText(new_name)
        self.live_rule_engine.set_rois(self.live_rois)
        self.live_rule_engine.set_rules(self.live_rules)
        self._refresh_live_panel_state()
        self._persist_live_detection_settings()
        self._on_status_update(f"Updated ROI '{new_name}': {format_roi_properties(self.live_rois[new_name])}")

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

    def _current_occupied_live_roi_names(self) -> set[str]:
        if not self._live_roi_overlays_visible():
            return set()
        return occupied_roi_names(self.live_rois, self._current_live_overlay_result())

    def _live_roi_overlays_visible(self) -> bool:
        preview_visible = bool(self.is_camera_connected and self.last_frame_size is not None)
        return bool(preview_visible or self.live_detection_enabled or self.live_roi_draw_mode)

    def _live_roi_preview_color(
        self,
        roi_name: str,
        roi: BehaviorROI,
        occupied_names: set[str],
    ) -> tuple[int, int, int]:
        return self.LIVE_ROI_OCCUPIED_COLOR if str(roi_name) in occupied_names else roi.color

    def _live_roi_preview_line_width(self, roi_name: str, occupied_names: set[str]) -> int:
        return 3 if str(roi_name) in occupied_names else 2

    def _sync_live_circle_roi_items(self):
        if not self._live_roi_overlays_visible():
            for roi_name in list(self.live_circle_roi_items.keys()):
                self._remove_live_circle_roi_item(roi_name)
            return
        occupied_names = self._current_occupied_live_roi_names()
        circle_names = {
            roi_name
            for roi_name, roi in self.live_rois.items()
            if roi.roi_type == "circle" and bool(roi.data)
        }
        for roi_name in list(self.live_circle_roi_items.keys()):
            if roi_name not in circle_names:
                self._remove_live_circle_roi_item(roi_name)
        for roi_name in circle_names:
            self._sync_live_circle_roi_item(roi_name, self.live_rois[roi_name], occupied_names)

    def _sync_live_circle_roi_item(self, roi_name: str, roi: BehaviorROI, occupied_names: set[str]):
        parent = self._live_roi_graphics_parent()
        if parent is None or not roi.data:
            return
        try:
            cx, cy, radius = roi.data[0]
            cx = float(cx)
            cy = float(cy)
            radius = max(1.0, float(radius))
        except Exception:
            return

        diameter = radius * 2.0
        self._syncing_live_circle_roi_item = True
        try:
            item = self.live_circle_roi_items.get(roi_name)
            if item is None:
                item = pg.CircleROI(
                    [cx - radius, cy - radius],
                    [diameter, diameter],
                    pen=pg.mkPen(
                        self._live_roi_preview_color(roi_name, roi, occupied_names),
                        width=self._live_roi_preview_line_width(roi_name, occupied_names),
                    ),
                    movable=True,
                    resizable=True,
                )
                item.sigRegionChangeFinished.connect(
                    lambda *args, roi_name=roi_name: self._on_live_circle_roi_item_changed(roi_name)
                )
                parent.addItem(item)
                self.live_circle_roi_items[roi_name] = item
            else:
                item.setPen(
                    pg.mkPen(
                        self._live_roi_preview_color(roi_name, roi, occupied_names),
                        width=self._live_roi_preview_line_width(roi_name, occupied_names),
                    )
                )
                item.setPos([cx - radius, cy - radius])
                item.setSize([diameter, diameter])
        finally:
            self._syncing_live_circle_roi_item = False

    def _update_live_circle_roi_item_pens(self, occupied_names: set[str]):
        for roi_name, item in self.live_circle_roi_items.items():
            roi = self.live_rois.get(roi_name)
            if roi is None:
                continue
            item.setPen(
                pg.mkPen(
                    self._live_roi_preview_color(roi_name, roi, occupied_names),
                    width=self._live_roi_preview_line_width(roi_name, occupied_names),
                )
            )

    def _remove_live_circle_roi_item(self, roi_name: str):
        item = self.live_circle_roi_items.pop(str(roi_name), None)
        if item is None:
            return
        try:
            parent = self._live_roi_graphics_parent()
            if parent is not None and hasattr(parent, "removeItem"):
                parent.removeItem(item)
                return
            view_box = item.getViewBox() if hasattr(item, "getViewBox") else None
            if view_box is not None:
                view_box.removeItem(item)
        except Exception:
            pass

    def _on_live_circle_roi_item_changed(self, roi_name: str):
        if self._syncing_live_circle_roi_item:
            return
        roi = self.live_rois.get(str(roi_name))
        item = self.live_circle_roi_items.get(str(roi_name))
        if roi is None or item is None or roi.roi_type != "circle":
            return

        pos = item.pos()
        size = item.size()
        diameter = max(2.0, float(size.x()), float(size.y()))
        radius = diameter / 2.0
        cx = float(pos.x()) + float(size.x()) / 2.0
        cy = float(pos.y()) + float(size.y()) / 2.0
        roi.data = [(cx, cy, radius)]
        self.live_rule_engine.set_rois(self.live_rois)
        self._persist_live_detection_settings()

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
        view_box = self._live_view_box()
        if view_box is None or not hasattr(view_box, "mapSceneToView"):
            return None

        point = view_box.mapSceneToView(scene_pos)
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
        if (
            self.planner_table is not None
            and obj is self.planner_table.viewport()
            and event.type() == QEvent.Resize
        ):
            self._schedule_planner_column_fit()
        if (
            self.recording_overlay is not None
            and obj is self.live_image_view
            and event.type() == QEvent.Resize
        ):
            self._position_recording_overlay()
        return super().eventFilter(obj, event)

    # ===== Arduino Slots =====

    def _scan_arduino_ports(self):
        """Scan for Arduino ports."""
        self.arduino_worker.scan_ports()

    @Slot(list)
    def _on_port_list_updated(self, ports):
        """Update port list."""
        self.combo_arduino_port.clear()
        if not ports:
            self.combo_arduino_port.addItem("No ports found — check the USB cable")
            self._on_status_update("Arduino scan: no serial ports found.")
            return
        self.combo_arduino_port.addItems(ports)
        self._on_status_update(f"Arduino scan: {len(ports)} port(s) found.")

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
            port = self.combo_arduino_port.currentText().strip()
            if not port or port.startswith("No ports"):
                self._on_error_occurred(
                    "No Arduino port selected. Plug the board in, press Scan, and pick its COM port."
                )
                return

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
            self.latest_ttl_states = {}
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
            self._set_status_chip(self.label_arduino_status, str(message), "success")
        else:
            self._set_status_chip(self.label_arduino_status, str(message), "danger")
            if self.is_arduino_connected:
                self.is_arduino_connected = False
                self.is_testing_ttl = False
                self.latest_ttl_states = {}
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
        self._update_user_flag_pin_summary()

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
        self.latest_ttl_states = dict(states or {})
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

        # Update hardware barcode status strip
        self._update_hw_barcode_display(states)

    def _update_camera_line_plot_from_metadata(self, metadata: Optional[Dict[str, object]]):
        """Plot camera chunk line states from per-frame metadata."""
        if not metadata:
            return

        active_line_keys = self._active_camera_line_keys()
        if not active_line_keys:
            return

        line_values = {key: metadata.get(key) for key in active_line_keys}
        if all(value is None for value in line_values.values()):
            return

        try:
            timestamp_value = float(metadata.get("timestamp_software", time.time()) or time.time())
        except (TypeError, ValueError):
            timestamp_value = time.time()

        signature = (metadata.get("frame_id", None), round(timestamp_value, 6))
        if signature == self.camera_line_last_signature:
            return
        self.camera_line_last_signature = signature

        if self.camera_line_plot_start_time_s is None or timestamp_value < self.camera_line_plot_start_time_s:
            self.camera_line_plot_start_time_s = timestamp_value
            self.camera_line_time_data.clear()
            for series in self.camera_line_plot_data.values():
                series.clear()

        current_time = max(0.0, timestamp_value - float(self.camera_line_plot_start_time_s or timestamp_value))
        self.camera_line_time_data.append(current_time)

        amplitude = 0.35
        for key in active_line_keys:
            level = self.camera_line_levels.get(key, 0.0)
            state = bool(line_values.get(key, False))
            self.camera_line_plot_data[key].append(level + amplitude if state else level - amplitude)

        # Throttle the pyqtgraph repaint: data is still fully buffered in
        # self.camera_line_plot_data above, so skipping a paint here only
        # delays the visual refresh — the recording stream is unaffected.
        now_monotonic = time.monotonic()
        if (now_monotonic - self._camera_line_plot_last_paint_s) < self._camera_line_plot_min_interval_s:
            return
        self._camera_line_plot_last_paint_s = now_monotonic

        times = np.fromiter(self.camera_line_time_data, dtype=float)
        if times.size == 0:
            return
        if times.size == 1:
            step = 0.03
        else:
            step = max(0.01, times[-1] - times[-2])
        times_step = np.append(times, times[-1] + step)

        for key, curve in self.camera_line_curves.items():
            curve.setData(times_step, np.fromiter(self.camera_line_plot_data[key], dtype=float))

        end_time = times[-1]
        start_time = max(0.0, end_time - self.ttl_window_seconds)
        self.camera_line_plot.setXRange(start_time, end_time)

    def _update_hw_barcode_display(self, states: dict):
        """Refresh the hardware barcode status strip in the TTL monitor panel."""
        hw_enabled = bool(states.get("hw_barcode_enabled", False))
        if self.hw_barcode_frame is not None:
            self.hw_barcode_frame.setVisible(hw_enabled)
        if not hw_enabled:
            return

        hw_running = bool(states.get("hw_barcode_running", False))
        hw_status = states.get("hw_barcode_status")

        if self.label_hw_barcode_mode is not None:
            if hw_running:
                self.label_hw_barcode_mode.setText("RUNNING")
                self.label_hw_barcode_mode.setStyleSheet("color: #7ef0ac; font-weight: 700;")
            else:
                self.label_hw_barcode_mode.setText("STOPPED")
                self.label_hw_barcode_mode.setStyleSheet("color: #8fa6bf;")

        if hw_status:
            counter = hw_status.get("counter", 0)
            bit_idx = hw_status.get("bit_index", 0)
            bit_width = hw_status.get("bit_width_ms", 0)
            if self.label_hw_barcode_counter is not None:
                self.label_hw_barcode_counter.setText(f"Counter: {counter}")
            if self.label_hw_barcode_bit is not None:
                self.label_hw_barcode_bit.setText(f"Bit: {bit_idx}/32")
            if self.label_hw_barcode_bitwidth is not None:
                self.label_hw_barcode_bitwidth.setText(f"Width: {bit_width} ms")
        else:
            if self.label_hw_barcode_counter is not None:
                self.label_hw_barcode_counter.setText("Counter: —")
            if self.label_hw_barcode_bit is not None:
                self.label_hw_barcode_bit.setText("Bit: —")
            if self.label_hw_barcode_bitwidth is not None:
                self.label_hw_barcode_bitwidth.setText("Width: — ms")

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
        timestamp_value = None
        try:
            timestamp_value = float(frame_metadata.get("timestamp_software"))
        except (TypeError, ValueError):
            timestamp_value = None

        if timestamp_value is not None:
            if self.recording_first_frame_wallclock is None:
                self.recording_first_frame_wallclock = timestamp_value
                if not self.recording_start_anchor_locked:
                    try:
                        self.recording_start_time = datetime.fromtimestamp(timestamp_value)
                    except Exception:
                        self.recording_start_time = datetime.now()
                    self.recording_start_anchor_locked = True
                    if self.active_recording_timing_audit:
                        start_requested = self.active_recording_timing_audit.get(
                            "video_start_requested_wallclock"
                        )
                        if start_requested is not None:
                            self.active_recording_timing_audit["duration_timer_reanchored_ms"] = round(
                                (timestamp_value - float(start_requested)) * 1000.0,
                                3,
                            )
                    self._restart_recording_duration_timer()
                    self._update_recording_time()
                if self.active_recording_timing_audit:
                    self.active_recording_timing_audit["video_first_frame_wallclock"] = float(timestamp_value)
                    video_started_wallclock = self.active_recording_timing_audit.get(
                        "video_start_requested_wallclock"
                    )
                    if video_started_wallclock is not None:
                        self.active_recording_timing_audit["video_start_to_first_frame_ms"] = round(
                            (timestamp_value - float(video_started_wallclock)) * 1000.0,
                            3,
                        )
                    arduino_started_wallclock = self.active_recording_timing_audit.get(
                        "arduino_start_requested_wallclock"
                    )
                    if arduino_started_wallclock is not None:
                        self.active_recording_timing_audit["arduino_to_first_frame_ms"] = round(
                            (timestamp_value - float(arduino_started_wallclock)) * 1000.0,
                            3,
                        )
                if self.audio_panel is not None and self.audio_panel.is_recording() and not self._audio_video_start_marked:
                    self.audio_panel.notify_video_started(timestamp_value)
                    self._audio_video_start_marked = True
            self.recording_last_frame_wallclock = timestamp_value

        self._update_camera_line_plot_from_metadata(frame_metadata)
        if self.is_arduino_connected and self.arduino_worker.is_generating:
            # Sample TTL state for this frame
            self.arduino_worker.sample_ttl_state(frame_metadata)
        if self.aux_arduino_manager is not None:
            # Sample auxiliary board inputs in lock-step with the camera frame.
            self.aux_arduino_manager.sample_state(frame_metadata)

    def _reset_live_recording_exports(self):
        self.live_recording_detection_rows = []
        self.live_recording_frame_rows = {}
        self.live_recording_roi_states = {}
        self.live_recording_coco_images = {}
        self.live_recording_coco_annotations = []
        self.live_recording_coco_categories = {}
        self.live_recording_coco_next_annotation_id = 1
        self.user_flag_events = []
        self.live_overlay_pending_packets.clear()
        self.live_overlay_last_written_frame_index = None

    def _should_save_live_overlay_video(self) -> bool:
        if self.live_detection_panel is None:
            return False
        return bool(self.live_detection_panel.detection_config().get("save_overlay_video", False))

    def _should_save_live_tracking_csv(self) -> bool:
        if self.live_detection_panel is None:
            return False
        return bool(self.live_detection_panel.detection_config().get("save_tracking_csv", False))

    def _should_save_live_masks_coco(self) -> bool:
        if self.live_detection_panel is None:
            return False
        return bool(self.live_detection_panel.detection_config().get("save_masks_coco", False))

    def _start_live_overlay_video_recording(self, base_path: str):
        self._stop_live_overlay_video_recording()
        self.live_overlay_video_enabled = True
        self.live_overlay_video_recorder = None
        self.live_overlay_video_path = f"{base_path}_overlay.mp4"
        self.live_overlay_pending_packets.clear()
        self.live_overlay_last_written_frame_index = None
        try:
            if self.worker is not None:
                self.worker.set_record_frame_packets_enabled(True)
            source_fps = None
            if self.worker is not None:
                source_fps = (
                    getattr(self.worker, "recording_output_fps", None)
                    or getattr(self.worker, "last_recording_output_fps", None)
                    or getattr(self.worker, "camera_reported_fps", None)
                    or getattr(self.worker, "fps_target", None)
                )
            self.live_overlay_video_fps = max(1.0, float(source_fps or self.spin_preview_fps.value()))
        except Exception:
            self.live_overlay_video_fps = 25.0
        self.live_overlay_video_recorder = OverlayVideoRecorder(
            self.live_overlay_video_path,
            float(self.live_overlay_video_fps or 25.0),
        )
        self.live_overlay_video_recorder.start()
        self._on_status_update(f"Overlay video recording: {Path(self.live_overlay_video_path).name}")

    def _record_live_overlay_frame(self, packet: object):
        if not self.live_overlay_video_enabled or self.live_overlay_video_recorder is None:
            return
        if not isinstance(packet, PreviewFramePacket):
            return
        if self.live_overlay_video_recorder.error_message:
            self._on_error_occurred(self.live_overlay_video_recorder.error_message)
            self._stop_live_overlay_video_recording()
            return
        try:
            frame_id = int(packet.frame_index)
            if not self._live_detection_overlay_visible():
                self._enqueue_live_overlay_video_frame(packet, None, repeat_count=1)
                return
            if (
                self.live_overlay_last_written_frame_index is not None
                and frame_id <= int(self.live_overlay_last_written_frame_index)
            ):
                return
            result = self.live_detection_results_by_frame.get(frame_id)
            if result is not None:
                self._enqueue_synced_live_overlay_frame(packet, result)
                return
            self.live_overlay_pending_packets[frame_id] = packet
            self._trim_live_overlay_pending_packets()
        except Exception as exc:
            self._on_error_occurred(f"Overlay video queue error: {str(exc)}")
            self._stop_live_overlay_video_recording()

    def _flush_live_overlay_frame_for_result(self, result: LiveDetectionResult):
        if not self.live_overlay_video_enabled or self.live_overlay_video_recorder is None:
            return
        frame_id = int(result.frame_index)
        packet = self.live_overlay_pending_packets.get(frame_id)
        if packet is None:
            return
        self._enqueue_synced_live_overlay_frame(packet, result)

    def _enqueue_synced_live_overlay_frame(
        self,
        packet: PreviewFramePacket,
        result: LiveDetectionResult,
    ):
        frame_id = int(result.frame_index)
        if (
            self.live_overlay_last_written_frame_index is not None
            and frame_id <= int(self.live_overlay_last_written_frame_index)
        ):
            self.live_overlay_pending_packets.pop(frame_id, None)
            return

        if self.live_overlay_last_written_frame_index is None:
            repeat_count = 1
        else:
            gap = frame_id - int(self.live_overlay_last_written_frame_index)
            repeat_count = max(1, min(30, int(gap)))
        self.live_overlay_last_written_frame_index = frame_id
        self.live_overlay_pending_packets.pop(frame_id, None)
        self._trim_live_overlay_pending_packets()
        self._enqueue_live_overlay_video_frame(packet, result, repeat_count=repeat_count)

    def _trim_live_overlay_pending_packets(self, max_pending: int = 64):
        last_written = self.live_overlay_last_written_frame_index
        if last_written is not None:
            last_written = int(last_written)
            for frame_id in list(self.live_overlay_pending_packets):
                if int(frame_id) <= last_written:
                    self.live_overlay_pending_packets.pop(frame_id, None)
        while len(self.live_overlay_pending_packets) > max(1, int(max_pending)):
            oldest_key = min(self.live_overlay_pending_packets)
            self.live_overlay_pending_packets.pop(oldest_key, None)

    def _enqueue_live_overlay_video_frame(
        self,
        packet: PreviewFramePacket,
        overlay_result: Optional[LiveDetectionResult],
        *,
        repeat_count: int,
    ):
        recorder = self.live_overlay_video_recorder
        if not self.live_overlay_video_enabled or recorder is None:
            return
        if recorder.error_message:
            self._on_error_occurred(recorder.error_message)
            self._stop_live_overlay_video_recording()
            return
        overlay_options = (
            self.live_detection_panel.overlay_options()
            if self.live_detection_panel is not None
            else {"show_masks": True, "show_boxes": True, "show_keypoints": True}
        )
        task = OverlayVideoFrameTask(
            frame_rgb=np.asarray(packet.frame),
            timestamp_s=float(packet.timestamp_s),
            overlay_result=overlay_result,
            rois=tuple(self.live_rois.values()),
            show_masks=bool(overlay_options.get("show_masks", True)),
            show_boxes=bool(overlay_options.get("show_boxes", True)),
            show_keypoints=bool(overlay_options.get("show_keypoints", True)),
            mask_opacity=clamp_mask_opacity(overlay_options.get("mask_opacity", 0.18)),
            repeat_count=max(1, int(repeat_count)),
        )
        recorder.enqueue(task)

    def _stop_live_overlay_video_recording(self):
        recorder = self.live_overlay_video_recorder
        if self.worker is not None:
            try:
                self.worker.set_record_frame_packets_enabled(False)
            except Exception:
                pass
        self.live_overlay_video_recorder = None
        self.live_overlay_video_enabled = False
        self.live_overlay_pending_packets.clear()
        self.live_overlay_last_written_frame_index = None
        if recorder is not None:
            recorder.stop()
            if recorder.error_message:
                self._on_error_occurred(recorder.error_message)
            elif self.live_overlay_video_path:
                status_message = f"Overlay video saved: {Path(self.live_overlay_video_path).name}"
                if recorder.dropped_frames > 0:
                    status_message += f" ({recorder.dropped_frames} overlay frames dropped)"
                self._on_status_update(status_message)

    def _live_roi_export_column_map(self) -> Dict[str, str]:
        """Return CSV column names for binary ROI occupancy export."""
        column_map: Dict[str, str] = {}
        used_columns: set[str] = set()
        for index, roi_name in enumerate(self.live_rois.keys(), start=1):
            slug = self._slugify_export_label(roi_name, f"roi_{index}")
            base_column = f"in_zone_roi_{slug}"
            column = base_column
            suffix = 2
            while column in used_columns:
                column = f"{base_column}_{suffix}"
                suffix += 1
            used_columns.add(column)
            column_map[str(roi_name)] = column
        return column_map

    def _live_roi_binary_export_values(self, tracked_mice: List[object]) -> Dict[str, int]:
        """
        Return one binary ROI occupancy column per ROI.

        Missing segmentation for the selected animal is treated as unknown, so
        the last ROI state is held until an animal is actually detected outside.
        """
        column_map = self._live_roi_export_column_map()
        if not column_map:
            return {}

        mice = list(tracked_mice or [])
        values: Dict[str, int] = {}
        for roi_name, column in column_map.items():
            roi = self.live_rois.get(roi_name)
            previous_state = bool(self.live_recording_roi_states.get(roi_name, False))
            occupied = previous_state
            if roi is not None and mice:
                occupied = False
                for mouse in mice:
                    try:
                        cx, cy = mouse.center
                        if roi.contains_point(float(cx), float(cy)):
                            occupied = True
                            break
                    except Exception:
                        continue
                self.live_recording_roi_states[roi_name] = occupied
            else:
                self.live_recording_roi_states.setdefault(roi_name, occupied)
            values[column] = int(bool(occupied))
        return values

    def _mask_to_coco_segmentation(self, mask: np.ndarray) -> list[list[float]]:
        """Convert one binary mask to COCO polygon segmentation."""
        try:
            mask_u8 = np.asarray(mask, dtype=np.uint8)
        except Exception:
            return []
        if mask_u8.ndim != 2 or not bool(mask_u8.any()):
            return []
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[float]] = []
        for contour in contours:
            contour = np.asarray(contour, dtype=float).reshape(-1, 2)
            if len(contour) < 3:
                continue
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append([float(value) for value in polygon])
        return polygons

    def _record_live_mask_coco_annotations(self, result: LiveDetectionResult, tracked_mice: List[object]) -> None:
        """Accumulate per-frame mask annotations into one COCO-style export payload."""
        if not self._should_save_live_masks_coco():
            return
        frame_id = int(result.frame_index)
        self.live_recording_coco_images[frame_id] = {
            "id": frame_id,
            "file_name": f"{Path(str(self.current_recording_filepath or 'recording')).name}:frame_{frame_id:06d}",
            "width": int(result.width),
            "height": int(result.height),
            "timestamp_s": float(result.timestamp_s),
        }
        for mouse in tracked_mice:
            mask = getattr(mouse, "mask", None)
            if mask is None or not np.size(mask):
                continue
            segmentation = self._mask_to_coco_segmentation(mask)
            if not segmentation:
                continue
            class_id = int(getattr(mouse, "class_id", 0))
            category_id = class_id + 1
            self.live_recording_coco_categories.setdefault(
                category_id,
                {"id": category_id, "name": f"class_{class_id}", "supercategory": "animal"},
            )
            bbox = tuple(float(value) for value in getattr(mouse, "bbox", (0.0, 0.0, 0.0, 0.0)))
            self.live_recording_coco_annotations.append(
                {
                    "id": int(self.live_recording_coco_next_annotation_id),
                    "image_id": frame_id,
                    "category_id": category_id,
                    "iscrowd": 0,
                    "bbox": [
                        float(bbox[0]),
                        float(bbox[1]),
                        max(0.0, float(bbox[2]) - float(bbox[0])),
                        max(0.0, float(bbox[3]) - float(bbox[1])),
                    ],
                    "area": float(np.count_nonzero(mask)),
                    "segmentation": segmentation,
                    "score": float(getattr(mouse, "confidence", 0.0)),
                    "track_id": int(getattr(mouse, "mouse_id", 0)),
                    "label": str(getattr(mouse, "label", "") or ""),
                    "timestamp_s": float(result.timestamp_s),
                }
            )
            self.live_recording_coco_next_annotation_id += 1

    def _live_tracking_scorer_name(self) -> str:
        """Return a DLC-style scorer name for tracking exports."""
        if self.live_detection_panel is not None:
            config = self.live_detection_panel.detection_config()
            keypoint_source = str(config.get("keypoint_source", "yolo_pose") or "yolo_pose")
            if keypoint_source == "mask_geometry":
                return "PyKabooMaskGeometry"
            if keypoint_source == "none":
                return "PyKabooNoKeypoints"
            pose_path = Path(str(config.get("pose_checkpoint_path", "") or "").strip())
            if pose_path.stem:
                return pose_path.stem
            model_key = str(config.get("model_key", "") or "").strip()
            if model_key:
                return model_key.replace("-", "_")
        return "PyKabooLive"

    def _tracking_mouse_ids_for_export(self, frame_df) -> List[int]:
        """Infer which tracked individuals should appear in the tracking export."""
        observed_ids: set[int] = set()
        for column in getattr(frame_df, "columns", []):
            match = re.match(r"mouse_(\d+)_", str(column or ""))
            if match:
                observed_ids.add(int(match.group(1)))
        expected = 1
        try:
            if self.live_detection_panel is not None:
                expected = max(1, int(self.live_detection_panel.spin_expected_mice.value()))
        except Exception:
            expected = 1
        if observed_ids:
            max_id = max(max(observed_ids), expected)
            return list(range(1, max_id + 1))
        return list(range(1, expected + 1))

    def _tracking_keypoint_indices_for_export(self, frame_df) -> List[int]:
        """Infer which generic keypoint ids were recorded."""
        indices: set[int] = set()
        for column in getattr(frame_df, "columns", []):
            match = re.match(r"mouse_\d+_kp_(\d+)_x$", str(column or ""))
            if match:
                indices.add(int(match.group(1)))
        return sorted(indices)

    def _save_live_tracking_dlc_csv(self, filepath: str, frame_df) -> Optional[Path]:
        """Write one DLC-style multi-index CSV with time, body center, and keypoints."""
        if frame_df is None or frame_df.empty:
            return None

        import pandas as pd

        scorer = self._live_tracking_scorer_name()
        mouse_ids = self._tracking_mouse_ids_for_export(frame_df)
        keypoint_indices = self._tracking_keypoint_indices_for_export(frame_df)
        bodyparts = ["bodycenter", *[f"kp{index}" for index in keypoint_indices]]

        column_tuples = [("time", "", "", "")]
        for mouse_id in mouse_ids:
            individual = f"mouse{mouse_id}"
            for bodypart in bodyparts:
                for coord in ("x", "y", "likelihood"):
                    column_tuples.append((scorer, individual, bodypart, coord))
        columns = pd.MultiIndex.from_tuples(
            column_tuples,
            names=["scorer", "individuals", "bodyparts", "coords"],
        )

        export_rows: list[list[object]] = []
        for _, row in frame_df.iterrows():
            values: list[object] = [row.get("timestamp_software", np.nan)]
            for mouse_id in mouse_ids:
                center_x = row.get(f"mouse_{mouse_id}_center_x", np.nan)
                center_y = row.get(f"mouse_{mouse_id}_center_y", np.nan)
                confidence = row.get(f"mouse_{mouse_id}_confidence", np.nan)
                values.extend([center_x, center_y, confidence])
                for kp_index in keypoint_indices:
                    values.extend(
                        [
                            row.get(f"mouse_{mouse_id}_kp_{kp_index}_x", np.nan),
                            row.get(f"mouse_{mouse_id}_kp_{kp_index}_y", np.nan),
                            row.get(f"mouse_{mouse_id}_kp_{kp_index}_likelihood", np.nan),
                        ]
                    )
            export_rows.append(values)

        tracking_df = pd.DataFrame(export_rows, columns=columns)
        csv_path = Path(f"{filepath}_tracking_dlc.csv")
        tracking_df.to_csv(csv_path, index=True)
        return csv_path

    def _save_live_tracking_roi_csv(self, filepath: str, frame_df) -> Optional[Path]:
        """Write one compact timestamp + ROI-binary CSV for live tracking."""
        if frame_df is None or frame_df.empty or not self.live_rois:
            return None

        import pandas as pd

        roi_column_map = self._live_roi_export_column_map()
        available_columns = [column for column in roi_column_map.values() if column in frame_df.columns]
        if not available_columns:
            return None
        roi_df = pd.DataFrame()
        roi_df["time"] = pd.to_numeric(frame_df.get("timestamp_software"), errors="coerce")
        inverse_map = {column: roi_name for roi_name, column in roi_column_map.items()}
        for column in available_columns:
            roi_df[str(inverse_map.get(column, column))] = pd.to_numeric(frame_df[column], errors="coerce").fillna(0).astype(int)
        csv_path = Path(f"{filepath}_tracking_rois.csv")
        roi_df.to_csv(csv_path, index=False)
        return csv_path

    def _save_live_masks_coco_export(self, filepath: str) -> Optional[Path]:
        """Write one COCO JSON sidecar with tracked mask polygons."""
        if not self.live_recording_coco_annotations:
            return None
        payload = {
            "info": {
                "description": "PyKaboo live mask export",
                "video": Path(str(filepath)).name,
            },
            "images": [
                self.live_recording_coco_images[image_id]
                for image_id in sorted(self.live_recording_coco_images.keys())
            ],
            "annotations": list(self.live_recording_coco_annotations),
            "categories": [
                self.live_recording_coco_categories[category_id]
                for category_id in sorted(self.live_recording_coco_categories.keys())
            ],
        }
        json_path = Path(f"{filepath}_masks_coco.json")
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return json_path

    def _record_live_detection_export(
        self,
        result: LiveDetectionResult,
        output_states: Dict[str, bool],
        active_rule_ids: List[str],
    ):
        if self.worker is None or not bool(getattr(self.worker, "is_recording", False)):
            return

        try:
            frame_id = int(result.frame_index)
        except Exception:
            return

        tracked_mice = sorted(
            list(result.tracked_mice or []),
            key=lambda mouse: int(getattr(mouse, "mouse_id", 0)),
        )
        output_snapshot = {
            f"live_do{index}_state": int(bool(output_states.get(f"DO{index}", False)))
            for index in range(1, 9)
        }
        roi_binary_values = self._live_roi_binary_export_values(tracked_mice)
        active_rule_text = "|".join(str(rule_id) for rule_id in active_rule_ids)

        frame_row: Dict[str, object] = {
            "live_detection_frame_id": frame_id,
            "live_detection_timestamp_software": float(result.timestamp_s),
            "live_detection_completed_timestamp_software": float(
                getattr(result, "completed_timestamp_s", 0.0) or 0.0
            ),
            "live_detection_age_ms": float(getattr(result, "end_to_end_ms", 0.0) or 0.0),
            "live_frame_width": int(result.width),
            "live_frame_height": int(result.height),
            "live_inference_ms": float(result.inference_ms),
            "live_predict_ms": float(getattr(result, "predict_ms", 0.0) or 0.0),
            "live_preprocess_ms": float(getattr(result, "preprocess_ms", 0.0) or 0.0),
            "live_postprocess_ms": float(getattr(result, "postprocess_ms", 0.0) or 0.0),
            "live_queue_wait_ms": float(getattr(result, "queue_wait_ms", 0.0) or 0.0),
            "live_end_to_end_ms": float(getattr(result, "end_to_end_ms", 0.0) or 0.0),
            "live_inference_input_width": int(getattr(result, "inference_width", 0) or 0),
            "live_inference_input_height": int(getattr(result, "inference_height", 0) or 0),
            "live_detection_count": int(len(tracked_mice)),
            "live_model_key": str(result.model_key or ""),
            "live_active_rule_ids": active_rule_text,
        }
        frame_row.update(output_snapshot)
        frame_row.update(roi_binary_values)

        if tracked_mice:
            first_mouse = tracked_mice[0]
            frame_row.update(
                {
                    "animal_track_id": int(first_mouse.mouse_id),
                    "animal_class_id": int(first_mouse.class_id),
                    "animal_confidence": float(first_mouse.confidence),
                    "animal_center_x": float(first_mouse.center[0]),
                    "animal_center_y": float(first_mouse.center[1]),
                }
            )

        for mouse in tracked_mice:
            mouse_prefix = f"mouse_{int(mouse.mouse_id)}"
            frame_row[f"{mouse_prefix}_class_id"] = int(mouse.class_id)
            frame_row[f"{mouse_prefix}_confidence"] = float(mouse.confidence)
            frame_row[f"{mouse_prefix}_center_x"] = float(mouse.center[0])
            frame_row[f"{mouse_prefix}_center_y"] = float(mouse.center[1])
            keypoints = getattr(mouse, "keypoints", None)
            keypoint_scores = getattr(mouse, "keypoint_scores", None)
            if keypoints is not None:
                keypoints_arr = np.asarray(keypoints, dtype=float).reshape(-1, 2)
                score_arr = (
                    np.asarray(keypoint_scores, dtype=float).reshape(-1)
                    if keypoint_scores is not None
                    else None
                )
                for kp_index, (kx, ky) in enumerate(keypoints_arr, start=1):
                    frame_row[f"{mouse_prefix}_kp_{kp_index}_x"] = float(kx)
                    frame_row[f"{mouse_prefix}_kp_{kp_index}_y"] = float(ky)
                    frame_row[f"{mouse_prefix}_kp_{kp_index}_likelihood"] = (
                        float(score_arr[kp_index - 1])
                        if score_arr is not None and kp_index - 1 < len(score_arr)
                        else np.nan
                    )
            for roi_name, column in self._live_roi_export_column_map().items():
                roi = self.live_rois.get(roi_name)
                in_zone = 0
                if roi is not None:
                    try:
                        in_zone = int(bool(roi.contains_point(float(mouse.center[0]), float(mouse.center[1]))))
                    except Exception:
                        in_zone = 0
                frame_row[f"{mouse_prefix}_{column}"] = in_zone

            bbox = tuple(float(value) for value in mouse.bbox)
            detail_row: Dict[str, object] = {
                "frame_id": frame_id,
                "timestamp_software": float(result.timestamp_s),
                "mouse_id": int(mouse.mouse_id),
                "class_id": int(mouse.class_id),
                "label": str(mouse.label or ""),
                "confidence": float(mouse.confidence),
                "center_x": float(mouse.center[0]),
                "center_y": float(mouse.center[1]),
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "mask_area_px": int(np.count_nonzero(mouse.mask)) if mouse.mask is not None else 0,
                "live_inference_ms": float(result.inference_ms),
                "live_predict_ms": float(getattr(result, "predict_ms", 0.0) or 0.0),
                "live_preprocess_ms": float(getattr(result, "preprocess_ms", 0.0) or 0.0),
                "live_postprocess_ms": float(getattr(result, "postprocess_ms", 0.0) or 0.0),
                "live_queue_wait_ms": float(getattr(result, "queue_wait_ms", 0.0) or 0.0),
                "live_end_to_end_ms": float(getattr(result, "end_to_end_ms", 0.0) or 0.0),
                "live_detection_completed_timestamp_software": float(
                    getattr(result, "completed_timestamp_s", 0.0) or 0.0
                ),
                "live_inference_input_width": int(getattr(result, "inference_width", 0) or 0),
                "live_inference_input_height": int(getattr(result, "inference_height", 0) or 0),
                "live_model_key": str(result.model_key or ""),
                "live_active_rule_ids": active_rule_text,
            }
            if keypoints is not None:
                for kp_index, (kx, ky) in enumerate(keypoints_arr, start=1):
                    detail_row[f"kp_{kp_index}_x"] = float(kx)
                    detail_row[f"kp_{kp_index}_y"] = float(ky)
                    detail_row[f"kp_{kp_index}_likelihood"] = (
                        float(score_arr[kp_index - 1])
                        if score_arr is not None and kp_index - 1 < len(score_arr)
                        else np.nan
                    )
            detail_row.update(output_snapshot)
            for roi_name, column in self._live_roi_export_column_map().items():
                roi = self.live_rois.get(roi_name)
                in_zone = 0
                if roi is not None:
                    try:
                        in_zone = int(bool(roi.contains_point(float(mouse.center[0]), float(mouse.center[1]))))
                    except Exception:
                        in_zone = 0
                detail_row[column] = in_zone
            self.live_recording_detection_rows.append(detail_row)

        self._record_live_mask_coco_annotations(result, tracked_mice)
        self.live_recording_frame_rows[frame_id] = frame_row

    def _fill_live_roi_binary_columns(self, df):
        roi_columns = [
            column
            for column in df.columns
            if column.startswith("in_zone_roi_") or "_in_zone_roi_" in column
        ]
        if not roi_columns:
            return df

        import pandas as pd

        for column in roi_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
        return df

    def _merge_live_detections_into_frame_df(self, df):
        if not self.live_recording_frame_rows:
            return df

        import pandas as pd

        live_df = pd.DataFrame(list(self.live_recording_frame_rows.values()))
        if live_df.empty or "live_detection_frame_id" not in live_df.columns:
            return df

        merged = df.copy()
        if (
            "timestamp_software" in merged.columns
            and "live_detection_timestamp_software" in live_df.columns
        ):
            row_order_column = "_recording_row_order"
            merged[row_order_column] = np.arange(len(merged), dtype=int)
            left = merged.copy()
            left["timestamp_software"] = pd.to_numeric(left["timestamp_software"], errors="coerce")
            right = live_df.copy()
            right["live_detection_timestamp_software"] = pd.to_numeric(
                right["live_detection_timestamp_software"],
                errors="coerce",
            )
            if left["timestamp_software"].notna().all() and right["live_detection_timestamp_software"].notna().any():
                left = left.sort_values("timestamp_software")
                right = right.dropna(subset=["live_detection_timestamp_software"]).sort_values("live_detection_timestamp_software")
                preview_fps = 0.0
                try:
                    preview_fps = float(self.spin_preview_fps.value())
                except Exception:
                    preview_fps = 0.0
                tolerance_s = max(0.25, 2.5 / preview_fps) if preview_fps > 0 else 0.5
                merged = pd.merge_asof(
                    left,
                    right,
                    left_on="timestamp_software",
                    right_on="live_detection_timestamp_software",
                    direction="backward",
                    tolerance=tolerance_s,
                )
                matched = merged["live_detection_timestamp_software"].notna()
                merged.loc[matched, "live_detection_age_ms"] = (
                    merged.loc[matched, "timestamp_software"]
                    - merged.loc[matched, "live_detection_timestamp_software"]
                ) * 1000.0
                merged = merged.sort_values(row_order_column).drop(columns=[row_order_column])
                return self._fill_live_roi_binary_columns(merged)
            merged = merged.drop(columns=[row_order_column])

        if "frame_id" not in merged.columns:
            return merged
        merged = merged.merge(
            live_df,
            left_on="frame_id",
            right_on="live_detection_frame_id",
            how="left",
        )
        return self._fill_live_roi_binary_columns(merged)

    def _merge_ttl_history_into_frame_df(self, df):
        if not self.is_arduino_connected or self.arduino_worker is None:
            return df
        if "frame_id" not in df.columns:
            return df

        import pandas as pd

        ttl_history = self.arduino_worker.get_ttl_history()
        if not ttl_history:
            return df

        ttl_df = pd.DataFrame(ttl_history)
        if ttl_df.empty or "frame_id" not in ttl_df.columns:
            return df
        ttl_df = self._apply_line_label_suffixes(ttl_df)
        ttl_df = self._augment_ttl_state_columns(ttl_df)

        ttl_columns = [
            column
            for column in ttl_df.columns
            if column == "frame_id" or column not in df.columns
        ]
        if len(ttl_columns) <= 1:
            return df
        return df.merge(ttl_df[ttl_columns], on="frame_id", how="left")

    def _merge_user_flag_events_into_frame_df(self, df):
        """Project manual user-flag events onto the per-frame recording export."""
        if df is None or df.empty:
            return df

        import pandas as pd

        metadata_user_flags = (self.metadata or {}).get("user_flags", [])
        if isinstance(metadata_user_flags, list):
            configs = [
                self._normalize_user_flag_config(item, fallback_index=index + 1)
                for index, item in enumerate(metadata_user_flags)
                if isinstance(item, dict)
            ]
        else:
            configs = self._current_user_flag_configs()
        primary_config = configs[0] if configs else self._current_user_flag_config()
        has_config = any(bool(str(config.get("shortcut", "") or "").strip()) for config in configs)
        if not has_config and not self.user_flag_events:
            return df

        merged = df.copy()
        merged["user_flag_label"] = str(primary_config.get("label", "User Flag") or "User Flag")
        merged["user_flag_shortcut"] = str(primary_config.get("shortcut", "") or "")
        merged["user_flag_output"] = str(primary_config.get("output_id", "") or "")
        merged["user_flag_pulse_ms"] = int(primary_config.get("pulse_ms", 100) or 100)

        if "timestamp_software" not in merged.columns:
            merged["user_flag_event"] = 0
            merged["user_flag_ttl"] = 0
            merged["user_flag_count"] = 0
            merged["user_flag_event_timestamp_software"] = np.nan
            merged["user_flag_event_label"] = ""
            merged["user_flag_event_shortcut"] = ""
            merged["user_flag_event_output"] = ""
            merged["user_flag_event_pulse_ms"] = np.nan
            return merged

        projected = project_user_flag_events(
            pd.to_numeric(merged["timestamp_software"], errors="coerce"),
            self.user_flag_events,
        )
        merged["user_flag_event"] = projected["event"]
        merged["user_flag_ttl"] = projected["ttl"]
        merged["user_flag_count"] = projected["count"]
        merged["user_flag_event_timestamp_software"] = projected["event_timestamp"]
        merged["user_flag_event_label"] = projected["event_label"]
        merged["user_flag_event_shortcut"] = projected["event_shortcut"]
        merged["user_flag_event_output"] = projected["event_output"]
        merged["user_flag_event_pulse_ms"] = projected["event_pulse_ms"]
        return merged

    def _drop_low_value_frame_export_columns(self, df):
        """Remove raw-frame diagnostic columns from the exported metadata CSV."""
        if df is None or df.empty:
            return df

        drop_candidates = [
            "raw_dtype",
            "raw_height",
            "raw_width",
            "raw_min",
            "raw_max",
            "raw_mean",
        ]
        removable = [column for column in drop_candidates if column in df.columns]
        if removable:
            df = df.drop(columns=removable)
        return df

    def _infer_timestamp_tick_scale(self, tick_series, software_series=None) -> float:
        """Infer how many camera ticks correspond to one second."""
        return infer_timestamp_tick_scale(tick_series, software_series)

    def _normalize_recording_timestamps(self, df):
        """Express exported timestamps in elapsed seconds from frame 0."""
        return normalize_recording_timestamps(df)

    def _save_recording_frame_csv_outputs(self, filepath: str):
        metadata_csv = Path(f"{filepath}_metadata.csv")
        if not metadata_csv.exists():
            return

        try:
            import pandas as pd

            metadata_context = {
                "animal_id": str((self.metadata or {}).get("animal_id", "")),
                "session": str((self.metadata or {}).get("session", "")),
                "trial": str((self.metadata or {}).get("trial", "")),
                "experiment": str((self.metadata or {}).get("experiment", "")),
                "condition": str((self.metadata or {}).get("condition", "")),
                "arena": str((self.metadata or {}).get("arena", "")),
                "date": str((self.metadata or {}).get("date", "")),
                "filename_preview": str((self.metadata or {}).get("filename_preview", "")),
            }

            frame_df = pd.read_csv(metadata_csv)
            for column, value in metadata_context.items():
                frame_df[column] = value
            frame_df = self._apply_line_label_suffixes(frame_df)
            frame_df = self._merge_ttl_history_into_frame_df(frame_df)
            if self.aux_arduino_manager is not None:
                frame_df = self.aux_arduino_manager.merge_into_frame_df(frame_df)
            if self._has_frame_aligned_signal_sources(frame_df):
                frame_df = self._augment_ttl_state_columns(frame_df)
            frame_df = self._merge_user_flag_events_into_frame_df(frame_df)
            frame_df = self._merge_live_detections_into_frame_df(frame_df)
            frame_df = self._normalize_recording_timestamps(frame_df)
            frame_df = self._drop_low_value_frame_export_columns(frame_df)
            frame_df = self._drop_unselected_signal_export_columns(frame_df)
            frame_df = self._reorder_signal_export_columns(frame_df)
            frame_df.to_csv(metadata_csv, index=False)
            self._on_status_update(f"Frame CSV updated: {metadata_csv}")

            if self._should_save_live_tracking_csv():
                tracking_csv = self._save_live_tracking_dlc_csv(filepath, frame_df)
                if tracking_csv is not None:
                    self._on_status_update(f"Tracking CSV saved: {tracking_csv.name}")
                roi_csv = self._save_live_tracking_roi_csv(filepath, frame_df)
                if roi_csv is not None:
                    self._on_status_update(f"Tracking ROI CSV saved: {roi_csv.name}")
            if self._should_save_live_masks_coco():
                coco_json = self._save_live_masks_coco_export(filepath)
                if coco_json is not None:
                    self._on_status_update(f"Mask COCO saved: {coco_json.name}")

            if self.live_recording_detection_rows:
                detections_csv = Path(f"{filepath}_live_detections.csv")
                pd.DataFrame(self.live_recording_detection_rows).to_csv(detections_csv, index=False)
                self._on_status_update(f"Live detections saved: {detections_csv}")
        except Exception as exc:
            self._on_error_occurred(f"Frame CSV export error: {str(exc)}")

    def _augment_ttl_state_columns(self, df):
        """Add aggregate TTL/behavior state columns for CSV readability."""
        if df is None or df.empty:
            return df

        import pandas as pd

        df = df.copy()
        definitions = self._signal_export_definitions()
        active_ttl_keys = self._active_signal_keys(group="ttl")
        active_behavior_keys = self._active_signal_keys(group="behavior")
        active_keys = active_ttl_keys + active_behavior_keys
        resolved = {key: self._resolve_display_signal_series(df, key) for key in active_keys}

        for key in active_keys:
            series = resolved[key]
            labeled_column = definitions[key]["state_column"]
            df[labeled_column] = series

            count_series = self._resolve_display_signal_count_series(df, key)
            labeled_count_column = definitions[key]["count_column"]
            if count_series is not None:
                df[labeled_count_column] = count_series

        if active_ttl_keys:
            ttl_active = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
            ttl_vector = None
            for key in active_ttl_keys:
                ttl_active = (ttl_active | resolved[key]).astype(int)
                segment = f"{definitions[key]['label']}=" + resolved[key].astype(str)
                ttl_vector = segment if ttl_vector is None else ttl_vector + "|" + segment
            df["ttl_state"] = np.where(ttl_active > 0, "HIGH", "LOW")
            if ttl_vector is not None:
                df["ttl_state_vector"] = ttl_vector
        else:
            for column in ("ttl_state", "ttl_state_vector"):
                if column in df.columns:
                    df = df.drop(columns=[column])

        if active_behavior_keys:
            behavior_active = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
            behavior_vector = None
            for key in active_behavior_keys:
                behavior_active = (behavior_active | resolved[key]).astype(int)
                segment = f"{definitions[key]['label']}=" + resolved[key].astype(str)
                behavior_vector = segment if behavior_vector is None else behavior_vector + "|" + segment
            df["behavior_state"] = np.where(behavior_active > 0, "ACTIVE", "IDLE")
            if behavior_vector is not None:
                df["behavior_state_vector"] = behavior_vector
        else:
            for column in ("behavior_state", "behavior_state_vector"):
                if column in df.columns:
                    df = df.drop(columns=[column])

        df = self._drop_unselected_signal_export_columns(df)
        return self._reorder_signal_export_columns(df)

    def _build_behavior_summary_df(self, source_df, ttl_counts: Dict):
        """Build behavior summary DataFrame (counts and cumulative HIGH durations)."""
        import pandas as pd

        signals = self._active_signal_keys(group="behavior")
        definitions = self._signal_export_definitions()
        rows = []

        if not signals:
            return pd.DataFrame(rows)

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
        t = t.ffill().bfill()
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
            count_value = rises
            count_series = self._resolve_display_signal_count_series(df, signal)
            if count_series is not None and len(count_series) > 0:
                numeric_counts = pd.to_numeric(count_series, errors="coerce").dropna()
                if not numeric_counts.empty:
                    count_value = int(numeric_counts.iloc[-1])
            if count_value <= 0:
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
            active_signal_keys = self._active_signal_keys()
            df_history = None
            df_live = None

            ttl_history = self.arduino_worker.get_ttl_history()
            if ttl_history:
                df_history = pd.DataFrame(ttl_history)
                df_history = self._apply_line_label_suffixes(df_history)
                df_history = self._augment_ttl_state_columns(df_history)
                df_history = self._normalize_recording_timestamps(df_history)
                csv_file = filepath + "_ttl_states.csv"
                df_history.to_csv(csv_file, index=False)
                self._on_status_update(f"TTL states saved: {csv_file}")

            live_history = self.arduino_worker.get_live_state_history()
            if live_history:
                df_live = pd.DataFrame(live_history)
                df_live = self._augment_ttl_state_columns(df_live)

            if ttl_counts and active_signal_keys:
                count_row = {}
                for key in active_signal_keys:
                    definition = export_definitions[key]
                    count_series = self._resolve_display_signal_count_series(df_history, key) if df_history is not None else None
                    if count_series is not None and len(count_series) > 0:
                        numeric_counts = pd.to_numeric(count_series, errors="coerce").dropna()
                        if not numeric_counts.empty:
                            count_row[definition["count_column"]] = int(numeric_counts.iloc[-1])
                            continue
                    count_row[definition["count_column"]] = self._resolve_display_signal_count(key, ttl_counts)
                df = pd.DataFrame([count_row])
                df = self._reorder_signal_export_columns(df)
                csv_file = filepath + "_ttl_counts.csv"
                df.to_csv(csv_file, index=False)
                self._on_status_update(f"TTL counts saved: {csv_file}")

            summary_source = df_history if df_history is not None else df_live
            if summary_source is not None:
                behavior_summary = self._build_behavior_summary_df(summary_source, ttl_counts)
                if not behavior_summary.empty:
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
        """Create a branded standby frame styled like a camera viewfinder."""
        W, H = 1280, 720
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # Deep blue radial-gradient background (BGR order).
        y_coords = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        x_coords = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        dist = np.sqrt((x_coords - 0.5) ** 2 + (y_coords - 0.42) ** 2)
        canvas[:, :, 0] = np.clip(30 - dist * 22, 8, 30).astype(np.uint8)   # B
        canvas[:, :, 1] = np.clip(18 - dist * 12, 5, 18).astype(np.uint8)   # G
        canvas[:, :, 2] = np.clip(11 - dist * 7, 3, 11).astype(np.uint8)    # R

        # Faint sensor grid.
        grid_color = (34, 24, 16)
        for gx in range(0, W, 80):
            cv2.line(canvas, (gx, 0), (gx, H), grid_color, 1, cv2.LINE_AA)
        for gy in range(0, H, 80):
            cv2.line(canvas, (0, gy), (W, gy), grid_color, 1, cv2.LINE_AA)

        # Soft cyan glow behind the lens emblem.
        overlay = canvas.copy()
        cv2.circle(overlay, (W // 2, H // 2 - 40), 230, (64, 44, 18), -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

        # Viewfinder corner brackets.
        bracket = (132, 96, 44)
        margin, arm = 36, 56
        for cx_b, cy_b, dx, dy in (
            (margin, margin, 1, 1),
            (W - margin, margin, -1, 1),
            (margin, H - margin, 1, -1),
            (W - margin, H - margin, -1, -1),
        ):
            cv2.line(canvas, (cx_b, cy_b), (cx_b + dx * arm, cy_b), bracket, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx_b, cy_b), (cx_b, cy_b + dy * arm), bracket, 2, cv2.LINE_AA)

        # Lens emblem: concentric rings + iris dot.
        cx, cy = W // 2, H // 2 - 40
        cv2.circle(canvas, (cx, cy), 58, (210, 150, 70), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 40, (150, 105, 50), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 14, (255, 196, 110), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 14, (255, 226, 160), 1, cv2.LINE_AA)

        # Title block centered under the emblem (skip duplicate app name).
        def centered_text(text: str, y: int, scale: float, color, thickness: int = 1):
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
            cv2.putText(canvas, text, (W // 2 - size[0] // 2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

        centered_text(APP_NAME, cy + 112, 1.0, (255, 205, 130), 2)
        line_y = cy + 142
        if title and title != APP_NAME:
            centered_text(title, line_y, 0.6, (225, 205, 185))
            line_y += 34
        if subtitle:
            centered_text(subtitle, line_y, 0.52, (190, 165, 135))
            line_y += 30

        # Status chip: STANDBY
        chip_text = "STANDBY"
        chip_size = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        chip_w, chip_h = chip_size[0] + 28, 30
        chip_x, chip_y = W // 2 - chip_w // 2, line_y + 6
        cv2.rectangle(canvas, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h),
                      (52, 34, 16), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h),
                      (120, 86, 40), 1, cv2.LINE_AA)
        cv2.circle(canvas, (chip_x + 14, chip_y + chip_h // 2), 4, (90, 200, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, chip_text, (chip_x + 24, chip_y + chip_h // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 215, 190), 1, cv2.LINE_AA)

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
        """Return elapsed wall-clock recording time in whole seconds."""
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

    def _preview_active_signal_labels(self) -> List[str]:
        """Return the labels that should appear in the live preview signal strip."""
        labels: List[str] = []
        seen: set[str] = set()
        states = dict(self.latest_ttl_states or {})

        for key in self._active_signal_keys():
            state_key = self._state_key_for_display(key)
            if not bool(states.get(state_key, False)):
                continue
            label = self._signal_label(key).strip() or str(self.DISPLAY_SIGNAL_META.get(key, {}).get("name", key))
            lowered = label.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            labels.append(label)

        user_flag_output_labels: Dict[str, List[str]] = {}
        for config in self._current_user_flag_configs():
            output_id = str(config.get("output_id", "") or "").strip().upper()
            label = str(config.get("label", "") or "").strip()
            if not output_id or not label:
                continue
            user_flag_output_labels.setdefault(output_id, [])
            if label not in user_flag_output_labels[output_id]:
                user_flag_output_labels[output_id].append(label)
        for index in range(1, 9):
            output_id = f"DO{index}"
            is_high = bool(states.get(output_id.lower(), False)) or bool(self.live_output_states.get(output_id, False))
            if not is_high:
                continue
            output_labels = user_flag_output_labels.get(output_id, [])
            if output_labels:
                for label in output_labels:
                    lowered = label.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    labels.append(label)
            else:
                lowered = output_id.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                labels.append(output_id)

        return labels

    def _draw_user_flag_preview_banner(self, display_bgr: np.ndarray):
        """Draw a transient visual confirmation when a user flag shortcut is pressed."""
        if time.monotonic() >= float(self.user_flag_preview_until_s or 0.0):
            return

        text = str(self.user_flag_preview_text or "").strip()
        if not text:
            return

        x1, y1 = 18, 48
        padding_x = 14
        padding_y = 10
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        x2 = min(display_bgr.shape[1] - 18, x1 + text_size[0] + (2 * padding_x))
        y2 = min(display_bgr.shape[0] - 18, y1 + text_size[1] + (2 * padding_y))

        overlay = display_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 74, 34), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (102, 255, 166), 2)
        cv2.addWeighted(overlay, 0.78, display_bgr, 0.22, 0, display_bgr)
        cv2.putText(
            display_bgr,
            text,
            (x1 + padding_x, y2 - padding_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (225, 255, 236),
            2,
            cv2.LINE_AA,
        )

    def _decorate_live_frame(
        self,
        frame: np.ndarray,
        include_recording_hud: bool = True,
        include_info: bool = True,
        overlay_result_override: Optional[LiveDetectionResult] = None,
    ) -> np.ndarray:
        """Render the live frame through an OpenCV overlay pipeline before display."""
        if frame.ndim == 2:
            display_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            display_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if include_recording_hud and self.worker and self.worker.is_recording:
            # Small in-frame REC dot only; the elapsed/remaining countdown is
            # now the crisp Qt overlay pinned to the live view's top-right.
            cv2.circle(display_bgr, (28, 28), 8, (32, 59, 240), -1)
            cv2.putText(display_bgr, "REC", (48, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        self._draw_live_detection_overlay(display_bgr, overlay_result_override=overlay_result_override)
        self._draw_user_flag_preview_banner(display_bgr)

        if include_info:
            info_text = f"{display_bgr.shape[1]}x{display_bgr.shape[0]}  {self.combo_image_format.currentText()}"
            info_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            info_x = max(18, display_bgr.shape[1] - info_size[0] - 22)
            cv2.putText(display_bgr, info_text, (info_x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (195, 216, 236), 2, cv2.LINE_AA)
        return cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)

    def _draw_live_detection_overlay(
        self,
        display_bgr: np.ndarray,
        *,
        overlay_result_override: Optional[LiveDetectionResult] = None,
    ):
        overlay_options = (
            self.live_detection_panel.overlay_options()
            if self.live_detection_panel is not None
            else {"show_masks": True, "show_boxes": True, "show_keypoints": True}
        )
        show_masks = bool(overlay_options.get("show_masks", True))
        show_boxes = bool(overlay_options.get("show_boxes", True))
        show_keypoints = bool(overlay_options.get("show_keypoints", True))
        mask_opacity = clamp_mask_opacity(overlay_options.get("mask_opacity", 0.18))
        overlay = display_bgr.copy()
        source_result = overlay_result_override if overlay_result_override is not None else self._current_live_overlay_result()
        overlay_result = self._scaled_overlay_result_cached(source_result, display_bgr.shape)
        draw_live_rois = self._live_roi_overlays_visible()
        occupied_names = occupied_roi_names(self.live_rois, overlay_result) if draw_live_rois else set()
        if draw_live_rois:
            self._update_live_circle_roi_item_pens(occupied_names)

            for roi_name, roi in self.live_rois.items():
                color_rgb = self._live_roi_preview_color(roi_name, roi, occupied_names)
                color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                line_width = self._live_roi_preview_line_width(roi_name, occupied_names)
                if roi.roi_type == "rectangle" and roi.data:
                    x1, y1, x2, y2 = [int(round(value)) for value in roi.data[0]]
                    cv2.rectangle(display_bgr, (x1, y1), (x2, y2), color_bgr, line_width)
                    cv2.putText(display_bgr, roi_name, (x1 + 6, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)
                elif roi.roi_type == "circle" and roi.data:
                    cx, cy, radius = roi.data[0]
                    cv2.circle(display_bgr, (int(round(cx)), int(round(cy))), int(round(radius)), color_bgr, line_width)
                    cv2.putText(display_bgr, roi_name, (int(cx) + 6, max(20, int(cy) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)
                elif roi.roi_type == "polygon" and roi.data:
                    pts = np.array([(int(round(px)), int(round(py))) for px, py in roi.data], dtype=np.int32)
                    if len(pts) >= 3:
                        cv2.polylines(display_bgr, [pts], True, color_bgr, line_width, cv2.LINE_AA)
                        cv2.fillPoly(overlay, [pts], color_bgr)
                        cx = int(np.mean(pts[:, 0]))
                        cy = int(np.mean(pts[:, 1]))
                        cv2.putText(display_bgr, roi_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2, cv2.LINE_AA)

            if self.live_rois:
                cv2.addWeighted(overlay, 0.12, display_bgr, 0.88, 0, display_bgr)

        if overlay_result is not None:
            for mouse in overlay_result.tracked_mice:
                # One vivid colour per identity, shared by mask, box, keypoints
                # and skeleton so pose and mask always read as the same animal.
                color_rgb = identity_color_rgb(int(mouse.mouse_id))
                color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                if show_masks and mouse.mask is not None and mouse.mask.size > 0:
                    mask_bool = np.asarray(mouse.mask, dtype=bool)
                    if mask_bool.shape[:2] != display_bgr.shape[:2]:
                        continue
                    # Soft translucent fill plus a crisp anti-aliased contour so
                    # the silhouette stays sharp and limited to the mask area.
                    self._blend_live_mask_roi(display_bgr, mask_bool, color_bgr, mask_opacity, mouse.bbox)
                x1, y1, x2, y2 = [int(round(value)) for value in mouse.bbox]
                if show_boxes:
                    cv2.rectangle(display_bgr, (x1, y1), (x2, y2), color_bgr, 2)
                cx, cy = int(round(mouse.center[0])), int(round(mouse.center[1]))
                cv2.circle(display_bgr, (cx, cy), 4, color_bgr, -1)
                label = f"{mouse.label}  C{mouse.class_id}  {mouse.confidence:.2f}"
                label_x = x1 + 4 if show_boxes else min(max(8, cx + 8), max(8, display_bgr.shape[1] - 220))
                label_y = max(20, y1 - 8) if show_boxes else max(20, cy - 8)
                cv2.putText(display_bgr, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2, cv2.LINE_AA)

                if show_keypoints and getattr(mouse, "keypoints", None) is not None:
                    self._draw_pose_skeleton(display_bgr, mouse, color_bgr)

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

        active_signals = self._preview_active_signal_labels()
        if active_signals:
            text = "HIGH: " + ", ".join(active_signals)
            cv2.putText(
                display_bgr,
                text,
                (18, max(64, display_bgr.shape[0] - 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (120, 255, 180),
                2,
                cv2.LINE_AA,
            )

    def _scaled_overlay_result_cached(self, source_result, shape):
        """Scale an inference result to the display shape, reusing the last scale.

        Carry-forward means the same inference result is painted onto many
        preview frames. Re-resizing every mask each frame is wasted CPU that
        also competes with the inference thread, so we cache the scaled result
        per (result identity, display shape).
        """
        if source_result is None:
            self._scaled_overlay_cache = None
            return None
        key = (id(source_result), int(shape[0]), int(shape[1]))
        cached = getattr(self, "_scaled_overlay_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        scaled = scale_live_detection_result_to_shape(source_result, shape)
        self._scaled_overlay_cache = (key, scaled)
        return scaled

    def _draw_pose_skeleton(self, display_bgr: np.ndarray, mouse, color_bgr) -> None:
        """Draw the pose skeleton + joints for one animal in its identity colour."""
        keypoints = np.asarray(mouse.keypoints, dtype=float).reshape(-1, 2)
        if keypoints.size == 0:
            return
        scores = getattr(mouse, "keypoint_scores", None)
        score_arr = np.asarray(scores, dtype=float).reshape(-1) if scores is not None else None

        def _valid(index: int) -> Optional[tuple[int, int]]:
            if index < 0 or index >= len(keypoints):
                return None
            kx, ky = keypoints[index]
            if not (np.isfinite(kx) and np.isfinite(ky)):
                return None
            if kx <= 0 and ky <= 0:
                return None
            if score_arr is not None and index < len(score_arr) and float(score_arr[index]) < 0.1:
                return None
            return int(round(float(kx))), int(round(float(ky)))

        # Bones first (slightly darker tint of the identity colour), joints on top.
        bone_color = tuple(int(c * 0.75) for c in color_bgr)
        for a, b in skeleton_for_keypoint_count(len(keypoints)):
            pa = _valid(a)
            pb = _valid(b)
            if pa is None or pb is None:
                continue
            cv2.line(display_bgr, pa, pb, bone_color, 2, cv2.LINE_AA)
        for index in range(len(keypoints)):
            point = _valid(index)
            if point is None:
                continue
            cv2.circle(display_bgr, point, 4, (20, 24, 32), -1, cv2.LINE_AA)
            cv2.circle(display_bgr, point, 3, color_bgr, -1, cv2.LINE_AA)

    def _blend_live_mask_roi(
        self,
        display_bgr: np.ndarray,
        mask_bool: np.ndarray,
        color_bgr,
        mask_opacity: float,
        bbox,
    ) -> None:
        """Blend and contour one mask in its bbox ROI instead of the full frame."""
        if mask_bool.ndim != 2 or mask_bool.shape[:2] != display_bgr.shape[:2] or not bool(mask_bool.any()):
            return
        h, w = mask_bool.shape[:2]
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        except Exception:
            ys, xs = np.nonzero(mask_bool)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1 = max(0, min(w, x1 - 3))
        x2 = max(0, min(w, x2 + 4))
        y1 = max(0, min(h, y1 - 3))
        y2 = max(0, min(h, y2 + 4))
        if x2 <= x1 or y2 <= y1:
            return
        local_mask = mask_bool[y1:y2, x1:x2]
        if not bool(local_mask.any()):
            return
        roi = display_bgr[y1:y2, x1:x2]
        base = roi[local_mask].astype(np.float32)
        tint = np.asarray(color_bgr, dtype=np.float32).reshape(1, 3)
        roi[local_mask] = np.clip(
            base * (1.0 - float(mask_opacity)) + tint * float(mask_opacity),
            0.0,
            255.0,
        ).astype(np.uint8)
        contours, _ = cv2.findContours(local_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            shifted = [contour + np.array([[[x1, y1]]], dtype=contour.dtype) for contour in contours]
            cv2.drawContours(display_bgr, shifted, -1, color_bgr, 2, cv2.LINE_AA)

    def _current_live_overlay_result(self) -> Optional[LiveDetectionResult]:
        if self.live_detection_last_result is None:
            return None
        if not self.live_detection_enabled:
            return None
        if self.live_preview_frame_index < 0:
            return self.live_detection_last_result
        return self._overlay_result_for_frame(
            int(self.live_preview_frame_index),
            float(self.live_preview_timestamp_s),
            frame_rate=max(1.0, float(self.spin_preview_fps.value())),
        )

    def _overlay_result_for_frame(
        self,
        frame_index: int,
        timestamp_s: float,
        *,
        frame_rate: Optional[float] = None,
    ) -> Optional[LiveDetectionResult]:
        if self.live_detection_last_result is None or not self.live_detection_enabled:
            return None
        if int(frame_index) < 0:
            return self.live_detection_last_result

        preview_fps = max(1.0, float(frame_rate or self.spin_preview_fps.value()))
        preview_timestamp_s = float(timestamp_s)
        # Carry the most recent *non-empty* result forward across the window so a
        # single-frame detection dropout never blinks the mask/skeleton off. If
        # nothing within the window has animals, fall back to the newest current
        # result (which may legitimately be empty when the arena is empty).
        newest_current: Optional[LiveDetectionResult] = None
        for result in reversed(self.live_detection_result_history):
            result_timestamp_s = float(result.timestamp_s)
            if result_timestamp_s > (preview_timestamp_s + 1e-6):
                continue
            if not overlay_result_is_current(
                preview_frame_index=int(frame_index),
                preview_timestamp_s=preview_timestamp_s,
                result_frame_index=int(result.frame_index),
                result_timestamp_s=result_timestamp_s,
                preview_fps=preview_fps,
                inference_ms=live_result_retention_ms(result),
            ):
                break
            if newest_current is None:
                newest_current = result
            if getattr(result, "tracked_mice", None):
                return result
        return newest_current

    @Slot(np.ndarray)
    def _on_frame_ready(self, frame: np.ndarray):
        """
        Update the docked ImageView with a frame from the worker.
        """
        try:
            height, width = frame.shape[:2]
            self.last_frame_size = (width, height)
            # Render every preview frame at the full preview rate (up to 60 fps)
            # and paint the most recent inference result on top with carry
            # forward. Decoupling the preview from the inference rate keeps the
            # live view smooth and Full-HD-fluid even when the mask+pose model
            # delivers results more slowly, while the carry-forward retention in
            # _current_live_overlay_result keeps the overlay flicker-free.
            image_rgb = self._decorate_live_frame(frame)
            auto_range = not self.live_frame_auto_ranged
            self._apply_live_image(image_rgb, auto_range=auto_range)
            self.live_frame_auto_ranged = True
            self._sync_camera_roi_overlay()
            self._sync_live_circle_roi_items()
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
            self.label_buffer.setStyleSheet("QLabel { color: #ffb3b3; font-weight: 700; }")
        elif buffer_percent > 50:
            self.label_buffer.setStyleSheet("QLabel { color: #ffd89c; font-weight: 700; }")
        else:
            self.label_buffer.setStyleSheet("")

    @Slot(str)
    def _on_error_occurred(self, error_message: str):
        """Handle error messages."""
        self.status_bar.showMessage(f"ERROR: {error_message}", 10000)
        print(f"Error: {error_message}")

    def resizeEvent(self, event):
        """Keep shell widths responsive as the main window changes size."""
        super().resizeEvent(event)
        self._update_side_panel_bounds()
        self._schedule_planner_column_fit()
        self._schedule_responsive_layout_refresh()

    def changeEvent(self, event):
        """Refresh responsive bounds after maximize, restore, or fullscreen toggles."""
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            self._schedule_responsive_layout_refresh()

    def closeEvent(self, event):
        """Handle window close event - cleanup resources."""
        self._save_planner_state_snapshot()
        try:
            self._apply_behavior_pin_configuration(persist=True)
        except Exception:
            pass
        try:
            self._persist_live_detection_settings()
        except Exception:
            pass

        if self.audio_panel is not None:
            try:
                self.audio_panel.shutdown()
            except Exception:
                pass

        if self.is_camera_connected:
            self._disconnect_camera()

        if self.camera_stream_manager is not None:
            try:
                self.camera_stream_manager.shutdown()
            except Exception:
                pass

        if self.is_arduino_connected:
            self.arduino_worker.clear_live_outputs()
            self._stop_arduino_generation()
            self.arduino_worker.stop()
            self.arduino_worker.wait()

        if self.aux_arduino_manager is not None:
            self.aux_arduino_manager.stop_all()

        if self.live_inference_worker is not None:
            self.live_inference_worker.shutdown()

        event.accept()

    def _reset_ttl_plot(self):
        """Reset TTL plot data and time base."""
        for data in self.ttl_plot_data.values():
            data.clear()
        for data in self.camera_line_plot_data.values():
            data.clear()
        self.time_data.clear()
        self.camera_line_time_data.clear()
        self.plot_start_time = datetime.now()
        self.camera_line_plot_start_time_s = None
        self.camera_line_last_signature = None
        self.ttl_plot.setXRange(0, self.ttl_window_seconds)
        self.behavior_plot.setXRange(0, self.ttl_window_seconds)
        self.camera_line_plot.setXRange(0, self.ttl_window_seconds)
        for curve in list(self.ttl_output_curves.values()) + list(self.behavior_curves.values()) + list(self.camera_line_curves.values()):
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
