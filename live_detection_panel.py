"""Right-rail widget for model setup, ROI drawing, and live TTL trigger rules."""

from __future__ import annotations

import uuid
from typing import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from live_detection_logic import build_rule_label, format_roi_properties, normalize_output_id
from live_detection_types import BehaviorROI, LiveTriggerRule

_PATH_EDGE_QUOTES = "\"'“”‘’"


def _normalize_pasted_path(value: object) -> str:
    """Accept plain or quoted paths pasted from Explorer/PowerShell."""
    return str(value or "").strip().strip(_PATH_EDGE_QUOTES).strip()


class _CollapsibleSection(QWidget):
    """A togglable section with a clickable header and animated content reveal."""

    def __init__(self, title: str, parent: QWidget | None = None, expanded: bool = False) -> None:
        super().__init__(parent)
        self._expanded = expanded

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._toggle_btn = QPushButton(f"  {title}")
        self._toggle_btn.setObjectName("ghostButton")
        self._toggle_btn.setStyleSheet(
            "QPushButton#ghostButton { text-align: left; font-weight: 600; "
            "font-size: 11px; color: #8dd0ff; padding: 4px 6px; "
            "border: none; border-radius: 8px; background: transparent; }"
            "QPushButton#ghostButton:hover { background: #12202f; }"
        )
        self._toggle_btn.setCursor(Qt.PointingHandCursor)
        self._toggle_btn.clicked.connect(self.toggle)
        root.addWidget(self._toggle_btn)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 4, 0, 0)
        self._content_layout.setSpacing(6)
        root.addWidget(self._content)

        self._content.setVisible(expanded)
        self._update_arrow()

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._update_arrow()

    def _update_arrow(self) -> None:
        arrow = "\u25BC" if self._expanded else "\u25B6"
        text = self._toggle_btn.text().lstrip(" \u25BC\u25B6").strip()
        self._toggle_btn.setText(f"{arrow}  {text}")


class LiveDetectionPanel(QWidget):
    """Control surface for live segmentation, ROI drawing, and TTL triggering."""

    toggle_detection_requested = Signal(bool)
    start_roi_draw_requested = Signal(str)
    center_circle_roi_requested = Signal(str)
    edit_roi_requested = Signal(str)
    finish_polygon_requested = Signal()
    remove_roi_requested = Signal(str)
    clear_rois_requested = Signal()
    save_rois_requested = Signal()
    load_rois_requested = Signal()
    output_mapping_changed = Signal(dict)
    output_labels_changed = Signal(dict)
    add_rule_requested = Signal(object)
    edit_rule_requested = Signal(str)
    test_rule_requested = Signal(str)
    remove_rule_requested = Signal(str)
    overlay_options_changed = Signal()
    run_behavior_toggled = Signal(bool)
    behavior_backend_changed = Signal(str)
    flip_orientation_requested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._roi_names: list[str] = []
        self._rule_ids: list[str] = []
        self._roi_dialog: QDialog | None = None
        self._rule_dialog: QDialog | None = None
        self._visible_output_ids: list[str] = ["DO1"]
        # Friendly display names per logical output ("DO1" -> "Laser 473nm"). The DO
        # id stays the canonical key everywhere (arbiter, Arduino, persistence); the
        # label is cosmetic and surfaces in the rule Output dropdowns / rule table.
        self._output_labels: dict[str, str] = {}
        # Output dropdowns in the rule builders, repopulated with labels on change.
        self._output_combos: list[QComboBox] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        section_tabs = QTabWidget()
        section_tabs.setObjectName("liveDetectionSectionTabs")
        section_tabs.setTabPosition(QTabWidget.North)
        section_tabs.setDocumentMode(True)
        section_tabs.setUsesScrollButtons(False)
        section_tabs.setElideMode(Qt.ElideNone)

        section_tabs.addTab(self._scrollable_section(self._build_model_group()), "Model")
        section_tabs.addTab(self._scrollable_section(self._build_roi_group()), "ROIs")
        section_tabs.addTab(self._scrollable_section(self._build_output_group()), "TTL")
        section_tabs.addTab(self._scrollable_section(self._build_rule_group()), "Rules")
        section_tabs.setCurrentIndex(0)
        layout.addWidget(section_tabs, 1)

    @staticmethod
    def _scrollable_section(widget: QWidget) -> QScrollArea:
        # Anchor the section to the top of the viewport so extra vertical
        # space never gets distributed between rows as awkward gaps.
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(widget)
        container_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(container)
        return scroll

    @staticmethod
    def _section_divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #1c3045;")
        return line

    def _build_model_group(self) -> QWidget:
        group = QGroupBox("Live Detection")
        group.setMinimumWidth(0)
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        root = QVBoxLayout(group)
        root.setSpacing(8)

        # ── Model & checkpoint (always visible) ──────────────────────
        model_form = QFormLayout()
        model_form.setHorizontalSpacing(10)
        model_form.setVerticalSpacing(6)

        self.combo_model_key = QComboBox()
        self.combo_model_key.addItem("RF-DETR Seg Nano", "rfdetr-seg-nano")
        self.combo_model_key.addItem("RF-DETR Seg Small", "rfdetr-seg-small")
        self.combo_model_key.addItem("RF-DETR Seg Medium", "rfdetr-seg-medium")
        self.combo_model_key.addItem("RF-DETR Seg Large", "rfdetr-seg-large")
        self.combo_model_key.addItem("RF-DETR Seg xLarge", "rfdetr-seg-xlarge")
        self.combo_model_key.addItem("YOLO Seg", "yolo-seg")
        model_form.addRow("Model:", self.combo_model_key)

        checkpoint_row = QHBoxLayout()
        self.edit_checkpoint = QLineEdit()
        self.edit_checkpoint.setPlaceholderText("Segmentation .pt / .pth / .ckpt")
        self._prepare_path_edit(self.edit_checkpoint)
        self.edit_checkpoint.editingFinished.connect(
            lambda: self._normalize_path_edit(self.edit_checkpoint)
        )
        checkpoint_row.addWidget(self.edit_checkpoint, 1)
        btn_browse = QPushButton("Browse")
        btn_browse.setObjectName("ghostButton")
        btn_browse.setFixedWidth(78)
        btn_browse.setCursor(Qt.PointingHandCursor)
        btn_browse.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(btn_browse)
        model_form.addRow("Checkpoint:", checkpoint_row)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.01, 1.0)
        self.spin_threshold.setDecimals(2)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(0.35)
        model_form.addRow("Confidence:", self.spin_threshold)

        self.spin_expected_mice = QSpinBox()
        self.spin_expected_mice.setRange(1, 8)
        self.spin_expected_mice.setValue(1)
        model_form.addRow("Mouse count:", self.spin_expected_mice)

        root.addLayout(model_form)

        # ── Overlay checkboxes (compact row) ─────────────────────────
        root.addWidget(self._section_divider())
        overlay_label = QLabel("Overlay")
        overlay_label.setStyleSheet("color: #8dd0ff; font-weight: 600; font-size: 11px;")
        root.addWidget(overlay_label)

        overlay_grid = QGridLayout()
        overlay_grid.setHorizontalSpacing(12)
        overlay_grid.setVerticalSpacing(6)
        self.check_show_masks = QCheckBox("Masks")
        self.check_show_masks.setChecked(True)
        self.check_show_masks.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        self.check_show_boxes = QCheckBox("Boxes")
        self.check_show_boxes.setChecked(True)
        self.check_show_boxes.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        self.check_show_keypoints = QCheckBox("Keypoints")
        self.check_show_keypoints.setChecked(True)
        self.check_show_keypoints.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        self.check_show_behavior = QCheckBox("Behavior")
        self.check_show_behavior.setToolTip(
            "Run the live behavior model and overlay per-mouse behavior subtitles "
            "(e.g. 'mounting (0.93)'). Requires the behavior model package + a GPU."
        )
        self.check_show_behavior.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        self.check_show_behavior.toggled.connect(lambda checked: self.run_behavior_toggled.emit(bool(checked)))
        self.check_save_overlay_video = QCheckBox("Overlay MP4")
        self.check_save_overlay_video.setToolTip(
            "Save a sidecar preview video with boxes, masks, and ROI overlays while recording."
        )
        self.check_save_overlay_video.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        self.check_save_tracking_csv = QCheckBox("Track CSV")
        self.check_save_tracking_csv.setToolTip(
            "Save a DLC-style tracking CSV with body center and keypoints aligned to recorded frame timestamps."
        )
        self.check_save_tracking_csv.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        self.check_save_masks_coco = QCheckBox("Mask COCO")
        self.check_save_masks_coco.setToolTip(
            "Save tracked segmentation masks in COCO JSON format while recording."
        )
        self.check_save_masks_coco.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        for idx, checkbox in enumerate(
            (
                self.check_show_masks,
                self.check_show_boxes,
                self.check_show_keypoints,
                self.check_show_behavior,
            )
        ):
            overlay_grid.addWidget(checkbox, idx // 3, idx % 3)
        root.addLayout(overlay_grid)

        # Disk-output toggles are a separate concern from live-display toggles;
        # group them under their own header so "saved while recording" reads
        # distinctly from "drawn on the live preview".
        save_label = QLabel("Save while recording")
        save_label.setStyleSheet("color: #8dd0ff; font-weight: 600; font-size: 11px;")
        root.addWidget(save_label)
        save_grid = QGridLayout()
        save_grid.setHorizontalSpacing(12)
        save_grid.setVerticalSpacing(6)
        for idx, checkbox in enumerate(
            (
                self.check_save_overlay_video,
                self.check_save_tracking_csv,
                self.check_save_masks_coco,
            )
        ):
            save_grid.addWidget(checkbox, idx // 3, idx % 3)
        root.addLayout(save_grid)

        # Behavior method: which detector computes the behavior subtitles / triggers
        # when the "Behavior" overlay is on. Lives next to the toggle for visibility.
        behavior_method_row = QHBoxLayout()
        behavior_method_row.addWidget(QLabel("Behavior method:"))
        self.combo_behavior_backend = QComboBox()
        self.combo_behavior_backend.addItem("Rule-based (fast)", "rules")
        self.combo_behavior_backend.addItem("ML model (EmbTCN)", "ml")
        self.combo_behavior_backend.setToolTip(
            "Rule-based: sub-ms geometric/kinematic tests on keypoints + mask contours "
            "(use this for closed-loop TTL). ML model: the trained EmbTCN temporal model "
            "(more behaviors but ~0.3 s/decision)."
        )
        self.combo_behavior_backend.currentIndexChanged.connect(
            lambda _i: self.behavior_backend_changed.emit(str(self.combo_behavior_backend.currentData() or "rules"))
        )
        self.combo_behavior_backend.currentIndexChanged.connect(lambda _i: self.overlay_options_changed.emit())
        behavior_method_row.addWidget(self.combo_behavior_backend, 1)
        behavior_method_row.addStretch()
        root.addLayout(behavior_method_row)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Mask opacity:"))
        self.spin_mask_opacity = QSpinBox()
        self.spin_mask_opacity.setRange(0, 100)
        self.spin_mask_opacity.setValue(30)
        self.spin_mask_opacity.setSuffix(" %")
        self.spin_mask_opacity.setToolTip("Transparency of live mask overlays in preview and overlay video")
        self.spin_mask_opacity.valueChanged.connect(lambda _value: self.overlay_options_changed.emit())
        opacity_row.addWidget(self.spin_mask_opacity)
        opacity_row.addStretch()
        root.addLayout(opacity_row)

        # ── Advanced settings (collapsible) ──────────────────────────
        root.addWidget(self._section_divider())
        adv = _CollapsibleSection("Advanced", parent=group, expanded=False)
        adv_form = QFormLayout()
        adv_form.setHorizontalSpacing(10)
        adv_form.setVerticalSpacing(6)

        self.combo_keypoint_source = QComboBox()
        self.combo_keypoint_source.addItem("YOLO pose", "yolo_pose")
        self.combo_keypoint_source.addItem("Mask geometry", "mask_geometry")
        self.combo_keypoint_source.addItem("None", "none")
        self.combo_keypoint_source.setToolTip(
            "Choose whether keypoints come from a YOLO pose model or directly from segmentation masks."
        )
        self.combo_keypoint_source.currentIndexChanged.connect(lambda _index: self._update_keypoint_source_controls())
        adv_form.addRow("Keypoints:", self.combo_keypoint_source)

        self.check_closed_loop_fast = QCheckBox("Closed-loop fast")
        self.check_closed_loop_fast.setChecked(True)
        self.check_closed_loop_fast.setToolTip(
            "Use mask geometry without full-resolution mask output for low-latency TTL feedback."
        )
        self.check_closed_loop_fast.toggled.connect(lambda _checked: self._update_keypoint_source_controls())
        self.check_closed_loop_fast.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        adv_form.addRow("Loop mode:", self.check_closed_loop_fast)

        self.check_clean_masks = QCheckBox("Clean masks")
        self.check_clean_masks.setChecked(True)
        self.check_clean_masks.setToolTip(
            "Keep only the largest solid mask blob. Disable for maximum frame rate when model masks are already clean."
        )
        self.check_clean_masks.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        adv_form.addRow("Mask quality:", self.check_clean_masks)

        pose_checkpoint_row = QHBoxLayout()
        self.edit_pose_checkpoint = QLineEdit()
        self.edit_pose_checkpoint.setPlaceholderText("Optional YOLO pose .pt")
        self._prepare_path_edit(self.edit_pose_checkpoint)
        self.edit_pose_checkpoint.editingFinished.connect(
            lambda: self._normalize_path_edit(self.edit_pose_checkpoint)
        )
        pose_checkpoint_row.addWidget(self.edit_pose_checkpoint, 1)
        self.btn_browse_pose = QPushButton("Browse")
        self.btn_browse_pose.setFixedWidth(78)
        self.btn_browse_pose.clicked.connect(self._browse_pose_checkpoint)
        pose_checkpoint_row.addWidget(self.btn_browse_pose)
        adv_form.addRow("Pose ckpt:", pose_checkpoint_row)

        self.spin_pose_threshold = QDoubleSpinBox()
        self.spin_pose_threshold.setRange(0.01, 1.0)
        self.spin_pose_threshold.setDecimals(2)
        self.spin_pose_threshold.setSingleStep(0.05)
        self.spin_pose_threshold.setValue(0.25)
        self.spin_pose_threshold.setToolTip(
            "Minimum keypoint confidence for the paired YOLO pose model."
        )
        adv_form.addRow("Pose conf:", self.spin_pose_threshold)

        self.spin_min_pose_kp = QSpinBox()
        self.spin_min_pose_kp.setRange(0, 32)
        self.spin_min_pose_kp.setValue(0)
        self.spin_min_pose_kp.setToolTip(
            "Drop pose keypoints when fewer than this many are confident. 0 = off."
        )
        adv_form.addRow("Min kp:", self.spin_min_pose_kp)

        self.edit_selected_classes = QLineEdit("0")
        self.edit_selected_classes.setPlaceholderText("e.g. 0 or 0,1")
        adv_form.addRow("Classes:", self.edit_selected_classes)

        self.combo_identity_mode = QComboBox()
        self.combo_identity_mode.addItem("Tracker IDs", "tracker")
        self.combo_identity_mode.addItem("Model Class IDs", "model_class")
        adv_form.addRow("Identity:", self.combo_identity_mode)

        self.spin_inference_width = QSpinBox()
        self.spin_inference_width.setRange(0, 4096)
        self.spin_inference_width.setSingleStep(64)
        self.spin_inference_width.setSpecialValueText("Preview resolution")
        self.spin_inference_width.setValue(960)
        self.spin_inference_width.setToolTip(
            "Downscale frames before model inference. Lower values reduce lag."
        )
        adv_form.addRow("Max width:", self.spin_inference_width)

        self.combo_acceleration_mode = QComboBox()
        self.combo_acceleration_mode.addItem("Balanced", "balanced")
        self.combo_acceleration_mode.addItem("Max GPU", "max_gpu")
        self.combo_acceleration_mode.addItem("Max GPU (TensorRT)", "max_gpu_trt")
        self.combo_acceleration_mode.addItem("Compatibility", "compatibility")
        self.combo_acceleration_mode.setToolTip(
            "Max GPU enables cuDNN benchmarking and TF32 on CUDA. "
            "It can improve throughput, but may not help if CPU preprocessing is the bottleneck.\n"
            "Max GPU (TensorRT) loads a prebuilt .engine next to the RF-DETR checkpoint "
            "(build it with scripts/build_rfdetr_engine.py) for the fastest inference; "
            "it falls back to Max GPU if no engine is found."
        )
        adv_form.addRow("GPU mode:", self.combo_acceleration_mode)

        adv.content_layout().addLayout(adv_form)
        root.addWidget(adv)
        self._update_keypoint_source_controls()

        # ── Status + action ──────────────────────────────────────────
        root.addWidget(self._section_divider())
        status_row = QHBoxLayout()
        status_label = QLabel("Status:")
        status_label.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        self.label_status = QLabel("Idle")
        self.label_status.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        status_row.addWidget(status_label)
        status_row.addWidget(self.label_status, 1)
        root.addLayout(status_row)

        # ── Manual head/tail correction ──────────────────────────────
        # Geometry orientation is automatic, but a motionless subject can lock the
        # wrong way round; these buttons let the user swap a mouse's nose<->tail live.
        flip_row = QHBoxLayout()
        flip_row.setSpacing(6)
        flip_caption = QLabel("Fix head↔tail:")
        flip_caption.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        flip_row.addWidget(flip_caption)
        self._flip_buttons_container = QHBoxLayout()
        self._flip_buttons_container.setSpacing(4)
        flip_row.addLayout(self._flip_buttons_container, 1)
        root.addLayout(flip_row)
        self._flip_buttons: list[QPushButton] = []
        self._rebuild_flip_buttons()
        self.spin_expected_mice.valueChanged.connect(self._rebuild_flip_buttons)

        self.btn_toggle_detection = QPushButton("Start Live Inference")
        self.btn_toggle_detection.setCheckable(True)
        self.btn_toggle_detection.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_detection.setToolTip(
            "Run the segmentation model on the live preview. Requires a connected "
            "camera with preview enabled."
        )
        self.btn_toggle_detection.toggled.connect(self._on_toggle_detection)
        root.addWidget(self.btn_toggle_detection)
        return group

    def _build_roi_group(self) -> QWidget:
        group = QGroupBox("Behavioural ROIs")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        self.edit_roi_name = QLineEdit("ROI 1")
        self.combo_roi_shape = QComboBox()
        self.combo_roi_shape.addItems(["Rectangle", "Circle", "Polygon"])
        self.btn_draw_roi = QPushButton("Draw")
        self.btn_draw_roi.setFixedWidth(56)
        self.btn_draw_roi.clicked.connect(self._request_roi_draw)
        name_row.addWidget(self.edit_roi_name, 1)
        name_row.addWidget(self.combo_roi_shape)
        name_row.addWidget(self.btn_draw_roi)
        layout.addLayout(name_row)

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        self.btn_center_circle_roi = QPushButton("Center Circle")
        self.btn_center_circle_roi.setObjectName("ghostButton")
        self.btn_center_circle_roi.clicked.connect(self._request_center_circle_roi)
        button_row.addWidget(self.btn_center_circle_roi)
        self.btn_finish_polygon = QPushButton("Finish Polygon")
        self.btn_finish_polygon.setObjectName("ghostButton")
        self.btn_finish_polygon.clicked.connect(self.finish_polygon_requested.emit)
        button_row.addWidget(self.btn_finish_polygon)
        button_row.addStretch()
        layout.addLayout(button_row)

        manage_row = QHBoxLayout()
        self.label_roi_summary = QLabel("ROIs: 0")
        self.label_roi_summary.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        manage_row.addWidget(self.label_roi_summary, 1)
        self.btn_manage_rois = QPushButton("Manage ROIs")
        self.btn_manage_rois.setObjectName("ghostButton")
        self.btn_manage_rois.clicked.connect(self._show_roi_dialog)
        manage_row.addWidget(self.btn_manage_rois)
        layout.addLayout(manage_row)

        self.roi_table = QTableWidget(0, 3)
        self.roi_table.setHorizontalHeaderLabels(["ROI", "Type", "Properties"])
        self.roi_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.roi_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.roi_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.roi_table.verticalHeader().setVisible(False)
        self.roi_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.roi_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.roi_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.roi_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.roi_table.setWordWrap(False)
        self.roi_table.setMinimumHeight(120)
        self.roi_table.cellDoubleClicked.connect(self._request_roi_edit)
        self.btn_remove_roi = QPushButton("Remove ROI")
        self.btn_remove_roi.clicked.connect(self._remove_selected_roi)
        self.btn_clear_rois = QPushButton("Clear ROIs")
        self.btn_clear_rois.clicked.connect(self.clear_rois_requested.emit)
        return group

    def _build_output_group(self) -> QWidget:
        group = QGroupBox("TTL Outputs")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        hint = QLabel("Name each output and map its Arduino pin(s). Names appear in the rule builders.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8fa6bf; font-size: 11px;")
        layout.addWidget(hint)

        header = QGridLayout()
        header.setHorizontalSpacing(8)
        for col, text, width in (
            (0, "Out", 32), (1, "Name", 0), (2, "Pin(s)", 0), (3, "", 24),
        ):
            cap = QLabel(text)
            cap.setStyleSheet("color: #6f8aa6; font-size: 10px; font-weight: 600;")
            if width:
                cap.setFixedWidth(width)
            header.addWidget(cap, 0, col)
        header.setColumnStretch(1, 2)
        header.setColumnStretch(2, 3)
        layout.addLayout(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 3)
        self.output_pin_edits: dict[str, QLineEdit] = {}
        self.output_name_edits: dict[str, QLineEdit] = {}
        self.output_remove_btns: dict[str, QPushButton] = {}
        # Each row's widgets, in column order, so visibility toggles uniformly.
        self.output_row_widgets: dict[str, tuple] = {}
        for index in range(1, 9):
            output_id = f"DO{index}"
            tag = QLabel(output_id)
            tag.setStyleSheet("color: #8dd0ff; font-weight: 600; font-size: 11px;")
            tag.setFixedWidth(32)
            name_edit = QLineEdit()
            name_edit.setPlaceholderText("e.g. Laser 473nm")
            pin_edit = QLineEdit()
            pin_edit.setPlaceholderText("Pin(s), e.g. 30 or 30,31")
            remove_btn = QPushButton("✕")
            remove_btn.setObjectName("ghostButton")
            remove_btn.setFixedWidth(24)
            remove_btn.setToolTip(f"Remove {output_id} from the list")
            remove_btn.clicked.connect(lambda _checked=False, oid=output_id: self._remove_output_row(oid))
            # DO1 is the always-present anchor output; it can be renamed but not removed.
            if output_id == "DO1":
                remove_btn.setEnabled(False)
                remove_btn.setVisible(False)
            self.output_name_edits[output_id] = name_edit
            self.output_pin_edits[output_id] = pin_edit
            self.output_remove_btns[output_id] = remove_btn
            self.output_row_widgets[output_id] = (tag, name_edit, pin_edit, remove_btn)
            grid.addWidget(tag, index - 1, 0)
            grid.addWidget(name_edit, index - 1, 1)
            grid.addWidget(pin_edit, index - 1, 2)
            grid.addWidget(remove_btn, index - 1, 3)
        layout.addLayout(grid)

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        self.btn_add_output = QPushButton("+ Add output")
        self.btn_add_output.setObjectName("ghostButton")
        self.btn_add_output.clicked.connect(self._show_next_output_row)
        button_row.addWidget(self.btn_add_output)
        self.btn_apply_output_map = QPushButton("Apply Mapping")
        self.btn_apply_output_map.clicked.connect(self._emit_output_mapping)
        button_row.addWidget(self.btn_apply_output_map, 1)
        layout.addLayout(button_row)
        self._set_visible_output_rows(["DO1"])
        return group

    def _build_rule_group(self) -> QWidget:
        group = QGroupBox("Trigger Rules")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        rules_row = QHBoxLayout()
        self.label_rules_summary = QLabel("Rules: 0")
        self.label_rules_summary.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        rules_row.addWidget(self.label_rules_summary, 1)
        self.label_active_outputs = QLabel("all low")
        self.label_active_outputs.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        rules_row.addWidget(self.label_active_outputs)
        self.btn_manage_rules = QPushButton("Manage")
        self.btn_manage_rules.setObjectName("ghostButton")
        self.btn_manage_rules.setFixedWidth(68)
        self.btn_manage_rules.clicked.connect(self._show_rule_dialog)
        rules_row.addWidget(self.btn_manage_rules)
        layout.addLayout(rules_row)

        self.rule_table = QTableWidget(0, 2)
        self.rule_table.setHorizontalHeaderLabels(["Rule", "State"])
        self.rule_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.rule_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.rule_table.verticalHeader().setVisible(False)
        self.rule_table.verticalHeader().setDefaultSectionSize(28)
        self.rule_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.rule_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.rule_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rule_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.rule_table.setWordWrap(False)
        self.rule_table.setMinimumHeight(100)
        self.rule_table.cellDoubleClicked.connect(self._request_rule_edit)
        self.btn_edit_rule = QPushButton("Edit Rule")
        self.btn_edit_rule.clicked.connect(lambda _checked=False: self._request_rule_edit())
        self.btn_test_rule = QPushButton("Test Rule")
        self.btn_test_rule.clicked.connect(lambda _checked=False: self._request_rule_test())
        self.btn_remove_rule = QPushButton("Remove Rule")
        self.btn_remove_rule.clicked.connect(self._remove_selected_rule)

        # ── ROI rule builder (collapsible) ───────────────────────────
        layout.addWidget(self._section_divider())
        roi_section = _CollapsibleSection("Add ROI Rule", parent=group, expanded=False)
        roi_rule_box = QFrame()
        roi_form = QFormLayout(roi_rule_box)
        self.spin_rule_mouse_id = QSpinBox()
        self.spin_rule_mouse_id.setRange(1, 8)
        self.combo_rule_roi = QComboBox()
        self.combo_rule_output = self._register_output_combo(QComboBox())
        self.combo_rule_mode = QComboBox()
        self.combo_rule_mode.addItem("Gate", "gate")
        self.combo_rule_mode.addItem("Level", "level")
        self.combo_rule_mode.addItem("Pulse", "pulse")
        self.spin_rule_duration = QSpinBox()
        self.spin_rule_duration.setRange(1, 600000)
        self.spin_rule_duration.setValue(250)
        self.spin_rule_pulse_count = QSpinBox()
        self.spin_rule_pulse_count.setRange(1, 10000)
        self.spin_rule_pulse_count.setValue(1)
        self.spin_rule_frequency = QDoubleSpinBox()
        self.spin_rule_frequency.setRange(0.001, 1000.0)
        self.spin_rule_frequency.setDecimals(3)
        self.spin_rule_frequency.setSingleStep(1.0)
        self.spin_rule_frequency.setValue(1.0)
        self.spin_rule_frequency.setSuffix(" Hz")
        self.combo_rule_activation = QComboBox()
        self.combo_rule_activation.addItem("At entry", "entry")
        self.combo_rule_activation.addItem("At exit", "exit")
        self.combo_rule_activation.addItem("Continuous", "continuous")
        self.spin_rule_inter_train_interval = QSpinBox()
        self.spin_rule_inter_train_interval.setRange(0, 600000)
        self.spin_rule_inter_train_interval.setValue(1000)
        self.spin_rule_inter_train_interval.setSuffix(" ms")
        self.spin_rule_min_active = QSpinBox()
        self.spin_rule_min_active.setRange(0, 600000)
        self.spin_rule_min_active.setValue(0)
        self.spin_rule_min_active.setSingleStep(50)
        self.spin_rule_min_active.setSuffix(" ms")
        self.spin_rule_min_active.setToolTip(
            "Mouse must stay in the ROI at least this long before the output fires (0 = immediate)."
        )
        roi_form.addRow("ROI mouse:", self.spin_rule_mouse_id)
        roi_form.addRow("ROI:", self.combo_rule_roi)
        roi_form.addRow("Min duration:", self.spin_rule_min_active)
        roi_form.addRow("Output:", self.combo_rule_output)
        roi_form.addRow("Mode:", self.combo_rule_mode)
        self.label_rule_duration = QLabel("Pulse ms:")
        self.label_rule_pulse_count = QLabel("Pulse count:")
        self.label_rule_frequency = QLabel("Frequency:")
        self.label_rule_activation = QLabel("Activation:")
        self.label_rule_inter_train_interval = QLabel("Inter-train interval:")
        roi_form.addRow(self.label_rule_duration, self.spin_rule_duration)
        roi_form.addRow(self.label_rule_pulse_count, self.spin_rule_pulse_count)
        roi_form.addRow(self.label_rule_frequency, self.spin_rule_frequency)
        roi_form.addRow(self.label_rule_activation, self.combo_rule_activation)
        roi_form.addRow(self.label_rule_inter_train_interval, self.spin_rule_inter_train_interval)
        btn_add_roi_rule = QPushButton("Add ROI Rule")
        btn_add_roi_rule.clicked.connect(self._add_roi_rule)
        roi_form.addRow("", btn_add_roi_rule)
        self.combo_rule_mode.currentIndexChanged.connect(lambda _index: self._update_rule_pulse_controls())
        self.combo_rule_activation.currentIndexChanged.connect(lambda _index: self._update_rule_pulse_controls())
        roi_section.content_layout().addWidget(roi_rule_box)
        layout.addWidget(roi_section)

        # ── Proximity rule builder (collapsible) ─────────────────────
        prox_section = _CollapsibleSection("Add Proximity / Contact Rule", parent=group, expanded=False)
        proximity_box = QFrame()
        proximity_form = QFormLayout(proximity_box)
        self.combo_prox_rule_type = QComboBox()
        self.combo_prox_rule_type.addItem("Center distance", "mouse_proximity")
        self.combo_prox_rule_type.addItem("Mask edge touch", "mask_contact")
        self.spin_prox_mouse_id = QSpinBox()
        self.spin_prox_mouse_id.setRange(1, 8)
        self.spin_prox_peer_id = QSpinBox()
        self.spin_prox_peer_id.setRange(1, 8)
        self.spin_prox_peer_id.setValue(2)
        self.spin_prox_distance = QSpinBox()
        self.spin_prox_distance.setRange(1, 2000)
        self.spin_prox_distance.setValue(80)
        self.combo_prox_output = self._register_output_combo(QComboBox())
        self.combo_prox_mode = QComboBox()
        self.combo_prox_mode.addItem("Gate", "gate")
        self.combo_prox_mode.addItem("Level", "level")
        self.combo_prox_mode.addItem("Pulse", "pulse")
        self.spin_prox_duration = QSpinBox()
        self.spin_prox_duration.setRange(1, 600000)
        self.spin_prox_duration.setValue(250)
        self.spin_prox_pulse_count = QSpinBox()
        self.spin_prox_pulse_count.setRange(1, 10000)
        self.spin_prox_pulse_count.setValue(1)
        self.spin_prox_frequency = QDoubleSpinBox()
        self.spin_prox_frequency.setRange(0.001, 1000.0)
        self.spin_prox_frequency.setDecimals(3)
        self.spin_prox_frequency.setSingleStep(1.0)
        self.spin_prox_frequency.setValue(1.0)
        self.spin_prox_frequency.setSuffix(" Hz")
        self.combo_prox_activation = QComboBox()
        self.combo_prox_activation.addItem("At entry", "entry")
        self.combo_prox_activation.addItem("At exit", "exit")
        self.combo_prox_activation.addItem("Continuous", "continuous")
        self.spin_prox_inter_train_interval = QSpinBox()
        self.spin_prox_inter_train_interval.setRange(0, 600000)
        self.spin_prox_inter_train_interval.setValue(1000)
        self.spin_prox_inter_train_interval.setSuffix(" ms")
        self.spin_prox_min_active = QSpinBox()
        self.spin_prox_min_active.setRange(0, 600000)
        self.spin_prox_min_active.setValue(0)
        self.spin_prox_min_active.setSingleStep(50)
        self.spin_prox_min_active.setSuffix(" ms")
        self.spin_prox_min_active.setToolTip(
            "Condition must hold at least this long before the output fires (0 = immediate)."
        )
        proximity_form.addRow("Condition:", self.combo_prox_rule_type)
        proximity_form.addRow("Mouse A:", self.spin_prox_mouse_id)
        proximity_form.addRow("Mouse B:", self.spin_prox_peer_id)
        self.label_prox_distance = QLabel("Distance px:")
        proximity_form.addRow(self.label_prox_distance, self.spin_prox_distance)
        proximity_form.addRow("Min duration:", self.spin_prox_min_active)
        proximity_form.addRow("Output:", self.combo_prox_output)
        proximity_form.addRow("Mode:", self.combo_prox_mode)
        self.label_prox_duration = QLabel("Pulse ms:")
        self.label_prox_pulse_count = QLabel("Pulse count:")
        self.label_prox_frequency = QLabel("Frequency:")
        self.label_prox_activation = QLabel("Activation:")
        self.label_prox_inter_train_interval = QLabel("Inter-train interval:")
        proximity_form.addRow(self.label_prox_duration, self.spin_prox_duration)
        proximity_form.addRow(self.label_prox_pulse_count, self.spin_prox_pulse_count)
        proximity_form.addRow(self.label_prox_frequency, self.spin_prox_frequency)
        proximity_form.addRow(self.label_prox_activation, self.combo_prox_activation)
        proximity_form.addRow(self.label_prox_inter_train_interval, self.spin_prox_inter_train_interval)
        btn_add_prox_rule = QPushButton("Add Proximity Rule")
        btn_add_prox_rule.clicked.connect(self._add_proximity_rule)
        proximity_form.addRow("", btn_add_prox_rule)
        self.combo_prox_rule_type.currentIndexChanged.connect(lambda _index: self._update_rule_pulse_controls())
        self.combo_prox_mode.currentIndexChanged.connect(lambda _index: self._update_rule_pulse_controls())
        self.combo_prox_activation.currentIndexChanged.connect(lambda _index: self._update_rule_pulse_controls())
        prox_section.content_layout().addWidget(proximity_box)
        layout.addWidget(prox_section)

        # ── Behavior rule builder (collapsible) ──────────────────────
        # Fire a TTL when the live temporal model reports a social behavior class
        # (scene-level: OR across the two mice, debounced by the streaming engine).
        behavior_section = _CollapsibleSection("Add Behavior Rule", parent=group, expanded=False)
        behavior_box = QFrame()
        behavior_form = QFormLayout(behavior_box)
        # The detector backend (rule-based vs ML) is chosen in the Overlay section next
        # to the "Behavior" toggle (self.combo_behavior_backend), so it is not repeated
        # here. This section just builds behavior_class trigger rules.
        self._default_behavior_classes = [
            "nose2nose", "nose2body", "nose2anogenital", "following",
            "chasing", "fighting", "approach", "sidebyside",
        ]
        self.combo_behavior_class = QComboBox()
        self.combo_behavior_class.addItems(self._default_behavior_classes)
        # Polarity: which mouse must be the ACTOR performing the behavior. "Any mouse"
        # keeps the legacy scene-level OR; Mouse 1/2 restrict to that directed actor
        # (e.g. fire only when the mouse of interest is the one doing anogenital).
        self.combo_behavior_subject = QComboBox()
        self.combo_behavior_subject.addItem("Any mouse", 0)
        self.combo_behavior_subject.addItem("Mouse 1", 1)
        self.combo_behavior_subject.addItem("Mouse 2", 2)
        self.combo_behavior_output = self._register_output_combo(QComboBox())
        self.combo_behavior_mode = QComboBox()
        self.combo_behavior_mode.addItem("Gate", "gate")
        self.combo_behavior_mode.addItem("Level", "level")
        self.combo_behavior_mode.addItem("Pulse", "pulse")
        self.spin_behavior_duration = QSpinBox()
        self.spin_behavior_duration.setRange(1, 600000)
        self.spin_behavior_duration.setValue(250)
        self.spin_behavior_pulse_count = QSpinBox()
        self.spin_behavior_pulse_count.setRange(1, 10000)
        self.spin_behavior_pulse_count.setValue(1)
        self.spin_behavior_frequency = QDoubleSpinBox()
        self.spin_behavior_frequency.setRange(0.001, 1000.0)
        self.spin_behavior_frequency.setDecimals(3)
        self.spin_behavior_frequency.setSingleStep(1.0)
        self.spin_behavior_frequency.setValue(1.0)
        self.spin_behavior_frequency.setSuffix(" Hz")
        self.combo_behavior_activation = QComboBox()
        self.combo_behavior_activation.addItem("At entry", "entry")
        self.combo_behavior_activation.addItem("At exit", "exit")
        self.combo_behavior_activation.addItem("Continuous", "continuous")
        self.spin_behavior_inter_train_interval = QSpinBox()
        self.spin_behavior_inter_train_interval.setRange(0, 600000)
        self.spin_behavior_inter_train_interval.setValue(1000)
        self.spin_behavior_inter_train_interval.setSuffix(" ms")
        # Minimum sustained duration: the behavior must stay active this long before
        # the TTL fires (e.g. only stimulate nose2nose that lasts > 200 ms). 0 = off.
        self.spin_behavior_min_active = QSpinBox()
        self.spin_behavior_min_active.setRange(0, 600000)
        self.spin_behavior_min_active.setValue(0)
        self.spin_behavior_min_active.setSingleStep(50)
        self.spin_behavior_min_active.setSuffix(" ms")
        self.spin_behavior_min_active.setToolTip(
            "Behavior must stay active at least this long before the output fires (0 = immediate)."
        )
        behavior_form.addRow("Behavior:", self.combo_behavior_class)
        behavior_form.addRow("Subject:", self.combo_behavior_subject)
        behavior_form.addRow("Min duration:", self.spin_behavior_min_active)
        behavior_form.addRow("Output:", self.combo_behavior_output)
        behavior_form.addRow("Mode:", self.combo_behavior_mode)
        self.label_behavior_duration = QLabel("Pulse ms:")
        self.label_behavior_pulse_count = QLabel("Pulse count:")
        self.label_behavior_frequency = QLabel("Frequency:")
        self.label_behavior_activation = QLabel("Activation:")
        self.label_behavior_inter_train_interval = QLabel("Inter-train interval:")
        behavior_form.addRow(self.label_behavior_duration, self.spin_behavior_duration)
        behavior_form.addRow(self.label_behavior_pulse_count, self.spin_behavior_pulse_count)
        behavior_form.addRow(self.label_behavior_frequency, self.spin_behavior_frequency)
        behavior_form.addRow(self.label_behavior_activation, self.combo_behavior_activation)
        behavior_form.addRow(self.label_behavior_inter_train_interval, self.spin_behavior_inter_train_interval)
        btn_add_behavior_rule = QPushButton("Add Behavior Rule")
        btn_add_behavior_rule.clicked.connect(self._add_behavior_rule)
        behavior_form.addRow("", btn_add_behavior_rule)
        self.label_behavior_status = QLabel("Behavior model: idle")
        self.label_behavior_status.setWordWrap(True)
        self.label_behavior_status.setStyleSheet("color: #8fa6bf;")
        behavior_form.addRow("Live state:", self.label_behavior_status)
        self.combo_behavior_mode.currentIndexChanged.connect(lambda _index: self._update_rule_pulse_controls())
        self.combo_behavior_activation.currentIndexChanged.connect(lambda _index: self._update_rule_pulse_controls())
        behavior_section.content_layout().addWidget(behavior_box)
        layout.addWidget(behavior_section)

        self._update_rule_pulse_controls()
        return group

    def _show_roi_dialog(self) -> None:
        if self._roi_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("Behavioural ROIs")
            dialog.resize(760, 360)
            layout = QVBoxLayout(dialog)
            hint = QLabel("Double-click a row to edit ROI geometry.")
            hint.setStyleSheet("color: #8fa6bf;")
            layout.addWidget(hint)
            layout.addWidget(self.roi_table)
            button_row = QHBoxLayout()
            btn_save_rois = QPushButton("Save ROIs…")
            btn_save_rois.setObjectName("ghostButton")
            btn_save_rois.setToolTip("Save the current ROI set to a .pkroi file")
            btn_save_rois.clicked.connect(self.save_rois_requested.emit)
            btn_load_rois = QPushButton("Load ROIs…")
            btn_load_rois.setObjectName("ghostButton")
            btn_load_rois.setToolTip("Merge ROIs from a .pkroi file into the current set")
            btn_load_rois.clicked.connect(self.load_rois_requested.emit)
            button_row.addWidget(btn_save_rois)
            button_row.addWidget(btn_load_rois)
            button_row.addStretch()
            button_row.addWidget(self.btn_remove_roi)
            button_row.addWidget(self.btn_clear_rois)
            layout.addLayout(button_row)
            self._roi_dialog = dialog
        self._present_dialog(self._roi_dialog)

    def _show_rule_dialog(self) -> None:
        if self._rule_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("Configured Trigger Rules")
            dialog.resize(780, 360)
            layout = QVBoxLayout(dialog)
            layout.addWidget(self.rule_table)
            button_row = QHBoxLayout()
            button_row.addStretch()
            button_row.addWidget(self.btn_test_rule)
            button_row.addWidget(self.btn_edit_rule)
            button_row.addWidget(self.btn_remove_rule)
            layout.addLayout(button_row)
            self._rule_dialog = dialog
        self._present_dialog(self._rule_dialog)

    def _present_dialog(self, dialog: QDialog) -> None:
        """Reliably bring a modeless dialog to the front.

        Fixes the intermittent "Manage doesn't pop up": once a dialog has been
        minimized or buried behind the main window, a bare ``show()`` is a no-op and
        ``activateWindow()`` can't un-minimize it on Windows. We explicitly clear the
        minimized state, re-center it if a stale geometry left it off every screen,
        then raise + activate so it always returns to view.
        """
        dialog.setWindowState(
            (dialog.windowState() & ~Qt.WindowMinimized) | Qt.WindowActive
        )
        dialog.show()
        try:
            screen = dialog.screen() or self.screen()
            off_screen = screen is not None and not screen.availableGeometry().intersects(
                dialog.frameGeometry()
            )
        except Exception:
            off_screen = False
        if off_screen:
            host = self.window()
            if host is not None:
                geo = dialog.frameGeometry()
                geo.moveCenter(host.frameGeometry().center())
                dialog.move(geo.topLeft())
        dialog.raise_()
        dialog.activateWindow()

    def _show_next_output_row(self) -> None:
        for output_id in self.output_pin_edits:
            if output_id not in self._visible_output_ids:
                self._set_visible_output_rows([*self._visible_output_ids, output_id])
                return

    def _remove_output_row(self, output_id: str) -> None:
        output_id = str(output_id).strip().upper()
        if output_id == "DO1":
            return  # anchor output is never removed
        # Clear its name + pins so the cleared row doesn't keep a stale mapping when
        # re-added later, then hide it and push the updated mapping immediately.
        if output_id in self.output_name_edits:
            self.output_name_edits[output_id].clear()
        if output_id in self.output_pin_edits:
            self.output_pin_edits[output_id].clear()
        self._output_labels.pop(output_id, None)
        self._set_visible_output_rows(
            [oid for oid in self._visible_output_ids if oid != output_id]
        )
        self._emit_output_mapping()

    def _set_visible_output_rows(self, visible_output_ids: Iterable[str]) -> None:
        available = [f"DO{i}" for i in range(1, 9)]
        requested = {str(output_id).strip().upper() for output_id in visible_output_ids}
        if "DO1" not in requested:
            requested.add("DO1")
        self._visible_output_ids = [output_id for output_id in available if output_id in requested]
        for output_id in available:
            visible = output_id in self._visible_output_ids
            for widget in self.output_row_widgets[output_id]:
                widget.setVisible(visible)
            # DO1's remove button stays hidden even when its row is visible.
            if output_id == "DO1":
                self.output_remove_btns[output_id].setVisible(False)
        if hasattr(self, "btn_add_output"):
            self.btn_add_output.setEnabled(len(self._visible_output_ids) < len(available))

    def _update_rule_pulse_controls(self) -> None:
        roi_pulse_visible = str(self.combo_rule_mode.currentData() or "gate") == "pulse"
        prox_distance_visible = str(self.combo_prox_rule_type.currentData() or "mouse_proximity") == "mouse_proximity"
        prox_pulse_visible = str(self.combo_prox_mode.currentData() or "gate") == "pulse"
        roi_continuous_visible = (
            roi_pulse_visible
            and str(self.combo_rule_activation.currentData() or "entry") == "continuous"
        )
        prox_continuous_visible = (
            prox_pulse_visible
            and str(self.combo_prox_activation.currentData() or "entry") == "continuous"
        )
        for widget in (
            self.label_rule_duration,
            self.spin_rule_duration,
            self.label_rule_pulse_count,
            self.spin_rule_pulse_count,
            self.label_rule_frequency,
            self.spin_rule_frequency,
            self.label_rule_activation,
            self.combo_rule_activation,
        ):
            widget.setVisible(roi_pulse_visible)
        for widget in (
            self.label_rule_inter_train_interval,
            self.spin_rule_inter_train_interval,
        ):
            widget.setVisible(roi_continuous_visible)
        for widget in (
            self.label_prox_distance,
            self.spin_prox_distance,
        ):
            widget.setVisible(prox_distance_visible)
        for widget in (
            self.label_prox_duration,
            self.spin_prox_duration,
            self.label_prox_pulse_count,
            self.spin_prox_pulse_count,
            self.label_prox_frequency,
            self.spin_prox_frequency,
            self.label_prox_activation,
            self.combo_prox_activation,
        ):
            widget.setVisible(prox_pulse_visible)
        for widget in (
            self.label_prox_inter_train_interval,
            self.spin_prox_inter_train_interval,
        ):
            widget.setVisible(prox_continuous_visible)
        if hasattr(self, "combo_behavior_mode"):
            behavior_pulse_visible = str(self.combo_behavior_mode.currentData() or "gate") == "pulse"
            behavior_continuous_visible = (
                behavior_pulse_visible
                and str(self.combo_behavior_activation.currentData() or "entry") == "continuous"
            )
            for widget in (
                self.label_behavior_duration,
                self.spin_behavior_duration,
                self.label_behavior_pulse_count,
                self.spin_behavior_pulse_count,
                self.label_behavior_frequency,
                self.spin_behavior_frequency,
                self.label_behavior_activation,
                self.combo_behavior_activation,
            ):
                widget.setVisible(behavior_pulse_visible)
            for widget in (
                self.label_behavior_inter_train_interval,
                self.spin_behavior_inter_train_interval,
            ):
                widget.setVisible(behavior_continuous_visible)

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select segmentation checkpoint",
            "",
            "Model files (*.pt *.pth *.ckpt *.bin);;All files (*.*)",
        )
        if path:
            self.edit_checkpoint.setText(_normalize_pasted_path(path))

    def _browse_pose_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO pose checkpoint",
            "",
            "YOLO pose models (*.pt);;All files (*.*)",
        )
        if path:
            self.edit_pose_checkpoint.setText(_normalize_pasted_path(path))

    def _update_keypoint_source_controls(self) -> None:
        source = str(self.combo_keypoint_source.currentData() or "yolo_pose")
        pose_enabled = source == "yolo_pose"
        fast_enabled = source == "mask_geometry"
        fast_active = fast_enabled and bool(self.check_closed_loop_fast.isChecked())
        for widget in (
            self.edit_pose_checkpoint,
            getattr(self, "btn_browse_pose", None),
            self.spin_pose_threshold,
            self.spin_min_pose_kp,
        ):
            if widget is not None:
                widget.setEnabled(pose_enabled)
        self.check_closed_loop_fast.setEnabled(fast_enabled)
        for checkbox in (
            self.check_show_masks,
            self.check_save_overlay_video,
            self.check_save_masks_coco,
        ):
            checkbox.setEnabled(not fast_active)
            if fast_active and checkbox.isChecked():
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)

    @staticmethod
    def _prepare_path_edit(edit: QLineEdit) -> None:
        edit.setMinimumWidth(0)
        edit.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        edit.setClearButtonEnabled(True)

    @staticmethod
    def _normalize_path_edit(edit: QLineEdit) -> None:
        edit.setText(_normalize_pasted_path(edit.text()))

    def _request_roi_draw(self) -> None:
        shape = self.combo_roi_shape.currentText().strip().lower()
        self.start_roi_draw_requested.emit(shape)

    def _request_center_circle_roi(self) -> None:
        self.center_circle_roi_requested.emit(self.selected_roi_name() or self.current_roi_name())

    def _remove_selected_roi(self) -> None:
        roi_name = self.selected_roi_name()
        if not roi_name:
            return
        self.remove_roi_requested.emit(roi_name)

    def _request_roi_edit(self, row: int, _column: int) -> None:
        roi_name = self._roi_name_for_row(int(row))
        if roi_name:
            self.edit_roi_requested.emit(roi_name)

    def _emit_output_mapping(self) -> None:
        # Only emit rows the user is actually using (visible). Hidden/removed rows
        # contribute no pins and no label so they stay out of the mapping.
        mapping = {
            output_id: (edit.text().strip() if output_id in self._visible_output_ids else "")
            for output_id, edit in self.output_pin_edits.items()
        }
        labels = {
            output_id: name_edit.text().strip()
            for output_id, name_edit in self.output_name_edits.items()
            if output_id in self._visible_output_ids and name_edit.text().strip()
        }
        self._output_labels = dict(labels)
        self._refresh_output_combos()
        self.output_mapping_changed.emit(mapping)
        self.output_labels_changed.emit(labels)

    @staticmethod
    def _selected_output_id(combo: QComboBox) -> str:
        """Resolve a rule-builder output combo to its canonical DO id (data, not label)."""
        return normalize_output_id(str(combo.currentData() or combo.currentText()))

    def _output_display(self, output_id: str) -> str:
        """Combo display text for an output: 'Laser 473nm (DO1)' or just 'DO1'."""
        output_id = str(output_id).strip().upper()
        label = str(self._output_labels.get(output_id, "")).strip()
        return f"{label} ({output_id})" if label else output_id

    def _register_output_combo(self, combo: QComboBox) -> QComboBox:
        """Populate an output-selector combo (data = DO id) and track it for relabel."""
        for index in range(1, 9):
            output_id = f"DO{index}"
            combo.addItem(self._output_display(output_id), output_id)
        self._output_combos.append(combo)
        return combo

    def _refresh_output_combos(self) -> None:
        """Re-render every rule Output dropdown so labels stay in sync, keeping the
        currently-selected DO id selected."""
        for combo in self._output_combos:
            current = combo.currentData()
            combo.blockSignals(True)
            for index in range(combo.count()):
                output_id = combo.itemData(index)
                combo.setItemText(index, self._output_display(output_id))
            if current is not None:
                restore = combo.findData(current)
                if restore >= 0:
                    combo.setCurrentIndex(restore)
            combo.blockSignals(False)

    def set_output_labels(self, labels: dict[str, str]) -> None:
        """Apply persisted/external output names to the name fields and dropdowns."""
        self._output_labels = {
            str(k).strip().upper(): str(v).strip()
            for k, v in dict(labels or {}).items()
            if str(v).strip()
        }
        visible = list(self._visible_output_ids)
        for output_id, name_edit in self.output_name_edits.items():
            text = self._output_labels.get(output_id, "")
            name_edit.setText(text)
            if text and output_id not in visible:
                visible.append(output_id)
        self._set_visible_output_rows(visible)
        self._refresh_output_combos()

    def _add_roi_rule(self) -> None:
        roi_name = self.combo_rule_roi.currentText().strip()
        if not roi_name:
            return
        payload = LiveTriggerRule(
            rule_id=f"rule-{uuid.uuid4().hex[:8]}",
            rule_type="roi_occupancy",
            output_id=self._selected_output_id(self.combo_rule_output),
            mode=str(self.combo_rule_mode.currentData() or "gate"),
            duration_ms=int(self.spin_rule_duration.value()),
            pulse_count=int(self.spin_rule_pulse_count.value()),
            pulse_frequency_hz=float(self.spin_rule_frequency.value()),
            inter_train_interval_ms=int(self.spin_rule_inter_train_interval.value()),
            activation_pattern=str(self.combo_rule_activation.currentData() or "entry"),
            mouse_id=int(self.spin_rule_mouse_id.value()),
            roi_name=roi_name,
            min_active_ms=int(self.spin_rule_min_active.value()),
        )
        self.add_rule_requested.emit(payload)

    def _add_proximity_rule(self) -> None:
        rule_type = str(self.combo_prox_rule_type.currentData() or "mouse_proximity")
        payload = LiveTriggerRule(
            rule_id=f"rule-{uuid.uuid4().hex[:8]}",
            rule_type=rule_type,
            output_id=self._selected_output_id(self.combo_prox_output),
            mode=str(self.combo_prox_mode.currentData() or "gate"),
            duration_ms=int(self.spin_prox_duration.value()),
            pulse_count=int(self.spin_prox_pulse_count.value()),
            pulse_frequency_hz=float(self.spin_prox_frequency.value()),
            inter_train_interval_ms=int(self.spin_prox_inter_train_interval.value()),
            activation_pattern=str(self.combo_prox_activation.currentData() or "entry"),
            mouse_id=int(self.spin_prox_mouse_id.value()),
            peer_mouse_id=int(self.spin_prox_peer_id.value()),
            distance_px=float(self.spin_prox_distance.value()) if rule_type == "mouse_proximity" else 0.0,
            min_active_ms=int(self.spin_prox_min_active.value()),
        )
        self.add_rule_requested.emit(payload)

    def _add_behavior_rule(self) -> None:
        behavior_name = str(self.combo_behavior_class.currentText()).strip()
        if not behavior_name:
            return
        payload = LiveTriggerRule(
            rule_id=f"rule-{uuid.uuid4().hex[:8]}",
            rule_type="behavior_class",
            output_id=self._selected_output_id(self.combo_behavior_output),
            mode=str(self.combo_behavior_mode.currentData() or "gate"),
            duration_ms=int(self.spin_behavior_duration.value()),
            pulse_count=int(self.spin_behavior_pulse_count.value()),
            pulse_frequency_hz=float(self.spin_behavior_frequency.value()),
            inter_train_interval_ms=int(self.spin_behavior_inter_train_interval.value()),
            activation_pattern=str(self.combo_behavior_activation.currentData() or "entry"),
            behavior_name=behavior_name,
            behavior_subject_id=int(self.combo_behavior_subject.currentData() or 0),
            min_active_ms=int(self.spin_behavior_min_active.value()),
        )
        self.add_rule_requested.emit(payload)

    def set_behavior_classes(self, labels: list) -> None:
        """Populate the behavior-class combo from the loaded model (drop background)."""
        if not hasattr(self, "combo_behavior_class"):
            return
        names = [str(x) for x in (labels or []) if str(x).lower() not in ("background", "none")]
        if not names:
            return
        current = self.combo_behavior_class.currentText()
        self.combo_behavior_class.blockSignals(True)
        self.combo_behavior_class.clear()
        self.combo_behavior_class.addItems(names)
        idx = self.combo_behavior_class.findText(current)
        if idx >= 0:
            self.combo_behavior_class.setCurrentIndex(idx)
        self.combo_behavior_class.blockSignals(False)

    def set_behavior_state(self, state: object) -> None:
        """Show the live scene-level behavior decision (active classes + latency)."""
        if not hasattr(self, "label_behavior_status"):
            return
        active = dict(getattr(state, "active", {}) or {})
        probs = dict(getattr(state, "probs", {}) or {})
        latency = float(getattr(state, "latency_ms", 0.0) or 0.0)
        on = [name for name, flag in active.items() if flag and str(name).lower() != "background"]
        if on:
            parts = ", ".join(f"{n} ({probs.get(n, 0.0):.2f})" for n in on)
            self.label_behavior_status.setText(f"ACTIVE: {parts}  [{latency:.0f} ms]")
        else:
            top = sorted(
                ((n, p) for n, p in probs.items() if str(n).lower() != "background"),
                key=lambda kv: kv[1], reverse=True,
            )
            hint = f"  top: {top[0][0]} {top[0][1]:.2f}" if top else ""
            self.label_behavior_status.setText(f"Behavior model: none active [{latency:.0f} ms]{hint}")

    def _remove_selected_rule(self) -> None:
        rule_id = self.selected_rule_id()
        if not rule_id:
            return
        self.remove_rule_requested.emit(rule_id)

    def _request_rule_edit(self, row: int | None = None, _column: int | None = None) -> None:
        if isinstance(row, int):
            rule_id = self._rule_id_for_row(row)
        else:
            rule_id = self.selected_rule_id()
        if not rule_id:
            return
        self.edit_rule_requested.emit(rule_id)

    def _request_rule_test(self) -> None:
        rule_id = self.selected_rule_id()
        if not rule_id:
            return
        self.test_rule_requested.emit(rule_id)

    def _on_toggle_detection(self, checked: bool) -> None:
        self.btn_toggle_detection.setText("Stop Live Inference" if checked else "Start Live Inference")
        self.toggle_detection_requested.emit(bool(checked))

    def _rebuild_flip_buttons(self) -> None:
        """One compact 'M<n>' button per expected mouse; emits flip_orientation_requested."""
        while self._flip_buttons_container.count():
            item = self._flip_buttons_container.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._flip_buttons = []
        count = int(self.spin_expected_mice.value())
        for mouse_id in range(1, count + 1):
            btn = QPushButton(f"⇄ M{mouse_id}")
            btn.setObjectName("toggleButton")
            btn.setFixedHeight(24)
            btn.setMaximumWidth(64)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(
                f"Swap mouse {mouse_id}'s head and tail. Use when a motionless animal "
                f"is tracked the wrong way round (nose where the tail should be)."
            )
            btn.clicked.connect(lambda _checked=False, mid=mouse_id: self.flip_orientation_requested.emit(int(mid)))
            self._flip_buttons_container.addWidget(btn)
            self._flip_buttons.append(btn)
        self._flip_buttons_container.addStretch(1)

    def detection_config(self) -> dict:
        selected_classes: list[int] = []
        raw_classes = str(self.edit_selected_classes.text() or "").replace(";", ",")
        for token in raw_classes.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                selected_classes.append(int(token))
            except ValueError:
                continue
        return {
            "model_key": str(self.combo_model_key.currentData() or "rfdetr-seg-medium"),
            "checkpoint_path": _normalize_pasted_path(self.edit_checkpoint.text()),
            "keypoint_source": str(self.combo_keypoint_source.currentData() or "yolo_pose"),
            "closed_loop_fast": bool(self.check_closed_loop_fast.isChecked()),
            "pose_checkpoint_path": _normalize_pasted_path(self.edit_pose_checkpoint.text()),
            "pose_threshold": float(self.spin_pose_threshold.value()),
            "min_pose_keypoints": int(self.spin_min_pose_kp.value()),
            "clean_masks": bool(self.check_clean_masks.isChecked()),
            "threshold": float(self.spin_threshold.value()),
            "selected_class_ids": selected_classes,
            "identity_mode": str(self.combo_identity_mode.currentData() or "tracker"),
            "expected_mouse_count": int(self.spin_expected_mice.value()),
            "inference_max_width": int(self.spin_inference_width.value()),
            "acceleration_mode": str(self.combo_acceleration_mode.currentData() or "balanced"),
            "show_masks": bool(self.check_show_masks.isChecked()),
            "show_boxes": bool(self.check_show_boxes.isChecked()),
            "show_keypoints": bool(self.check_show_keypoints.isChecked()),
            "show_behavior": bool(self.check_show_behavior.isChecked()),
            "mask_opacity": float(self.spin_mask_opacity.value()) / 100.0,
            "save_overlay_video": bool(self.check_save_overlay_video.isChecked()),
            "save_tracking_csv": bool(self.check_save_tracking_csv.isChecked()),
            "save_masks_coco": bool(self.check_save_masks_coco.isChecked()),
        }

    def current_roi_name(self) -> str:
        return str(self.edit_roi_name.text() or "").strip() or "ROI"

    def selected_roi_name(self) -> str:
        return self._roi_name_for_row(self.roi_table.currentRow())

    def select_roi_row(self, roi_name: str) -> None:
        """Highlight the ROI table row for ``roi_name`` (kept in sync with the
        preview selection). No-op if the name is not present."""
        name = str(roi_name or "").strip()
        if not name:
            return
        try:
            row = self._roi_names.index(name)
        except ValueError:
            return
        if row != self.roi_table.currentRow():
            self.roi_table.blockSignals(True)
            self.roi_table.selectRow(row)
            self.roi_table.blockSignals(False)

    def selected_rule_id(self) -> str:
        return self._rule_id_for_row(self.rule_table.currentRow())

    def _roi_name_for_row(self, row: int) -> str:
        if row < 0 or row >= len(self._roi_names):
            return ""
        item = self.roi_table.item(row, 0)
        if item is None:
            return self._roi_names[row]
        value = item.data(Qt.UserRole)
        return str(value if value is not None else self._roi_names[row]).strip()

    def _rule_id_for_row(self, row: int) -> str:
        if row < 0 or row >= len(self._rule_ids):
            return ""
        item = self.rule_table.item(row, 0)
        if item is None:
            return self._rule_ids[row]
        value = item.data(Qt.UserRole)
        return str(value if value is not None else self._rule_ids[row]).strip()

    def set_detection_running(self, running: bool) -> None:
        if self.btn_toggle_detection.isChecked() != bool(running):
            self.btn_toggle_detection.blockSignals(True)
            self.btn_toggle_detection.setChecked(bool(running))
            self.btn_toggle_detection.blockSignals(False)
        self.btn_toggle_detection.setText("Stop Live Inference" if running else "Start Live Inference")

    def set_status(self, text: str) -> None:
        self.label_status.setText(str(text))

    def overlay_options(self) -> dict[str, object]:
        return {
            "show_masks": bool(self.check_show_masks.isChecked()),
            "show_boxes": bool(self.check_show_boxes.isChecked()),
            "show_keypoints": bool(self.check_show_keypoints.isChecked()),
            "show_behavior": bool(self.check_show_behavior.isChecked()),
            "behavior_backend": str(self.combo_behavior_backend.currentData() or "rules"),
            "mask_opacity": float(self.spin_mask_opacity.value()) / 100.0,
            "save_overlay_video": bool(self.check_save_overlay_video.isChecked()),
            "save_tracking_csv": bool(self.check_save_tracking_csv.isChecked()),
            "save_masks_coco": bool(self.check_save_masks_coco.isChecked()),
        }

    def behavior_backend(self) -> str:
        return str(self.combo_behavior_backend.currentData() or "rules")

    def set_behavior_backend(self, backend: str) -> None:
        idx = self.combo_behavior_backend.findData("ml" if str(backend) == "ml" else "rules")
        if idx >= 0:
            self.combo_behavior_backend.blockSignals(True)
            self.combo_behavior_backend.setCurrentIndex(idx)
            self.combo_behavior_backend.blockSignals(False)

    def set_overlay_options(
        self,
        show_masks: bool,
        show_boxes: bool,
        save_overlay_video: bool = False,
        show_keypoints: bool = True,
        save_tracking_csv: bool = False,
        save_masks_coco: bool = False,
        mask_opacity: float = 0.18,
        show_behavior: bool = False,
        behavior_backend: str = "rules",
    ) -> None:
        self.set_behavior_backend(behavior_backend)
        self.check_show_masks.blockSignals(True)
        self.check_show_masks.setChecked(bool(show_masks))
        self.check_show_masks.blockSignals(False)
        self.check_show_behavior.blockSignals(True)
        self.check_show_behavior.setChecked(bool(show_behavior))
        self.check_show_behavior.blockSignals(False)
        self.check_show_boxes.blockSignals(True)
        self.check_show_boxes.setChecked(bool(show_boxes))
        self.check_show_boxes.blockSignals(False)
        self.check_show_keypoints.blockSignals(True)
        self.check_show_keypoints.setChecked(bool(show_keypoints))
        self.check_show_keypoints.blockSignals(False)
        self.check_save_overlay_video.blockSignals(True)
        self.check_save_overlay_video.setChecked(bool(save_overlay_video))
        self.check_save_overlay_video.blockSignals(False)
        self.check_save_tracking_csv.blockSignals(True)
        self.check_save_tracking_csv.setChecked(bool(save_tracking_csv))
        self.check_save_tracking_csv.blockSignals(False)
        self.check_save_masks_coco.blockSignals(True)
        self.check_save_masks_coco.setChecked(bool(save_masks_coco))
        self.check_save_masks_coco.blockSignals(False)
        self.spin_mask_opacity.blockSignals(True)
        try:
            opacity_percent = int(round(float(mask_opacity) * 100.0))
        except (TypeError, ValueError):
            opacity_percent = 18
        self.spin_mask_opacity.setValue(max(0, min(100, opacity_percent)))
        self.spin_mask_opacity.blockSignals(False)

    def set_output_mapping(self, mapping: dict[str, Iterable[int]]) -> None:
        visible_output_ids = ["DO1"]
        for output_id, edit in self.output_pin_edits.items():
            pins = mapping.get(output_id, [])
            edit.setText(",".join(str(int(pin)) for pin in pins))
            if pins:
                visible_output_ids.append(output_id)
        self._set_visible_output_rows(visible_output_ids)

    def set_rois(self, rois: dict[str, BehaviorROI]) -> None:
        current_name = self.selected_roi_name() or self.current_roi_name()
        self._roi_names = list(rois.keys())
        self.label_roi_summary.setText(f"ROIs: {len(self._roi_names)}")
        self.roi_table.clearSpans()
        self.roi_table.clearContents()
        if not self._roi_names:
            self.roi_table.setRowCount(1)
            self.roi_table.setSpan(0, 0, 1, 3)
            self.roi_table.setItem(0, 0, QTableWidgetItem("No behavioural ROIs configured"))
        else:
            self.roi_table.setRowCount(len(self._roi_names))
        selected_row = -1
        for row, name in enumerate(self._roi_names):
            roi = rois[name]
            name_item = QTableWidgetItem(name)
            name_item.setData(Qt.UserRole, name)
            type_item = QTableWidgetItem(str(roi.roi_type).title())
            type_item.setData(Qt.UserRole, name)
            properties = format_roi_properties(roi)
            properties_item = QTableWidgetItem(properties)
            properties_item.setData(Qt.UserRole, name)
            properties_item.setToolTip(properties)
            self.roi_table.setItem(row, 0, name_item)
            self.roi_table.setItem(row, 1, type_item)
            self.roi_table.setItem(row, 2, properties_item)
            if name == current_name:
                selected_row = row
        if selected_row < 0 and self._roi_names:
            selected_row = 0
        if selected_row >= 0:
            self.roi_table.selectRow(selected_row)
        self.roi_table.resizeRowsToContents()
        self.combo_rule_roi.blockSignals(True)
        self.combo_rule_roi.clear()
        self.combo_rule_roi.addItems(self._roi_names)
        self.combo_rule_roi.blockSignals(False)

    def set_rules(self, rules: list[LiveTriggerRule], active_rule_ids: Iterable[str] = ()) -> None:
        current_rule_id = self.selected_rule_id()
        active_rule_set = set(active_rule_ids)
        self._rule_ids = [rule.rule_id for rule in rules]
        self.label_rules_summary.setText(f"Rules: {len(rules)}")
        self.rule_table.clearSpans()
        self.rule_table.clearContents()
        if not rules:
            self.rule_table.setRowCount(1)
            self.rule_table.setSpan(0, 0, 1, 2)
            empty_item = QTableWidgetItem("No trigger rules configured")
            self.rule_table.setItem(0, 0, empty_item)
            return
        self.rule_table.setRowCount(len(rules))
        selected_row = -1
        for row, rule in enumerate(rules):
            label = build_rule_label(rule, self._output_labels)
            label_item = QTableWidgetItem(label)
            label_item.setData(Qt.UserRole, rule.rule_id)
            label_item.setToolTip(label)
            is_active = rule.rule_id in active_rule_set
            # For an active ROI rule, surface the human-readable zone status the user
            # asked for ("in <ROI> zone") instead of a bare ACTIVE.
            if is_active and str(getattr(rule, "rule_type", "")) == "roi_occupancy" and getattr(rule, "roi_name", ""):
                state_text = f"in {rule.roi_name} zone"
            elif is_active:
                state_text = "ACTIVE"
            else:
                state_text = "idle"
            active_item = QTableWidgetItem(state_text)
            active_item.setData(Qt.UserRole, rule.rule_id)
            if is_active:
                active_item.setForeground(QColor("#7ce0a3"))
            self.rule_table.setItem(row, 0, label_item)
            self.rule_table.setItem(row, 1, active_item)
            if rule.rule_id == current_rule_id:
                selected_row = row
        if selected_row < 0 and rules:
            selected_row = 0
        if selected_row >= 0:
            self.rule_table.selectRow(selected_row)
        self.rule_table.resizeRowsToContents()

    def set_active_outputs(self, output_states: dict[str, bool]) -> None:
        active = [output_id for output_id, state in sorted(output_states.items()) if bool(state)]
        if active:
            self.label_active_outputs.setText(
                "Outputs high: " + ", ".join(self._output_display(oid) for oid in active)
            )
        else:
            self.label_active_outputs.setText("Outputs: all low")
