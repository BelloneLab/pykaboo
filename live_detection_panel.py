"""Right-rail widget for model setup, ROI drawing, and live TTL trigger rules."""

from __future__ import annotations

import uuid
from typing import Iterable

from PySide6.QtCore import Qt, Signal
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
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from live_detection_logic import build_rule_label, format_roi_properties, normalize_output_id
from live_detection_types import BehaviorROI, LiveTriggerRule


class LiveDetectionPanel(QWidget):
    """Control surface for live segmentation, ROI drawing, and TTL triggering."""

    toggle_detection_requested = Signal(bool)
    start_roi_draw_requested = Signal(str)
    center_circle_roi_requested = Signal(str)
    edit_roi_requested = Signal(str)
    finish_polygon_requested = Signal()
    remove_roi_requested = Signal(str)
    clear_rois_requested = Signal()
    output_mapping_changed = Signal(dict)
    add_rule_requested = Signal(object)
    edit_rule_requested = Signal(str)
    test_rule_requested = Signal(str)
    remove_rule_requested = Signal(str)
    overlay_options_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._roi_names: list[str] = []
        self._rule_ids: list[str] = []
        self._roi_dialog: QDialog | None = None
        self._rule_dialog: QDialog | None = None
        self._visible_output_ids: list[str] = ["DO1"]
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_roi_group())
        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_rule_group())

    def _build_model_group(self) -> QWidget:
        group = QGroupBox("Live Detection")
        form = QFormLayout(group)

        self.combo_model_key = QComboBox()
        self.combo_model_key.addItem("RF-DETR Seg Medium", "rfdetr-seg-medium")
        self.combo_model_key.addItem("RF-DETR Seg Large", "rfdetr-seg-large")
        self.combo_model_key.addItem("YOLO Seg", "yolo-seg")
        form.addRow("Model:", self.combo_model_key)

        checkpoint_row = QHBoxLayout()
        self.edit_checkpoint = QLineEdit()
        checkpoint_row.addWidget(self.edit_checkpoint, 1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(btn_browse)
        form.addRow("Checkpoint:", checkpoint_row)

        pose_checkpoint_row = QHBoxLayout()
        self.edit_pose_checkpoint = QLineEdit()
        self.edit_pose_checkpoint.setPlaceholderText(
            "Optional YOLO pose .pt — runs on each detection bbox"
        )
        pose_checkpoint_row.addWidget(self.edit_pose_checkpoint, 1)
        btn_browse_pose = QPushButton("Browse")
        btn_browse_pose.clicked.connect(self._browse_pose_checkpoint)
        pose_checkpoint_row.addWidget(btn_browse_pose)
        form.addRow("Pose checkpoint:", pose_checkpoint_row)

        self.spin_pose_threshold = QDoubleSpinBox()
        self.spin_pose_threshold.setRange(0.01, 1.0)
        self.spin_pose_threshold.setDecimals(2)
        self.spin_pose_threshold.setSingleStep(0.05)
        self.spin_pose_threshold.setValue(0.25)
        self.spin_pose_threshold.setToolTip(
            "Minimum keypoint confidence for the paired YOLO pose model."
        )
        form.addRow("Pose conf:", self.spin_pose_threshold)

        self.spin_min_pose_kp = QSpinBox()
        self.spin_min_pose_kp.setRange(0, 32)
        self.spin_min_pose_kp.setValue(0)
        self.spin_min_pose_kp.setToolTip(
            "Drop pose keypoints when fewer than this many are confident "
            "(seg detection still kept). 0 disables the gate."
        )
        form.addRow("Min kp:", self.spin_min_pose_kp)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.01, 1.0)
        self.spin_threshold.setDecimals(2)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(0.35)
        form.addRow("Confidence:", self.spin_threshold)

        self.edit_selected_classes = QLineEdit("0")
        self.edit_selected_classes.setPlaceholderText("Comma-separated class ids, e.g. 0 or 0,1")
        form.addRow("Mouse classes:", self.edit_selected_classes)

        self.combo_identity_mode = QComboBox()
        self.combo_identity_mode.addItem("Tracker IDs", "tracker")
        self.combo_identity_mode.addItem("Model Class IDs", "model_class")
        form.addRow("Identity:", self.combo_identity_mode)

        self.spin_expected_mice = QSpinBox()
        self.spin_expected_mice.setRange(1, 8)
        self.spin_expected_mice.setValue(1)
        form.addRow("Mouse count:", self.spin_expected_mice)

        self.spin_inference_width = QSpinBox()
        self.spin_inference_width.setRange(0, 4096)
        self.spin_inference_width.setSingleStep(64)
        self.spin_inference_width.setSpecialValueText("Preview resolution")
        self.spin_inference_width.setValue(960)
        self.spin_inference_width.setToolTip(
            "Downscale frames before model inference. Lower values reduce lag."
        )
        form.addRow("Inference Max Width:", self.spin_inference_width)

        overlay_row = QHBoxLayout()
        self.check_show_masks = QCheckBox("Show masks")
        self.check_show_masks.setChecked(True)
        self.check_show_masks.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        overlay_row.addWidget(self.check_show_masks)
        self.check_show_boxes = QCheckBox("Show boxes")
        self.check_show_boxes.setChecked(True)
        self.check_show_boxes.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        overlay_row.addWidget(self.check_show_boxes)
        self.check_show_keypoints = QCheckBox("Show keypoints")
        self.check_show_keypoints.setChecked(True)
        self.check_show_keypoints.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        overlay_row.addWidget(self.check_show_keypoints)
        overlay_row.addStretch()
        form.addRow("Overlay:", overlay_row)

        recording_overlay_row = QHBoxLayout()
        self.check_save_overlay_video = QCheckBox("Save overlay MP4")
        self.check_save_overlay_video.setToolTip(
            "Save a sidecar preview video with boxes, masks, and ROI overlays while recording."
        )
        self.check_save_overlay_video.toggled.connect(lambda _checked: self.overlay_options_changed.emit())
        recording_overlay_row.addWidget(self.check_save_overlay_video)
        recording_overlay_row.addStretch()
        form.addRow("Recording:", recording_overlay_row)

        self.label_status = QLabel("Idle")
        self.label_status.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        form.addRow("Status:", self.label_status)

        self.btn_toggle_detection = QPushButton("Start Live Inference")
        self.btn_toggle_detection.setCheckable(True)
        self.btn_toggle_detection.toggled.connect(self._on_toggle_detection)
        form.addRow("", self.btn_toggle_detection)
        return group

    def _build_roi_group(self) -> QWidget:
        group = QGroupBox("Behavioural ROIs")
        layout = QVBoxLayout(group)

        name_row = QHBoxLayout()
        self.edit_roi_name = QLineEdit("ROI 1")
        self.combo_roi_shape = QComboBox()
        self.combo_roi_shape.addItems(["Rectangle", "Circle", "Polygon"])
        name_row.addWidget(self.edit_roi_name, 1)
        name_row.addWidget(self.combo_roi_shape)
        layout.addLayout(name_row)

        button_row = QHBoxLayout()
        self.btn_draw_roi = QPushButton("Draw")
        self.btn_draw_roi.clicked.connect(self._request_roi_draw)
        button_row.addWidget(self.btn_draw_roi)
        self.btn_center_circle_roi = QPushButton("Center Circle")
        self.btn_center_circle_roi.clicked.connect(self._request_center_circle_roi)
        button_row.addWidget(self.btn_center_circle_roi)
        self.btn_finish_polygon = QPushButton("Finish Polygon")
        self.btn_finish_polygon.clicked.connect(self.finish_polygon_requested.emit)
        button_row.addWidget(self.btn_finish_polygon)
        layout.addLayout(button_row)

        manage_row = QHBoxLayout()
        self.label_roi_summary = QLabel("ROIs: 0")
        self.label_roi_summary.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        manage_row.addWidget(self.label_roi_summary, 1)
        self.btn_manage_rois = QPushButton("Manage ROIs")
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
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        self.output_pin_edits: dict[str, QLineEdit] = {}
        self.output_row_widgets: dict[str, tuple[QLabel, QLineEdit]] = {}
        for index in range(1, 9):
            output_id = f"DO{index}"
            label = QLabel(output_id)
            edit = QLineEdit()
            edit.setPlaceholderText("Pin list, e.g. 30 or 30,31")
            self.output_pin_edits[output_id] = edit
            self.output_row_widgets[output_id] = (label, edit)
            layout.addWidget(label, index - 1, 0)
            layout.addWidget(edit, index - 1, 1)
        button_row = QHBoxLayout()
        self.btn_add_output = QPushButton("Add DO")
        self.btn_add_output.clicked.connect(self._show_next_output_row)
        button_row.addWidget(self.btn_add_output)
        self.btn_apply_output_map = QPushButton("Apply DO Mapping")
        self.btn_apply_output_map.clicked.connect(self._emit_output_mapping)
        button_row.addWidget(self.btn_apply_output_map, 1)
        layout.addLayout(button_row, 8, 0, 1, 2)
        self._set_visible_output_rows(["DO1"])
        return group

    def _build_rule_group(self) -> QWidget:
        group = QGroupBox("Trigger Rules")
        layout = QVBoxLayout(group)

        rules_row = QHBoxLayout()
        self.label_rules_summary = QLabel("Rules: 0")
        self.label_rules_summary.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        rules_row.addWidget(self.label_rules_summary, 1)
        self.btn_manage_rules = QPushButton("Manage Rules")
        self.btn_manage_rules.clicked.connect(self._show_rule_dialog)
        rules_row.addWidget(self.btn_manage_rules)
        layout.addLayout(rules_row)

        self.label_active_outputs = QLabel("Outputs: all low")
        self.label_active_outputs.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        layout.addWidget(self.label_active_outputs)

        self.rule_table = QTableWidget(0, 2)
        self.rule_table.setHorizontalHeaderLabels(["Rule", "State"])
        self.rule_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.rule_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.rule_table.verticalHeader().setVisible(False)
        self.rule_table.verticalHeader().setDefaultSectionSize(32)
        self.rule_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.rule_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.rule_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rule_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.rule_table.setWordWrap(False)
        self.rule_table.setMinimumHeight(150)
        self.rule_table.cellDoubleClicked.connect(self._request_rule_edit)
        self.btn_edit_rule = QPushButton("Edit Rule")
        self.btn_edit_rule.clicked.connect(lambda _checked=False: self._request_rule_edit())
        self.btn_test_rule = QPushButton("Test Rule")
        self.btn_test_rule.clicked.connect(lambda _checked=False: self._request_rule_test())
        self.btn_remove_rule = QPushButton("Remove Rule")
        self.btn_remove_rule.clicked.connect(self._remove_selected_rule)

        add_label = QLabel("Add rules")
        add_label.setStyleSheet("color: #9dd9ff; font-weight: 700;")
        layout.addWidget(add_label)

        roi_rule_box = QFrame()
        roi_form = QFormLayout(roi_rule_box)
        self.spin_rule_mouse_id = QSpinBox()
        self.spin_rule_mouse_id.setRange(1, 8)
        self.combo_rule_roi = QComboBox()
        self.combo_rule_output = QComboBox()
        self.combo_rule_output.addItems([f"DO{i}" for i in range(1, 9)])
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
        roi_form.addRow("ROI mouse:", self.spin_rule_mouse_id)
        roi_form.addRow("ROI:", self.combo_rule_roi)
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
        layout.addWidget(roi_rule_box)

        proximity_box = QFrame()
        proximity_form = QFormLayout(proximity_box)
        self.spin_prox_mouse_id = QSpinBox()
        self.spin_prox_mouse_id.setRange(1, 8)
        self.spin_prox_peer_id = QSpinBox()
        self.spin_prox_peer_id.setRange(1, 8)
        self.spin_prox_peer_id.setValue(2)
        self.spin_prox_distance = QSpinBox()
        self.spin_prox_distance.setRange(1, 2000)
        self.spin_prox_distance.setValue(80)
        self.combo_prox_output = QComboBox()
        self.combo_prox_output.addItems([f"DO{i}" for i in range(1, 9)])
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
        proximity_form.addRow("Mouse A:", self.spin_prox_mouse_id)
        proximity_form.addRow("Mouse B:", self.spin_prox_peer_id)
        proximity_form.addRow("Distance px:", self.spin_prox_distance)
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
        self.combo_prox_mode.currentIndexChanged.connect(lambda _index: self._update_rule_pulse_controls())
        self.combo_prox_activation.currentIndexChanged.connect(lambda _index: self._update_rule_pulse_controls())
        layout.addWidget(proximity_box)
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
            button_row.addStretch()
            button_row.addWidget(self.btn_remove_roi)
            button_row.addWidget(self.btn_clear_rois)
            layout.addLayout(button_row)
            self._roi_dialog = dialog
        self._roi_dialog.show()
        self._roi_dialog.raise_()
        self._roi_dialog.activateWindow()

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
        self._rule_dialog.show()
        self._rule_dialog.raise_()
        self._rule_dialog.activateWindow()

    def _show_next_output_row(self) -> None:
        for output_id in self.output_pin_edits:
            if output_id not in self._visible_output_ids:
                self._set_visible_output_rows([*self._visible_output_ids, output_id])
                return

    def _set_visible_output_rows(self, visible_output_ids: Iterable[str]) -> None:
        available = [f"DO{i}" for i in range(1, 9)]
        requested = {str(output_id).strip().upper() for output_id in visible_output_ids}
        if "DO1" not in requested:
            requested.add("DO1")
        self._visible_output_ids = [output_id for output_id in available if output_id in requested]
        for output_id in available:
            visible = output_id in self._visible_output_ids
            label, edit = self.output_row_widgets[output_id]
            label.setVisible(visible)
            edit.setVisible(visible)
        if hasattr(self, "btn_add_output"):
            self.btn_add_output.setEnabled(len(self._visible_output_ids) < len(available))

    def _update_rule_pulse_controls(self) -> None:
        roi_pulse_visible = str(self.combo_rule_mode.currentData() or "gate") == "pulse"
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

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select segmentation checkpoint",
            "",
            "Model files (*.pt *.pth *.ckpt *.bin);;All files (*.*)",
        )
        if path:
            self.edit_checkpoint.setText(path)

    def _browse_pose_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO pose checkpoint",
            "",
            "YOLO pose models (*.pt);;All files (*.*)",
        )
        if path:
            self.edit_pose_checkpoint.setText(path)

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
        mapping = {
            output_id: edit.text().strip()
            for output_id, edit in self.output_pin_edits.items()
        }
        self.output_mapping_changed.emit(mapping)

    def _add_roi_rule(self) -> None:
        roi_name = self.combo_rule_roi.currentText().strip()
        if not roi_name:
            return
        payload = LiveTriggerRule(
            rule_id=f"rule-{uuid.uuid4().hex[:8]}",
            rule_type="roi_occupancy",
            output_id=normalize_output_id(self.combo_rule_output.currentText()),
            mode=str(self.combo_rule_mode.currentData() or "gate"),
            duration_ms=int(self.spin_rule_duration.value()),
            pulse_count=int(self.spin_rule_pulse_count.value()),
            pulse_frequency_hz=float(self.spin_rule_frequency.value()),
            inter_train_interval_ms=int(self.spin_rule_inter_train_interval.value()),
            activation_pattern=str(self.combo_rule_activation.currentData() or "entry"),
            mouse_id=int(self.spin_rule_mouse_id.value()),
            roi_name=roi_name,
        )
        self.add_rule_requested.emit(payload)

    def _add_proximity_rule(self) -> None:
        payload = LiveTriggerRule(
            rule_id=f"rule-{uuid.uuid4().hex[:8]}",
            rule_type="mouse_proximity",
            output_id=normalize_output_id(self.combo_prox_output.currentText()),
            mode=str(self.combo_prox_mode.currentData() or "gate"),
            duration_ms=int(self.spin_prox_duration.value()),
            pulse_count=int(self.spin_prox_pulse_count.value()),
            pulse_frequency_hz=float(self.spin_prox_frequency.value()),
            inter_train_interval_ms=int(self.spin_prox_inter_train_interval.value()),
            activation_pattern=str(self.combo_prox_activation.currentData() or "entry"),
            mouse_id=int(self.spin_prox_mouse_id.value()),
            peer_mouse_id=int(self.spin_prox_peer_id.value()),
            distance_px=float(self.spin_prox_distance.value()),
        )
        self.add_rule_requested.emit(payload)

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
            "checkpoint_path": str(self.edit_checkpoint.text() or "").strip(),
            "pose_checkpoint_path": str(self.edit_pose_checkpoint.text() or "").strip(),
            "pose_threshold": float(self.spin_pose_threshold.value()),
            "min_pose_keypoints": int(self.spin_min_pose_kp.value()),
            "threshold": float(self.spin_threshold.value()),
            "selected_class_ids": selected_classes,
            "identity_mode": str(self.combo_identity_mode.currentData() or "tracker"),
            "expected_mouse_count": int(self.spin_expected_mice.value()),
            "inference_max_width": int(self.spin_inference_width.value()),
            "show_masks": bool(self.check_show_masks.isChecked()),
            "show_boxes": bool(self.check_show_boxes.isChecked()),
            "show_keypoints": bool(self.check_show_keypoints.isChecked()),
            "save_overlay_video": bool(self.check_save_overlay_video.isChecked()),
        }

    def current_roi_name(self) -> str:
        return str(self.edit_roi_name.text() or "").strip() or "ROI"

    def selected_roi_name(self) -> str:
        return self._roi_name_for_row(self.roi_table.currentRow())

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

    def overlay_options(self) -> dict[str, bool]:
        return {
            "show_masks": bool(self.check_show_masks.isChecked()),
            "show_boxes": bool(self.check_show_boxes.isChecked()),
            "show_keypoints": bool(self.check_show_keypoints.isChecked()),
            "save_overlay_video": bool(self.check_save_overlay_video.isChecked()),
        }

    def set_overlay_options(
        self,
        show_masks: bool,
        show_boxes: bool,
        save_overlay_video: bool = False,
        show_keypoints: bool = True,
    ) -> None:
        self.check_show_masks.blockSignals(True)
        self.check_show_masks.setChecked(bool(show_masks))
        self.check_show_masks.blockSignals(False)
        self.check_show_boxes.blockSignals(True)
        self.check_show_boxes.setChecked(bool(show_boxes))
        self.check_show_boxes.blockSignals(False)
        self.check_show_keypoints.blockSignals(True)
        self.check_show_keypoints.setChecked(bool(show_keypoints))
        self.check_show_keypoints.blockSignals(False)
        self.check_save_overlay_video.blockSignals(True)
        self.check_save_overlay_video.setChecked(bool(save_overlay_video))
        self.check_save_overlay_video.blockSignals(False)

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
            label = build_rule_label(rule)
            label_item = QTableWidgetItem(label)
            label_item.setData(Qt.UserRole, rule.rule_id)
            label_item.setToolTip(label)
            active_item = QTableWidgetItem("ACTIVE" if rule.rule_id in active_rule_set else "idle")
            active_item.setData(Qt.UserRole, rule.rule_id)
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
            self.label_active_outputs.setText("Outputs high: " + ", ".join(active))
        else:
            self.label_active_outputs.setText("Outputs: all low")
