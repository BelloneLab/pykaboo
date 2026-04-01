"""Right-rail widget for model setup, ROI drawing, and live TTL trigger rules."""

from __future__ import annotations

import uuid
from typing import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
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
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from live_detection_logic import build_rule_label, normalize_output_id
from live_detection_types import BehaviorROI, LiveTriggerRule


class LiveDetectionPanel(QWidget):
    """Control surface for live segmentation, ROI drawing, and TTL triggering."""

    toggle_detection_requested = Signal(bool)
    start_roi_draw_requested = Signal(str)
    finish_polygon_requested = Signal()
    remove_roi_requested = Signal(str)
    clear_rois_requested = Signal()
    output_mapping_changed = Signal(dict)
    add_rule_requested = Signal(object)
    remove_rule_requested = Signal(str)
    overlay_options_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._roi_names: list[str] = []
        self._rule_ids: list[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_roi_group())
        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_rule_group(), 1)

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

        overlay_row = QHBoxLayout()
        self.check_show_masks = QCheckBox("Show masks")
        self.check_show_masks.setChecked(True)
        self.check_show_masks.toggled.connect(self.overlay_options_changed.emit)
        overlay_row.addWidget(self.check_show_masks)
        self.check_show_boxes = QCheckBox("Show boxes")
        self.check_show_boxes.setChecked(True)
        self.check_show_boxes.toggled.connect(self.overlay_options_changed.emit)
        overlay_row.addWidget(self.check_show_boxes)
        overlay_row.addStretch()
        form.addRow("Overlay:", overlay_row)

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
        self.btn_finish_polygon = QPushButton("Finish Polygon")
        self.btn_finish_polygon.clicked.connect(self.finish_polygon_requested.emit)
        button_row.addWidget(self.btn_finish_polygon)
        layout.addLayout(button_row)

        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.roi_list)

        manage_row = QHBoxLayout()
        self.btn_remove_roi = QPushButton("Remove ROI")
        self.btn_remove_roi.clicked.connect(self._remove_selected_roi)
        manage_row.addWidget(self.btn_remove_roi)
        self.btn_clear_rois = QPushButton("Clear ROIs")
        self.btn_clear_rois.clicked.connect(self.clear_rois_requested.emit)
        manage_row.addWidget(self.btn_clear_rois)
        layout.addLayout(manage_row)
        return group

    def _build_output_group(self) -> QWidget:
        group = QGroupBox("TTL Outputs")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        self.output_pin_edits: dict[str, QLineEdit] = {}
        for index in range(1, 9):
            output_id = f"DO{index}"
            label = QLabel(output_id)
            edit = QLineEdit()
            edit.setPlaceholderText("Pin list, e.g. 30 or 30,31")
            self.output_pin_edits[output_id] = edit
            layout.addWidget(label, index - 1, 0)
            layout.addWidget(edit, index - 1, 1)
        self.btn_apply_output_map = QPushButton("Apply DO Mapping")
        self.btn_apply_output_map.clicked.connect(self._emit_output_mapping)
        layout.addWidget(self.btn_apply_output_map, 8, 0, 1, 2)
        return group

    def _build_rule_group(self) -> QWidget:
        group = QGroupBox("Trigger Rules")
        layout = QVBoxLayout(group)

        roi_rule_box = QFrame()
        roi_form = QFormLayout(roi_rule_box)
        self.spin_rule_mouse_id = QSpinBox()
        self.spin_rule_mouse_id.setRange(1, 8)
        self.combo_rule_roi = QComboBox()
        self.combo_rule_output = QComboBox()
        self.combo_rule_output.addItems([f"DO{i}" for i in range(1, 9)])
        self.combo_rule_mode = QComboBox()
        self.combo_rule_mode.addItem("Level", "level")
        self.combo_rule_mode.addItem("Pulse", "pulse")
        self.spin_rule_duration = QSpinBox()
        self.spin_rule_duration.setRange(1, 600000)
        self.spin_rule_duration.setValue(250)
        roi_form.addRow("ROI mouse:", self.spin_rule_mouse_id)
        roi_form.addRow("ROI:", self.combo_rule_roi)
        roi_form.addRow("Output:", self.combo_rule_output)
        roi_form.addRow("Mode:", self.combo_rule_mode)
        roi_form.addRow("Duration ms:", self.spin_rule_duration)
        btn_add_roi_rule = QPushButton("Add ROI Rule")
        btn_add_roi_rule.clicked.connect(self._add_roi_rule)
        roi_form.addRow("", btn_add_roi_rule)
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
        self.combo_prox_mode.addItem("Level", "level")
        self.combo_prox_mode.addItem("Pulse", "pulse")
        self.spin_prox_duration = QSpinBox()
        self.spin_prox_duration.setRange(1, 600000)
        self.spin_prox_duration.setValue(250)
        proximity_form.addRow("Mouse A:", self.spin_prox_mouse_id)
        proximity_form.addRow("Mouse B:", self.spin_prox_peer_id)
        proximity_form.addRow("Distance px:", self.spin_prox_distance)
        proximity_form.addRow("Output:", self.combo_prox_output)
        proximity_form.addRow("Mode:", self.combo_prox_mode)
        proximity_form.addRow("Duration ms:", self.spin_prox_duration)
        btn_add_prox_rule = QPushButton("Add Proximity Rule")
        btn_add_prox_rule.clicked.connect(self._add_proximity_rule)
        proximity_form.addRow("", btn_add_prox_rule)
        layout.addWidget(proximity_box)

        self.rule_table = QTableWidget(0, 2)
        self.rule_table.setHorizontalHeaderLabels(["Rule", "State"])
        self.rule_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.rule_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.rule_table.verticalHeader().setVisible(False)
        self.rule_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.rule_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.rule_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rule_table.setMinimumHeight(180)
        layout.addWidget(self.rule_table, 1)

        footer_row = QHBoxLayout()
        self.label_active_outputs = QLabel("Outputs: all low")
        self.label_active_outputs.setStyleSheet("color: #8fa6bf; font-weight: 600;")
        footer_row.addWidget(self.label_active_outputs, 1)
        self.btn_remove_rule = QPushButton("Remove Rule")
        self.btn_remove_rule.clicked.connect(self._remove_selected_rule)
        footer_row.addWidget(self.btn_remove_rule)
        layout.addLayout(footer_row)
        return group

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select segmentation checkpoint",
            "",
            "Model files (*.pt *.pth *.ckpt *.bin);;All files (*.*)",
        )
        if path:
            self.edit_checkpoint.setText(path)

    def _request_roi_draw(self) -> None:
        shape = self.combo_roi_shape.currentText().strip().lower()
        self.start_roi_draw_requested.emit(shape)

    def _remove_selected_roi(self) -> None:
        item = self.roi_list.currentItem()
        if item is None:
            return
        self.remove_roi_requested.emit(item.text())

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
            mode=str(self.combo_rule_mode.currentData() or "level"),
            duration_ms=int(self.spin_rule_duration.value()),
            mouse_id=int(self.spin_rule_mouse_id.value()),
            roi_name=roi_name,
        )
        self.add_rule_requested.emit(payload)

    def _add_proximity_rule(self) -> None:
        payload = LiveTriggerRule(
            rule_id=f"rule-{uuid.uuid4().hex[:8]}",
            rule_type="mouse_proximity",
            output_id=normalize_output_id(self.combo_prox_output.currentText()),
            mode=str(self.combo_prox_mode.currentData() or "level"),
            duration_ms=int(self.spin_prox_duration.value()),
            mouse_id=int(self.spin_prox_mouse_id.value()),
            peer_mouse_id=int(self.spin_prox_peer_id.value()),
            distance_px=float(self.spin_prox_distance.value()),
        )
        self.add_rule_requested.emit(payload)

    def _remove_selected_rule(self) -> None:
        row = self.rule_table.currentRow()
        if row < 0 or row >= len(self._rule_ids):
            return
        self.remove_rule_requested.emit(self._rule_ids[row])

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
            "threshold": float(self.spin_threshold.value()),
            "selected_class_ids": selected_classes,
            "identity_mode": str(self.combo_identity_mode.currentData() or "tracker"),
            "expected_mouse_count": int(self.spin_expected_mice.value()),
            "show_masks": bool(self.check_show_masks.isChecked()),
            "show_boxes": bool(self.check_show_boxes.isChecked()),
        }

    def current_roi_name(self) -> str:
        return str(self.edit_roi_name.text() or "").strip() or "ROI"

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
        }

    def set_overlay_options(self, show_masks: bool, show_boxes: bool) -> None:
        self.check_show_masks.blockSignals(True)
        self.check_show_masks.setChecked(bool(show_masks))
        self.check_show_masks.blockSignals(False)
        self.check_show_boxes.blockSignals(True)
        self.check_show_boxes.setChecked(bool(show_boxes))
        self.check_show_boxes.blockSignals(False)

    def set_output_mapping(self, mapping: dict[str, Iterable[int]]) -> None:
        for output_id, edit in self.output_pin_edits.items():
            pins = mapping.get(output_id, [])
            edit.setText(",".join(str(int(pin)) for pin in pins))

    def set_rois(self, rois: dict[str, BehaviorROI]) -> None:
        self._roi_names = list(rois.keys())
        self.roi_list.clear()
        for name in self._roi_names:
            item = QListWidgetItem(name)
            self.roi_list.addItem(item)
        self.combo_rule_roi.blockSignals(True)
        self.combo_rule_roi.clear()
        self.combo_rule_roi.addItems(self._roi_names)
        self.combo_rule_roi.blockSignals(False)

    def set_rules(self, rules: list[LiveTriggerRule], active_rule_ids: Iterable[str] = ()) -> None:
        active_rule_set = set(active_rule_ids)
        self._rule_ids = [rule.rule_id for rule in rules]
        self.rule_table.setRowCount(len(rules))
        for row, rule in enumerate(rules):
            label_item = QTableWidgetItem(build_rule_label(rule))
            active_item = QTableWidgetItem("ACTIVE" if rule.rule_id in active_rule_set else "idle")
            self.rule_table.setItem(row, 0, label_item)
            self.rule_table.setItem(row, 1, active_item)

    def set_active_outputs(self, output_states: dict[str, bool]) -> None:
        active = [output_id for output_id, state in sorted(output_states.items()) if bool(state)]
        if active:
            self.label_active_outputs.setText("Outputs high: " + ", ".join(active))
        else:
            self.label_active_outputs.setText("Outputs: all low")
