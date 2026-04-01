"""Stable live mouse identity assignment for realtime detection streams."""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.optimize

from live_detection_types import TrackedMouseState


def compute_body_center(mask: Optional[np.ndarray], bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    if mask is not None:
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            return float(np.mean(xs)), float(np.mean(ys))
    x1, y1, x2, y2 = bbox
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


class LiveIdentityTracker:
    """Lightweight centroid tracker adapted from the offline mousetracker matcher."""

    def __init__(self, expected_mice: int = 1) -> None:
        self.expected_mice = max(1, int(expected_mice))
        self._last_centers: dict[int, tuple[float, float]] = {}
        self._last_boxes: dict[int, tuple[float, float, float, float]] = {}

    def reset(self, expected_mice: Optional[int] = None) -> None:
        if expected_mice is not None:
            self.expected_mice = max(1, int(expected_mice))
        self._last_centers.clear()
        self._last_boxes.clear()

    def update(self, detections: list[dict]) -> list[TrackedMouseState]:
        if not detections:
            return []

        working = sorted(
            detections,
            key=lambda item: (-float(item.get("confidence", 0.0)), float(item["center"][0]), float(item["center"][1])),
        )[: self.expected_mice]

        if not self._last_centers:
            working.sort(key=lambda item: (float(item["center"][0]), float(item["center"][1])))
            assigned: list[TrackedMouseState] = []
            for index, det in enumerate(working, start=1):
                assigned.append(self._build_state(index, det))
            self._commit(assigned)
            return assigned

        prev_ids = sorted(self._last_centers.keys())
        cost = np.full((len(prev_ids), len(working)), 1e6, dtype=float)
        for row, mouse_id in enumerate(prev_ids):
            prev_center = self._last_centers.get(mouse_id, (0.0, 0.0))
            for col, det in enumerate(working):
                cx, cy = det["center"]
                dx = float(prev_center[0]) - float(cx)
                dy = float(prev_center[1]) - float(cy)
                cost[row, col] = (dx * dx) + (dy * dy)

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
        assigned_indices: set[int] = set()
        assigned_states: list[TrackedMouseState] = []
        for row, col in zip(row_ind, col_ind):
            mouse_id = int(prev_ids[row])
            det = working[col]
            assigned_indices.add(col)
            assigned_states.append(self._build_state(mouse_id, det))

        next_mouse_id = 1
        while next_mouse_id in self._last_centers:
            next_mouse_id += 1
        for index, det in enumerate(working):
            if index in assigned_indices:
                continue
            assigned_states.append(self._build_state(next_mouse_id, det))
            next_mouse_id += 1

        assigned_states.sort(key=lambda item: item.mouse_id)
        self._commit(assigned_states)
        return assigned_states

    def assign_by_model_class(self, detections: list[dict], selected_class_ids: list[int]) -> list[TrackedMouseState]:
        if not detections:
            return []
        normalized_classes = [int(value) for value in selected_class_ids if value is not None]
        if not normalized_classes:
            unique_classes = sorted({int(det.get("class_id", 0)) for det in detections})
            normalized_classes = unique_classes[: max(1, self.expected_mice)]

        assigned: list[TrackedMouseState] = []
        for mouse_id, class_id in enumerate(normalized_classes, start=1):
            matching = [det for det in detections if int(det.get("class_id", -1)) == class_id]
            if not matching:
                continue
            best = max(matching, key=lambda item: float(item.get("confidence", 0.0)))
            assigned.append(self._build_state(mouse_id, best))

        assigned.sort(key=lambda item: item.mouse_id)
        self._commit(assigned)
        return assigned

    def _build_state(self, mouse_id: int, det: dict) -> TrackedMouseState:
        return TrackedMouseState(
            mouse_id=int(mouse_id),
            class_id=int(det.get("class_id", 0)),
            confidence=float(det.get("confidence", 0.0)),
            center=(float(det["center"][0]), float(det["center"][1])),
            bbox=tuple(float(value) for value in det.get("bbox", (0.0, 0.0, 0.0, 0.0))),
            mask=det.get("mask"),
            label=f"Mouse {int(mouse_id)}",
        )

    def _commit(self, states: list[TrackedMouseState]) -> None:
        self._last_centers = {state.mouse_id: state.center for state in states}
        self._last_boxes = {state.mouse_id: state.bbox for state in states}
