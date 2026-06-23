"""Stable live mouse identity assignment for realtime detection streams."""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.optimize

from live_detection_types import TrackedMouseState
from live_overlay_quality import smooth_keypoints


# Default body scale (px) used before any detection is seen; the tracker then tracks
# the live median bbox diagonal so the motion-prediction clamp and the association gate
# auto-scale to the camera resolution and the animal size.
_DEFAULT_BODY_SCALE_PX = 90.0
# Velocity clamp as a fraction of the body scale: a single noisy centroid jump cannot
# push the predicted position more than this many body-lengths, so a bad frame can never
# fling a track's prediction across the arena.
_VELOCITY_CLAMP_FRAC = 0.75
# EMA weight on the freshest velocity sample (rest on the previous velocity).
_VELOCITY_EMA = 0.6
# Association gate: a previous track is never matched to a detection farther than this
# many body-lengths away (kept generous so only true teleports/ID-steals are rejected).
_GATE_BODY_FACTOR = 3.0
_GATE_FLOOR_PX = 120.0


def _bbox_diagonal(bbox) -> float:
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except Exception:
        return 0.0
    return float(np.hypot(x2 - x1, y2 - y1))


def compute_body_center(mask: Optional[np.ndarray], bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    if mask is not None:
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            return float(np.mean(xs)), float(np.mean(ys))
    x1, y1, x2, y2 = bbox
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


class LiveIdentityTracker:
    """Lightweight centroid tracker adapted from the offline mousetracker matcher."""

    def __init__(
        self,
        expected_mice: int = 1,
        *,
        use_motion_prediction: bool = True,
        use_distance_gate: bool = True,
    ) -> None:
        self.expected_mice = max(1, int(expected_mice))
        self._last_centers: dict[int, tuple[float, float]] = {}
        self._last_boxes: dict[int, tuple[float, float, float, float]] = {}
        self._last_keypoints: dict[int, np.ndarray] = {}
        # Per-track velocity (px/frame) so association matches on the PREDICTED next
        # position, not the last one. This is what keeps identities stable when two
        # mice cross or move fast. Reduces exactly to nearest-centroid when velocity
        # is zero (first frames / stationary animals), so it can never do worse.
        self._last_velocity: dict[int, tuple[float, float]] = {}
        self._body_scale = _DEFAULT_BODY_SCALE_PX
        self.use_motion_prediction = bool(use_motion_prediction)
        self.use_distance_gate = bool(use_distance_gate)
        # Temporal keypoint smoothing removes the per-frame jitter that makes a
        # skeleton look like it is vibrating. Disabled for a single update has
        # no effect; the worker toggles this from the live config.
        self.smooth_keypoints_enabled = True

    def reset(self, expected_mice: Optional[int] = None) -> None:
        if expected_mice is not None:
            self.expected_mice = max(1, int(expected_mice))
        self._last_centers.clear()
        self._last_boxes.clear()
        self._last_keypoints.clear()
        self._last_velocity.clear()
        self._body_scale = _DEFAULT_BODY_SCALE_PX

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

        self._update_body_scale(working)

        prev_ids = sorted(self._last_centers.keys())
        cost = np.full((len(prev_ids), len(working)), 1e6, dtype=float)
        for row, mouse_id in enumerate(prev_ids):
            prev_center = self._last_centers.get(mouse_id, (0.0, 0.0))
            if self.use_motion_prediction:
                vx, vy = self._last_velocity.get(mouse_id, (0.0, 0.0))
                pred_x = float(prev_center[0]) + float(vx)
                pred_y = float(prev_center[1]) + float(vy)
            else:
                pred_x, pred_y = float(prev_center[0]), float(prev_center[1])
            for col, det in enumerate(working):
                cx, cy = det["center"]
                dx = pred_x - float(cx)
                dy = pred_y - float(cy)
                cost[row, col] = (dx * dx) + (dy * dy)

        gate_sq = self._gate_distance() ** 2 if self.use_distance_gate else float("inf")
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
        assigned_indices: set[int] = set()
        assigned_states: list[TrackedMouseState] = []
        for row, col in zip(row_ind, col_ind):
            if float(cost[row, col]) > gate_sq:
                # Too far to plausibly be the same animal: refuse the force-match so a
                # vanished track cannot teleport its identity onto a distant detection.
                # The detection falls through to the new-ID branch below.
                continue
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
        keypoints = det.get("keypoints")
        if self.smooth_keypoints_enabled and keypoints is not None:
            keypoints = smooth_keypoints(self._last_keypoints.get(int(mouse_id)), keypoints)
        return TrackedMouseState(
            mouse_id=int(mouse_id),
            class_id=int(det.get("class_id", 0)),
            confidence=float(det.get("confidence", 0.0)),
            center=(float(det["center"][0]), float(det["center"][1])),
            bbox=tuple(float(value) for value in det.get("bbox", (0.0, 0.0, 0.0, 0.0))),
            mask=det.get("mask"),
            label=f"Mouse {int(mouse_id)}",
            keypoints=keypoints,
            keypoint_scores=det.get("keypoint_scores"),
        )

    def _update_body_scale(self, detections: list[dict]) -> None:
        """Track the live median body size (bbox diagonal) so the velocity clamp and
        the association gate scale with the camera resolution and the animal."""
        diags = [
            d for d in (_bbox_diagonal(det.get("bbox")) for det in detections) if d > 1.0
        ]
        if diags:
            self._body_scale = 0.5 * float(np.median(diags)) + 0.5 * float(self._body_scale)

    def _gate_distance(self) -> float:
        return max(_GATE_FLOOR_PX, _GATE_BODY_FACTOR * float(self._body_scale))

    def _commit(self, states: list[TrackedMouseState]) -> None:
        new_centers = {state.mouse_id: state.center for state in states}
        if self.use_motion_prediction:
            # Velocity from old->new centroid, clamped so one noisy frame cannot
            # dominate, then EMA-smoothed. Computed BEFORE _last_centers is replaced.
            clamp = max(8.0, _VELOCITY_CLAMP_FRAC * float(self._body_scale))
            new_velocity: dict[int, tuple[float, float]] = {}
            for state in states:
                mouse_id = int(state.mouse_id)
                old_center = self._last_centers.get(mouse_id)
                old_vel = self._last_velocity.get(mouse_id, (0.0, 0.0))
                if old_center is None:
                    new_velocity[mouse_id] = (0.0, 0.0)
                    continue
                raw_vx = float(state.center[0]) - float(old_center[0])
                raw_vy = float(state.center[1]) - float(old_center[1])
                raw_vx = float(np.clip(raw_vx, -clamp, clamp))
                raw_vy = float(np.clip(raw_vy, -clamp, clamp))
                new_velocity[mouse_id] = (
                    _VELOCITY_EMA * raw_vx + (1.0 - _VELOCITY_EMA) * float(old_vel[0]),
                    _VELOCITY_EMA * raw_vy + (1.0 - _VELOCITY_EMA) * float(old_vel[1]),
                )
            self._last_velocity = new_velocity
        self._last_centers = new_centers
        self._last_boxes = {state.mouse_id: state.bbox for state in states}
        self._last_keypoints = {
            state.mouse_id: np.asarray(state.keypoints, dtype=float).reshape(-1, 2).copy()
            for state in states
            if getattr(state, "keypoints", None) is not None
        }
