"""Shared runtime types for live detection, ROI triggers, and TTL outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def _point_in_polygon(x: float, y: float, vertices: list[tuple[float, float]]) -> bool:
    inside = False
    total = len(vertices)
    if total < 3:
        return False
    j = total - 1
    for i in range(total):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if ((yi > y) != (yj > y)) and (x < ((xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)):
            inside = not inside
        j = i
    return inside


@dataclass
class PreviewFramePacket:
    frame: np.ndarray
    frame_index: int
    timestamp_s: float
    width: int
    height: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackedMouseState:
    mouse_id: int
    class_id: int
    confidence: float
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None
    label: str = ""


@dataclass
class LiveDetectionResult:
    frame_index: int
    timestamp_s: float
    width: int
    height: int
    inference_ms: float
    tracked_mice: list[TrackedMouseState] = field(default_factory=list)
    model_key: str = ""
    status: str = ""


@dataclass
class BehaviorROI:
    name: str
    roi_type: str
    data: list[Any]
    color: tuple[int, int, int] = (255, 220, 120)

    def contains_point(self, x: float, y: float) -> bool:
        if self.roi_type == "rectangle" and self.data:
            x1, y1, x2, y2 = self.data[0]
            return x1 <= x <= x2 and y1 <= y <= y2
        if self.roi_type == "circle" and self.data:
            cx, cy, radius = self.data[0]
            return ((x - cx) ** 2 + (y - cy) ** 2) <= (radius ** 2)
        if self.roi_type == "polygon":
            points = [(float(px), float(py)) for px, py in self.data]
            return _point_in_polygon(float(x), float(y), points)
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "roi_type": self.roi_type,
            "data": self.data,
            "color": list(self.color),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BehaviorROI":
        color = payload.get("color", [255, 220, 120])
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            color_value = (int(color[0]), int(color[1]), int(color[2]))
        else:
            color_value = (255, 220, 120)
        return cls(
            name=str(payload.get("name", "ROI")).strip() or "ROI",
            roi_type=str(payload.get("roi_type", "rectangle")).strip().lower(),
            data=list(payload.get("data", [])),
            color=color_value,
        )


@dataclass
class LiveTriggerRule:
    rule_id: str
    rule_type: str
    output_id: str
    mode: str = "level"
    duration_ms: int = 250
    mouse_id: int = 1
    peer_mouse_id: int = 2
    roi_name: str = ""
    distance_px: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "output_id": self.output_id,
            "mode": self.mode,
            "duration_ms": int(self.duration_ms),
            "mouse_id": int(self.mouse_id),
            "peer_mouse_id": int(self.peer_mouse_id),
            "roi_name": self.roi_name,
            "distance_px": float(self.distance_px),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiveTriggerRule":
        return cls(
            rule_id=str(payload.get("rule_id", "")).strip(),
            rule_type=str(payload.get("rule_type", "roi_occupancy")).strip(),
            output_id=str(payload.get("output_id", "DO1")).strip().upper(),
            mode=str(payload.get("mode", "level")).strip().lower(),
            duration_ms=max(1, int(payload.get("duration_ms", 250))),
            mouse_id=max(1, int(payload.get("mouse_id", 1))),
            peer_mouse_id=max(1, int(payload.get("peer_mouse_id", 2))),
            roi_name=str(payload.get("roi_name", "")).strip(),
            distance_px=max(0.0, float(payload.get("distance_px", 0.0))),
        )


@dataclass
class LiveOutputArbiterState:
    output_states: dict[str, bool] = field(default_factory=dict)
    pulse_until_ms: dict[str, int] = field(default_factory=dict)


@dataclass
class LiveRuleEvaluation:
    active_rule_ids: list[str] = field(default_factory=list)
    triggered_pulses: list[tuple[str, int]] = field(default_factory=list)
    output_states: dict[str, bool] = field(default_factory=dict)
