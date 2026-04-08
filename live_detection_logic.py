"""Pure-Python ROI and live trigger evaluation for real-time TTL outputs."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from live_detection_types import (
    BehaviorROI,
    LiveDetectionResult,
    LiveOutputArbiterState,
    LiveRuleEvaluation,
    LiveTriggerRule,
    TrackedMouseState,
    normalize_activation_pattern,
)


def normalize_output_id(value: str) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("DO"):
        return text
    if text.isdigit():
        return f"DO{text}"
    return "DO1"


def occupied_roi_names(
    rois: dict[str, BehaviorROI],
    result: Optional[LiveDetectionResult],
) -> set[str]:
    if result is None:
        return set()

    occupied: set[str] = set()
    for roi_name, roi in rois.items():
        for mouse in result.tracked_mice:
            try:
                cx, cy = mouse.center
                if roi.contains_point(cx, cy):
                    occupied.add(roi_name)
                    break
            except Exception:
                continue
    return occupied


def roi_geometry_properties(roi: BehaviorROI) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "name": str(roi.name),
        "roi_type": str(roi.roi_type),
    }
    if roi.roi_type == "rectangle" and roi.data:
        x1, y1, x2, y2 = [float(value) for value in roi.data[0]]
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        properties.update(
            {
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
                "x2": right,
                "y2": bottom,
            }
        )
    elif roi.roi_type == "circle" and roi.data:
        cx, cy, radius = [float(value) for value in roi.data[0]]
        properties.update(
            {
                "center_x": cx,
                "center_y": cy,
                "radius": radius,
                "diameter": radius * 2.0,
                "x": cx,
                "y": cy,
            }
        )
    elif roi.roi_type == "polygon" and roi.data:
        points = [(float(px), float(py)) for px, py in roi.data]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        properties.update(
            {
                "points": points,
                "point_count": len(points),
                "centroid_x": sum(xs) / len(xs) if xs else 0.0,
                "centroid_y": sum(ys) / len(ys) if ys else 0.0,
                "x": min(xs) if xs else 0.0,
                "y": min(ys) if ys else 0.0,
                "width": (max(xs) - min(xs)) if xs else 0.0,
                "height": (max(ys) - min(ys)) if ys else 0.0,
            }
        )
    return properties


def _format_float(value: float) -> str:
    return f"{float(value):.1f}".rstrip("0").rstrip(".")


def format_roi_properties(roi: BehaviorROI) -> str:
    properties = roi_geometry_properties(roi)
    if roi.roi_type == "circle":
        return (
            f"x={_format_float(float(properties.get('center_x', 0.0)))}, "
            f"y={_format_float(float(properties.get('center_y', 0.0)))}, "
            f"diameter={_format_float(float(properties.get('diameter', 0.0)))}"
        )
    if roi.roi_type == "rectangle":
        return (
            f"x={_format_float(float(properties.get('x', 0.0)))}, "
            f"y={_format_float(float(properties.get('y', 0.0)))}, "
            f"w={_format_float(float(properties.get('width', 0.0)))}, "
            f"h={_format_float(float(properties.get('height', 0.0)))}"
        )
    if roi.roi_type == "polygon":
        return (
            f"points={int(properties.get('point_count', 0))}, "
            f"centroid=({_format_float(float(properties.get('centroid_x', 0.0)))}, "
            f"{_format_float(float(properties.get('centroid_y', 0.0)))}), "
            f"bounds x={_format_float(float(properties.get('x', 0.0)))}, "
            f"y={_format_float(float(properties.get('y', 0.0)))}, "
            f"w={_format_float(float(properties.get('width', 0.0)))}, "
            f"h={_format_float(float(properties.get('height', 0.0)))}"
        )
    return "no geometry"


class LiveOutputArbiter:
    """Merge multiple level and pulse requests into stable logical DO states."""

    def __init__(self, outputs: Optional[Iterable[str]] = None) -> None:
        self.outputs = [normalize_output_id(entry) for entry in (outputs or [f"DO{i}" for i in range(1, 9)])]
        self._level_rules: dict[str, set[str]] = {output_id: set() for output_id in self.outputs}
        self._pulse_until_ms: dict[str, int] = {output_id: 0 for output_id in self.outputs}
        self._pulse_trains_ms: dict[str, list[tuple[int, int, int, int]]] = {
            output_id: [] for output_id in self.outputs
        }

    def clear(self) -> None:
        for output_id in self.outputs:
            self._level_rules[output_id].clear()
            self._pulse_until_ms[output_id] = 0
            self._pulse_trains_ms[output_id] = []

    def set_level(self, output_id: str, rule_id: str, active: bool) -> None:
        normalized_output = normalize_output_id(output_id)
        if normalized_output not in self._level_rules:
            self._level_rules[normalized_output] = set()
            self._pulse_until_ms.setdefault(normalized_output, 0)
            self._pulse_trains_ms.setdefault(normalized_output, [])
            self.outputs.append(normalized_output)
        if active:
            self._level_rules[normalized_output].add(rule_id)
        else:
            self._level_rules[normalized_output].discard(rule_id)

    def trigger_pulse(
        self,
        output_id: str,
        duration_ms: int,
        now_ms: int,
        pulse_count: int = 1,
        pulse_frequency_hz: float = 1.0,
    ) -> None:
        normalized_output = normalize_output_id(output_id)
        if normalized_output not in self._pulse_until_ms:
            self._pulse_until_ms[normalized_output] = 0
            self._level_rules.setdefault(normalized_output, set())
            self._pulse_trains_ms.setdefault(normalized_output, [])
            self.outputs.append(normalized_output)
        duration = max(1, int(duration_ms))
        count = max(1, int(pulse_count))
        if count > 1:
            frequency_hz = max(0.001, float(pulse_frequency_hz))
            period_ms = max(duration, int(round(1000.0 / frequency_hz)))
        else:
            period_ms = duration
        train = (int(now_ms), duration, period_ms, count)
        self._pulse_trains_ms.setdefault(normalized_output, []).append(train)
        self._pulse_until_ms[normalized_output] = max(
            int(self._pulse_until_ms.get(normalized_output, 0)),
            int(now_ms) + ((count - 1) * period_ms) + duration,
        )

    @staticmethod
    def _pulse_train_active(train: tuple[int, int, int, int], now_ms: int) -> bool:
        start_ms, duration_ms, period_ms, count = train
        elapsed_ms = int(now_ms) - int(start_ms)
        if elapsed_ms < 0:
            return False
        end_ms = int(start_ms) + ((int(count) - 1) * int(period_ms)) + int(duration_ms)
        if int(now_ms) >= end_ms:
            return False
        pulse_index = elapsed_ms // int(period_ms)
        if pulse_index >= int(count):
            return False
        return (elapsed_ms - (pulse_index * int(period_ms))) < int(duration_ms)

    @staticmethod
    def _pulse_train_end_ms(train: tuple[int, int, int, int]) -> int:
        start_ms, duration_ms, period_ms, count = train
        return int(start_ms) + ((int(count) - 1) * int(period_ms)) + int(duration_ms)

    def level_states(self) -> dict[str, bool]:
        return {output_id: bool(self._level_rules.get(output_id)) for output_id in self.outputs}

    def snapshot(self, now_ms: int) -> LiveOutputArbiterState:
        states: dict[str, bool] = {}
        for output_id in self.outputs:
            level_active = bool(self._level_rules.get(output_id))
            trains = [
                train
                for train in self._pulse_trains_ms.get(output_id, [])
                if self._pulse_train_end_ms(train) > int(now_ms)
            ]
            self._pulse_trains_ms[output_id] = trains
            self._pulse_until_ms[output_id] = max(
                [self._pulse_train_end_ms(train) for train in trains] or [0]
            )
            pulse_active = any(self._pulse_train_active(train, int(now_ms)) for train in trains)
            states[output_id] = level_active or pulse_active
        return LiveOutputArbiterState(
            output_states=states,
            pulse_until_ms=dict(self._pulse_until_ms),
        )


class LiveRuleEngine:
    """Evaluate ROI and proximity rules against tracked live detections."""

    def __init__(self) -> None:
        self.rois: dict[str, BehaviorROI] = {}
        self.rules: list[LiveTriggerRule] = []
        self._previous_truth: dict[str, bool] = {}
        self._next_continuous_pulse_ms: dict[str, int] = {}
        self.arbiter = LiveOutputArbiter()

    def set_rois(self, rois: dict[str, BehaviorROI]) -> None:
        self.rois = dict(rois)

    def set_rules(self, rules: list[LiveTriggerRule]) -> None:
        self.rules = list(rules)
        valid_ids = {rule.rule_id for rule in self.rules}
        self._previous_truth = {
            rule_id: truth
            for rule_id, truth in self._previous_truth.items()
            if rule_id in valid_ids
        }
        self._next_continuous_pulse_ms = {
            rule_id: next_ms
            for rule_id, next_ms in self._next_continuous_pulse_ms.items()
            if rule_id in valid_ids
        }
        for output_id in list(self.arbiter._level_rules.keys()):
            self.arbiter._level_rules[output_id] = {
                rule_id
                for rule_id in self.arbiter._level_rules[output_id]
                if rule_id in valid_ids
            }

    def clear_runtime_state(self) -> None:
        self._previous_truth.clear()
        self._next_continuous_pulse_ms.clear()
        self.arbiter.clear()

    def evaluate(self, result: Optional[LiveDetectionResult], now_ms: int) -> LiveRuleEvaluation:
        active_rule_ids: list[str] = []
        triggered_pulses: list[tuple[str, int, int, float]] = []
        if result is None:
            snapshot = self.arbiter.snapshot(now_ms)
            return LiveRuleEvaluation(
                active_rule_ids=[],
                triggered_pulses=[],
                output_states=snapshot.output_states,
                level_output_states=self.arbiter.level_states(),
            )

        mouse_lookup = {mouse.mouse_id: mouse for mouse in result.tracked_mice}

        for rule in self.rules:
            previous_truth = bool(self._previous_truth.get(rule.rule_id, False))
            truth_value = self._rule_truth(rule, mouse_lookup)
            truth = previous_truth if truth_value is None else bool(truth_value)
            output_id = normalize_output_id(rule.output_id)

            mode = str(rule.mode or "gate").strip().lower()
            if mode in {"gate", "level"}:
                self.arbiter.set_level(output_id, rule.rule_id, truth)
            elif self._should_trigger_pulse(rule, truth, previous_truth, now_ms):
                triggered_pulses.append(self._trigger_rule_pulse(rule, output_id, now_ms))

            if truth:
                active_rule_ids.append(rule.rule_id)
            if truth_value is not None:
                self._previous_truth[rule.rule_id] = truth

        snapshot = self.arbiter.snapshot(now_ms)
        return LiveRuleEvaluation(
            active_rule_ids=active_rule_ids,
            triggered_pulses=triggered_pulses,
            output_states=snapshot.output_states,
            level_output_states=self.arbiter.level_states(),
        )

    def _rule_truth(self, rule: LiveTriggerRule, mouse_lookup: dict[int, TrackedMouseState]) -> Optional[bool]:
        if rule.rule_type == "roi_occupancy":
            mouse = mouse_lookup.get(rule.mouse_id)
            roi = self.rois.get(rule.roi_name)
            if roi is None:
                return False
            if mouse is None:
                return None
            cx, cy = mouse.center
            return roi.contains_point(cx, cy)

        if rule.rule_type == "mouse_proximity":
            mouse_a = mouse_lookup.get(rule.mouse_id)
            mouse_b = mouse_lookup.get(rule.peer_mouse_id)
            if mouse_a is None or mouse_b is None:
                return None
            dx = float(mouse_a.center[0]) - float(mouse_b.center[0])
            dy = float(mouse_a.center[1]) - float(mouse_b.center[1])
            distance = math.sqrt((dx * dx) + (dy * dy))
            return distance <= float(rule.distance_px)

        return False

    @staticmethod
    def _rule_pulse_train_duration_ms(rule: LiveTriggerRule) -> int:
        duration_ms = max(1, int(rule.duration_ms))
        pulse_count = max(1, int(rule.pulse_count))
        if pulse_count > 1:
            period_ms = max(duration_ms, int(round(1000.0 / max(0.001, float(rule.pulse_frequency_hz)))))
        else:
            period_ms = duration_ms
        return ((pulse_count - 1) * period_ms) + duration_ms

    def _should_trigger_pulse(
        self,
        rule: LiveTriggerRule,
        truth: bool,
        previous_truth: bool,
        now_ms: int,
    ) -> bool:
        activation_pattern = normalize_activation_pattern(rule.activation_pattern)
        if activation_pattern == "entry":
            return truth and not previous_truth
        if activation_pattern == "exit":
            return previous_truth and not truth
        if activation_pattern == "continuous":
            if not truth:
                self._next_continuous_pulse_ms.pop(rule.rule_id, None)
                return False
            next_ms = int(self._next_continuous_pulse_ms.get(rule.rule_id, int(now_ms)))
            return int(now_ms) >= next_ms
        return truth and not previous_truth

    def _trigger_rule_pulse(
        self,
        rule: LiveTriggerRule,
        output_id: str,
        now_ms: int,
    ) -> tuple[str, int, int, float]:
        duration_ms = max(1, int(rule.duration_ms))
        pulse_count = max(1, int(rule.pulse_count))
        pulse_frequency_hz = max(0.001, float(rule.pulse_frequency_hz))
        self.arbiter.trigger_pulse(
            output_id,
            duration_ms,
            now_ms,
            pulse_count=pulse_count,
            pulse_frequency_hz=pulse_frequency_hz,
        )
        if normalize_activation_pattern(rule.activation_pattern) == "continuous":
            self._next_continuous_pulse_ms[rule.rule_id] = (
                int(now_ms)
                + self._rule_pulse_train_duration_ms(rule)
                + max(0, int(rule.inter_train_interval_ms))
            )
        return output_id, duration_ms, pulse_count, pulse_frequency_hz


def build_rule_label(rule: LiveTriggerRule) -> str:
    mode = str(rule.mode or "gate").strip().lower()
    if mode in {"gate", "level"}:
        mode_label = "Gate"
    else:
        pulse_count = max(1, int(rule.pulse_count))
        pulse_frequency_hz = max(0.001, float(rule.pulse_frequency_hz))
        if pulse_count == 1:
            train_label = f"Pulse {int(rule.duration_ms)}ms"
        else:
            train_label = f"Pulse {pulse_count}x {int(rule.duration_ms)}ms @ {pulse_frequency_hz:g}Hz"
        activation_pattern = normalize_activation_pattern(rule.activation_pattern)
        if activation_pattern == "continuous":
            pattern_label = f"continuous ITI {max(0, int(rule.inter_train_interval_ms))}ms"
        elif activation_pattern == "exit":
            pattern_label = "exit"
        else:
            pattern_label = "entry"
        mode_label = f"{train_label}, {pattern_label}"
    if rule.rule_type == "mouse_proximity":
        return (
            f"M{rule.mouse_id} close to M{rule.peer_mouse_id} "
            f"({rule.distance_px:.0f}px) [{mode_label}] -> {normalize_output_id(rule.output_id)}"
        )
    return f"M{rule.mouse_id} in {rule.roi_name or 'ROI'} [{mode_label}] -> {normalize_output_id(rule.output_id)}"
