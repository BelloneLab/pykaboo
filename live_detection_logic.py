"""Pure-Python ROI and live trigger evaluation for real-time TTL outputs."""

from __future__ import annotations

import math
from typing import Iterable, Optional

from live_detection_types import (
    BehaviorROI,
    LiveDetectionResult,
    LiveOutputArbiterState,
    LiveRuleEvaluation,
    LiveTriggerRule,
    TrackedMouseState,
)


def normalize_output_id(value: str) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("DO"):
        return text
    if text.isdigit():
        return f"DO{text}"
    return "DO1"


class LiveOutputArbiter:
    """Merge multiple level and pulse requests into stable logical DO states."""

    def __init__(self, outputs: Optional[Iterable[str]] = None) -> None:
        self.outputs = [normalize_output_id(entry) for entry in (outputs or [f"DO{i}" for i in range(1, 9)])]
        self._level_rules: dict[str, set[str]] = {output_id: set() for output_id in self.outputs}
        self._pulse_until_ms: dict[str, int] = {output_id: 0 for output_id in self.outputs}

    def clear(self) -> None:
        for output_id in self.outputs:
            self._level_rules[output_id].clear()
            self._pulse_until_ms[output_id] = 0

    def set_level(self, output_id: str, rule_id: str, active: bool) -> None:
        normalized_output = normalize_output_id(output_id)
        if normalized_output not in self._level_rules:
            self._level_rules[normalized_output] = set()
            self._pulse_until_ms.setdefault(normalized_output, 0)
            self.outputs.append(normalized_output)
        if active:
            self._level_rules[normalized_output].add(rule_id)
        else:
            self._level_rules[normalized_output].discard(rule_id)

    def trigger_pulse(self, output_id: str, duration_ms: int, now_ms: int) -> None:
        normalized_output = normalize_output_id(output_id)
        if normalized_output not in self._pulse_until_ms:
            self._pulse_until_ms[normalized_output] = 0
            self._level_rules.setdefault(normalized_output, set())
            self.outputs.append(normalized_output)
        self._pulse_until_ms[normalized_output] = max(
            int(self._pulse_until_ms.get(normalized_output, 0)),
            int(now_ms) + max(1, int(duration_ms)),
        )

    def snapshot(self, now_ms: int) -> LiveOutputArbiterState:
        states: dict[str, bool] = {}
        for output_id in self.outputs:
            level_active = bool(self._level_rules.get(output_id))
            pulse_active = int(self._pulse_until_ms.get(output_id, 0)) > int(now_ms)
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
        for output_id in list(self.arbiter._level_rules.keys()):
            self.arbiter._level_rules[output_id] = {
                rule_id
                for rule_id in self.arbiter._level_rules[output_id]
                if rule_id in valid_ids
            }

    def clear_runtime_state(self) -> None:
        self._previous_truth.clear()
        self.arbiter.clear()

    def evaluate(self, result: Optional[LiveDetectionResult], now_ms: int) -> LiveRuleEvaluation:
        active_rule_ids: list[str] = []
        triggered_pulses: list[tuple[str, int]] = []
        if result is None:
            snapshot = self.arbiter.snapshot(now_ms)
            return LiveRuleEvaluation(
                active_rule_ids=[],
                triggered_pulses=[],
                output_states=snapshot.output_states,
            )

        mouse_lookup = {mouse.mouse_id: mouse for mouse in result.tracked_mice}

        for rule in self.rules:
            truth = self._rule_truth(rule, mouse_lookup)
            previous_truth = bool(self._previous_truth.get(rule.rule_id, False))
            output_id = normalize_output_id(rule.output_id)

            if rule.mode == "level":
                self.arbiter.set_level(output_id, rule.rule_id, truth)
            elif truth and not previous_truth:
                duration_ms = max(1, int(rule.duration_ms))
                self.arbiter.trigger_pulse(output_id, duration_ms, now_ms)
                triggered_pulses.append((output_id, duration_ms))

            if truth:
                active_rule_ids.append(rule.rule_id)
            self._previous_truth[rule.rule_id] = truth

        snapshot = self.arbiter.snapshot(now_ms)
        return LiveRuleEvaluation(
            active_rule_ids=active_rule_ids,
            triggered_pulses=triggered_pulses,
            output_states=snapshot.output_states,
        )

    def _rule_truth(self, rule: LiveTriggerRule, mouse_lookup: dict[int, TrackedMouseState]) -> bool:
        if rule.rule_type == "roi_occupancy":
            mouse = mouse_lookup.get(rule.mouse_id)
            roi = self.rois.get(rule.roi_name)
            if mouse is None or roi is None:
                return False
            cx, cy = mouse.center
            return roi.contains_point(cx, cy)

        if rule.rule_type == "mouse_proximity":
            mouse_a = mouse_lookup.get(rule.mouse_id)
            mouse_b = mouse_lookup.get(rule.peer_mouse_id)
            if mouse_a is None or mouse_b is None:
                return False
            dx = float(mouse_a.center[0]) - float(mouse_b.center[0])
            dy = float(mouse_a.center[1]) - float(mouse_b.center[1])
            distance = math.sqrt((dx * dx) + (dy * dy))
            return distance <= float(rule.distance_px)

        return False


def build_rule_label(rule: LiveTriggerRule) -> str:
    if rule.rule_type == "mouse_proximity":
        return (
            f"M{rule.mouse_id} close to M{rule.peer_mouse_id} "
            f"({rule.distance_px:.0f}px) -> {normalize_output_id(rule.output_id)}"
        )
    return f"M{rule.mouse_id} in {rule.roi_name or 'ROI'} -> {normalize_output_id(rule.output_id)}"
