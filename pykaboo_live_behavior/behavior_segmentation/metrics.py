"""Framewise and bout-level evaluation metrics for behavior segmentation.

Framewise metrics treat every frame independently (accuracy, precision, recall,
F1, support, confusion matrix). Bout-level metrics first segment predictions and
ground truth into contiguous behavior bouts, then match them by temporal IoU,
which better reflects how a human reads a segmentation result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .labels import LabelMap


@dataclass
class FramewiseMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    micro_f1: float
    per_class: pd.DataFrame
    confusion: np.ndarray
    class_names: list[str]
    scored_class_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "micro_f1": self.micro_f1,
            "scored_class_names": self.scored_class_names,
            "per_class": self.per_class.to_dict(orient="records"),
        }


def scored_class_ids(
    label_map: LabelMap, include_background: bool = False
) -> list[int]:
    """Return classes that should contribute to behavior metrics."""

    return [
        class_id
        for class_id in range(label_map.num_classes)
        if include_background or class_id != label_map.background_id
    ]


def compute_framewise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: LabelMap,
    include_background: bool = False,
) -> FramewiseMetrics:
    """Compute framewise accuracy, macro/weighted F1, and per-class scores."""

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    labels = list(range(label_map.num_classes))
    class_names = [label_map.id_to_name[i] for i in labels]
    scored_labels = scored_class_ids(label_map, include_background)
    scored_names = [label_map.id_to_name[i] for i in scored_labels]

    accuracy = float(accuracy_score(y_true, y_pred))
    if scored_labels:
        macro = float(
            f1_score(
                y_true,
                y_pred,
                labels=scored_labels,
                average="macro",
                zero_division=0,
            )
        )
        weighted = float(
            f1_score(
                y_true,
                y_pred,
                labels=scored_labels,
                average="weighted",
                zero_division=0,
            )
        )
        micro = float(
            f1_score(
                y_true,
                y_pred,
                labels=scored_labels,
                average="micro",
                zero_division=0,
            )
        )
    else:
        macro = 0.0
        weighted = 0.0
        micro = 0.0
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "behavior": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "scored": [class_id in scored_labels for class_id in labels],
        }
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return FramewiseMetrics(
        accuracy=accuracy,
        macro_f1=macro,
        weighted_f1=weighted,
        micro_f1=micro,
        per_class=per_class,
        confusion=cm,
        class_names=class_names,
        scored_class_names=scored_names,
    )


@dataclass
class Bout:
    behavior: int
    start: int
    end: int  # inclusive

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def extract_bouts(labels: np.ndarray, background_id: int) -> list[Bout]:
    """Extract contiguous foreground bouts from a framewise label array."""

    labels = np.asarray(labels).reshape(-1)
    bouts: list[Bout] = []
    if labels.size == 0:
        return bouts
    start = 0
    current = int(labels[0])
    for i in range(1, labels.size + 1):
        if i == labels.size or int(labels[i]) != current:
            if current != background_id:
                bouts.append(Bout(current, start, i - 1))
            if i < labels.size:
                current = int(labels[i])
                start = i
    return bouts


def bout_iou(a: Bout, b: Bout) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start) + 1)
    union = (a.end - a.start + 1) + (b.end - b.start + 1) - inter
    return inter / union if union > 0 else 0.0


def compute_bout_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: LabelMap,
    iou_thresholds: tuple[float, ...] = (0.1, 0.25, 0.5),
) -> pd.DataFrame:
    """Match predicted and true bouts by IoU and report precision/recall/F1."""

    background = label_map.background_id
    true_bouts = extract_bouts(y_true, background)
    pred_bouts = extract_bouts(y_pred, background)

    rows: list[dict[str, Any]] = []
    for threshold in iou_thresholds:
        matched_true: set[int] = set()
        true_positive = 0
        for pred in pred_bouts:
            best_iou = 0.0
            best_idx = -1
            for idx, true in enumerate(true_bouts):
                if idx in matched_true or true.behavior != pred.behavior:
                    continue
                iou = bout_iou(pred, true)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= threshold and best_idx >= 0:
                matched_true.add(best_idx)
                true_positive += 1
        false_positive = len(pred_bouts) - true_positive
        false_negative = len(true_bouts) - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {
                "iou_threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "num_true_bouts": len(true_bouts),
                "num_pred_bouts": len(pred_bouts),
            }
        )
    return pd.DataFrame(rows)


def edit_score(y_true: np.ndarray, y_pred: np.ndarray, background_id: int) -> float:
    """Segmental edit score: 1 - normalized Levenshtein distance over bout labels."""

    true_seq = [b.behavior for b in extract_bouts(y_true, background_id)]
    pred_seq = [b.behavior for b in extract_bouts(y_pred, background_id)]
    if not true_seq and not pred_seq:
        return 1.0
    distance = _levenshtein(true_seq, pred_seq)
    norm = max(len(true_seq), len(pred_seq), 1)
    return float(1.0 - distance / norm)


def _levenshtein(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1]


@dataclass
class EvaluationReport:
    framewise: FramewiseMetrics
    bouts: pd.DataFrame
    edit: float
    extras: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        out = self.framewise.to_dict()
        out["edit_score"] = self.edit
        out["bouts"] = self.bouts.to_dict(orient="records")
        out.update(self.extras)
        return out


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: LabelMap,
    include_background: bool = False,
) -> EvaluationReport:
    """Full evaluation: framewise + bout-level + edit score."""

    framewise = compute_framewise_metrics(
        y_true, y_pred, label_map, include_background=include_background
    )
    bouts = compute_bout_metrics(y_true, y_pred, label_map)
    edit = edit_score(y_true, y_pred, label_map.background_id)
    return EvaluationReport(framewise=framewise, bouts=bouts, edit=edit)
