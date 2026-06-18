"""Behavior role constraints for interaction-specific models."""

from __future__ import annotations

import re

import numpy as np

from .config import BehaviorRolesConfig
from .labels import LabelMap


def normalize_identity(value: object) -> str:
    return str(value or "").strip()


def behavior_prefix_role(
    behavior: str, config: BehaviorRolesConfig
) -> str | None:
    """Return the role implied by a behavior-name prefix, if any."""

    text = str(behavior).strip()
    for role, prefixes in (
        ("BL6", config.bl6_behavior_prefixes),
        ("CD1", config.cd1_behavior_prefixes),
    ):
        for prefix in prefixes:
            prefix = str(prefix).strip()
            if not prefix:
                continue
            pattern = rf"^{re.escape(prefix)}(?:$|[\s_\-:/])"
            if re.match(pattern, text, flags=re.IGNORECASE):
                return role
    return None


def subject_role(subject_id: str, config: BehaviorRolesConfig) -> str | None:
    """Map a tracked identity to its configured biological/social role."""

    subject = normalize_identity(subject_id)
    if config.interaction_mode == "aggression_cd1_bl6":
        if subject == normalize_identity(config.bl6_identity):
            return "BL6"
        if subject == normalize_identity(config.cd1_identity):
            return "CD1"
    if config.interaction_mode == "single_mouse":
        if subject == normalize_identity(config.primary_identity):
            return "primary"
        return "other"
    return None


def should_expand_global_labels_to_subject(
    subject_id: str, config: BehaviorRolesConfig
) -> bool:
    """Whether blank-subject annotations should become a track for this subject."""

    if config.interaction_mode == "single_mouse":
        return normalize_identity(subject_id) == normalize_identity(
            config.primary_identity
        )
    return True


def behavior_allowed_for_subject(
    behavior: str,
    subject_id: str,
    config: BehaviorRolesConfig,
) -> bool:
    """Return whether ``subject_id`` is allowed to display/learn ``behavior``."""

    if config.interaction_mode == "classic_free_interaction":
        return True
    if config.interaction_mode == "single_mouse":
        return normalize_identity(subject_id) == normalize_identity(
            config.primary_identity
        )

    behavior_role = behavior_prefix_role(behavior, config)
    if behavior_role is None:
        return bool(config.unprefixed_behaviors_allowed_for_all)
    role = subject_role(subject_id, config)
    if role is None:
        return True
    return role == behavior_role


def allowed_class_mask(
    label_map: LabelMap, config: BehaviorRolesConfig, subject_id: str
) -> np.ndarray:
    """Boolean class mask for behavior labels allowed on ``subject_id``."""

    mask = np.ones(label_map.num_classes, dtype=bool)
    for class_id, behavior in label_map.id_to_name.items():
        if class_id == label_map.background_id:
            mask[class_id] = True
        else:
            mask[class_id] = behavior_allowed_for_subject(
                behavior, subject_id, config
            )
    return mask


def constrain_label_array_for_subject(
    labels: np.ndarray,
    label_map: LabelMap,
    config: BehaviorRolesConfig,
    subject_id: str,
) -> np.ndarray:
    """Replace labels disallowed for a subject with background."""

    if config.interaction_mode == "classic_free_interaction":
        return labels
    allowed = allowed_class_mask(label_map, config, subject_id)
    if allowed.all():
        return labels
    out = labels.copy()
    if out.ndim == 2:
        out[~allowed, :] = 0
        active = out.sum(axis=0) > 0
        out[label_map.background_id, ~active] = 1
        return out
    valid = (out >= 0) & (out < len(allowed))
    disallowed = valid & ~allowed[out]
    out[disallowed] = label_map.background_id
    return out


def constrain_probabilities_for_subject(
    probabilities: np.ndarray,
    label_map: LabelMap,
    config: BehaviorRolesConfig,
    subject_id: str,
    multilabel: bool = False,
) -> np.ndarray:
    """Zero probabilities for classes that this subject cannot display."""

    if config.interaction_mode == "classic_free_interaction":
        return probabilities
    allowed = allowed_class_mask(label_map, config, subject_id)
    # Guard against a label-map / prediction class-count mismatch (e.g. stale
    # inference outputs from a different model still in view when the role config
    # changes): the mask is sized to label_map, so it cannot index a prediction
    # with a different number of classes. Skip constraining rather than crash.
    if allowed.all() or len(allowed) != probabilities.shape[0]:
        return probabilities
    out = probabilities.copy()
    out[~allowed, :] = 0.0
    if multilabel:
        return out
    sums = out.sum(axis=0, keepdims=True)
    empty = sums.squeeze(0) <= 1e-9
    sums = np.where(sums <= 1e-9, 1.0, sums)
    out = out / sums
    if np.any(empty):
        out[:, empty] = 0.0
        out[label_map.background_id, empty] = 1.0
    return out
