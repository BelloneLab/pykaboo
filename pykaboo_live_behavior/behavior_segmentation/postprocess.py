"""Postprocessing of framewise probabilities into clean behavior segments.

Steps, in order:

1. merge overlapping sliding-window probabilities by averaging
2. temporal smoothing of probabilities (Gaussian over time)
3. argmax (multiclass) or threshold (multilabel) to framewise labels
4. enforce a minimum bout duration
5. merge same-behavior bouts separated by tiny gaps
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .labels import LabelMap
from .metrics import extract_bouts


def merge_window_logits(
    num_frames: int,
    num_classes: int,
    window_logits: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    """Average overlapping per-window logits onto a single ``[num_classes, T]`` grid.

    Each entry is ``(start, end, logits[num_classes, end-start])``.
    """

    accum = np.zeros((num_classes, num_frames), dtype=np.float64)
    counts = np.zeros(num_frames, dtype=np.float64)
    for start, end, logits in window_logits:
        width = min(end, num_frames) - start
        if width <= 0:
            continue
        accum[:, start : start + width] += logits[:, :width]
        counts[start : start + width] += 1.0
    counts = np.where(counts < 1.0, 1.0, counts)
    return accum / counts


def smooth_probabilities(
    probabilities: np.ndarray,
    smoothing_seconds: float,
    frame_rate: float,
    normalize: bool = True,
) -> np.ndarray:
    """Gaussian-smooth probabilities over time, optionally renormalizing."""

    if smoothing_seconds <= 0:
        return probabilities
    sigma = max(smoothing_seconds * frame_rate / 2.0, 0.0)
    if sigma <= 0:
        return probabilities
    smoothed = gaussian_filter1d(probabilities, sigma=sigma, axis=-1, mode="nearest")
    if not normalize:
        return np.clip(smoothed, 0.0, 1.0)
    column_sums = smoothed.sum(axis=0, keepdims=True)
    column_sums = np.where(column_sums < 1e-9, 1.0, column_sums)
    return smoothed / column_sums


def probabilities_to_labels(
    probabilities: np.ndarray,
    multilabel: bool = False,
    threshold: float | np.ndarray = 0.5,
    background_id: int = 0,
    transition_penalty: float = 0.0,
) -> np.ndarray:
    """Convert probabilities to class ids or binary multi-label rows."""

    if multilabel:
        thresholds = np.asarray(threshold, dtype=np.float32)
        if thresholds.ndim == 0:
            thresholds = np.full(probabilities.shape[0], float(thresholds))
        if transition_penalty > 0:
            active = np.vstack(
                [
                    binary_viterbi_decode(
                        probabilities[class_id],
                        float(thresholds[class_id]),
                        transition_penalty,
                    )
                    for class_id in range(probabilities.shape[0])
                ]
            ).astype(bool)
        else:
            active = probabilities >= thresholds[:, None]
        if 0 <= background_id < active.shape[0]:
            active[background_id, :] = False
        labels = active.astype(np.int8)
        has_foreground = labels.sum(axis=0) > 0
        if 0 <= background_id < labels.shape[0]:
            labels[background_id, :] = 0
            labels[background_id, ~has_foreground] = 1
        return labels
    if transition_penalty > 0:
        return multiclass_viterbi_decode(probabilities, transition_penalty)
    return probabilities.argmax(axis=0).astype(np.int64)


def calibrate_thresholds_to_priors(
    probabilities: np.ndarray,
    thresholds: float | np.ndarray,
    class_priors: np.ndarray,
    background_id: int,
    strength: float = 0.0,
    mode: str = "dampen",
    min_rate: float = 1e-4,
) -> np.ndarray:
    """Adjust multilabel thresholds toward stored training positive rates.

    ``dampen`` only raises thresholds, which suppresses cross-video overprediction
    while keeping the base detector free to miss rare classes. ``match`` can move
    thresholds in either direction and is more aggressive.
    """

    base = np.asarray(thresholds, dtype=np.float32)
    if base.ndim == 0:
        base = np.full(probabilities.shape[0], float(base), dtype=np.float32)
    else:
        base = base.copy()
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0 or probabilities.size == 0:
        return base
    priors = np.asarray(class_priors, dtype=np.float64).reshape(-1)
    if priors.size < probabilities.shape[0]:
        return base
    adjusted = base.copy()
    for class_id in range(probabilities.shape[0]):
        if class_id == background_id:
            continue
        prior = float(priors[class_id])
        if not np.isfinite(prior) or prior < min_rate:
            continue
        target_rate = float(np.clip(prior, min_rate, 1.0 - min_rate))
        quantile_threshold = float(
            np.quantile(probabilities[class_id], 1.0 - target_rate)
        )
        candidate = quantile_threshold
        if mode == "dampen":
            candidate = max(float(base[class_id]), candidate)
        adjusted[class_id] = (1.0 - strength) * float(base[class_id]) + strength * candidate
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)


def binary_viterbi_decode(
    probability: np.ndarray,
    threshold: float,
    transition_penalty: float,
) -> np.ndarray:
    """Two-state temporal decoding for one binary behavior probability row."""

    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    threshold = float(np.clip(threshold, 1e-4, 1.0 - 1e-4))
    emit0 = np.log1p(-p) - np.log1p(-threshold)
    emit1 = np.log(p) - np.log(threshold)
    stay = 0.0
    switch = -float(transition_penalty)
    scores = np.zeros((2, p.size), dtype=np.float64)
    back = np.zeros((2, p.size), dtype=np.int8)
    scores[:, 0] = [emit0[0], emit1[0]]
    for t in range(1, p.size):
        prev0 = np.asarray(
            [scores[0, t - 1] + stay, scores[1, t - 1] + switch]
        )
        prev1 = np.asarray(
            [scores[0, t - 1] + switch, scores[1, t - 1] + stay]
        )
        back[0, t] = int(np.argmax(prev0))
        back[1, t] = int(np.argmax(prev1))
        scores[0, t] = prev0[back[0, t]] + emit0[t]
        scores[1, t] = prev1[back[1, t]] + emit1[t]
    states = np.zeros(p.size, dtype=np.int8)
    states[-1] = int(np.argmax(scores[:, -1]))
    for t in range(p.size - 1, 0, -1):
        states[t - 1] = back[states[t], t]
    return states


def multiclass_viterbi_decode(
    probabilities: np.ndarray,
    transition_penalty: float,
) -> np.ndarray:
    """Viterbi decode class probabilities with a uniform label-switch penalty."""

    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0)
    logp = np.log(probs)
    num_classes, length = logp.shape
    scores = np.zeros((num_classes, length), dtype=np.float64)
    back = np.zeros((num_classes, length), dtype=np.int64)
    scores[:, 0] = logp[:, 0]
    change = np.full((num_classes, num_classes), -float(transition_penalty))
    np.fill_diagonal(change, 0.0)
    for t in range(1, length):
        candidates = scores[:, t - 1][:, None] + change
        back[:, t] = np.argmax(candidates, axis=0)
        scores[:, t] = candidates[back[:, t], np.arange(num_classes)] + logp[:, t]
    states = np.zeros(length, dtype=np.int64)
    states[-1] = int(np.argmax(scores[:, -1]))
    for t in range(length - 1, 0, -1):
        states[t - 1] = back[states[t], t]
    return states


def enforce_min_bout(
    labels: np.ndarray, min_bout_frames: int, background_id: int
) -> np.ndarray:
    """Replace foreground bouts shorter than ``min_bout_frames`` with background."""

    if min_bout_frames <= 1:
        return labels
    if labels.ndim == 2:
        out = labels.copy()
        for class_id in range(out.shape[0]):
            if class_id == background_id:
                continue
            row = out[class_id]
            start = 0
            current = int(row[0]) if row.size else 0
            for pos in range(1, row.size + 1):
                changed = pos == row.size or int(row[pos]) != current
                if changed:
                    if current == 1 and pos - start < min_bout_frames:
                        row[start:pos] = 0
                    if pos < row.size:
                        current = int(row[pos])
                        start = pos
        if 0 <= background_id < out.shape[0]:
            out[background_id, :] = 0
            active = out.sum(axis=0) > 0
            out[background_id, ~active] = 1
        return out
    out = labels.copy()
    for bout in extract_bouts(labels, background_id):
        if bout.length < min_bout_frames:
            out[bout.start : bout.end + 1] = background_id
    return out


def merge_short_gaps(
    labels: np.ndarray, merge_gap_frames: int, background_id: int
) -> np.ndarray:
    """Bridge short background gaps between two bouts of the same behavior."""

    if merge_gap_frames <= 0:
        return labels
    if labels.ndim == 2:
        out = labels.copy()
        for class_id in range(out.shape[0]):
            if class_id == background_id:
                continue
            row = out[class_id]
            idx = 0
            while idx < row.size:
                if row[idx] == 0:
                    end = idx
                    while end < row.size and row[end] == 0:
                        end += 1
                    if (
                        idx > 0
                        and end < row.size
                        and row[idx - 1] == 1
                        and row[end] == 1
                        and end - idx <= merge_gap_frames
                    ):
                        row[idx:end] = 1
                    idx = end
                else:
                    idx += 1
        if 0 <= background_id < out.shape[0]:
            out[background_id, :] = 0
            active = out.sum(axis=0) > 0
            out[background_id, ~active] = 1
        return out
    out = labels.copy()
    bouts = extract_bouts(labels, background_id)
    for prev, nxt in zip(bouts, bouts[1:]):
        gap = nxt.start - prev.end - 1
        if prev.behavior == nxt.behavior and 0 < gap <= merge_gap_frames:
            out[prev.end + 1 : nxt.start] = prev.behavior
    return out


def freezing_class_ids(label_map: LabelMap) -> list[int]:
    """Return class ids whose behavior names describe freezing."""

    return [
        class_id
        for class_id, name in label_map.id_to_name.items()
        if class_id != label_map.background_id and "freez" in str(name).lower()
    ]


def stationary_motion_mask(
    features: np.ndarray,
    feature_names: list[str],
    frame_indices: np.ndarray | None,
    frame_rate: float,
    min_stationary_seconds: float = 2.0,
    speed_threshold: float = 0.2,
) -> np.ndarray | None:
    """Detect stationary runs from per-track motion features.

    The preferred cue is ``speed_body_norm`` (body-lengths per second). If that
    feature is unavailable, this falls back to ``speed`` and then to center
    deltas converted to per-second speed. Only continuous runs lasting at least
    ``min_stationary_seconds`` are marked.
    """

    if features is None or len(feature_names) == 0:
        return None
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return None
    names = {name: idx for idx, name in enumerate(feature_names)}
    speed: np.ndarray | None = None
    for candidate in ("speed_body_norm", "speed", "nearest_subject_speed"):
        idx = names.get(candidate)
        if idx is not None and idx < matrix.shape[1]:
            speed = matrix[:, idx].astype(np.float64, copy=False)
            break
    if speed is None and {"delta_center_x", "delta_center_y"}.issubset(names):
        dx = matrix[:, names["delta_center_x"]].astype(np.float64, copy=False)
        dy = matrix[:, names["delta_center_y"]].astype(np.float64, copy=False)
        speed = np.hypot(dx, dy) * float(frame_rate)
    if speed is None:
        return None

    speed = np.nan_to_num(np.abs(speed), nan=np.inf, posinf=np.inf, neginf=np.inf)
    candidate = speed <= float(speed_threshold)
    min_frames = max(int(np.ceil(float(min_stationary_seconds) * float(frame_rate))), 1)
    frames = (
        np.asarray(frame_indices).reshape(-1)[: len(candidate)]
        if frame_indices is not None
        else None
    )
    return _keep_long_true_runs(candidate, min_frames, frames)


def _keep_long_true_runs(
    mask: np.ndarray,
    min_frames: int,
    frame_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Keep true runs whose duration is at least ``min_frames``."""

    out = np.zeros(len(mask), dtype=bool)
    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        end = idx + 1
        while end < len(mask) and mask[end]:
            end += 1
        if frame_indices is not None and len(frame_indices) >= end:
            duration = int(frame_indices[end - 1] - frame_indices[idx] + 1)
        else:
            duration = end - idx
        if duration >= min_frames:
            out[idx:end] = True
        idx = end
    return out


def refine_freezing_by_stationary_motion(
    labels: np.ndarray,
    probabilities: np.ndarray,
    label_map: LabelMap,
    features: np.ndarray,
    feature_names: list[str],
    frame_indices: np.ndarray | None,
    frame_rate: float,
    enabled: bool = True,
    min_stationary_seconds: float = 2.0,
    speed_threshold: float = 0.2,
) -> np.ndarray:
    """Refine freezing labels using the ethological no-movement >=2s rule."""

    if not enabled:
        return labels
    freeze_ids = freezing_class_ids(label_map)
    if not freeze_ids:
        return labels
    stationary = stationary_motion_mask(
        features,
        feature_names,
        frame_indices,
        frame_rate,
        min_stationary_seconds,
        speed_threshold,
    )
    if stationary is None:
        return labels

    probs = np.asarray(probabilities) if probabilities is not None else None
    allowed_freeze_ids = []
    for class_id in freeze_ids:
        if probs is None or class_id >= probs.shape[0]:
            allowed_freeze_ids.append(class_id)
            continue
        row = np.nan_to_num(probs[class_id], nan=0.0)
        if row.size and np.nanmax(row) > 1e-8:
            allowed_freeze_ids.append(class_id)
    out = labels.copy()
    limit = min(out.shape[-1], len(stationary))
    stationary = stationary[:limit]
    bg = label_map.background_id

    if out.ndim == 2:
        for class_id in freeze_ids:
            if class_id < out.shape[0]:
                out[class_id, :limit] = 0
        for class_id in allowed_freeze_ids:
            if class_id < out.shape[0]:
                out[class_id, :limit] = stationary.astype(out.dtype)
        if 0 <= bg < out.shape[0]:
            out[bg, :] = 0
            active = out.sum(axis=0) > 0
            out[bg, ~active] = 1
        return out

    if not allowed_freeze_ids:
        segment = out[:limit].copy()
        segment[np.isin(segment, freeze_ids)] = bg
        out[:limit] = segment
        return out
    segment = out[:limit].copy()
    freeze_mask = np.isin(segment, freeze_ids)
    segment[freeze_mask & ~stationary] = bg
    if probs is not None and probs.ndim == 2:
        freeze_probs = probs[np.asarray(allowed_freeze_ids), :limit]
        best = np.asarray(allowed_freeze_ids)[np.argmax(freeze_probs, axis=0)]
        segment[stationary] = best[stationary]
    else:
        segment[stationary] = allowed_freeze_ids[0]
    out[:limit] = segment
    return out


def postprocess_predictions(
    probabilities: np.ndarray,
    label_map: LabelMap,
    frame_rate: float,
    smoothing_seconds: float = 0.2,
    min_bout_frames: int = 3,
    merge_gap_frames: int = 2,
    confidence_threshold: float | np.ndarray = 0.5,
    multilabel: bool = False,
    transition_penalty: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full postprocessing chain. Returns (labels, smoothed_probabilities)."""

    smoothed = smooth_probabilities(
        probabilities, smoothing_seconds, frame_rate, normalize=not multilabel
    )
    labels = probabilities_to_labels(
        smoothed,
        multilabel,
        confidence_threshold,
        label_map.background_id,
        transition_penalty,
    )
    labels = enforce_min_bout(labels, min_bout_frames, label_map.background_id)
    labels = merge_short_gaps(labels, merge_gap_frames, label_map.background_id)
    return labels, smoothed
