"""Temporal windowing utilities for framewise segmentation training.

A window is a contiguous slice of a track's frame sequence. Windows are produced
with a configurable length and stride (in frames). For inference we use
overlapping windows whose predictions are later averaged; for training we can
sample windows in a class-balanced way so rare behaviors are seen often enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowSpan:
    """A half-open window ``[start, end)`` over a track's frame positions."""

    track_index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def seconds_to_frames(seconds: float, frame_rate: float) -> int:
    return max(int(round(seconds * frame_rate)), 1)


def build_windows(
    num_frames: int,
    window_length: int,
    stride: int,
    track_index: int = 0,
    drop_last_partial: bool = False,
) -> list[WindowSpan]:
    """Build sliding windows covering ``num_frames`` positions."""

    if num_frames <= 0:
        return []
    window_length = min(window_length, num_frames)
    stride = max(stride, 1)
    spans: list[WindowSpan] = []
    start = 0
    while start < num_frames:
        end = min(start + window_length, num_frames)
        if end - start < window_length and drop_last_partial and spans:
            break
        spans.append(WindowSpan(track_index, start, end))
        if end >= num_frames:
            break
        start += stride
    return spans


def _behavior_category(name: str) -> str:
    n = str(name).lower()
    if "attack" in n:
        return "attack"
    if "freez" in n or "groom" in n:
        return "freeze_groom"
    return "other"


DEFAULT_SUPPORT_QUOTAS = {
    "attack": 0.25,
    "freeze_groom": 0.20,
    "other": 0.20,
    "hard_neg": 0.20,
    "background": 0.15,
}


def support_balanced_window_weights(
    spans: list[WindowSpan],
    tracks: list,
    names: list[str],
    background_id: int,
    quotas: dict[str, float] | None = None,
    cap_fraction: float = 0.40,
    hard_neg_radius: int = 2,
) -> np.ndarray:
    """Sampler weights that enforce category QUOTAS and a per-(video, behavior) CAP.

    Decouples the effective training distribution from raw frame frequency so a
    freezing-heavy video (M13 at 55%) cannot flood the batches. Each window is sorted
    into a category (attack / freeze_groom / other / hard_neg / background); categories
    are sampled to the requested quotas; within a category, groups are sqrt-tempered by
    size and no single (video, dominant-behavior) group may exceed ``cap_fraction`` of
    that category. Returns per-window probabilities (sum 1).
    """

    quotas = quotas or DEFAULT_SUPPORT_QUOTAS
    cat_of_class = {
        k: _behavior_category(names[k]) for k in range(len(names)) if k != background_id
    }
    fg_masks: list = []
    for t in tracks:
        lab = getattr(t, "labels", None)
        if lab is None:
            fg_masks.append(None)
        elif lab.ndim == 2:
            fg_masks.append(np.delete(lab, background_id, axis=0).sum(axis=0) > 0)
        else:
            fg_masks.append(lab != background_id)

    win_cat: list[str] = []
    win_group: list[tuple] = []
    for span in spans:
        t = tracks[span.track_index]
        lab = getattr(t, "labels", None)
        vid = str(getattr(t, "video_id", span.track_index))
        present: set[str] = set()
        dom_name = "bg"
        if lab is not None:
            if lab.ndim == 2:
                seg = lab[:, span.start : span.end]
                counts = seg.sum(axis=1).astype(float)
                counts[background_id] = -1.0
                for k in range(len(names)):
                    if k != background_id and seg[k].any():
                        present.add(cat_of_class[k])
                if counts.max() > 0:
                    dom_name = names[int(counts.argmax())]
            else:
                seg = lab[span.start : span.end]
                fg = seg[seg != background_id]
                if fg.size:
                    dom = int(np.bincount(fg, minlength=len(names)).argmax())
                    dom_name = names[dom]
                    present.add(cat_of_class[dom])
        if present:
            cat = "attack" if "attack" in present else (
                "freeze_groom" if "freeze_groom" in present else "other"
            )
        else:
            fgm = fg_masks[span.track_index]
            w = span.end - span.start
            lo, hi = max(0, span.start - hard_neg_radius * w), span.end + hard_neg_radius * w
            cat = "hard_neg" if (fgm is not None and fgm[lo:hi].any()) else "background"
        win_cat.append(cat)
        win_group.append((vid, dom_name))

    weights = np.zeros(len(spans), dtype=np.float64)
    by_cat: dict[str, list[int]] = {}
    for i, c in enumerate(win_cat):
        by_cat.setdefault(c, []).append(i)
    for cat, idxs in by_cat.items():
        target = quotas.get(cat, 0.0)
        if target <= 0 or not idxs:
            continue
        gcount: dict[tuple, int] = {}
        for i in idxs:
            gcount[win_group[i]] = gcount.get(win_group[i], 0) + 1
        gtot = {g: float(np.sqrt(n)) for g, n in gcount.items()}
        s = sum(gtot.values()) or 1.0
        gtot = {g: v / s for g, v in gtot.items()}
        for _ in range(3):  # water-fill the per-group cap
            over = {g: v for g, v in gtot.items() if v > cap_fraction}
            if not over:
                break
            excess = sum(v - cap_fraction for v in over.values())
            for g in over:
                gtot[g] = cap_fraction
            under = [g for g in gtot if gtot[g] < cap_fraction]
            room = sum(cap_fraction - gtot[g] for g in under) or 1.0
            for g in under:
                gtot[g] += excess * (cap_fraction - gtot[g]) / room
        for i in idxs:
            weights[i] = target * gtot[win_group[i]] / gcount[win_group[i]]
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


def class_balanced_window_weights(
    spans: list[WindowSpan],
    labels_per_track: list[np.ndarray],
    num_classes: int,
    background_id: int,
) -> np.ndarray:
    """Weight windows inversely to the frequency of their dominant foreground class."""

    class_counts = np.ones(num_classes, dtype=np.float64)
    for labels in labels_per_track:
        flat = labels.reshape(-1) if labels.ndim == 1 else labels.argmax(axis=0)
        ids, counts = np.unique(flat, return_counts=True)
        for class_id, count in zip(ids, counts):
            class_counts[int(class_id)] += count
    class_weight = class_counts.sum() / (num_classes * class_counts)

    weights = np.zeros(len(spans), dtype=np.float64)
    for i, span in enumerate(spans):
        labels = labels_per_track[span.track_index]
        if labels.ndim == 2:
            segment = labels[:, span.start : span.end].argmax(axis=0)
        else:
            segment = labels[span.start : span.end]
        foreground = segment[segment != background_id]
        if foreground.size == 0:
            weights[i] = class_weight[background_id]
        else:
            dominant = np.bincount(foreground, minlength=num_classes).argmax()
            weights[i] = class_weight[int(dominant)]
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights
