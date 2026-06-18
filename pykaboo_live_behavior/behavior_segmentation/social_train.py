"""Multi-label temporal trainer and evaluator for social behavior segmentation.

The recall problem under heavy class imbalance is attacked on four fronts:

1. *Focal BCE with per-class ``pos_weight``* so rare behaviors are not drowned out.
2. *Window oversampling* that surfaces windows containing rare behaviors.
3. *Per-class decision-threshold tuning* on a validation split (maximize F1).
4. *Temporal smoothing + min-bout postprocessing* to turn noisy frame scores into
   clean bouts.

Everything is driven by explicit ``train/val/test`` track lists so the same code
serves leave-one-video-out and temporal-holdout protocols, and arbitrary feature
subsets for in-app A/B comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import WindowDataset, collate_windows
from .labels import LabelMap
from .metrics import extract_bouts
from .models.temporal_tcn import TemporalTcnModel
from .normalization import FeatureNormalizer
from .windows import seconds_to_frames

ProgressFn = Callable[[float, str], None]
LogFn = Callable[[str], None]


# --------------------------------------------------------------------------- #
# Feature-group selection (for ablations / in-app comparison)
# --------------------------------------------------------------------------- #

import re as _re

_ROLL_RE = _re.compile(r"_(mean|std|min|max|median|slope)_w\d+$")


def feature_group_of(col: str) -> str:
    """Fine-grained feature family for a column (used for ablation/compare)."""

    base = col[4:] if col.startswith("obj_") else col
    if "_cwt_p" in base:
        return "wavelet"
    if base.startswith("pp_") or base.startswith("pose_"):
        return "pose"
    if base.startswith("pair_"):
        return "maskpair"
    if _ROLL_RE.search(base):
        return "maskroll"
    return "maskcore"


# High-level group aliases -> the fine families they include.
GROUP_ALIASES = {
    "mask": {"maskcore", "maskroll", "maskpair"},
    "maskcore": {"maskcore", "maskpair"},
    "maskroll": {"maskroll"},
    "pose": {"pose"},
    "wavelet": {"wavelet"},
}


def select_feature_indices(
    feature_names: list[str], groups: list[str] | None
) -> list[int]:
    if not groups:
        return list(range(len(feature_names)))
    wanted: set[str] = set()
    for g in groups:
        wanted |= GROUP_ALIASES.get(g, {g})
    keep = [i for i, n in enumerate(feature_names) if feature_group_of(n) in wanted]
    return keep or list(range(len(feature_names)))


def subset_tracks(tracks, keep_idx: list[int], feature_names: list[str]):
    new_names = [feature_names[i] for i in keep_idx]
    out = []
    for t in tracks:
        import copy

        nt = copy.copy(t)
        nt.features = t.features[:, keep_idx]
        nt.feature_names = new_names
        out.append(nt)
    return out, new_names


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss with per-class positive weighting, time-masked.

    ``logits``/``targets`` are ``[B, K, T]``; ``mask`` is ``[B, T]``.
    """

    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight[None, :, None]
    )
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    focal = (1.0 - pt).clamp_min(1e-6) ** gamma * bce
    m = mask.unsqueeze(1)
    return (focal * m).sum() / m.sum().clamp_min(1.0) / max(logits.shape[1], 1)


def temporal_smoothness(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (probs[:, :, 1:] - probs[:, :, :-1]) ** 2
    pair = (mask[:, 1:] * mask[:, :-1]).unsqueeze(1)
    return (diff * pair).sum() / pair.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def multilabel_window_weights(dataset: WindowDataset, num_classes: int, bg_id: int):
    """Weight each window by the rarity of the behaviors it contains."""

    counts = np.ones(num_classes)
    for t in dataset.tracks:
        if t.labels is not None:
            counts += t.labels.sum(axis=1)
    inv = 1.0 / np.sqrt(counts)
    inv[bg_id] = 0.0
    weights = []
    for span in dataset.spans:
        t = dataset.tracks[span.track_index]
        seg = t.labels[:, span.start : span.end]
        present = (seg.sum(axis=1) > 0).astype(float)
        weights.append(1.0 + float((present * inv).sum()))
    return np.asarray(weights, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict_track_probs(
    model: TemporalTcnModel,
    track,
    normalizer: FeatureNormalizer,
    window_frames: int,
    stride_frames: int,
    device: str,
    want_embeddings: bool = False,
):
    """Sliding-window multilabel inference -> probs ``[K, T]`` (+ optional emb)."""

    model.eval()
    feats = normalizer.transform(track.features).astype(np.float32)
    T = feats.shape[0]
    K = model.num_classes
    prob_sum = np.zeros((K, T), dtype=np.float64)
    counts = np.zeros(T, dtype=np.float64)
    emb_sum = None
    starts = list(range(0, max(T - window_frames, 0) + 1, max(stride_frames, 1)))
    if not starts or starts[-1] + window_frames < T:
        starts.append(max(T - window_frames, 0))
    for s in starts:
        e = min(s + window_frames, T)
        chunk = feats[s:e]
        x = torch.from_numpy(chunk.T[None]).to(device)
        out = model(x)
        p = out.probabilities[0].cpu().numpy()  # [K, e-s]
        prob_sum[:, s:e] += p
        counts[s:e] += 1.0
        if want_embeddings:
            emb = out.embeddings[0].cpu().numpy()
            if emb_sum is None:
                emb_sum = np.zeros((emb.shape[0], T), dtype=np.float64)
            emb_sum[:, s:e] += emb
    counts = np.clip(counts, 1.0, None)
    probs = prob_sum / counts
    embeddings = (emb_sum / counts) if emb_sum is not None else None
    return probs, embeddings


# --------------------------------------------------------------------------- #
# Threshold tuning + metrics
# --------------------------------------------------------------------------- #

def tune_thresholds(probs: np.ndarray, targets: np.ndarray, bg_id: int):
    """Per-class threshold maximizing F1 on (probs, targets) ``[K, N]``."""

    K = probs.shape[0]
    thr = np.full(K, 0.5)
    grid = np.linspace(0.05, 0.95, 19)
    for k in range(K):
        y = targets[k]
        if y.sum() < 5:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (probs[k] >= t).astype(int)
            tp = float((pred & y).sum())
            fp = float((pred & (1 - y)).sum())
            fn = float(((1 - pred) & y).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[k] = best_t
    return thr


def smooth_probs(probs: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return probs
    kernel = np.ones(win) / win
    out = np.empty_like(probs)
    for k in range(probs.shape[0]):
        out[k] = np.convolve(probs[k], kernel, mode="same")
    return out


def enforce_min_bout(binary: np.ndarray, min_frames: int) -> np.ndarray:
    """Remove positive runs shorter than ``min_frames`` (per class row)."""

    if min_frames <= 1:
        return binary
    out = binary.copy()
    for k in range(out.shape[0]):
        row = out[k]
        idx = 0
        n = len(row)
        while idx < n:
            if row[idx] == 1:
                j = idx
                while j < n and row[j] == 1:
                    j += 1
                if j - idx < min_frames:
                    row[idx:j] = 0
                idx = j
            else:
                idx += 1
    return out


def merge_short_gaps_binary(binary: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Bridge short zero gaps between positive runs, per class row."""

    if max_gap_frames <= 0:
        return binary
    out = binary.copy()
    for k in range(out.shape[0]):
        row = out[k]
        idx = 0
        n = len(row)
        while idx < n:
            if row[idx] == 0:
                j = idx
                while j < n and row[j] == 0:
                    j += 1
                gap = j - idx
                bounded_by_positive = (
                    idx > 0 and j < n and row[idx - 1] == 1 and row[j] == 1
                )
                if bounded_by_positive and gap <= max_gap_frames:
                    row[idx:j] = 1
                idx = j
            else:
                idx += 1
    return out


def per_behavior_report(
    probs_list: list[np.ndarray],
    targets_list: list[np.ndarray],
    thresholds: np.ndarray,
    label_map: LabelMap,
    frame_rate: float,
    smooth_win: int = 5,
    min_bout: int = 5,
    merge_gap: int = 0,
) -> pd.DataFrame:
    """Framewise + bout metrics per behavior, pooled over the given tracks."""

    K = label_map.num_classes
    bg = label_map.background_id
    # concat targets
    Y = np.concatenate(targets_list, axis=1)  # [K, N]
    # build predictions per track (so smoothing/min-bout respect track edges)
    preds = []
    for probs in probs_list:
        sp = smooth_probs(probs, smooth_win)
        pred = (sp >= thresholds[:, None]).astype(np.int8)
        pred = merge_short_gaps_binary(pred, merge_gap)
        pred = enforce_min_bout(pred, min_bout)
        preds.append(pred)
    P = np.concatenate(preds, axis=1)

    rows = []
    for k in range(K):
        if k == bg:
            continue
        y = Y[k].astype(int)
        p = P[k].astype(int)
        tp = float((p & y).sum())
        fp = float((p & (1 - y)).sum())
        fn = float(((1 - p) & y).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        # bout-level F1 at IoU 0.5 (pooled across tracks)
        bf1 = bout_f1(preds, targets_list, k, iou=0.5)
        rows.append(
            {
                "behavior": label_map.id_to_name[k],
                "support": int(y.sum()),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "bout_f1": bf1,
                "threshold": float(thresholds[k]),
            }
        )
    return pd.DataFrame(rows)


def bout_f1(preds, targets_list, k, iou=0.5) -> float:
    tp = fp = fn = 0
    for pred, tgt in zip(preds, targets_list):
        gt = intervals_from_binary(tgt[k])
        pr = intervals_from_binary(pred[k])
        matched = set()
        for a in pr:
            hit = False
            for j, b in enumerate(gt):
                if j in matched:
                    continue
                if interval_iou(a, b) >= iou:
                    matched.add(j)
                    hit = True
                    break
            tp += int(hit)
            fp += int(not hit)
        fn += len(gt) - len(matched)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def intervals_from_binary(row: np.ndarray):
    out = []
    n = len(row)
    i = 0
    while i < n:
        if row[i]:
            j = i
            while j < n and row[j]:
                j += 1
            out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def interval_iou(a, b) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union > 0 else 0.0


def weighted_macro(report: pd.DataFrame):
    sup = report["support"].to_numpy(float)
    has = sup > 0
    wf1 = float(np.average(report["f1"][has], weights=sup[has])) if has.any() else 0.0
    macro = float(report["f1"][has].mean()) if has.any() else 0.0
    return wf1, macro


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    window_seconds: float = 16.0
    stride_seconds: float = 2.0
    eval_stride_seconds: float = 4.0
    batch_size: int = 16
    max_epochs: int = 60
    learning_rate: float = 7e-4
    weight_decay: float = 2e-4
    hidden_channels: int = 128
    num_stages: int = 4
    num_layers_per_stage: int = 10
    dropout: float = 0.3
    embedding_dim: int = 64
    focal_gamma: float = 1.5
    pos_weight_cap: float = 12.0
    smoothness_weight: float = 4.0
    early_stopping_patience: int = 16
    grad_clip: float = 1.0
    smooth_win: int = 15
    min_bout_frames: int = 12
    merge_gap_frames: int = 6
    # augmentation (combats overfitting with few training windows)
    feature_noise: float = 0.08
    feature_dropout: float = 0.1
    seed: int = 42


@dataclass
class TrainResult:
    report: pd.DataFrame
    summary: dict[str, Any]
    thresholds: np.ndarray
    history: list[dict[str, float]] = field(default_factory=list)
    model: Any = None
    normalizer: Any = None
    feature_names: list[str] = field(default_factory=list)


def pos_weights_from_tracks(tracks, num_classes, cap):
    pos = np.ones(num_classes)
    tot = 0
    for t in tracks:
        if t.labels is not None:
            pos += t.labels.sum(axis=1)
            tot += t.labels.shape[1]
    neg = np.clip(tot - pos, 1, None)
    pw = np.clip(neg / pos, 1.0, cap)
    return torch.tensor(pw, dtype=torch.float32)


def train_social_model(
    train_tracks,
    val_tracks,
    test_tracks,
    feature_names: list[str],
    label_map: LabelMap,
    frame_rate: float,
    tc: TrainConfig,
    device: str = "cuda",
    feature_groups: list[str] | None = None,
    log: LogFn | None = None,
    progress: ProgressFn | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TrainResult:
    def emit(frac, msg):
        if log:
            log(msg)
        if progress:
            progress(frac, msg)

    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    if feature_groups:
        orig_names = list(feature_names)
        keep = select_feature_indices(orig_names, feature_groups)
        train_tracks, feature_names = subset_tracks(train_tracks, keep, orig_names)
        if val_tracks:
            val_tracks, _ = subset_tracks(val_tracks, keep, orig_names)
        if test_tracks:
            test_tracks, _ = subset_tracks(test_tracks, keep, orig_names)

    normalizer = FeatureNormalizer.fit([t.features for t in train_tracks], feature_names)
    win = seconds_to_frames(tc.window_seconds, frame_rate)
    stride = seconds_to_frames(tc.stride_seconds, frame_rate)
    eval_stride = seconds_to_frames(tc.eval_stride_seconds, frame_rate)

    train_ds = WindowDataset(
        train_tracks, tc.window_seconds, tc.stride_seconds, frame_rate,
        normalizer, multilabel=True,
    )
    w = multilabel_window_weights(train_ds, label_map.num_classes, label_map.background_id)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    loader = DataLoader(
        train_ds, batch_size=tc.batch_size, sampler=sampler,
        collate_fn=collate_windows, num_workers=0,
    )

    model = TemporalTcnModel(
        num_features=len(feature_names),
        num_classes=label_map.num_classes,
        hidden_channels=tc.hidden_channels,
        num_stages=tc.num_stages,
        num_layers_per_stage=tc.num_layers_per_stage,
        dropout=tc.dropout,
        embedding_dim=tc.embedding_dim,
        multilabel=True,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.max_epochs)
    pos_weight = pos_weights_from_tracks(train_tracks, label_map.num_classes, tc.pos_weight_cap).to(device)

    best_score = -1.0
    best_state = None
    best_thr = np.full(label_map.num_classes, 0.5)
    patience = 0
    history = []

    val_for_select = val_tracks or test_tracks

    for epoch in range(tc.max_epochs):
        if should_stop and should_stop():
            emit(1.0, "Stopped by user.")
            break
        model.train()
        tot = 0.0
        nb = 0
        for batch in loader:
            feats = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)  # [B, K, T]
            if tc.feature_noise > 0:
                feats = feats + tc.feature_noise * torch.randn_like(feats)
            if tc.feature_dropout > 0:
                keep = (
                    torch.rand(feats.shape[0], feats.shape[1], 1, device=device)
                    > tc.feature_dropout
                ).float()
                feats = feats * keep
            opt.zero_grad()
            out = model(feats)
            loss = focal_bce(out.logits, labels, mask, pos_weight, tc.focal_gamma)
            for sl in out.stage_logits[:-1]:
                loss = loss + 0.5 * focal_bce(sl, labels, mask, pos_weight, tc.focal_gamma)
            loss = loss + tc.smoothness_weight * temporal_smoothness(out.probabilities, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()

        # validation: tune thresholds, score weighted-F1
        vp, vt = _collect(model, val_for_select, normalizer, win, eval_stride, device)
        thr = tune_thresholds(np.concatenate(vp, axis=1), np.concatenate(vt, axis=1), label_map.background_id)
        rep = per_behavior_report(
            vp,
            vt,
            thr,
            label_map,
            frame_rate,
            tc.smooth_win,
            tc.min_bout_frames,
            tc.merge_gap_frames,
        )
        wf1, macro = weighted_macro(rep)
        history.append({"epoch": epoch, "loss": tot / max(nb, 1), "val_wf1": wf1, "val_macro": macro})
        emit(0.1 + 0.8 * (epoch + 1) / tc.max_epochs,
             f"epoch {epoch+1}/{tc.max_epochs} loss={tot/max(nb,1):.3f} val_wF1={wf1:.3f} macro={macro:.3f}")
        if wf1 > best_score:
            best_score = wf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr = thr
            patience = 0
        else:
            patience += 1
            if patience >= tc.early_stopping_patience:
                emit(0.9, f"Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final test evaluation
    tp_, tt_ = _collect(model, test_tracks, normalizer, win, eval_stride, device)
    report = per_behavior_report(
        tp_,
        tt_,
        best_thr,
        label_map,
        frame_rate,
        tc.smooth_win,
        tc.min_bout_frames,
        tc.merge_gap_frames,
    )
    wf1, macro = weighted_macro(report)
    # scene-level (OR across tracks within a video)
    scene = scene_level_report(model, test_tracks, normalizer, win, eval_stride, device,
                               best_thr, label_map, tc)
    swf1, smacro = weighted_macro(scene)
    summary = {
        "test_weighted_f1": wf1,
        "test_macro_f1": macro,
        "scene_weighted_f1": swf1,
        "scene_macro_f1": smacro,
        "best_val_weighted_f1": best_score,
        "num_features": len(feature_names),
    }
    emit(1.0, f"DONE test_wF1={wf1:.3f} macro={macro:.3f} scene_wF1={swf1:.3f}")
    return TrainResult(report=report, summary=summary, thresholds=best_thr,
                       history=history, model=model, normalizer=normalizer,
                       feature_names=feature_names)


def _collect(model, tracks, normalizer, win, stride, device):
    probs_list, tgt_list = [], []
    for t in tracks:
        if t.labels is None:
            continue
        probs, _ = predict_track_probs(model, t, normalizer, win, stride, device)
        n = min(probs.shape[1], t.labels.shape[1])
        probs_list.append(probs[:, :n])
        tgt_list.append(t.labels[:, :n].astype(int))
    return probs_list, tgt_list


def scene_level_report(model, tracks, normalizer, win, stride, device, thr, label_map, tc):
    """OR predictions/targets across the tracks of each video -> scene ethogram."""

    by_video: dict[str, list] = {}
    for t in tracks:
        by_video.setdefault(t.video_id, []).append(t)
    scene_preds, scene_tgts = [], []
    for vid, ts in by_video.items():
        n = min(t.labels.shape[1] for t in ts)
        pred_or = np.zeros((label_map.num_classes, n), dtype=np.int8)
        tgt_or = np.zeros((label_map.num_classes, n), dtype=np.int8)
        for t in ts:
            probs, _ = predict_track_probs(model, t, normalizer, win, stride, device)
            sp = smooth_probs(probs[:, :n], tc.smooth_win)
            pred = (sp >= thr[:, None]).astype(np.int8)
            pred = merge_short_gaps_binary(pred, tc.merge_gap_frames)
            pred = enforce_min_bout(pred, tc.min_bout_frames)
            pred_or |= pred
            tgt_or |= t.labels[:, :n].astype(np.int8)
        scene_preds.append(pred_or)
        scene_tgts.append(tgt_or)
    return per_behavior_report(
        scene_preds,
        scene_tgts,
        thr,
        label_map,
        0,
        smooth_win=1,
        min_bout=1,
        merge_gap=0,
    )
