"""Training entry point for the temporal behavior segmentation model.

Provides :func:`run_training`, a callable that the CLI and the GUI both use. It
extracts features, aligns labels, splits by group, fits the normalizer on the
train split only, trains the multi-stage TCN with early stopping on validation
macro F1, evaluates on the held-out split, and writes a full set of metric
reports plus a self-contained checkpoint.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .checkpoint import Checkpoint, save_checkpoint
from .config import AppConfig, load_config, save_config
from .dataset import (
    WindowDataset,
    build_tracks,
    collate_windows,
    split_tracks_by_group,
)
from .labels import LabelMap
from .metrics import evaluate
from .models.embtcn_attention import EmbTCNAttention, EmbTCNConfig
from .models.mask_video_encoder import (
    EmbTCNMaskVideoFusion,
    MaskVideoConfig as TorchMaskVideoConfig,
)
from .models.temporal_tcn import TemporalTcnModel
from .normalization import FeatureNormalizer
from .pipeline import (
    align_all_labels,
    attach_mask_clips_to_tracks,
    extract_features_for_videos,
    _forward_model,
    logits_to_probabilities,
    resolve_device,
    sliding_window_inference,
)
from .postprocess import postprocess_predictions, smooth_probabilities
from .roles import constrain_probabilities_for_subject
from .ssl_pretrain import (
    augmented_ssl_view,
    hierarchical_temporal_contrastive_loss,
    vicreg_regularizer,
)
from .storage import write_json, write_table
from .windows import support_balanced_window_weights, class_balanced_window_weights

ProgressFn = Callable[[float, str], None]
StopFn = Callable[[], bool]


@dataclass
class TrainingResult:
    checkpoint_path: Path
    metrics: dict[str, Any]
    history: list[dict[str, float]] = field(default_factory=list)
    label_map: LabelMap | None = None


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.weight * grad_output, None


class DomainAdversary(nn.Module):
    def __init__(self, embedding_dim: int, num_domains: int) -> None:
        super().__init__()
        hidden = max(embedding_dim, 16)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor, grl_weight: float) -> torch.Tensor:
        valid = mask.unsqueeze(1).float()
        pooled = (embeddings * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        pooled = GradientReversal.apply(pooled, grl_weight)
        return self.net(pooled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights_from_tracks(
    tracks, num_classes: int, scheme: str, effective_beta: float = 0.999
) -> torch.Tensor:
    counts = np.ones(num_classes, dtype=np.float64)
    for track in tracks:
        if track.labels is None:
            continue
        labels = track.labels
        flat = labels.reshape(-1) if labels.ndim == 1 else labels.argmax(axis=0)
        ids, c = np.unique(flat, return_counts=True)
        for class_id, count in zip(ids, c):
            counts[int(class_id)] += count
    if scheme == "inverse_frequency":
        weights = counts.sum() / counts
    elif scheme == "inverse_sqrt_frequency":
        weights = np.sqrt(counts.sum() / counts)
    elif scheme == "effective_number":
        beta = float(np.clip(effective_beta, 0.0, 0.999999))
        weights = (1.0 - beta) / np.maximum(1.0 - np.power(beta, counts), 1e-12)
    else:
        weights = np.ones(num_classes)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def add_grouped_attack_targets(aligned, config: AppConfig) -> LabelMap:
    """Add a parent attack-family target for multilabel aggression training."""

    label_map = aligned.label_map
    if not config.training.grouped_attack_label or config.labels.mode != "multilabel":
        return label_map
    name = config.training.grouped_attack_name
    names = list(label_map.names)
    if name not in names:
        names.append(name)
    new_map = LabelMap(names=names, background_label=label_map.background_label)
    old_ids = label_map.name_to_id
    new_parent = new_map.name_to_id[name]
    member_ids = [
        old_ids[member]
        for member in config.training.grouped_attack_members
        if member in old_ids
    ]
    if not member_ids:
        aligned.label_map = new_map
        return new_map
    for key, labels in list(aligned.tracks.items()):
        if labels.ndim != 2:
            continue
        expanded = np.zeros((new_map.num_classes, labels.shape[1]), dtype=labels.dtype)
        expanded[: labels.shape[0]] = labels
        grouped = labels[member_ids].max(axis=0)
        expanded[new_parent] = np.maximum(expanded[new_parent], grouped)
        aligned.tracks[key] = expanded
    aligned.label_map = new_map
    return new_map


def _attack_member_ids(label_map: LabelMap, config: AppConfig) -> list[int]:
    name_to_id = label_map.name_to_id
    ids = [
        name_to_id[name]
        for name in config.training.grouped_attack_members
        if name in name_to_id
    ]
    parent = name_to_id.get(config.training.grouped_attack_name)
    if parent is not None:
        ids.append(parent)
    return sorted(set(ids))


def attack_boundary_window_weights(
    spans,
    labels_per_track: list[np.ndarray],
    label_map: LabelMap,
    config: AppConfig,
) -> np.ndarray:
    """Boost windows touching attack-family onset/offset boundaries."""

    boost = float(config.training.attack_boundary_oversample_weight)
    if boost <= 0:
        return np.ones(len(spans), dtype=np.float64)
    member_ids = _attack_member_ids(label_map, config)
    if not member_ids:
        return np.ones(len(spans), dtype=np.float64)
    weights = np.ones(len(spans), dtype=np.float64)
    boundaries_by_track: list[np.ndarray] = []
    for labels in labels_per_track:
        if labels.ndim != 2:
            boundaries_by_track.append(np.array([], dtype=np.int64))
            continue
        active = labels[member_ids].max(axis=0).astype(bool)
        if active.size <= 1:
            boundaries_by_track.append(np.array([], dtype=np.int64))
            continue
        boundaries_by_track.append(np.flatnonzero(active[1:] != active[:-1]) + 1)
    for idx, span in enumerate(spans):
        boundaries = boundaries_by_track[span.track_index]
        if boundaries.size and ((boundaries >= span.start) & (boundaries < span.end)).any():
            weights[idx] += boost
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


def positive_weights_from_tracks(
    tracks,
    num_classes: int,
    scheme: str,
    cap: float = 20.0,
    effective_beta: float = 0.999,
) -> torch.Tensor:
    if scheme == "none":
        return torch.ones(num_classes, dtype=torch.float32)
    positives = np.ones(num_classes, dtype=np.float64)
    total = 0.0
    for track in tracks:
        labels = track.labels
        if labels is None:
            continue
        if labels.ndim == 2:
            positives += labels.sum(axis=1)
            total += labels.shape[1]
        else:
            ids, counts = np.unique(labels, return_counts=True)
            for class_id, count in zip(ids, counts):
                positives[int(class_id)] += count
            total += len(labels)
    negatives = np.clip(total - positives, 1.0, None)
    if scheme == "effective_number":
        beta = float(np.clip(effective_beta, 0.0, 0.999999))
        weights = (1.0 - beta) / np.maximum(
            1.0 - np.power(beta, positives), 1e-12
        )
        weights = weights / np.maximum(weights.mean(), 1e-12)
    else:
        weights = negatives / positives
    if scheme == "inverse_sqrt_frequency":
        weights = np.sqrt(weights)
    weights = np.clip(weights, 1.0, cap)
    return torch.tensor(weights, dtype=torch.float32)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    gamma: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=1)
    prob = log_prob.exp()
    ce = F.nll_loss(log_prob, targets, weight=weight, ignore_index=ignore_index, reduction="none")
    valid = targets != ignore_index
    safe_targets = targets.clamp_min(0)
    pt = prob.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    focal = (1.0 - pt) ** gamma * ce
    focal = focal[valid]
    return focal.mean() if focal.numel() > 0 else logits.sum() * 0.0


def temporal_smoothing_loss(
    logits: torch.Tensor, mask: torch.Tensor, max_value: float = 4.0
) -> torch.Tensor:
    """Truncated MSE on adjacent log-probabilities to discourage flicker."""

    log_prob = F.log_softmax(logits, dim=1)
    diff = (log_prob[:, :, 1:] - log_prob[:, :, :-1]) ** 2
    diff = torch.clamp(diff, max=max_value)
    pair_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(1)
    masked = diff * pair_mask
    denom = pair_mask.sum().clamp_min(1)
    return masked.sum() / denom


def temporal_smoothing_loss_multilabel(
    logits: torch.Tensor,
    mask: torch.Tensor,
    max_value: float = 4.0,
    background_id: int = 0,
) -> torch.Tensor:
    """Per-class truncated-MSE smoothing for multi-label (independent sigmoids).

    The MS-TCN smoothing loss extended to multi-label: penalize per-class jumps in
    log-sigmoid between adjacent frames, which suppresses flicker / over-segmentation
    while leaving real bout boundaries (large, truncated) unpenalized beyond the cap.
    """

    log_p = F.logsigmoid(logits)                       # [B, K, T]
    diff = (log_p[:, :, 1:] - log_p[:, :, :-1]) ** 2
    diff = torch.clamp(diff, max=max_value)
    if 0 <= background_id < diff.shape[1]:
        keep_classes = torch.ones(diff.shape[1], dtype=torch.bool, device=diff.device)
        keep_classes[background_id] = False
        diff = diff[:, keep_classes, :]
    if diff.shape[1] == 0:
        return logits.sum() * 0.0
    pair_mask = (mask[:, 1:] & mask[:, :-1]).unsqueeze(1).float()
    denom = pair_mask.sum().clamp_min(1.0) * diff.shape[1]
    return (diff * pair_mask).sum() / denom


def feature_stat_augment(
    features: torch.Tensor,
    mask: torch.Tensor,
    scale_jitter: float,
    offset_jitter: float,
) -> torch.Tensor:
    """Apply per-sample feature scale/offset jitter to normalized windows."""

    if scale_jitter <= 0 and offset_jitter <= 0:
        return features
    scale = 1.0
    if scale_jitter > 0:
        scale = 1.0 + scale_jitter * torch.randn(
            features.shape[0],
            features.shape[1],
            1,
            device=features.device,
            dtype=features.dtype,
        )
    offset = 0.0
    if offset_jitter > 0:
        offset = offset_jitter * torch.randn(
            features.shape[0],
            features.shape[1],
            1,
            device=features.device,
            dtype=features.dtype,
        )
    out = features * scale + offset
    return torch.where(mask.unsqueeze(1), out, features)


def dominant_targets(labels: torch.Tensor, multilabel: bool, background_id: int = 0) -> torch.Tensor:
    """Return one class id per frame, using strongest active label for multilabel."""

    if not multilabel:
        return labels.long()
    active = labels.clone()
    if 0 <= background_id < active.shape[1]:
        active[:, background_id, :] = 0.0
    has_active = active.sum(dim=1) > 0
    targets = active.argmax(dim=1).long()
    targets[~has_active] = background_id
    return targets


def bout_transition_regularization(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    multilabel: bool,
    background_id: int = 0,
) -> torch.Tensor:
    """Discourage probability jumps where manual labels do not change."""

    if logits.shape[-1] < 2:
        return logits.sum() * 0.0
    if multilabel:
        probs = torch.sigmoid(logits)
        target_change = (labels[:, :, 1:] != labels[:, :, :-1]).any(dim=1)
    else:
        probs = torch.softmax(logits, dim=1)
        target = dominant_targets(labels, False, background_id)
        target_change = target[:, 1:] != target[:, :-1]
    stable = (~target_change) & mask[:, 1:] & mask[:, :-1]
    if not stable.any():
        return logits.sum() * 0.0
    jumps = (probs[:, :, 1:] - probs[:, :, :-1]).abs().sum(dim=1)
    return jumps[stable].mean()


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    multilabel: bool,
    background_id: int = 0,
    max_tokens: int = 256,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Label-aware contrastive loss on frame embeddings in a batch."""

    z = embeddings.transpose(1, 2)[mask]  # [N, E]
    if z.shape[0] < 4:
        return embeddings.sum() * 0.0
    targets = dominant_targets(labels, multilabel, background_id)[mask]
    keep = targets != background_id
    z = z[keep]
    targets = targets[keep]
    if z.shape[0] < 4 or torch.unique(targets).numel() < 2:
        return embeddings.sum() * 0.0
    if z.shape[0] > max_tokens:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_tokens]
        z = z[idx]
        targets = targets[idx]
    z = F.normalize(z, dim=1)
    logits = z @ z.T / max(temperature, 1e-6)
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    positives = (targets[:, None] == targets[None, :]) & ~eye
    has_pos = positives.any(dim=1)
    if not has_pos.any():
        return embeddings.sum() * 0.0
    logits = logits.masked_fill(eye, -1e9)
    log_prob = F.log_softmax(logits[has_pos], dim=1)
    target = positives[has_pos].to(dtype=log_prob.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(target * log_prob).sum(dim=1).mean()


def input_feature_group_lasso(model: nn.Module) -> torch.Tensor:
    """Group-lasso penalty over input feature channels for feature selection."""

    conv = getattr(model, "input_proj", None)
    if conv is None and hasattr(model, "embtcn"):
        conv = getattr(model.embtcn, "input_proj", None)
    if conv is None and hasattr(model, "stages") and len(model.stages) > 0:
        conv = getattr(model.stages[0], "input_proj", None)
    if conv is None or not hasattr(conv, "weight"):
        return next(model.parameters()).sum() * 0.0
    weight = conv.weight
    if weight.ndim == 3:
        weight = weight.squeeze(-1)
    if weight.ndim != 2:
        return conv.weight.sum() * 0.0
    per_feature = torch.sqrt((weight ** 2).sum(dim=0) + 1e-8)
    return per_feature.mean()


def soften_multilabel_boundaries(
    labels: torch.Tensor,
    radius: int,
    soft_value: float,
    background_id: int = 0,
) -> torch.Tensor:
    """Give foreground behaviors a soft halo so tiny boundary shifts cost less."""

    if radius <= 0 or soft_value <= 0 or labels.shape[-1] == 0:
        return labels
    kernel = radius * 2 + 1
    target = labels.clone()
    classes = [
        class_id
        for class_id in range(labels.shape[1])
        if class_id != background_id
    ]
    if not classes:
        return target
    fg = labels[:, classes, :].float()
    halo = F.max_pool1d(fg, kernel_size=kernel, stride=1, padding=radius)
    softened = torch.maximum(fg, halo * float(soft_value))
    target[:, classes, :] = softened
    return target


def overlap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    class_weight: torch.Tensor,
    mode: str,
    alpha: float,
    beta: float,
    background_id: int = 0,
) -> torch.Tensor:
    """Differentiable per-class F1/Dice/Tversky loss for multilabel segments."""

    probs = torch.sigmoid(logits)
    valid = mask.unsqueeze(1).float()
    keep = torch.ones(targets.shape[1], dtype=torch.bool, device=targets.device)
    if 0 <= background_id < targets.shape[1]:
        keep[background_id] = False
    if not keep.any():
        return logits.sum() * 0.0
    probs = probs[:, keep, :]
    targets = targets[:, keep, :]
    valid = valid.expand_as(targets)
    weights = class_weight.to(device=targets.device, dtype=targets.dtype)[keep]
    weights = weights / weights.mean().clamp_min(1e-6)
    tp = (probs * targets * valid).sum(dim=(0, 2))
    fp = (probs * (1.0 - targets) * valid).sum(dim=(0, 2))
    fn = ((1.0 - probs) * targets * valid).sum(dim=(0, 2))
    eps = 1e-6
    if mode in {"soft_f1", "dice"}:
        score = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    elif mode == "tversky":
        score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    else:
        return logits.sum() * 0.0
    present = (targets * valid).sum(dim=(0, 2)) > 0
    if present.any():
        score = score[present]
        weights = weights[present]
    return ((1.0 - score) * weights).sum() / weights.sum().clamp_min(1e-6)


def reduce_multilabel_frame_loss(
    loss: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    ohem_fraction: float,
    background_id: int = 0,
) -> torch.Tensor:
    """Reduce per-class multilabel losses, optionally keeping hard valid frames."""

    if loss.ndim == 3 and 0 <= background_id < loss.shape[1]:
        keep_classes = torch.ones(loss.shape[1], dtype=torch.bool, device=loss.device)
        keep_classes[background_id] = False
        loss = loss[:, keep_classes, :]
        targets = targets[:, keep_classes, :]
    frame_loss = (loss * mask.unsqueeze(1)).sum(dim=1)
    valid = mask
    if not valid.any():
        return loss.sum() * 0.0
    if ohem_fraction <= 0:
        return frame_loss[valid].mean()
    foreground = targets.clone()
    if 0 <= background_id < foreground.shape[1]:
        foreground[:, background_id, :] = 0
    positive = (foreground > 0.5).any(dim=1) & valid
    selected = positive.clone()
    remaining = valid & ~selected
    if remaining.any():
        valid_count = int(valid.sum().item())
        budget = max(int(round(valid_count * float(ohem_fraction))), 1)
        budget = max(budget - int(selected.sum().item()), 0)
        budget = min(budget, int(remaining.sum().item()))
        if budget > 0:
            hard_values = frame_loss[remaining]
            hard_indices = torch.topk(hard_values, k=budget, largest=True).indices
            remaining_indices = remaining.nonzero(as_tuple=False)
            picked = remaining_indices[hard_indices]
            selected[picked[:, 0], picked[:, 1]] = True
    if not selected.any():
        selected = valid
    return frame_loss[selected].mean()


def compute_loss(
    output,
    labels: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    config: AppConfig,
    multilabel: bool,
    background_id: int = 0,
    label_map: LabelMap | None = None,
) -> torch.Tensor:
    train_cfg = config.training
    total = output.logits.sum() * 0.0
    for stage_logits in output.stage_logits:
        if multilabel:
            target = soften_multilabel_boundaries(
                labels,
                train_cfg.boundary_tolerance_frames,
                train_cfg.boundary_soft_target,
                background_id,
            )
            stage_loss = stage_logits.sum() * 0.0
            use_bce = train_cfg.loss in {"bce", "focal", "cross_entropy"}
            if use_bce:
                bce = F.binary_cross_entropy_with_logits(
                    stage_logits,
                    target,
                    reduction="none",
                    pos_weight=weight.view(1, -1, 1),
                )
                if train_cfg.loss == "focal":
                    prob = torch.sigmoid(stage_logits)
                    pt = torch.where(target > 0.5, prob, 1.0 - prob)
                    bce = ((1.0 - pt).clamp_min(1e-6) ** train_cfg.focal_gamma) * bce
                stage_loss = stage_loss + reduce_multilabel_frame_loss(
                    bce,
                    target,
                    mask,
                    train_cfg.ohem_fraction,
                    background_id,
                )
            overlap_mode = (
                train_cfg.loss
                if train_cfg.loss in {"soft_f1", "dice", "tversky"}
                else "tversky"
            )
            overlap_weight = (
                1.0
                if train_cfg.loss in {"soft_f1", "dice", "tversky"}
                else train_cfg.overlap_loss_weight
            )
            if overlap_weight > 0:
                stage_loss = stage_loss + overlap_weight * overlap_loss(
                    stage_logits,
                    target,
                    mask,
                    weight,
                    overlap_mode,
                    train_cfg.tversky_alpha,
                    train_cfg.tversky_beta,
                    background_id,
                )
        elif train_cfg.loss == "focal":
            masked_labels = labels.clone()
            masked_labels[~mask] = -100
            stage_loss = focal_loss(
                stage_logits, masked_labels, weight, train_cfg.focal_gamma
            )
        else:
            masked_labels = labels.clone()
            masked_labels[~mask] = -100
            stage_loss = F.cross_entropy(
                stage_logits, masked_labels, weight=weight, ignore_index=-100
            )
        total = total + stage_loss
        if (
            multilabel
            and label_map is not None
            and train_cfg.attack_parent_consistency_weight > 0
        ):
            total = total + train_cfg.attack_parent_consistency_weight * (
                attack_parent_consistency_loss(stage_logits, mask, label_map, config)
            )
        if train_cfg.temporal_smoothing_weight > 0 and not multilabel:
            total = total + train_cfg.temporal_smoothing_weight * temporal_smoothing_loss(
                stage_logits, mask
            )
        if multilabel and getattr(train_cfg, "multilabel_temporal_smoothing_weight", 0.0) > 0:
            total = total + train_cfg.multilabel_temporal_smoothing_weight * (
                temporal_smoothing_loss_multilabel(
                    stage_logits, mask, background_id=background_id
                )
            )
    return total / max(len(output.stage_logits), 1)


def attack_parent_consistency_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    label_map: LabelMap,
    config: AppConfig,
) -> torch.Tensor:
    name_to_id = label_map.name_to_id
    parent = name_to_id.get(config.training.grouped_attack_name)
    children = [
        name_to_id[name]
        for name in config.training.grouped_attack_members
        if name in name_to_id and name_to_id[name] != parent
    ]
    if parent is None or not children:
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits)
    parent_prob = probs[:, parent : parent + 1]
    child_prob = probs[:, children]
    violation = F.relu(child_prob - parent_prob)
    valid = mask.unsqueeze(1).float()
    return (violation * valid).sum() / valid.sum().clamp_min(1.0) / len(children)


def tune_multilabel_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    background_id: int,
    metric: str = "frame_f1",
    bout_iou_threshold: float = 0.25,
    boundary_tolerance_frames: int = 0,
) -> np.ndarray:
    thresholds = np.full(probabilities.shape[0], 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 19)
    for class_id in range(probabilities.shape[0]):
        if class_id == background_id:
            continue
        y = targets[class_id].astype(bool)
        if int(y.sum()) < 5:
            continue
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in grid:
            pred = probabilities[class_id] >= threshold
            if metric == "bout_f1":
                f1 = binary_bout_f1(
                    y,
                    pred,
                    iou_threshold=bout_iou_threshold,
                    tolerance=boundary_tolerance_frames,
                )
            else:
                tp = float((pred & y).sum())
                fp = float((pred & ~y).sum())
                fn = float((~pred & y).sum())
                denom = 2.0 * tp + fp + fn
                f1 = 2.0 * tp / denom if denom else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds[class_id] = best_threshold
    return thresholds


def _binary_bouts(row: np.ndarray) -> list[tuple[int, int]]:
    row = np.asarray(row).astype(bool).reshape(-1)
    bouts: list[tuple[int, int]] = []
    if row.size == 0:
        return bouts
    start: int | None = None
    for idx, active in enumerate(row):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            bouts.append((start, idx - 1))
            start = None
    if start is not None:
        bouts.append((start, row.size - 1))
    return bouts


def _interval_iou(
    pred: tuple[int, int],
    truth: tuple[int, int],
    tolerance: int = 0,
) -> float:
    ps, pe = pred
    ts, te = truth
    if tolerance > 0:
        ps, pe = ps - tolerance, pe + tolerance
        ts, te = ts - tolerance, te + tolerance
    inter = max(0, min(pe, te) - max(ps, ts) + 1)
    union = (pe - ps + 1) + (te - ts + 1) - inter
    return inter / union if union > 0 else 0.0


def binary_bout_f1(
    truth: np.ndarray,
    prediction: np.ndarray,
    iou_threshold: float = 0.25,
    tolerance: int = 0,
) -> float:
    true_bouts = _binary_bouts(truth)
    pred_bouts = _binary_bouts(prediction)
    if not true_bouts and not pred_bouts:
        return 1.0
    matched: set[int] = set()
    tp = 0
    for pred in pred_bouts:
        best_iou = 0.0
        best_idx = -1
        for idx, truth_bout in enumerate(true_bouts):
            if idx in matched:
                continue
            iou = _interval_iou(pred, truth_bout, tolerance)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched.add(best_idx)
            tp += 1
    fp = len(pred_bouts) - tp
    fn = len(true_bouts) - tp
    return 2.0 * tp / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def collect_multilabel_probabilities(
    model,
    tracks,
    normalizer: FeatureNormalizer,
    label_map: LabelMap,
    config: AppConfig,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for track in tracks:
        if track.labels is None or track.labels.ndim != 2:
            continue
        logits, _ = sliding_window_inference(
            model,
            track,
            normalizer,
            config.inference.window_seconds,
            config.inference.stride_seconds,
            config.data.frame_rate,
            device,
        )
        p = logits_to_probabilities(logits, multilabel=True)
        p = constrain_probabilities_for_subject(
            p,
            label_map,
            config.behavior_roles,
            track.subject_id,
            multilabel=True,
        )
        n = min(p.shape[1], track.labels.shape[1])
        probs.append(p[:, :n])
        targets.append(track.labels[:, :n].astype(np.int8))
    if not probs:
        return np.empty((label_map.num_classes, 0)), np.empty((label_map.num_classes, 0))
    return np.concatenate(probs, axis=1), np.concatenate(targets, axis=1)


def collect_multilabel_probability_tracks(
    model,
    tracks,
    normalizer: FeatureNormalizer,
    label_map: LabelMap,
    config: AppConfig,
    device: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect per-track probabilities and labels for temporal postprocess tuning."""

    collected: list[tuple[np.ndarray, np.ndarray]] = []
    for track in tracks:
        if track.labels is None or track.labels.ndim != 2:
            continue
        logits, _ = sliding_window_inference(
            model,
            track,
            normalizer,
            config.inference.window_seconds,
            config.inference.stride_seconds,
            config.data.frame_rate,
            device,
        )
        probabilities = logits_to_probabilities(logits, multilabel=True)
        probabilities = constrain_probabilities_for_subject(
            probabilities,
            label_map,
            config.behavior_roles,
            track.subject_id,
            multilabel=True,
        )
        n = min(probabilities.shape[1], track.labels.shape[1])
        collected.append(
            (
                probabilities[:, :n],
                track.labels[:, :n].astype(np.int8),
            )
        )
    return collected


def _concat_multilabel_tracks(
    probability_tracks: list[tuple[np.ndarray, np.ndarray]],
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not probability_tracks:
        return np.empty((num_classes, 0)), np.empty((num_classes, 0))
    return (
        np.concatenate([p for p, _ in probability_tracks], axis=1),
        np.concatenate([y for _, y in probability_tracks], axis=1),
    )


def multilabel_class_priors(tracks, num_classes: int) -> np.ndarray:
    """Positive frame rate per class over labeled training tracks."""

    positives = np.zeros(num_classes, dtype=np.float64)
    total = 0
    for track in tracks:
        if track.labels is None or track.labels.ndim != 2:
            continue
        labels = track.labels.astype(bool)
        n = labels.shape[1]
        positives[: labels.shape[0]] += labels.sum(axis=1)
        total += n
    if total <= 0:
        return np.zeros(num_classes, dtype=np.float32)
    return (positives / float(total)).astype(np.float32)


def _postprocess_settings_from_config(config: AppConfig) -> dict[str, float | int]:
    return {
        "probability_smoothing_seconds": float(
            config.inference.probability_smoothing_seconds
        ),
        "min_bout_frames": int(config.inference.min_bout_frames),
        "merge_gap_frames": int(config.inference.merge_gap_frames),
        "transition_penalty": float(config.inference.transition_penalty),
    }


def apply_multilabel_postprocess_tracks(
    probability_tracks: list[tuple[np.ndarray, np.ndarray]],
    label_map: LabelMap,
    frame_rate: float,
    thresholds: np.ndarray,
    settings: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for probabilities, truth in probability_tracks:
        labels, _ = postprocess_predictions(
            probabilities,
            label_map,
            frame_rate,
            float(settings["probability_smoothing_seconds"]),
            int(settings["min_bout_frames"]),
            int(settings["merge_gap_frames"]),
            thresholds,
            multilabel=True,
            transition_penalty=float(settings["transition_penalty"]),
        )
        n = min(labels.shape[1], truth.shape[1])
        predictions.append(labels[:, :n].astype(np.int8))
        targets.append(truth[:, :n].astype(np.int8))
    if not predictions:
        return (
            np.empty((label_map.num_classes, 0), dtype=np.int8),
            np.empty((label_map.num_classes, 0), dtype=np.int8),
        )
    return np.concatenate(predictions, axis=1), np.concatenate(targets, axis=1)


def _multilabel_postprocess_score(
    targets: np.ndarray,
    predictions: np.ndarray,
    label_map: LabelMap,
    metric: str,
    iou_threshold: float,
    tolerance: int,
) -> tuple[float, dict[str, float]]:
    frame_summary = multilabel_summary(
        multilabel_frame_report(targets, predictions, label_map)
    )
    bout_summary = multilabel_bout_summary(
        multilabel_bout_report(
            targets,
            predictions,
            label_map,
            iou_threshold,
            tolerance,
        )
    )
    summary = {**frame_summary, **bout_summary}
    return float(summary.get(metric, summary["macro_f1"])), summary


def tune_multilabel_postprocessing(
    probability_tracks: list[tuple[np.ndarray, np.ndarray]],
    label_map: LabelMap,
    config: AppConfig,
) -> tuple[np.ndarray, dict[str, float | int], float, dict[str, float]]:
    """Tune thresholds plus temporal cleanup on validation tracks."""

    best_thresholds = np.full(
        label_map.num_classes,
        config.inference.confidence_threshold,
        dtype=np.float32,
    )
    best_settings = _postprocess_settings_from_config(config)
    best_score = -1.0
    best_summary: dict[str, float] = {}
    seen_smoothing: dict[float, np.ndarray] = {}

    smoothing_grid = sorted(
        {float(v) for v in config.training.postprocess_smoothing_grid}
        | {float(config.inference.probability_smoothing_seconds)}
    )
    min_bout_grid = sorted(
        {max(1, int(v)) for v in config.training.postprocess_min_bout_grid}
        | {int(config.inference.min_bout_frames)}
    )
    merge_gap_grid = sorted(
        {max(0, int(v)) for v in config.training.postprocess_merge_gap_grid}
        | {int(config.inference.merge_gap_frames)}
    )
    transition_grid = sorted(
        {max(0.0, float(v)) for v in config.training.postprocess_transition_grid}
        | {float(config.inference.transition_penalty)}
    )

    for smoothing in smoothing_grid:
        if smoothing not in seen_smoothing:
            smoothed_tracks = [
                (
                    smooth_probabilities(
                        probabilities,
                        smoothing,
                        config.data.frame_rate,
                        normalize=False,
                    ),
                    targets,
                )
                for probabilities, targets in probability_tracks
            ]
            smoothed_probs, smoothed_targets = _concat_multilabel_tracks(
                smoothed_tracks,
                label_map.num_classes,
            )
            seen_smoothing[smoothing] = tune_multilabel_thresholds(
                smoothed_probs,
                smoothed_targets,
                label_map.background_id,
                config.training.threshold_tuning_metric,
                config.training.bout_tuning_iou_threshold,
                config.training.boundary_tolerance_frames,
            )
        thresholds = seen_smoothing[smoothing]
        for min_bout in min_bout_grid:
            for merge_gap in merge_gap_grid:
                for transition in transition_grid:
                    settings = {
                        "probability_smoothing_seconds": smoothing,
                        "min_bout_frames": min_bout,
                        "merge_gap_frames": merge_gap,
                        "transition_penalty": transition,
                    }
                    predictions, targets = apply_multilabel_postprocess_tracks(
                        probability_tracks,
                        label_map,
                        config.data.frame_rate,
                        thresholds,
                        settings,
                    )
                    score, summary = _multilabel_postprocess_score(
                        targets,
                        predictions,
                        label_map,
                        config.training.postprocess_tuning_metric,
                        config.training.bout_tuning_iou_threshold,
                        config.training.boundary_tolerance_frames,
                    )
                    if score > best_score:
                        best_score = score
                        best_settings = settings
                        best_thresholds = thresholds.copy()
                        best_summary = summary
    return best_thresholds, best_settings, best_score, best_summary


def multilabel_frame_report(
    targets: np.ndarray,
    predictions: np.ndarray,
    label_map: LabelMap,
) -> pd.DataFrame:
    rows = []
    for class_id in range(label_map.num_classes):
        y = targets[class_id].astype(bool)
        p = predictions[class_id].astype(bool)
        tp = float((p & y).sum())
        fp = float((p & ~y).sum())
        fn = float((~p & y).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "behavior": label_map.id_to_name[class_id],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(y.sum()),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "scored": class_id != label_map.background_id,
            }
        )
    return pd.DataFrame(rows)


def multilabel_summary(per_class: pd.DataFrame) -> dict[str, Any]:
    scored = (
        per_class["scored"].to_numpy(dtype=bool)
        if "scored" in per_class
        else np.ones(len(per_class), dtype=bool)
    )
    scored_rows = per_class.loc[scored]
    support = scored_rows["support"].to_numpy(dtype=float)
    f1 = scored_rows["f1"].to_numpy(dtype=float)
    active = support > 0
    weighted = (
        float(np.average(f1[active], weights=support[active]))
        if active.any()
        else 0.0
    )
    macro = float(f1[active].mean()) if active.any() else 0.0
    if {"true_positive", "false_positive", "false_negative"}.issubset(
        scored_rows.columns
    ):
        true_positive = scored_rows["true_positive"].to_numpy(dtype=float)
        false_positive = scored_rows["false_positive"].to_numpy(dtype=float)
        false_negative = scored_rows["false_negative"].to_numpy(dtype=float)
        denom = (
            2.0 * true_positive.sum()
            + false_positive.sum()
            + false_negative.sum()
        )
        micro = float(2.0 * true_positive.sum() / denom) if denom > 0 else 0.0
    else:
        micro = 0.0
    return {"macro_f1": macro, "weighted_f1": weighted, "micro_f1": micro}


def multilabel_bout_report(
    targets: np.ndarray,
    predictions: np.ndarray,
    label_map: LabelMap,
    iou_threshold: float = 0.25,
    tolerance: int = 0,
) -> pd.DataFrame:
    rows = []
    for class_id in range(label_map.num_classes):
        if class_id == label_map.background_id:
            continue
        y = targets[class_id].astype(bool)
        p = predictions[class_id].astype(bool)
        score = binary_bout_f1(y, p, iou_threshold, tolerance)
        rows.append(
            {
                "behavior": label_map.id_to_name[class_id],
                "bout_f1": score,
                "support": int(y.sum()),
                "true_bouts": len(_binary_bouts(y)),
                "pred_bouts": len(_binary_bouts(p)),
                "iou_threshold": float(iou_threshold),
                "boundary_tolerance_frames": int(tolerance),
            }
        )
    return pd.DataFrame(rows)


def multilabel_bout_summary(report: pd.DataFrame) -> dict[str, float]:
    if report.empty:
        return {"bout_macro_f1": 0.0, "bout_weighted_f1": 0.0}
    support = report["support"].to_numpy(dtype=float)
    f1 = report["bout_f1"].to_numpy(dtype=float)
    active = support > 0
    weighted = float(np.average(f1[active], weights=support[active])) if active.any() else 0.0
    macro = float(f1[active].mean()) if active.any() else 0.0
    return {"bout_macro_f1": macro, "bout_weighted_f1": weighted}


def validation_score_from_summary(summary: dict[str, float], metric: str) -> float:
    """Pick the validation objective, falling back to macro-F1 when unavailable."""

    if metric in summary:
        return float(summary[metric])
    if metric.startswith("bout_") and "weighted_f1" in summary:
        return float(summary["weighted_f1"])
    return float(summary.get("macro_f1", 0.0))


@torch.no_grad()
def evaluate_tracks(
    model: TemporalTcnModel,
    tracks,
    normalizer: FeatureNormalizer,
    label_map: LabelMap,
    config: AppConfig,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run sliding-window inference over tracks and stack predictions vs truth."""

    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for track in tracks:
        if track.labels is None:
            continue
        logits, _ = sliding_window_inference(
            model,
            track,
            normalizer,
            config.inference.window_seconds,
            config.inference.stride_seconds,
            config.data.frame_rate,
            device,
        )
        multilabel = track.labels.ndim == 2
        probabilities = logits_to_probabilities(logits, multilabel)
        probabilities = constrain_probabilities_for_subject(
            probabilities,
            label_map,
            config.behavior_roles,
            track.subject_id,
            multilabel,
        )
        if multilabel:
            thresholds = np.asarray(
                model_thresholds(model, config, label_map.num_classes),
                dtype=np.float32,
            )
            pred = (probabilities >= thresholds[:, None]).astype(np.int8)
            truth = track.labels.astype(np.int8)
            n = min(pred.shape[1], truth.shape[1])
            y_pred.append(pred[:, :n].T)
            y_true.append(truth[:, :n].T)
        else:
            pred = probabilities.argmax(axis=0)
            truth = track.labels
            n = min(len(pred), len(truth))
            y_pred.append(pred[:n])
            y_true.append(truth[:n])
    if not y_true:
        return np.array([]), np.array([])
    return np.concatenate(y_true), np.concatenate(y_pred)


def model_thresholds(model, config: AppConfig, num_classes: int) -> np.ndarray:
    thresholds = getattr(model, "decision_thresholds", None)
    if thresholds is None:
        return np.full(num_classes, config.inference.confidence_threshold, dtype=np.float32)
    if isinstance(thresholds, torch.Tensor):
        thresholds = thresholds.detach().cpu().numpy()
    return np.asarray(thresholds, dtype=np.float32)


def run_training(
    config: AppConfig,
    progress: ProgressFn | None = None,
    log: Callable[[str], None] | None = None,
    should_stop: StopFn | None = None,
) -> TrainingResult:
    """Train the temporal model end to end. Safe to call from a worker thread."""

    def emit(fraction: float, message: str) -> None:
        if log:
            log(message)
        if progress:
            progress(fraction, message)

    set_seed(config.project.seed)
    output_dir = Path(config.project.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "features"
    device = resolve_device(config.training.device)

    emit(0.02, f"Using device: {device}")
    identity_df, pair_df, frame_indices = extract_features_for_videos(
        config, cache_dir, log
    )
    if identity_df.empty:
        raise RuntimeError("No features extracted; check COCO paths.")

    emit(0.25, "Aligning manual labels to frames ...")
    aligned = align_all_labels(config, frame_indices)
    label_map = add_grouped_attack_targets(aligned, config)
    aligned.coverage().pipe(write_table, output_dir / "label_coverage.csv")

    tracks, feature_names = build_tracks(
        identity_df,
        pair_df,
        aligned.tracks,
        label_map=label_map,
        behavior_roles=config.behavior_roles,
    )
    if not tracks:
        raise RuntimeError(
            "No labeled tracks could be aligned to features. Check identities."
        )
    if config.model.architecture == "embtcn_mask_video":
        attach_mask_clips_to_tracks(
            tracks,
            config.data.coco_jsons,
            config.data,
            size=config.mask_video.clip_size,
            log=log,
        )
    is_pair = any(t.object_id for t in tracks)
    task_type = "pair" if is_pair else "single"

    splits = split_tracks_by_group(
        tracks,
        config.training.train_fraction,
        config.training.validation_fraction,
        config.project.seed,
    )
    emit(
        0.3,
        f"Tracks: {len(tracks)}  features: {len(feature_names)}  "
        f"train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}",
    )

    if config.training.coral_normalization:
        normalizer = FeatureNormalizer.fit_with_video_coral(
            splits["train"], feature_names
        )
    elif config.training.per_video_robust_normalization:
        normalizer = FeatureNormalizer.fit_with_video_robust(
            splits["train"], feature_names
        )
    else:
        normalizer = FeatureNormalizer.fit(
            [t.features for t in splits["train"]], feature_names
        )
    multilabel = config.labels.mode == "multilabel"

    train_ds = WindowDataset(
        splits["train"],
        config.training.window_seconds,
        config.training.stride_seconds,
        config.data.frame_rate,
        normalizer,
        multilabel,
    )
    val_ds = WindowDataset(
        splits["val"],
        config.training.window_seconds,
        config.training.window_seconds,
        config.data.frame_rate,
        normalizer,
        multilabel,
    )

    if getattr(config.training, "support_balanced_sampler", False):
        weights = support_balanced_window_weights(
            train_ds.spans,
            splits["train"],
            label_map.names,
            label_map.background_id,
            cap_fraction=getattr(config.training, "support_cap_fraction", 0.40),
        )
    else:
        weights = class_balanced_window_weights(
            train_ds.spans,
            [t.labels for t in splits["train"] if t.labels is not None],
            label_map.num_classes,
            label_map.background_id,
        )
    boundary_weights = attack_boundary_window_weights(
        train_ds.spans,
        train_ds.labels_per_track(),
        label_map,
        config,
    )
    weights = weights * boundary_weights
    if weights.sum() > 0:
        weights = weights / weights.sum()
    sampler = (
        WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        if len(weights) > 0
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collate_windows,
        num_workers=0,
    )

    if config.model.architecture in {"embtcn_attention", "embtcn_mask_video"}:
        emb_cfg = EmbTCNConfig(
            num_features=len(feature_names),
            num_classes=label_map.num_classes,
            d_model=config.embtcn.d_model,
            embedding_dim=config.embtcn.embedding_dim,
            tcn_dilations=tuple(config.embtcn.tcn_dilations),
            kernel_size=config.embtcn.kernel_size,
            num_encoder_layers=config.embtcn.num_encoder_layers,
            num_heads=config.embtcn.num_heads,
            ffn_mult=config.embtcn.ffn_mult,
            dropout=config.embtcn.dropout,
            temperature=config.embtcn.temperature,
            causal=config.embtcn.causal,
            max_len=config.embtcn.max_len,
            num_refinement_stages=config.embtcn.num_refinement_stages,
            refinement_hidden=config.embtcn.refinement_hidden,
            refinement_dilations=tuple(config.embtcn.refinement_dilations),
            use_supervised_head=True,
            use_fault_head=True,
            use_decoder=False,
            multilabel=multilabel,
        )
        if config.model.architecture == "embtcn_mask_video":
            mask_cfg = TorchMaskVideoConfig(
                in_channels=config.mask_video.in_channels,
                spatial_channels=tuple(config.mask_video.spatial_channels),
                temporal_channels=config.mask_video.temporal_channels,
                embedding_dim=config.mask_video.embedding_dim,
                temporal_dilations=tuple(config.mask_video.temporal_dilations),
                kernel_size=config.mask_video.kernel_size,
                dropout=config.mask_video.dropout,
                clip_size=config.mask_video.clip_size,
            )
            model = EmbTCNMaskVideoFusion(emb_cfg, mask_cfg).to(device)
        else:
            model = EmbTCNAttention(emb_cfg).to(device)
        pretrained = config.embtcn.pretrained_checkpoint
        if pretrained:
            payload = torch.load(pretrained, map_location=device, weights_only=False)
            state = payload.get("model_state", payload)
            current = model.state_dict()
            compatible = {
                key: value
                for key, value in state.items()
                if key in current and tuple(current[key].shape) == tuple(value.shape)
            }
            model.load_state_dict(compatible, strict=False)
            emit(0.31, f"Loaded {len(compatible)} EmbTCN pretrained tensors.")
        if config.embtcn.freeze_encoder:
            for name, param in model.named_parameters():
                if not name.startswith("cls_head"):
                    param.requires_grad = False
    else:
        model = TemporalTcnModel(
            num_features=len(feature_names),
            num_classes=label_map.num_classes,
            hidden_channels=config.model.hidden_channels,
            num_stages=config.model.num_stages,
            num_layers_per_stage=config.model.num_layers_per_stage,
            kernel_size=config.model.kernel_size,
            dropout=config.model.dropout,
            embedding_dim=config.model.embedding_dim,
            multilabel=multilabel,
        ).to(device)

    video_to_domain = {
        video_id: idx for idx, video_id in enumerate(sorted({t.video_id for t in tracks}))
    }
    domain_head = None
    if config.training.domain_adversary_weight > 0 and len(video_to_domain) > 1:
        domain_head = DomainAdversary(model.embedding_dim, len(video_to_domain)).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if domain_head is not None:
        trainable_params += list(domain_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    if multilabel:
        class_weight = positive_weights_from_tracks(
            splits["train"],
            label_map.num_classes,
            config.training.class_weighting,
            effective_beta=config.training.effective_number_beta,
        ).to(device)
        class_priors = multilabel_class_priors(splits["train"], label_map.num_classes)
    else:
        class_weight = class_weights_from_tracks(
            splits["train"],
            label_map.num_classes,
            config.training.class_weighting,
            config.training.effective_number_beta,
        ).to(device)
        class_priors = None

    best_score = -1.0
    best_state: dict[str, Any] | None = None
    best_thresholds = np.full(
        label_map.num_classes, config.inference.confidence_threshold, dtype=np.float32
    )
    best_postprocess = _postprocess_settings_from_config(config)
    best_validation_summary: dict[str, float] = {}
    patience = 0
    history: list[dict[str, float]] = []
    early_stopping_enabled = config.training.early_stopping_patience > 0
    if early_stopping_enabled:
        emit(
            0.29,
            "Early stopping enabled "
            f"(patience={config.training.early_stopping_patience}); "
            "best validation checkpoint will be restored.",
        )
    else:
        emit(
            0.29,
            "Early stopping disabled; running all requested epochs and "
            "restoring the best validation checkpoint afterward.",
        )

    for epoch in range(config.training.max_epochs):
        if should_stop and should_stop():
            emit(1.0, "Training stopped by user.")
            break
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            mask_clip = (
                batch["mask_clip"].to(device)
                if "mask_clip" in batch and getattr(model, "requires_mask_clip", False)
                else None
            )
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            if config.training.feature_stat_augmentation:
                features = feature_stat_augment(
                    features,
                    mask,
                    config.training.feature_scale_jitter,
                    config.training.feature_offset_jitter,
                )
            output = _forward_model(model, features, mask_clip)
            loss = compute_loss(
                output,
                labels,
                mask,
                class_weight,
                config,
                multilabel,
                label_map.background_id,
                label_map,
            )
            if config.training.bout_transition_weight > 0:
                loss = loss + config.training.bout_transition_weight * bout_transition_regularization(
                    output.logits,
                    labels,
                    mask,
                    multilabel,
                    label_map.background_id,
                )
            if config.training.supervised_contrastive_weight > 0:
                loss = loss + config.training.supervised_contrastive_weight * supervised_contrastive_loss(
                    output.embeddings,
                    labels,
                    mask,
                    multilabel,
                    label_map.background_id,
                    temperature=config.embtcn.contrastive_temperature,
                )
            if config.training.input_feature_l1_weight > 0:
                loss = loss + config.training.input_feature_l1_weight * input_feature_group_lasso(
                    model
                )
            if domain_head is not None:
                domain_targets = torch.tensor(
                    [video_to_domain[str(meta["video_id"])] for meta in batch["meta"]],
                    dtype=torch.long,
                    device=device,
                )
                domain_logits = domain_head(
                    output.embeddings,
                    mask,
                    1.0,
                )
                loss = loss + config.training.domain_adversary_weight * F.cross_entropy(
                    domain_logits, domain_targets
                )
            if (
                config.model.architecture == "embtcn_attention"
                and config.embtcn.supervised_ssl_aux_weight > 0
                and config.embtcn.ssl_objective in {"contrastive", "recon+contrastive"}
            ):
                view1 = augmented_ssl_view(features, config)
                view2 = augmented_ssl_view(features, config)
                z1 = model(view1).embeddings
                z2 = model(view2).embeddings
                aux = hierarchical_temporal_contrastive_loss(
                    z1,
                    z2,
                    mask,
                    config.embtcn.contrastive_temperature,
                    config.embtcn.contrastive_radius_frames,
                )
                aux = aux + 0.5 * (
                    vicreg_regularizer(
                        z1,
                        mask,
                        config.embtcn.vicreg_variance_weight,
                        config.embtcn.vicreg_covariance_weight,
                    )
                    + vicreg_regularizer(
                        z2,
                        mask,
                        config.embtcn.vicreg_variance_weight,
                        config.embtcn.vicreg_covariance_weight,
                    )
                )
                loss = loss + config.embtcn.supervised_ssl_aux_weight * aux
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip_norm
            )
            if domain_head is not None:
                nn.utils.clip_grad_norm_(
                    domain_head.parameters(), config.training.gradient_clip_norm
                )
            optimizer.step()
            epoch_loss += float(loss.item())
            num_batches += 1
        mean_loss = epoch_loss / max(num_batches, 1)

        model.eval()
        if multilabel:
            val_probability_tracks = collect_multilabel_probability_tracks(
                model, splits["val"], normalizer, label_map, config, device
            )
            val_probs, val_targets = _concat_multilabel_tracks(
                val_probability_tracks, label_map.num_classes
            )
            val_summary: dict[str, float] = {}
            if val_probs.size > 0 and config.training.tune_postprocessing:
                thresholds, postprocess_settings, tuned_score, _summary = (
                    tune_multilabel_postprocessing(
                        val_probability_tracks,
                        label_map,
                        config,
                    )
                )
                val_summary = dict(_summary)
                val_f1 = validation_score_from_summary(
                    val_summary,
                    config.training.validation_metric,
                )
                if not np.isfinite(val_f1):
                    val_f1 = float(tuned_score)
            elif val_probs.size > 0:
                thresholds = tune_multilabel_thresholds(
                    val_probs,
                    val_targets,
                    label_map.background_id,
                    config.training.threshold_tuning_metric,
                    config.training.bout_tuning_iou_threshold,
                    config.training.boundary_tolerance_frames,
                )
                postprocess_settings = _postprocess_settings_from_config(config)
                val_pred = (val_probs >= thresholds[:, None]).astype(np.int8)
                frame_summary = multilabel_summary(
                    multilabel_frame_report(val_targets, val_pred, label_map)
                )
                bout_summary = multilabel_bout_summary(
                    multilabel_bout_report(
                        val_targets,
                        val_pred,
                        label_map,
                        config.training.bout_tuning_iou_threshold,
                        config.training.boundary_tolerance_frames,
                    )
                )
                val_summary = {**frame_summary, **bout_summary}
                val_f1 = validation_score_from_summary(
                    val_summary,
                    config.training.validation_metric,
                )
            else:
                thresholds = best_thresholds
                postprocess_settings = best_postprocess
                val_summary = dict(best_validation_summary)
                val_f1 = 0.0
        else:
            y_true, y_pred = evaluate_tracks(
                model, splits["val"], normalizer, label_map, config, device
            )
            if y_true.size > 0:
                from sklearn.metrics import f1_score

                val_macro = float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=list(range(label_map.num_classes)),
                        average="macro",
                        zero_division=0,
                    )
                )
                val_weighted = float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=list(range(label_map.num_classes)),
                        average="weighted",
                        zero_division=0,
                    )
                )
                val_summary = {
                    "macro_f1": val_macro,
                    "weighted_f1": val_weighted,
                }
                val_f1 = validation_score_from_summary(
                    val_summary,
                    config.training.validation_metric,
                )
            else:
                val_summary = {}
                val_f1 = 0.0
        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "val_score": val_f1,
                "val_metric": config.training.validation_metric,
                "val_macro_f1": float(val_summary.get("macro_f1", 0.0)),
                "val_weighted_f1": float(val_summary.get("weighted_f1", 0.0)),
                "val_bout_macro_f1": float(val_summary.get("bout_macro_f1", 0.0)),
                "val_bout_weighted_f1": float(
                    val_summary.get("bout_weighted_f1", 0.0)
                ),
            }
        )
        fraction = 0.3 + 0.6 * (epoch + 1) / config.training.max_epochs
        emit(
            min(fraction, 0.9),
            f"Epoch {epoch + 1}/{config.training.max_epochs}  "
            f"loss={mean_loss:.4f}  "
            f"val_{config.training.validation_metric}={val_f1:.4f}",
        )

        if val_f1 > best_score:
            best_score = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if multilabel:
                best_thresholds = thresholds.copy()
                best_postprocess = dict(postprocess_settings)
            best_validation_summary = dict(val_summary)
            patience = 0
        else:
            patience += 1
            if early_stopping_enabled and patience >= config.training.early_stopping_patience:
                emit(0.9, f"Early stopping at epoch {epoch + 1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if multilabel:
        model.decision_thresholds = best_thresholds

    emit(0.92, "Evaluating on the test split ...")
    test_tracks = splits["test"] or splits["val"]
    metrics_payload: dict[str, Any] = {
        "best_val_score": best_score,
        "best_val_metric": config.training.validation_metric,
        "best_val_macro_f1": float(best_validation_summary.get("macro_f1", best_score)),
        "best_val_weighted_f1": float(
            best_validation_summary.get("weighted_f1", best_score)
        ),
        "best_val_bout_macro_f1": float(
            best_validation_summary.get("bout_macro_f1", 0.0)
        ),
        "best_val_bout_weighted_f1": float(
            best_validation_summary.get("bout_weighted_f1", 0.0)
        ),
    }
    if multilabel:
        test_probability_tracks = collect_multilabel_probability_tracks(
            model, test_tracks, normalizer, label_map, config, device
        )
        test_probs, test_targets = _concat_multilabel_tracks(
            test_probability_tracks, label_map.num_classes
        )
        if test_probs.size > 0:
            if config.training.tune_postprocessing:
                test_pred, test_targets = apply_multilabel_postprocess_tracks(
                    test_probability_tracks,
                    label_map,
                    config.data.frame_rate,
                    best_thresholds,
                    best_postprocess,
                )
            else:
                test_pred = (test_probs >= best_thresholds[:, None]).astype(np.int8)
            per_class = multilabel_frame_report(test_targets, test_pred, label_map)
            summary = multilabel_summary(per_class)
            bout_report = multilabel_bout_report(
                test_targets,
                test_pred,
                label_map,
                config.training.bout_tuning_iou_threshold,
                config.training.boundary_tolerance_frames,
            )
            bout_summary = multilabel_bout_summary(bout_report)
            metrics_payload.update(
                {
                    "accuracy": float((test_pred == test_targets).all(axis=0).mean()),
                    "macro_f1": summary["macro_f1"],
                    "weighted_f1": summary["weighted_f1"],
                    "bout_macro_f1": bout_summary["bout_macro_f1"],
                    "bout_weighted_f1": bout_summary["bout_weighted_f1"],
                    "per_class": per_class.to_dict(orient="records"),
                    "bout_per_class": bout_report.to_dict(orient="records"),
                    "thresholds": best_thresholds.tolist(),
                    "postprocess": best_postprocess if config.training.tune_postprocessing else None,
                }
            )
            write_table(per_class, output_dir / "classification_report.csv")
            write_table(bout_report, output_dir / "multilabel_bout_report.csv")
            low_recall = per_class.query("recall < 0.5 and support > 0")
            write_table(low_recall, output_dir / "low_recall_classes.csv")
    else:
        y_true, y_pred = evaluate_tracks(
            model, test_tracks, normalizer, label_map, config, device
        )
        if y_true.size > 0:
            report = evaluate(y_true, y_pred, label_map)
            metrics_payload.update(report.summary_dict())
            write_table(report.framewise.per_class, output_dir / "classification_report.csv")
            write_table(report.bouts, output_dir / "bout_metrics.csv")
            cm = pd.DataFrame(
                report.framewise.confusion,
                index=report.framewise.class_names,
                columns=report.framewise.class_names,
            )
            cm.to_csv(output_dir / "confusion_matrix.csv")
            low_recall = report.framewise.per_class.query("recall < 0.5 and support > 0")
            write_table(low_recall, output_dir / "low_recall_classes.csv")

    model_config = {
        "architecture": config.model.architecture,
        "num_features": len(feature_names),
        "num_classes": label_map.num_classes,
        "hidden_channels": config.model.hidden_channels,
        "num_stages": config.model.num_stages,
        "num_layers_per_stage": config.model.num_layers_per_stage,
        "kernel_size": config.model.kernel_size,
        "dropout": config.model.dropout,
        "embedding_dim": config.model.embedding_dim,
        "multilabel": multilabel,
        "thresholds": best_thresholds.tolist() if multilabel else None,
        "postprocess": best_postprocess if multilabel and config.training.tune_postprocessing else None,
        "class_priors": class_priors.tolist() if class_priors is not None else None,
        "prior_calibration": (
            {
                "strength": float(config.inference.prior_calibration_strength),
                "mode": config.inference.prior_calibration_mode,
                "min_rate": float(config.inference.prior_calibration_min_rate),
            }
            if multilabel and config.inference.prior_calibration_strength > 0
            else None
        ),
    }
    if config.model.architecture in {"embtcn_attention", "embtcn_mask_video"}:
        model_config.update(model.cfg.to_dict())
        model_config.update(
            {
                "architecture": config.model.architecture,
                "num_features": len(feature_names),
                "num_classes": label_map.num_classes,
                "use_decoder": False,
                "use_fault_head": True,
                "use_supervised_head": True,
                "thresholds": best_thresholds.tolist() if multilabel else None,
                "postprocess": (
                    best_postprocess if multilabel and config.training.tune_postprocessing else None
                ),
                "class_priors": class_priors.tolist() if class_priors is not None else None,
                "prior_calibration": (
                    {
                        "strength": float(config.inference.prior_calibration_strength),
                        "mode": config.inference.prior_calibration_mode,
                        "min_rate": float(config.inference.prior_calibration_min_rate),
                    }
                    if multilabel and config.inference.prior_calibration_strength > 0
                    else None
                ),
            }
        )
        if config.model.architecture == "embtcn_mask_video":
            model_config["mask_video"] = model.mask_cfg.to_dict()
    checkpoint = Checkpoint(
        model_state={k: v.cpu() for k, v in model.state_dict().items()},
        model_config=model_config,
        feature_names=feature_names,
        label_map=label_map,
        normalizer=normalizer,
        task_type=task_type,
        frame_rate=config.data.frame_rate,
        training_metadata={
            "best_val_score": best_score,
            "best_val_metric": config.training.validation_metric,
            "best_val_macro_f1": float(
                best_validation_summary.get("macro_f1", best_score)
            ),
            "best_val_weighted_f1": float(
                best_validation_summary.get("weighted_f1", best_score)
            ),
            "best_val_bout_macro_f1": float(
                best_validation_summary.get("bout_macro_f1", 0.0)
            ),
            "best_val_bout_weighted_f1": float(
                best_validation_summary.get("bout_weighted_f1", 0.0)
            ),
            "num_tracks": len(tracks),
            "epochs_run": len(history),
            "interaction_mode": config.behavior_roles.interaction_mode,
            "bl6_identity": config.behavior_roles.bl6_identity,
            "cd1_identity": config.behavior_roles.cd1_identity,
            "primary_identity": config.behavior_roles.primary_identity,
            "features_lean": config.features.lean,
            "features_pose": config.features.pose,
            "features_egocentric": config.features.egocentric,
            "features_rolling_stat_policy": config.features.rolling_stat_policy,
            "features_dyadic": config.features.dyadic,
            "features_mask_overlap": config.features.mask_overlap,
            "per_video_robust_normalization": config.training.per_video_robust_normalization,
            "coral_normalization": config.training.coral_normalization,
            "feature_stat_augmentation": config.training.feature_stat_augmentation,
            "domain_adversary_weight": config.training.domain_adversary_weight,
            "supervised_contrastive_weight": config.training.supervised_contrastive_weight,
            "bout_transition_weight": config.training.bout_transition_weight,
            "input_feature_l1_weight": config.training.input_feature_l1_weight,
            "overlap_loss_weight": config.training.overlap_loss_weight,
            "tversky_alpha": config.training.tversky_alpha,
            "tversky_beta": config.training.tversky_beta,
            "ohem_fraction": config.training.ohem_fraction,
            "boundary_tolerance_frames": config.training.boundary_tolerance_frames,
            "grouped_attack_label": config.training.grouped_attack_label,
            "grouped_attack_name": config.training.grouped_attack_name,
            "grouped_attack_members": list(config.training.grouped_attack_members),
            "attack_parent_consistency_weight": config.training.attack_parent_consistency_weight,
            "attack_boundary_oversample_weight": config.training.attack_boundary_oversample_weight,
            "threshold_tuning_metric": config.training.threshold_tuning_metric,
            "tune_postprocessing": config.training.tune_postprocessing,
            "postprocess_tuning_metric": config.training.postprocess_tuning_metric,
            "validation_metric": config.training.validation_metric,
            "postprocess": best_postprocess if multilabel and config.training.tune_postprocessing else None,
            "transition_penalty": config.inference.transition_penalty,
            "prior_calibration_strength": config.inference.prior_calibration_strength,
            "prior_calibration_mode": config.inference.prior_calibration_mode,
            "prior_calibration_min_rate": config.inference.prior_calibration_min_rate,
        },
    )
    ckpt_dir = output_dir / "checkpoints"
    ckpt_path = save_checkpoint(checkpoint, ckpt_dir / "best.pt")

    write_json(metrics_payload, output_dir / "metrics.json")
    write_table(pd.DataFrame(history), output_dir / "training_history.csv")
    save_config(config, output_dir / "config_used.yaml")
    emit(1.0, f"Training complete. Checkpoint: {ckpt_path}")

    return TrainingResult(
        checkpoint_path=ckpt_path,
        metrics=metrics_payload,
        history=history,
        label_map=label_map,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train behavior segmentation model.")
    parser.add_argument("--config", default=None, help="Path to YAML config.")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    def log(message: str) -> None:
        print(message, flush=True)

    result = run_training(config, log=log)
    metric = result.metrics.get("best_val_metric", config.training.validation_metric)
    print(f"\nBest val {metric}: {result.metrics.get('best_val_score', 0.0):.4f}")
    print(f"Checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
