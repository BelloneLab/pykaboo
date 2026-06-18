"""Train the free social-interaction model: SSL pretrain + supervised head.

Two architectures share one feature surface, one split and one metric so the
comparison is honest:

- ``embtcn_attention`` (the requested "unsupervised network"): self-supervised
  masked-reconstruction pretraining on ALL videos (no labels), then a multi-label
  sigmoid head trained on the labeled tracks with focal-BCE + per-class
  thresholds. This is the EmbTCN-Attention-Transformer from ``TCN.md``.
- ``temporal_tcn``: the proven MS-TCN baseline, supervised only.

The trainer reuses the losses, samplers, threshold tuning, post-processing and
metrics from :mod:`behavior_segmentation.social_train` so the two paths are
identical apart from the backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import TrackData, WindowDataset, collate_windows
from .labels import LabelMap
from .models.embtcn_attention import (
    EmbTCNAttention,
    EmbTCNConfig,
    apply_channel_dropout,
    make_span_mask,
    masked_reconstruction_loss,
)
from .models.temporal_tcn import TemporalTcnModel
from .normalization import FeatureNormalizer
from .social_train import (
    TrainConfig,
    _collect,
    focal_bce,
    multilabel_window_weights,
    per_behavior_report,
    pos_weights_from_tracks,
    scene_level_report,
    temporal_smoothness,
    tune_thresholds,
    weighted_macro,
)
from .windows import seconds_to_frames

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class FreeTrainConfig(TrainConfig):
    architecture: str = "embtcn_attention"  # or "temporal_tcn"
    # EmbTCN backbone
    d_model: int = 160
    emb_dim: int = 96
    num_encoder_layers: int = 4
    num_heads: int = 8
    tcn_dilations: tuple = (1, 2, 4, 8, 16, 32)
    kernel_size: int = 5
    num_refinement_stages: int = 1
    # SSL pretraining
    ssl_epochs: int = 40
    ssl_mask_ratio: float = 0.15
    ssl_mask_span_seconds: float = 0.5
    ssl_channel_dropout: float = 0.2
    ssl_noise: float = 0.1
    ssl_lr: float = 7e-4


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #

def embtcn_config(num_features: int, num_classes: int, tc: FreeTrainConfig) -> EmbTCNConfig:
    return EmbTCNConfig(
        num_features=num_features,
        num_classes=num_classes,
        d_model=tc.d_model,
        embedding_dim=tc.emb_dim,
        tcn_dilations=tuple(tc.tcn_dilations),
        kernel_size=tc.kernel_size,
        num_encoder_layers=tc.num_encoder_layers,
        num_heads=tc.num_heads,
        dropout=tc.dropout,
        use_supervised_head=True,
        use_fault_head=False,
        use_decoder=True,
        multilabel=True,
        num_refinement_stages=tc.num_refinement_stages,
    )


def build_model(num_features: int, num_classes: int, tc: FreeTrainConfig, device: str):
    if tc.architecture == "embtcn_attention":
        return EmbTCNAttention(embtcn_config(num_features, num_classes, tc)).to(device)
    return TemporalTcnModel(
        num_features=num_features,
        num_classes=num_classes,
        hidden_channels=tc.hidden_channels,
        num_stages=tc.num_stages,
        num_layers_per_stage=tc.num_layers_per_stage,
        dropout=tc.dropout,
        embedding_dim=tc.embedding_dim,
        multilabel=True,
    ).to(device)


# --------------------------------------------------------------------------- #
# Self-supervised pretraining (masked reconstruction on ALL videos)
# --------------------------------------------------------------------------- #

@dataclass
class SSLResult:
    state_dict: dict[str, Any]
    history: list[dict[str, float]] = field(default_factory=list)
    num_features: int = 0


def ssl_pretrain(
    all_tracks: list[TrackData],
    feature_names: list[str],
    num_classes: int,
    frame_rate: float,
    tc: FreeTrainConfig,
    device: str = "cuda",
    log: LogFn | None = None,
    progress: ProgressFn | None = None,
    should_stop: Callable[[], bool] | None = None,
    make_model_fn: Callable[[], nn.Module] | None = None,
) -> SSLResult:
    """Masked-reconstruction pretraining of the backbone on every track.

    No labels are used. Returns the trained encoder/decoder ``state_dict`` so the
    supervised stage can warm-start from a representation shaped by all sessions.
    ``make_model_fn`` builds an arbitrary SSL backbone (must accept
    ``forward(x, mask=, padding_mask=)`` and return ``out.reconstruction``); when
    omitted the default EmbTCN-Attention is used.
    """

    def emit(frac: float, msg: str) -> None:
        if log:
            log(msg)
        if progress:
            progress(frac, msg)

    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    normalizer = FeatureNormalizer.fit([t.features for t in all_tracks], feature_names)
    win = seconds_to_frames(tc.window_seconds, frame_rate)
    stride = seconds_to_frames(tc.stride_seconds, frame_rate)
    span = max(seconds_to_frames(tc.ssl_mask_span_seconds, frame_rate), 1)

    ds = WindowDataset(all_tracks, tc.window_seconds, tc.stride_seconds, frame_rate,
                       normalizer, multilabel=True)
    loader = DataLoader(ds, batch_size=tc.batch_size, shuffle=True,
                        collate_fn=collate_windows, num_workers=0)

    if make_model_fn is not None:
        model = make_model_fn().to(device)
    else:
        model = EmbTCNAttention(embtcn_config(len(feature_names), num_classes, tc)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.ssl_lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(tc.ssl_epochs, 1))

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(tc.ssl_epochs):
        if should_stop and should_stop():
            emit(1.0, "SSL stopped by user.")
            break
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            feats = batch["features"].to(device)       # [B, D, T]
            pad = ~batch["mask"].to(device).bool()      # [B, T] True where padded
            valid = ~pad
            x = feats
            corrupt = apply_channel_dropout(x, tc.ssl_channel_dropout)
            if tc.ssl_noise > 0:
                corrupt = corrupt + tc.ssl_noise * torch.randn_like(corrupt)
            span_mask = make_span_mask(x.shape[0], x.shape[-1], tc.ssl_mask_ratio,
                                       span, device=device)
            span_mask = span_mask & valid
            out = model(corrupt, mask=span_mask, padding_mask=pad)
            loss = masked_reconstruction_loss(x, out.reconstruction, span_mask, valid)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        avg = tot / max(nb, 1)
        history.append({"epoch": epoch, "ssl_loss": avg})
        emit(0.05 + 0.9 * (epoch + 1) / max(tc.ssl_epochs, 1),
             f"SSL epoch {epoch+1}/{tc.ssl_epochs} recon_loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    emit(1.0, f"SSL done best_recon={best_loss:.4f}")
    return SSLResult(state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                     history=history, num_features=len(feature_names))


def load_ssl_backbone(model: EmbTCNAttention, ssl_state: dict[str, Any]) -> int:
    """Copy SSL-pretrained encoder/decoder weights into a fresh model. Returns the
    number of tensors transferred (the supervised head is left random)."""

    own = model.state_dict()
    transfer = {k: v for k, v in ssl_state.items() if k in own and own[k].shape == v.shape}
    own.update(transfer)
    model.load_state_dict(own)
    return len(transfer)


# --------------------------------------------------------------------------- #
# Supervised training
# --------------------------------------------------------------------------- #

@dataclass
class FreeTrainResult:
    report: Any                       # per-behavior DataFrame (per-identity)
    scene_report: Any                 # per-behavior DataFrame (scene OR)
    summary: dict[str, Any]
    thresholds: np.ndarray
    history: list[dict[str, float]]
    model: Any
    normalizer: FeatureNormalizer
    feature_names: list[str]
    label_map: LabelMap
    frame_rate: float


def train_supervised(
    train_tracks: list[TrackData],
    val_tracks: list[TrackData],
    feature_names: list[str],
    label_map: LabelMap,
    frame_rate: float,
    tc: FreeTrainConfig,
    device: str = "cuda",
    ssl_state: dict[str, Any] | None = None,
    log: LogFn | None = None,
    progress: ProgressFn | None = None,
    should_stop: Callable[[], bool] | None = None,
    make_model_fn: Callable[[], nn.Module] | None = None,
) -> FreeTrainResult:
    def emit(frac: float, msg: str) -> None:
        if log:
            log(msg)
        if progress:
            progress(frac, msg)

    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)

    normalizer = FeatureNormalizer.fit([t.features for t in train_tracks], feature_names)
    win = seconds_to_frames(tc.window_seconds, frame_rate)
    stride = seconds_to_frames(tc.stride_seconds, frame_rate)
    eval_stride = seconds_to_frames(tc.eval_stride_seconds, frame_rate)

    train_ds = WindowDataset(train_tracks, tc.window_seconds, tc.stride_seconds,
                             frame_rate, normalizer, multilabel=True)
    w = multilabel_window_weights(train_ds, label_map.num_classes, label_map.background_id)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    loader = DataLoader(train_ds, batch_size=tc.batch_size, sampler=sampler,
                        collate_fn=collate_windows, num_workers=0)

    if make_model_fn is not None:
        model = make_model_fn().to(device)
    else:
        model = build_model(len(feature_names), label_map.num_classes, tc, device)
    # Backbones that accept a padding_mask (EmbTCN and the wrapped NEMBA backbones)
    # take the masked forward path; MS-TCN does not.
    use_pad = (tc.architecture == "embtcn_attention") or (make_model_fn is not None)
    if ssl_state is not None and use_pad:
        n = load_ssl_backbone(model, ssl_state)
        emit(0.0, f"warm-started backbone from SSL ({n} tensors)")

    opt = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc.max_epochs)
    pos_weight = pos_weights_from_tracks(train_tracks, label_map.num_classes, tc.pos_weight_cap).to(device)

    best_score, best_state, patience = -1.0, None, 0
    best_thr = np.full(label_map.num_classes, 0.5)
    history: list[dict[str, float]] = []
    is_embtcn = use_pad

    for epoch in range(tc.max_epochs):
        if should_stop and should_stop():
            emit(1.0, "Stopped by user.")
            break
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            feats = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            if tc.feature_noise > 0:
                feats = feats + tc.feature_noise * torch.randn_like(feats)
            if tc.feature_dropout > 0:
                keep = (torch.rand(feats.shape[0], feats.shape[1], 1, device=device)
                        > tc.feature_dropout).float()
                feats = feats * keep
            opt.zero_grad()
            if is_embtcn:
                out = model(feats, padding_mask=~mask.bool())
            else:
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

        vp, vt = _collect(model, val_tracks, normalizer, win, eval_stride, device)
        thr = tune_thresholds(np.concatenate(vp, axis=1), np.concatenate(vt, axis=1),
                              label_map.background_id)
        rep = per_behavior_report(vp, vt, thr, label_map, frame_rate,
                                  tc.smooth_win, tc.min_bout_frames, tc.merge_gap_frames)
        wf1, macro = weighted_macro(rep)
        history.append({"epoch": epoch, "loss": tot / max(nb, 1), "val_wf1": wf1, "val_macro": macro})
        emit(0.1 + 0.85 * (epoch + 1) / tc.max_epochs,
             f"epoch {epoch+1}/{tc.max_epochs} loss={tot/max(nb,1):.3f} val_wF1={wf1:.3f} macro={macro:.3f}")
        if wf1 > best_score:
            best_score, best_thr, patience = wf1, thr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= tc.early_stopping_patience:
                emit(0.95, f"Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final validation evaluation (per-identity + scene)
    vp, vt = _collect(model, val_tracks, normalizer, win, eval_stride, device)
    report = per_behavior_report(vp, vt, best_thr, label_map, frame_rate,
                                 tc.smooth_win, tc.min_bout_frames, tc.merge_gap_frames)
    wf1, macro = weighted_macro(report)
    scene = scene_level_report(model, val_tracks, normalizer, win, eval_stride, device,
                               best_thr, label_map, tc)
    swf1, smacro = weighted_macro(scene)
    summary = {
        "architecture": tc.architecture,
        "val_weighted_f1": wf1,
        "val_macro_f1": macro,
        "scene_weighted_f1": swf1,
        "scene_macro_f1": smacro,
        "best_val_weighted_f1": best_score,
        "num_features": len(feature_names),
        "num_params": int(sum(p.numel() for p in model.parameters())),
    }
    emit(1.0, f"DONE wF1={wf1:.3f} macro={macro:.3f} scene_wF1={swf1:.3f}")
    return FreeTrainResult(report=report, scene_report=scene, summary=summary,
                           thresholds=best_thr, history=history, model=model,
                           normalizer=normalizer, feature_names=feature_names,
                           label_map=label_map, frame_rate=frame_rate)


# --------------------------------------------------------------------------- #
# Checkpoint (architecture-aware, loadable by social_infer.load_social_checkpoint)
# --------------------------------------------------------------------------- #

def save_free_checkpoint(path: str | Path, res: FreeTrainResult, tc: FreeTrainConfig,
                         metadata: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = res.model
    if tc.architecture == "embtcn_attention":
        model_config = model.cfg.to_dict()
    else:
        model_config = {
            "num_features": model.num_features,
            "num_classes": model.num_classes,
            "hidden_channels": tc.hidden_channels,
            "num_stages": tc.num_stages,
            "num_layers_per_stage": tc.num_layers_per_stage,
            "dropout": tc.dropout,
            "embedding_dim": tc.embedding_dim,
            "multilabel": True,
        }
    payload = {
        "architecture": tc.architecture,
        "model_config": model_config,
        "feature_names": list(res.feature_names),
        "label_map": res.label_map.to_dict(),
        "normalizer": res.normalizer.to_dict(),
        "thresholds": np.asarray(res.thresholds, dtype=np.float64).tolist(),
        "frame_rate": float(res.frame_rate),
        "window_seconds": float(tc.window_seconds),
        "eval_stride_seconds": float(tc.eval_stride_seconds),
        "smooth_win": int(tc.smooth_win),
        "min_bout_frames": int(tc.min_bout_frames),
        "merge_gap_frames": int(tc.merge_gap_frames),
        "metadata": {**(metadata or {}), **res.summary},
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, path)
    return path
