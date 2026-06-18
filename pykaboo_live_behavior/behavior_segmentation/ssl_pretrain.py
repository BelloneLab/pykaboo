"""Self-supervised EmbTCN-AT pretraining on unlabeled feature tracks."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import AppConfig, load_config, save_config
from .dataset import WindowDataset, collate_windows, split_tracks_by_group
from .embtcn_infer import (
    encode_track,
    embeddings_dataframe,
    prepare_embtcn_tracks,
    save_embtcn_bundle,
)
from .models.embtcn_attention import (
    EmbTCNAttention,
    EmbTCNConfig,
    UncertaintyWeightedLoss,
    apply_channel_dropout,
    inject_synthetic_faults,
    make_span_mask,
    masked_reconstruction_loss,
)
from .normalization import FeatureNormalizer
from .pipeline import resolve_device
from .storage import write_table
from .windows import seconds_to_frames

ProgressFn = Callable[[float, str], None]
LogFn = Callable[[str], None]
StopFn = Callable[[], bool]


@dataclass
class SSLPretrainResult:
    checkpoint_path: Path
    history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_embtcn_model(config: AppConfig, num_features: int) -> EmbTCNAttention:
    cfg = EmbTCNConfig(
        num_features=num_features,
        num_classes=1,
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
        use_supervised_head=False,
        use_fault_head=True,
        use_decoder=True,
        multilabel=True,
    )
    return EmbTCNAttention(cfg)


def mean_baseline_reconstruction_loss(
    batch: torch.Tensor,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid = valid_mask.unsqueeze(1).float()
    mean = (batch * valid).sum(dim=2, keepdim=True) / valid.sum(dim=2, keepdim=True).clamp_min(1.0)
    return masked_reconstruction_loss(batch, mean.expand_as(batch), mask, valid_mask)


def temporal_jitter(x: torch.Tensor, max_shift: int) -> torch.Tensor:
    """Shift each sample by a small random amount, padding with edge frames."""

    if max_shift <= 0 or x.shape[-1] <= 1:
        return x
    out = x.clone()
    shifts = torch.randint(
        -max_shift,
        max_shift + 1,
        (x.shape[0],),
        device=x.device,
    )
    for idx, shift_t in enumerate(shifts):
        shift = int(shift_t.item())
        if shift > 0:
            out[idx, :, shift:] = x[idx, :, :-shift]
            out[idx, :, :shift] = x[idx, :, :1]
        elif shift < 0:
            k = -shift
            out[idx, :, :-k] = x[idx, :, k:]
            out[idx, :, -k:] = x[idx, :, -1:]
    return out


def augmented_ssl_view(x: torch.Tensor, config: AppConfig) -> torch.Tensor:
    """Build one SSL view with noise, channel dropout and small time jitter."""

    out = x
    if config.embtcn.gaussian_noise_std > 0:
        out = out + config.embtcn.gaussian_noise_std * torch.randn_like(out)
    out = apply_channel_dropout(out, config.embtcn.channel_dropout)
    return temporal_jitter(out, config.embtcn.ssl_time_jitter_frames)


def _flatten_valid(z: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return valid tokens plus their sample and time ids from ``[B, E, T]``."""

    b, _e, t = z.shape
    sample_ids = torch.arange(b, device=z.device).repeat_interleave(t)
    time_ids = torch.arange(t, device=z.device).repeat(b)
    flat = z.transpose(1, 2).reshape(b * t, -1)
    valid = valid_mask.reshape(b * t).bool()
    return flat[valid], sample_ids[valid], time_ids[valid]


def temporal_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 0.1,
    radius: int = 2,
) -> torch.Tensor:
    """InfoNCE with same or nearby frames across two augmented views as positives."""

    a, sample_a, time_a = _flatten_valid(z1, valid_mask)
    b, sample_b, time_b = _flatten_valid(z2, valid_mask)
    if a.shape[0] < 2 or b.shape[0] < 2:
        return z1.sum() * 0.0
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    positives = (sample_a[:, None] == sample_b[None, :]) & (
        (time_a[:, None] - time_b[None, :]).abs() <= max(int(radius), 0)
    )
    has_pos = positives.any(dim=1)
    if not has_pos.any():
        return z1.sum() * 0.0
    logits = a @ b.T / max(float(temperature), 1e-6)
    log_prob = F.log_softmax(logits[has_pos], dim=1)
    target = positives[has_pos].to(dtype=log_prob.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(target * log_prob).sum(dim=1).mean()


def hierarchical_temporal_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
    radius: int,
) -> torch.Tensor:
    """Average contrastive loss across raw and pooled temporal scales."""

    losses = [temporal_contrastive_loss(z1, z2, valid_mask, temperature, radius)]
    cur1, cur2, cur_mask = z1, z2, valid_mask
    for _ in range(3):
        if cur1.shape[-1] < 4:
            break
        cur1 = F.avg_pool1d(cur1, kernel_size=2, stride=2)
        cur2 = F.avg_pool1d(cur2, kernel_size=2, stride=2)
        cur_mask = F.max_pool1d(cur_mask.float().unsqueeze(1), kernel_size=2, stride=2).squeeze(1) > 0
        losses.append(
            temporal_contrastive_loss(
                cur1,
                cur2,
                cur_mask,
                temperature,
                max(radius // 2, 0),
            )
        )
    return torch.stack(losses).mean()


def vicreg_regularizer(
    z: torch.Tensor,
    valid_mask: torch.Tensor,
    variance_weight: float,
    covariance_weight: float,
) -> torch.Tensor:
    """VICReg-style variance and covariance penalties to discourage collapse."""

    tokens, _sample, _time = _flatten_valid(z, valid_mask)
    if tokens.shape[0] < 2:
        return z.sum() * 0.0
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    std = torch.sqrt(tokens.var(dim=0) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    cov = (tokens.T @ tokens) / max(tokens.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    covariance = off_diag.square().sum() / max(tokens.shape[1], 1)
    return variance_weight * variance + covariance_weight * covariance


def jepa_loss(
    online_z: torch.Tensor,
    target_z: torch.Tensor,
    predictor: nn.Module,
    mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Predict target-encoder latents at masked positions."""

    pred = predictor(online_z.transpose(1, 2)).transpose(1, 2)
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target_z.detach(), dim=1)
    effective = (mask & valid_mask).unsqueeze(1).float()
    se = (pred - target).square() * effective
    return se.sum() / effective.sum().clamp_min(1.0) / max(pred.shape[1], 1)


@torch.no_grad()
def update_ema_model(target: nn.Module, online: nn.Module, momentum: float) -> None:
    """EMA update for the JEPA target encoder."""

    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)


def run_ssl_pretraining(
    config: AppConfig,
    progress: ProgressFn | None = None,
    log: LogFn | None = None,
    should_stop: StopFn | None = None,
) -> SSLPretrainResult:
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

    tracks, feature_names = prepare_embtcn_tracks(config, cache_dir, log=log)
    if not tracks:
        raise RuntimeError("No tracks available for EmbTCN SSL pretraining.")
    splits = split_tracks_by_group(
        tracks,
        train_fraction=0.85,
        validation_fraction=0.15,
        seed=config.project.seed,
    )
    normalizer = FeatureNormalizer.fit([track.features for track in splits["train"]], feature_names)
    emit(
        0.12,
        f"EmbTCN SSL tracks: {len(tracks)}  features: {len(feature_names)}  "
        f"train/val={len(splits['train'])}/{len(splits['val'])}",
    )

    train_ds = WindowDataset(
        splits["train"],
        config.training.window_seconds,
        config.training.stride_seconds,
        config.data.frame_rate,
        normalizer,
        multilabel=False,
    )
    val_ds = WindowDataset(
        splits["val"],
        config.training.window_seconds,
        config.training.window_seconds,
        config.data.frame_rate,
        normalizer,
        multilabel=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.embtcn.ssl_batch_size,
        shuffle=True,
        collate_fn=collate_windows,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.embtcn.ssl_batch_size,
        shuffle=False,
        collate_fn=collate_windows,
        num_workers=0,
    )

    objective = config.embtcn.ssl_objective
    use_recon = objective in {"recon", "recon+contrastive"}
    use_contrastive = objective in {"contrastive", "recon+contrastive"}
    use_jepa = objective == "jepa"
    model = build_embtcn_model(config, len(feature_names)).to(device)
    target_model = copy.deepcopy(model).to(device) if use_jepa else None
    if target_model is not None:
        target_model.eval()
        for param in target_model.parameters():
            param.requires_grad = False
    predictor = (
        nn.Sequential(
            nn.Linear(config.embtcn.embedding_dim, config.embtcn.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embtcn.embedding_dim, config.embtcn.embedding_dim),
        ).to(device)
        if use_jepa
        else None
    )
    num_tasks = int(use_recon) + int(use_contrastive) + int(use_jepa) + 1
    task_loss = UncertaintyWeightedLoss(num_tasks).to(device)
    trainable = list(model.parameters()) + list(task_loss.parameters())
    if predictor is not None:
        trainable += list(predictor.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.embtcn.ssl_learning_rate,
        weight_decay=config.embtcn.ssl_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.embtcn.ssl_max_epochs, 1)
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mask_span = seconds_to_frames(config.embtcn.mask_span_seconds, config.data.frame_rate)
    best_val = float("inf")
    best_state = None
    best_loss_state = None
    best_predictor_state = None
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in range(config.embtcn.ssl_max_epochs):
        if should_stop and should_stop():
            emit(1.0, "EmbTCN SSL pretraining stopped by user.")
            break
        model.train()
        total = rec_total = con_total = jepa_total = fault_total = baseline_total = 0.0
        batches = 0
        for batch in train_loader:
            x = batch["features"].to(device)
            valid_mask = batch["mask"].to(device)
            span_mask = make_span_mask(
                x.shape[0],
                x.shape[-1],
                config.embtcn.mask_ratio,
                mask_span,
                device=x.device,
            ) & valid_mask
            noisy = augmented_ssl_view(x, config)
            corrupted, fault_target = inject_synthetic_faults(
                noisy,
                probability=config.embtcn.fault_probability,
                noise_std=config.embtcn.fault_noise_std,
            )
            optimizer.zero_grad()
            out = model(
                corrupted,
                mask=span_mask if (use_recon or use_jepa) else None,
                padding_mask=~valid_mask,
            )
            losses: list[torch.Tensor] = []
            rec_loss = x.sum() * 0.0
            con_loss = x.sum() * 0.0
            jepa_loss_value = x.sum() * 0.0
            if use_recon:
                rec_loss = masked_reconstruction_loss(
                    x, out.reconstruction, span_mask, valid_mask
                )
                losses.append(rec_loss)
            if use_contrastive:
                view1 = augmented_ssl_view(x, config)
                view2 = augmented_ssl_view(x, config)
                z1 = model(view1, padding_mask=~valid_mask).embeddings
                z2 = model(view2, padding_mask=~valid_mask).embeddings
                con_loss = hierarchical_temporal_contrastive_loss(
                    z1,
                    z2,
                    valid_mask,
                    config.embtcn.contrastive_temperature,
                    config.embtcn.contrastive_radius_frames,
                )
                con_loss = con_loss + 0.5 * (
                    vicreg_regularizer(
                        z1,
                        valid_mask,
                        config.embtcn.vicreg_variance_weight,
                        config.embtcn.vicreg_covariance_weight,
                    )
                    + vicreg_regularizer(
                        z2,
                        valid_mask,
                        config.embtcn.vicreg_variance_weight,
                        config.embtcn.vicreg_covariance_weight,
                    )
                )
                losses.append(con_loss)
            if use_jepa and target_model is not None and predictor is not None:
                with torch.no_grad():
                    target_z = target_model(x, padding_mask=~valid_mask).embeddings
                jepa_loss_value = jepa_loss(
                    out.embeddings,
                    target_z,
                    predictor,
                    span_mask,
                    valid_mask,
                )
                jepa_loss_value = jepa_loss_value + vicreg_regularizer(
                    out.embeddings,
                    valid_mask,
                    config.embtcn.vicreg_variance_weight,
                    config.embtcn.vicreg_covariance_weight,
                )
                losses.append(jepa_loss_value)
            fault_loss_raw = bce(out.fault, fault_target)
            fault_loss = (fault_loss_raw * valid_mask.float()).sum() / valid_mask.sum().clamp_min(1)
            losses.append(fault_loss)
            loss = task_loss(losses)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            if predictor is not None:
                nn.utils.clip_grad_norm_(predictor.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            if target_model is not None:
                update_ema_model(target_model, model, config.embtcn.jepa_ema_momentum)
            with torch.no_grad():
                baseline = mean_baseline_reconstruction_loss(x, span_mask, valid_mask)
            total += float(loss.item())
            rec_total += float(rec_loss.item())
            con_total += float(con_loss.item())
            jepa_total += float(jepa_loss_value.item())
            fault_total += float(fault_loss.item())
            baseline_total += float(baseline.item())
            batches += 1
        scheduler.step()

        val_rec = evaluate_reconstruction(model, val_loader, config, device, mask_span)
        val_ssl = evaluate_ssl_objective(
            model,
            val_loader,
            config,
            device,
            mask_span,
            objective,
            target_model=target_model,
            predictor=predictor,
        )
        row = {
            "epoch": epoch,
            "loss": total / max(batches, 1),
            "reconstruction_loss": rec_total / max(batches, 1),
            "contrastive_loss": con_total / max(batches, 1),
            "jepa_loss": jepa_total / max(batches, 1),
            "fault_loss": fault_total / max(batches, 1),
            "mean_baseline_loss": baseline_total / max(batches, 1),
            "val_reconstruction_loss": val_rec,
            "val_objective_loss": val_ssl,
        }
        history.append(row)
        emit(
            0.12 + 0.78 * (epoch + 1) / max(config.embtcn.ssl_max_epochs, 1),
            f"SSL epoch {epoch + 1}/{config.embtcn.ssl_max_epochs} "
            f"obj={objective} loss={row['loss']:.4f} val_obj={val_ssl:.4f}",
        )
        if val_ssl < best_val:
            best_val = val_ssl
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_loss_state = {key: value.detach().cpu().clone() for key, value in task_loss.state_dict().items()}
            if predictor is not None:
                best_predictor_state = {
                    key: value.detach().cpu().clone()
                    for key, value in predictor.state_dict().items()
                }
            patience = 0
        else:
            patience += 1
            if patience >= config.embtcn.early_stopping_patience:
                emit(0.9, f"EmbTCN SSL early stopping at epoch {epoch + 1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_loss_state is not None:
        task_loss.load_state_dict(best_loss_state)
    if predictor is not None and best_predictor_state is not None:
        predictor.load_state_dict(best_predictor_state)
    metadata = {
        "best_val_objective_loss": best_val,
        "best_val_reconstruction_loss": min(
            [row["val_reconstruction_loss"] for row in history],
            default=float("inf"),
        ),
        "ssl_objective": objective,
        "epochs_run": len(history),
        "num_tracks": len(tracks),
        "num_features": len(feature_names),
        "task_type": config.embtcn.task_type,
        "per_video_robust_normalization": config.embtcn.per_video_robust_normalization,
        "uncertainty_log_sigma": task_loss.log_sigma.detach().cpu().numpy().tolist(),
    }
    ckpt_path = save_embtcn_bundle(
        output_dir / "checkpoints" / "embtcn_ssl.pt",
        model,
        normalizer,
        feature_names,
        metadata,
    )
    write_table(pd.DataFrame(history), output_dir / "embtcn_ssl_history.csv")
    save_config(config, output_dir / "embtcn_ssl_config.yaml")
    export_embeddings(model, tracks, normalizer, config, output_dir, device, log)
    emit(1.0, f"EmbTCN SSL complete. Checkpoint: {ckpt_path}")
    return SSLPretrainResult(ckpt_path, history, metadata)


@torch.no_grad()
def evaluate_reconstruction(
    model: EmbTCNAttention,
    loader: DataLoader,
    config: AppConfig,
    device: str,
    mask_span: int,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        x = batch["features"].to(device)
        valid_mask = batch["mask"].to(device)
        span_mask = make_span_mask(
            x.shape[0],
            x.shape[-1],
            config.embtcn.mask_ratio,
            mask_span,
            device=x.device,
        ) & valid_mask
        out = model(x, mask=span_mask, padding_mask=~valid_mask)
        losses.append(float(masked_reconstruction_loss(x, out.reconstruction, span_mask, valid_mask).item()))
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def evaluate_ssl_objective(
    model: EmbTCNAttention,
    loader: DataLoader,
    config: AppConfig,
    device: str,
    mask_span: int,
    objective: str,
    target_model: EmbTCNAttention | None = None,
    predictor: nn.Module | None = None,
) -> float:
    """Validation loss for the selected SSL objective."""

    model.eval()
    if target_model is not None:
        target_model.eval()
    if predictor is not None:
        predictor.eval()
    values: list[float] = []
    use_recon = objective in {"recon", "recon+contrastive"}
    use_contrastive = objective in {"contrastive", "recon+contrastive"}
    use_jepa = objective == "jepa"
    for batch in loader:
        x = batch["features"].to(device)
        valid_mask = batch["mask"].to(device)
        span_mask = make_span_mask(
            x.shape[0],
            x.shape[-1],
            config.embtcn.mask_ratio,
            mask_span,
            device=x.device,
        ) & valid_mask
        losses: list[torch.Tensor] = []
        if use_recon:
            out = model(x, mask=span_mask, padding_mask=~valid_mask)
            losses.append(
                masked_reconstruction_loss(
                    x, out.reconstruction, span_mask, valid_mask
                )
            )
        if use_contrastive:
            z1 = model(augmented_ssl_view(x, config), padding_mask=~valid_mask).embeddings
            z2 = model(augmented_ssl_view(x, config), padding_mask=~valid_mask).embeddings
            losses.append(
                hierarchical_temporal_contrastive_loss(
                    z1,
                    z2,
                    valid_mask,
                    config.embtcn.contrastive_temperature,
                    config.embtcn.contrastive_radius_frames,
                )
            )
        if use_jepa and target_model is not None and predictor is not None:
            online = model(x, mask=span_mask, padding_mask=~valid_mask).embeddings
            target = target_model(x, padding_mask=~valid_mask).embeddings
            losses.append(jepa_loss(online, target, predictor, span_mask, valid_mask))
        if losses:
            values.append(float(torch.stack(losses).mean().item()))
    if predictor is not None:
        predictor.train()
    return float(np.mean(values)) if values else float("inf")


@torch.no_grad()
def export_embeddings(
    model: EmbTCNAttention,
    tracks,
    normalizer: FeatureNormalizer,
    config: AppConfig,
    output_dir: Path,
    device: str,
    log: LogFn | None,
) -> None:
    model.eval()
    embeddings, faults = [], []
    for idx, track in enumerate(tracks):
        emb, fault = encode_track(
            model,
            track,
            normalizer,
            config.training.window_seconds,
            config.inference.stride_seconds,
            config.data.frame_rate,
            device,
        )
        embeddings.append(emb)
        faults.append(fault)
        if log:
            log(f"Exported SSL embedding {idx + 1}/{len(tracks)}: {track.video_id}/{track.subject_id}")
    write_table(
        embeddings_dataframe(tracks, embeddings, faults),
        output_dir / "embtcn_ssl_embeddings.parquet",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretrain EmbTCN-AT with masked reconstruction.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--objective",
        choices=["recon", "contrastive", "jepa", "recon+contrastive"],
        default=None,
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.output_dir:
        config.project.output_dir = args.output_dir
    if args.objective:
        config.embtcn.ssl_objective = args.objective
    result = run_ssl_pretraining(config, log=lambda message: print(message, flush=True))
    print(f"Checkpoint: {result.checkpoint_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
