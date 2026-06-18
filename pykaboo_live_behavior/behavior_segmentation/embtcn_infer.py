"""Inference utilities for EmbTCN-AT self-supervised embeddings and fault scores."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import AppConfig, load_config
from .dataset import (
    TrackData,
    build_feature_column_list,
    build_inference_tracks,
)
from .models.embtcn_attention import (
    EmbTCNAttention,
    EmbTCNConfig,
    inject_synthetic_faults,
)
from .normalization import FeatureNormalizer
from .pipeline import extract_features_for_videos, resolve_device
from .storage import write_table
from .windows import build_windows, seconds_to_frames

LogFn = Callable[[str], None]


@dataclass
class EmbTCNBundle:
    model: EmbTCNAttention
    normalizer: FeatureNormalizer
    feature_names: list[str]
    model_config: dict
    metadata: dict


def save_embtcn_bundle(
    path: str | Path,
    model: EmbTCNAttention,
    normalizer: FeatureNormalizer,
    feature_names: list[str],
    metadata: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": model.cfg.to_dict(),
        "feature_names": list(feature_names),
        "normalizer": normalizer.to_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_embtcn_bundle(path: str | Path, device: str = "cpu") -> EmbTCNBundle:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = EmbTCNConfig.from_dict(payload["model_config"])
    model = EmbTCNAttention(cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return EmbTCNBundle(
        model=model,
        normalizer=FeatureNormalizer.from_dict(payload["normalizer"]),
        feature_names=list(payload["feature_names"]),
        model_config=dict(payload["model_config"]),
        metadata=dict(payload.get("metadata", {})),
    )


def robust_scale_tracks_by_video(tracks: list[TrackData]) -> list[TrackData]:
    """Apply per-video median/IQR scaling while preserving track metadata."""

    by_video: dict[str, list[TrackData]] = {}
    for track in tracks:
        by_video.setdefault(track.video_id, []).append(track)
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for video_id, group in by_video.items():
        matrix = np.concatenate([track.features for track in group], axis=0)
        median = np.nanmedian(matrix, axis=0)
        q25 = np.nanpercentile(matrix, 25, axis=0)
        q75 = np.nanpercentile(matrix, 75, axis=0)
        iqr = np.where((q75 - q25) < 1e-6, 1.0, q75 - q25)
        stats[video_id] = (median, iqr)
    scaled: list[TrackData] = []
    for track in tracks:
        median, iqr = stats[track.video_id]
        new_track = copy.copy(track)
        new_track.features = np.nan_to_num(
            (track.features.astype(np.float64) - median) / iqr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
        scaled.append(new_track)
    return scaled


def prepare_embtcn_tracks(
    config: AppConfig,
    cache_dir: str | Path | None = None,
    feature_names: list[str] | None = None,
    log: LogFn | None = None,
) -> tuple[list[TrackData], list[str]]:
    identity_df, pair_df, _frame_indices = extract_features_for_videos(
        config, cache_dir, log
    )
    if identity_df.empty:
        raise RuntimeError("No features extracted for EmbTCN.")
    is_pair = config.embtcn.task_type == "pair"
    if feature_names is None:
        feature_names = build_feature_column_list(identity_df, pair_df, is_pair)
    tracks = build_inference_tracks(identity_df, pair_df, feature_names, is_pair)
    if config.embtcn.per_video_robust_normalization:
        tracks = robust_scale_tracks_by_video(tracks)
    return tracks, feature_names


@torch.no_grad()
def encode_track(
    model: EmbTCNAttention,
    track: TrackData,
    normalizer: FeatureNormalizer,
    window_seconds: float,
    stride_seconds: float,
    frame_rate: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = track.num_frames
    window_length = seconds_to_frames(window_seconds, frame_rate)
    stride = seconds_to_frames(stride_seconds, frame_rate)
    spans = build_windows(num_frames, window_length, stride)
    features = normalizer.transform(track.features, track.video_id)
    embeddings = np.zeros((model.embedding_dim, num_frames), dtype=np.float64)
    faults = np.zeros(num_frames, dtype=np.float64)
    counts = np.zeros(num_frames, dtype=np.float64)
    for span in spans:
        segment = features[span.start : span.end]
        tensor = torch.from_numpy(segment.T.copy()).unsqueeze(0).to(device)
        out = model(tensor)
        width = span.end - span.start
        embeddings[:, span.start : span.end] += out.embeddings[0, :, :width].cpu().numpy()
        if out.fault is not None:
            faults[span.start : span.end] += torch.sigmoid(out.fault[0, :width]).cpu().numpy()
        counts[span.start : span.end] += 1.0
    counts = np.clip(counts, 1.0, None)
    return (embeddings / counts).astype(np.float32), (faults / counts).astype(np.float32)


def embeddings_dataframe(
    tracks: list[TrackData],
    embeddings: list[np.ndarray],
    faults: list[np.ndarray],
) -> pd.DataFrame:
    frames = []
    for track, emb, fault in zip(tracks, embeddings, faults):
        base = pd.DataFrame(
            {
                "video_id": track.video_id,
                "frame_idx": track.frame_indices.astype(int),
                "subject_id": track.subject_id,
                "object_id": track.object_id,
                "fault_score": fault,
            }
        )
        emb_df = pd.DataFrame(
            emb.T,
            columns=[f"embedding_{idx:03d}" for idx in range(emb.shape[0])],
        )
        frames.append(pd.concat([base, emb_df], axis=1))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_embtcn_embedding_export(
    checkpoint: str | Path,
    config: AppConfig,
    output_dir: str | Path,
    device: str | None = None,
    cache_dir: str | Path | None = None,
    log: LogFn | None = None,
) -> pd.DataFrame:
    device = device or resolve_device(config.training.device)
    bundle = load_embtcn_bundle(checkpoint, device)
    tracks, feature_names = prepare_embtcn_tracks(
        config, cache_dir, bundle.feature_names, log
    )
    if list(feature_names) != list(bundle.feature_names):
        raise RuntimeError("Feature schema mismatch for EmbTCN checkpoint.")
    normalizer = bundle.normalizer.with_video_stats_from_tracks(tracks)
    embeddings, faults = [], []
    for idx, track in enumerate(tracks):
        emb, fault = encode_track(
            bundle.model,
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
            log(f"Encoded EmbTCN track {idx + 1}/{len(tracks)}: {track.video_id}/{track.subject_id}")
    df = embeddings_dataframe(tracks, embeddings, faults)
    write_table(df, Path(output_dir) / "embtcn_embeddings.parquet")
    return df


@torch.no_grad()
def evaluate_synthetic_fault_auroc(
    bundle: EmbTCNBundle,
    tracks: list[TrackData],
    config: AppConfig,
    device: str,
    max_batches: int = 32,
) -> float:
    normalizer = bundle.normalizer
    window_length = seconds_to_frames(config.training.window_seconds, config.data.frame_rate)
    scores, targets = [], []
    seen = 0
    for track in tracks:
        if track.num_frames < 4:
            continue
        for span in build_windows(track.num_frames, window_length, window_length):
            x = normalizer.transform(track.features[span.start : span.end])
            tensor = torch.from_numpy(x.T.copy()).unsqueeze(0).to(device)
            corrupted, target = inject_synthetic_faults(
                tensor,
                probability=config.embtcn.fault_probability,
                noise_std=config.embtcn.fault_noise_std,
            )
            out = bundle.model(corrupted)
            if out.fault is None:
                return float("nan")
            scores.append(torch.sigmoid(out.fault).flatten().cpu().numpy())
            targets.append(target.flatten().cpu().numpy())
            seen += 1
            if seen >= max_batches:
                y = np.concatenate(targets)
                s = np.concatenate(scores)
                return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    if not scores:
        return float("nan")
    y = np.concatenate(targets)
    s = np.concatenate(scores)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export EmbTCN embeddings and fault scores.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fault-auroc", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or resolve_device(config.training.device)
    run_embtcn_embedding_export(
        args.checkpoint,
        config,
        output_dir,
        device=device,
        cache_dir=args.cache_dir,
        log=lambda message: print(message, flush=True),
    )
    if args.fault_auroc:
        bundle = load_embtcn_bundle(args.checkpoint, device)
        tracks, _ = prepare_embtcn_tracks(config, args.cache_dir, bundle.feature_names)
        auroc = evaluate_synthetic_fault_auroc(bundle, tracks, config, device)
        pd.DataFrame([{"synthetic_fault_auroc": auroc}]).to_csv(
            output_dir / "fault_metrics.csv", index=False
        )
        print(f"Synthetic fault AUROC: {auroc:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
