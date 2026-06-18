"""Checkpoint serialization that bundles everything needed for inference.

A checkpoint stores model weights together with the exact feature schema, label
map, and normalizer statistics used during training. Inference rebuilds the model
from this bundle so there is no chance of a feature or label mismatch going
unnoticed.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .labels import LabelMap
from .normalization import FeatureNormalizer
from .models.embtcn_attention import EmbTCNAttention, EmbTCNConfig
from .models.mask_video_encoder import EmbTCNMaskVideoFusion
from .models.temporal_tcn import TemporalTcnModel


class CheckpointMismatchError(RuntimeError):
    """Raised when loaded data does not match a checkpoint's feature/label schema."""


@dataclass
class Checkpoint:
    model_state: dict[str, Any]
    model_config: dict[str, Any]
    feature_names: list[str]
    label_map: LabelMap
    normalizer: FeatureNormalizer
    task_type: str
    frame_rate: float
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def build_model(self, device: str = "cpu") -> TemporalTcnModel:
        cfg = self.model_config
        architecture = cfg.get("architecture", "temporal_tcn")
        if architecture == "embtcn_attention":
            emb_cfg = EmbTCNConfig.from_dict(cfg)
            emb_cfg.use_supervised_head = True
            emb_cfg.use_decoder = bool(cfg.get("use_decoder", False))
            emb_cfg.use_fault_head = bool(cfg.get("use_fault_head", True))
            model = EmbTCNAttention(emb_cfg)
        elif architecture == "embtcn_mask_video":
            model = EmbTCNMaskVideoFusion.from_model_config(cfg)
        else:
            model = TemporalTcnModel(
                num_features=cfg["num_features"],
                num_classes=cfg["num_classes"],
                hidden_channels=cfg.get("hidden_channels", 128),
                num_stages=cfg.get("num_stages", 3),
                num_layers_per_stage=cfg.get("num_layers_per_stage", 8),
                kernel_size=cfg.get("kernel_size", 3),
                dropout=cfg.get("dropout", 0.2),
                embedding_dim=cfg.get("embedding_dim", 128),
                multilabel=cfg.get("multilabel", False),
            )
        model.load_state_dict(self.model_state)
        model.to(device)
        model.eval()
        return model

    def validate_features(self, feature_names: list[str]) -> None:
        if list(feature_names) != list(self.feature_names):
            missing = set(self.feature_names) - set(feature_names)
            extra = set(feature_names) - set(self.feature_names)
            raise CheckpointMismatchError(
                "Feature schema mismatch between checkpoint and data.\n"
                f"  expected {len(self.feature_names)} features, "
                f"got {len(feature_names)}.\n"
                f"  missing: {sorted(missing)}\n"
                f"  unexpected: {sorted(extra)}"
            )


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def save_checkpoint(checkpoint: Checkpoint, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": checkpoint.model_state,
        "model_config": checkpoint.model_config,
        "feature_names": checkpoint.feature_names,
        "label_map": checkpoint.label_map.to_dict(),
        "normalizer": checkpoint.normalizer.to_dict(),
        "task_type": checkpoint.task_type,
        "frame_rate": checkpoint.frame_rate,
        "training_metadata": {
            **checkpoint.training_metadata,
            "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit_hash(),
        },
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> Checkpoint:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    return Checkpoint(
        model_state=payload["model_state"],
        model_config=payload["model_config"],
        feature_names=list(payload["feature_names"]),
        label_map=LabelMap.from_dict(payload["label_map"]),
        normalizer=FeatureNormalizer.from_dict(payload["normalizer"]),
        task_type=payload.get("task_type", "single"),
        frame_rate=float(payload.get("frame_rate", 30.0)),
        training_metadata=payload.get("training_metadata", {}),
    )


def load_checkpoint_any(path: str | Path, device: str = "cpu") -> Checkpoint:
    """Load ANY checkpoint format into a display-ready Checkpoint.

    Main-pipeline checkpoints load directly. Social / nemba-fused checkpoints
    (no ``model_config``, e.g. the free-interaction model) are loaded via the
    social loader and adapted, so the GUI can load and show any checkpoint from
    any folder. Inference itself routes social/nemba models through
    ``run_social_inference``; this is for loading + display + label_map.
    """
    try:
        return load_checkpoint(path, device=device)
    except (KeyError, RuntimeError, TypeError):
        from .social_infer import load_social_checkpoint

        sc = load_social_checkpoint(path)
        model_config = {"architecture": sc.architecture}
        model_config.update(sc.backbone_config or {})
        model_config.update(sc.model_config or {})
        return Checkpoint(
            model_state=sc.model_state,
            model_config=model_config,
            feature_names=list(sc.feature_names),
            label_map=sc.label_map,
            normalizer=sc.normalizer,
            task_type="social",
            frame_rate=float(sc.frame_rate or 30.0),
            training_metadata=dict(sc.metadata or {}),
        )
