"""Wrap NEMBA self-supervised sequence backbones for the supervised free task.

NEMBA (``/home/andry/tracking_project/NEMBA/embedders_backbones``) provides a
family of SSL sequence encoders with a single interface:

    encode(x[B,C,T], mask[B,T]) -> z[B,E,T]
    forward(x, mask) -> out.reconstruction[B,C,T]   (masked-reconstruction SSL)

They have no supervised head. :class:`NembaSupervised` adds a per-frame
multi-label classification head on top of the embedding and exposes the same
output contract as ``EmbTCNAttention`` (``.embeddings, .reconstruction, .logits,
.probabilities, .stage_logits``), so it drops straight into
``free_train.ssl_pretrain`` / ``train_supervised`` and the existing metrics.

This lets us benchmark every NEMBA backbone on the free-interaction supervised
task with one identical training/eval surface.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from .labels import LabelMap
from .normalization import FeatureNormalizer

NEMBA_ROOT = "/home/andry/tracking_project/NEMBA"
if NEMBA_ROOT not in sys.path:
    sys.path.insert(0, NEMBA_ROOT)


def _nemba():
    from embedders_backbones import backbones as bb  # noqa: WPS433
    return bb


# Speed-ordered (fastest first), measured fwd+bwd at batch 4 on a 3090 with 432
# input channels. ``stt`` and ``gatr`` attend over all 432 feature channels as
# spatial tokens; they are memory-heavy at this width so they get a reduced
# footprint and run last (they may still OOM and are reported honestly).
BACKBONE_ORDER = [
    "tst", "vit", "rcnn", "patchtst", "embtcn",
    "enc_crf", "attn_bilstm", "bilstm", "dtc", "stt", "gatr",
]

# Per-backbone construction overrides (kept modest so the comparison is fair on
# compute and so the wide-channel spatial models have a chance to fit in memory).
BACKBONE_OVERRIDES: dict[str, dict[str, Any]] = {
    "stt": {"d_model": 32, "num_layers": 1, "num_heads": 2},
    "gatr": {"d_model": 32, "num_layers": 1, "num_heads": 2},
}
# Backbones that need a smaller batch to fit (spatial attention over 432 channels).
BACKBONE_BATCH: dict[str, int] = {"vit": 8, "patchtst": 8}

# Spatial-attention backbones (stt, gatr) attend over the input channels as tokens,
# which is O(C^2) and OOMs at C=432 (they target ~tens of neurons). A learned
# linear bottleneck (432 -> 64) before the backbone makes them tractable at a
# normal batch while still reconstructing the full 432 channels for SSL.
BACKBONE_BOTTLENECK: dict[str, int] = {"stt": 64, "gatr": 64}


@dataclass
class FreeBackboneOutput:
    embeddings: torch.Tensor
    reconstruction: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None

    @property
    def stage_logits(self) -> list[torch.Tensor]:
        return [self.logits] if self.logits is not None else []


class NembaSupervised(nn.Module):
    """A NEMBA backbone + a per-frame multi-label head, EmbTCN-compatible I/O."""

    def __init__(self, backbone_name: str, num_features: int, num_classes: int,
                 embedding_dim: int = 64, max_len: int = 512,
                 graph_adj: np.ndarray | None = None, channel_bottleneck: int | None = None,
                 **overrides) -> None:
        super().__init__()
        bb = _nemba()
        self.backbone_name = backbone_name
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.max_len = int(max_len)
        self.channel_bottleneck = int(channel_bottleneck) if channel_bottleneck else None
        self._overrides = dict(overrides)
        backbone_channels = self.channel_bottleneck or num_features
        if self.channel_bottleneck:
            # 432 -> bottleneck before the backbone; bottleneck -> 432 after, so SSL
            # still reconstructs the original feature space.
            self.proj_in = nn.Conv1d(num_features, self.channel_bottleneck, 1)
            self.proj_out = nn.Conv1d(self.channel_bottleneck, num_features, 1)
        self.backbone = bb.make_model(
            backbone_name, num_features=backbone_channels,
            embedding_dim=embedding_dim, max_len=max_len, **overrides)
        if bb.needs_graph(backbone_name) and graph_adj is not None and not self.channel_bottleneck:
            self.backbone.set_graph(graph_adj)
        self.head = nn.Conv1d(embedding_dim, num_classes, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                padding_mask: torch.Tensor | None = None) -> FreeBackboneOutput:
        # NEMBA backbones take only (x, mask); padding is handled at the loss level.
        xb = self.proj_in(x) if self.channel_bottleneck else x
        out = self.backbone(xb, mask)
        z = out.embeddings                       # [B, E, T]
        logits = self.head(z)                    # [B, K, T]
        rec = getattr(out, "reconstruction", None)
        if rec is not None and self.channel_bottleneck:
            rec = self.proj_out(rec)             # bottleneck -> original feature space
        return FreeBackboneOutput(
            embeddings=z,
            reconstruction=rec,
            logits=logits,
            probabilities=torch.sigmoid(logits),
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        xb = self.proj_in(x) if self.channel_bottleneck else x
        return self.backbone.encode(xb, mask)

    def config_dict(self) -> dict[str, Any]:
        return {
            "backbone_name": self.backbone_name,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
            "max_len": self.max_len,
            "channel_bottleneck": self.channel_bottleneck,
            "overrides": self._overrides,
        }


def build_channel_graph(train_features: list[np.ndarray], top_k: int = 8) -> np.ndarray:
    """Correlation graph over feature channels for graph-attention backbones."""
    bb = _nemba()
    X = np.concatenate(train_features, axis=0).T  # [N_features, T_total]
    # subsample time for speed
    if X.shape[1] > 4000:
        idx = np.linspace(0, X.shape[1] - 1, 4000).astype(int)
        X = X[:, idx]
    return bb.neuron_graph(X, top_k=top_k)


def make_backbone_factory(backbone_name: str, num_features: int, num_classes: int,
                          embedding_dim: int = 64, max_len: int = 512,
                          graph_adj: np.ndarray | None = None,
                          channel_bottleneck: int | None = "auto") -> Callable[[], NembaSupervised]:
    overrides = BACKBONE_OVERRIDES.get(backbone_name, {})
    bottleneck = (BACKBONE_BOTTLENECK.get(backbone_name)
                  if channel_bottleneck == "auto" else channel_bottleneck)

    def _factory() -> NembaSupervised:
        return NembaSupervised(
            backbone_name, num_features, num_classes,
            embedding_dim=embedding_dim, max_len=max_len,
            graph_adj=graph_adj, channel_bottleneck=bottleneck, **overrides)

    return _factory


# --------------------------------------------------------------------------- #
# Checkpoint (self-contained; loadable without the training script)
# --------------------------------------------------------------------------- #

def save_backbone_checkpoint(path: str | Path, model: NembaSupervised,
                             feature_names: list[str], label_map: LabelMap,
                             normalizer: FeatureNormalizer, thresholds: np.ndarray,
                             frame_rate: float, tc, metadata: dict[str, Any] | None = None,
                             graph_adj: np.ndarray | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": f"nemba:{model.backbone_name}",
        "backbone_config": model.config_dict(),
        "feature_names": list(feature_names),
        "label_map": label_map.to_dict(),
        "normalizer": normalizer.to_dict(),
        "thresholds": np.asarray(thresholds, dtype=np.float64).tolist(),
        "frame_rate": float(frame_rate),
        "window_seconds": float(tc.window_seconds),
        "eval_stride_seconds": float(tc.eval_stride_seconds),
        "smooth_win": int(tc.smooth_win),
        "min_bout_frames": int(tc.min_bout_frames),
        "merge_gap_frames": int(tc.merge_gap_frames),
        "metadata": metadata or {},
        "graph_adj": (np.asarray(graph_adj) if graph_adj is not None else None),
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, path)
    return path


@dataclass
class BackboneCheckpoint:
    backbone_config: dict[str, Any]
    feature_names: list[str]
    label_map: LabelMap
    normalizer: FeatureNormalizer
    thresholds: np.ndarray
    frame_rate: float
    window_seconds: float
    eval_stride_seconds: float
    smooth_win: int
    min_bout_frames: int
    merge_gap_frames: int
    metadata: dict[str, Any]
    model_state: dict[str, Any]
    graph_adj: np.ndarray | None = None

    def build_model(self, device: str = "cpu") -> NembaSupervised:
        cfg = self.backbone_config
        m = NembaSupervised(
            cfg["backbone_name"], cfg["num_features"], cfg["num_classes"],
            embedding_dim=cfg["embedding_dim"], max_len=cfg["max_len"],
            graph_adj=self.graph_adj, channel_bottleneck=cfg.get("channel_bottleneck"),
            **cfg.get("overrides", {})).to(device)
        m.load_state_dict(self.model_state)
        m.eval()
        return m


def load_backbone_checkpoint(path: str | Path) -> BackboneCheckpoint:
    p = torch.load(path, map_location="cpu", weights_only=False)
    return BackboneCheckpoint(
        backbone_config=p["backbone_config"],
        feature_names=p["feature_names"],
        label_map=LabelMap.from_dict(p["label_map"]),
        normalizer=FeatureNormalizer.from_dict(p["normalizer"]),
        thresholds=np.asarray(p["thresholds"], dtype=np.float64),
        frame_rate=p["frame_rate"],
        window_seconds=p["window_seconds"],
        eval_stride_seconds=p["eval_stride_seconds"],
        smooth_win=p["smooth_win"],
        min_bout_frames=p["min_bout_frames"],
        merge_gap_frames=p["merge_gap_frames"],
        metadata=p.get("metadata", {}),
        model_state=p["model_state"],
        graph_adj=p.get("graph_adj"),
    )
