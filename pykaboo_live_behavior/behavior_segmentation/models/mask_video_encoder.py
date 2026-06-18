"""Mask-video stream for clean interaction clips.

The input is a background-free clip shaped ``[B, T, C, H, W]`` where channels
separate subject mask, partner mask, and overlap. A shared 2D CNN embeds each
frame, then a light temporal TCN models fast mask deformation and contact bouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embtcn_attention import EmbTCNAttention, EmbTCNConfig, EmbTCNOutput


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class _Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@dataclass
class MaskVideoConfig:
    in_channels: int = 3
    spatial_channels: tuple[int, ...] = (16, 32, 64)
    temporal_channels: int = 96
    embedding_dim: int = 64
    temporal_dilations: tuple[int, ...] = (1, 2, 4, 8)
    kernel_size: int = 5
    dropout: float = 0.15
    clip_size: int = 48

    @classmethod
    def from_dict(cls, payload: dict) -> "MaskVideoConfig":
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in payload.items() if key in fields}
        if "spatial_channels" in values:
            values["spatial_channels"] = tuple(values["spatial_channels"])
        if "temporal_dilations" in values:
            values["temporal_dilations"] = tuple(values["temporal_dilations"])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


class MaskVideoEncoder(nn.Module):
    """2D-CNN per frame followed by a temporal TCN. Returns ``[B, E, T]``."""

    def __init__(self, cfg: MaskVideoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        channels = [cfg.in_channels, *cfg.spatial_channels]
        blocks = []
        for idx in range(len(channels) - 1):
            blocks.append(_Conv2dBlock(channels[idx], channels[idx + 1], stride=2))
        self.spatial = nn.Sequential(*blocks)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.temporal_proj = nn.Conv1d(cfg.spatial_channels[-1], cfg.temporal_channels, 1)
        self.temporal = nn.ModuleList(
            [
                _TemporalBlock(
                    cfg.temporal_channels,
                    cfg.kernel_size,
                    dilation,
                    cfg.dropout,
                )
                for dilation in cfg.temporal_dilations
            ]
        )
        self.out = nn.Sequential(
            nn.Conv1d(cfg.temporal_channels, cfg.embedding_dim, 1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.dim() != 5:
            raise ValueError("mask clip must be [B, T, C, H, W]")
        b, t, c, h, w = clip.shape
        x = clip.reshape(b * t, c, h, w)
        x = self.spatial(x)
        x = self.spatial_pool(x).flatten(1)
        x = x.view(b, t, -1).transpose(1, 2)
        h_temporal = self.temporal_proj(x)
        for block in self.temporal:
            h_temporal = block(h_temporal)
        return self.out(h_temporal)


class MaskVideoAutoencoder(nn.Module):
    """Small SSL wrapper for reconstructing clean mask clips from visual embeddings."""

    def __init__(self, cfg: MaskVideoConfig) -> None:
        super().__init__()
        self.encoder = MaskVideoEncoder(cfg)
        self.cfg = cfg
        pixels = cfg.in_channels * cfg.clip_size * cfg.clip_size
        self.decoder = nn.Sequential(
            nn.Conv1d(cfg.embedding_dim, cfg.temporal_channels, 1),
            nn.GELU(),
            nn.Conv1d(cfg.temporal_channels, pixels, 1),
        )

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(clip)
        b, _, t = z.shape
        recon = self.decoder(z).transpose(1, 2)
        recon = recon.reshape(
            b,
            t,
            self.cfg.in_channels,
            self.cfg.clip_size,
            self.cfg.clip_size,
        )
        return z, torch.sigmoid(recon)


class EmbTCNMaskVideoFusion(nn.Module):
    """EmbTCN-AT scalar stream fused with a clean mask-video stream."""

    def __init__(
        self,
        emb_cfg: EmbTCNConfig,
        mask_cfg: MaskVideoConfig | None = None,
    ) -> None:
        super().__init__()
        mask_cfg = mask_cfg or MaskVideoConfig()
        emb_cfg.use_supervised_head = False
        emb_cfg.use_decoder = False
        self.scalar = EmbTCNAttention(emb_cfg)
        self.mask_encoder = MaskVideoEncoder(mask_cfg)
        self.requires_mask_clip = True
        self.cfg = emb_cfg
        self.mask_cfg = mask_cfg
        self.num_features = emb_cfg.num_features
        self.num_classes = emb_cfg.num_classes
        self.embedding_dim = emb_cfg.embedding_dim
        self.multilabel = emb_cfg.multilabel
        fused_dim = emb_cfg.embedding_dim + mask_cfg.embedding_dim
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_dim, emb_cfg.embedding_dim, 1),
            nn.GroupNorm(_groups(emb_cfg.embedding_dim), emb_cfg.embedding_dim),
            nn.GELU(),
            nn.Dropout(emb_cfg.dropout),
        )
        self.cls_head = nn.Conv1d(emb_cfg.embedding_dim, emb_cfg.num_classes, 1)

    @classmethod
    def from_model_config(cls, payload: dict) -> "EmbTCNMaskVideoFusion":
        emb_cfg = EmbTCNConfig.from_dict(payload)
        mask_cfg = MaskVideoConfig.from_dict(payload.get("mask_video", {}))
        return cls(emb_cfg, mask_cfg)

    def forward(
        self,
        features: torch.Tensor,
        mask_clip: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> EmbTCNOutput:
        scalar_z = self.scalar.encode(features, padding_mask=padding_mask)
        if mask_clip is None:
            b, _, t = scalar_z.shape
            mask_z = torch.zeros(
                b,
                self.mask_cfg.embedding_dim,
                t,
                device=scalar_z.device,
                dtype=scalar_z.dtype,
            )
        else:
            mask_z = self.mask_encoder(mask_clip)
            if mask_z.shape[-1] != scalar_z.shape[-1]:
                mask_z = F.interpolate(
                    mask_z,
                    size=scalar_z.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
        z = self.fusion(torch.cat([scalar_z, mask_z], dim=1))
        logits = self.cls_head(z)
        if padding_mask is not None:
            z = z.masked_fill(padding_mask.unsqueeze(1), 0.0)
            logits = logits.masked_fill(padding_mask.unsqueeze(1), 0.0)
        return EmbTCNOutput(
            embeddings=z,
            logits=logits,
            probabilities=torch.sigmoid(logits),
        )
