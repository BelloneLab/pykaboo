"""Multi-stage dilated temporal convolutional network for action segmentation.

Architecture follows the MS-TCN family: a prediction-generation stage built from
dilated residual 1D convolutions, optionally followed by refinement stages that
take the previous stage's probabilities as input. The penultimate feature tensor
is exposed as a per-frame embedding for downstream clustering.

Tensors use the convention ``[batch, channels, time]`` throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """A dilated residual 1D convolution block with group normalization."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        num_groups = _pick_num_groups(channels)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.residual(out)
        return x + out


class TemporalStage(nn.Module):
    """One stage: input projection, a stack of dilated blocks, classifier head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
        kernel_size: int,
        dropout: float,
        embedding_dim: int,
        produce_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(hidden_channels, kernel_size, 2**i, dropout)
                for i in range(num_layers)
            ]
        )
        self.produce_embedding = produce_embedding
        self.embedding_proj = (
            nn.Conv1d(hidden_channels, embedding_dim, 1)
            if produce_embedding
            else None
        )
        head_in = embedding_dim if produce_embedding else hidden_channels
        self.classifier = nn.Conv1d(head_in, num_classes, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)
        embedding = None
        if self.produce_embedding and self.embedding_proj is not None:
            embedding = self.embedding_proj(out)
            logits = self.classifier(embedding)
        else:
            logits = self.classifier(out)
        return logits, embedding


@dataclass
class TemporalTcnOutput:
    """Container for a forward pass result."""

    logits: torch.Tensor  # [B, num_classes, T] from the final stage
    probabilities: torch.Tensor  # [B, num_classes, T]
    embeddings: torch.Tensor  # [B, embedding_dim, T]
    stage_logits: list[torch.Tensor]  # logits from every stage (for aux loss)


class TemporalTcnModel(nn.Module):
    """Multi-stage temporal convolutional segmentation model."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_channels: int = 128,
        num_stages: int = 3,
        num_layers_per_stage: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.2,
        embedding_dim: int = 128,
        multilabel: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.multilabel = multilabel

        stages: list[TemporalStage] = []
        stages.append(
            TemporalStage(
                in_channels=num_features,
                hidden_channels=hidden_channels,
                num_layers=num_layers_per_stage,
                num_classes=num_classes,
                kernel_size=kernel_size,
                dropout=dropout,
                embedding_dim=embedding_dim,
                produce_embedding=(num_stages == 1),
            )
        )
        for stage_idx in range(1, num_stages):
            stages.append(
                TemporalStage(
                    in_channels=num_classes,
                    hidden_channels=hidden_channels,
                    num_layers=num_layers_per_stage,
                    num_classes=num_classes,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    embedding_dim=embedding_dim,
                    produce_embedding=(stage_idx == num_stages - 1),
                )
            )
        self.stages = nn.ModuleList(stages)

    def forward(self, features: torch.Tensor) -> TemporalTcnOutput:
        stage_logits: list[torch.Tensor] = []
        embedding: torch.Tensor | None = None

        logits, embedding = self.stages[0](features)
        stage_logits.append(logits)
        for stage in self.stages[1:]:
            prob = self._activate(logits)
            logits, stage_embedding = stage(prob)
            stage_logits.append(logits)
            if stage_embedding is not None:
                embedding = stage_embedding

        if embedding is None:
            embedding = torch.zeros(
                features.shape[0],
                self.embedding_dim,
                features.shape[-1],
                device=features.device,
            )
        probabilities = self._activate(logits)
        return TemporalTcnOutput(
            logits=logits,
            probabilities=probabilities,
            embeddings=embedding,
            stage_logits=stage_logits,
        )

    def _activate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.multilabel:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=1)


def _pick_num_groups(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1
