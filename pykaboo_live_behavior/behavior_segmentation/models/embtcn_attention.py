"""EmbTCN-Attention-Transformer for self-supervised behavior representation.

Synthesizes two papers (see TCN.md):
- Paper B (EmbTCN-Transformer): a TCN is used as the *input embedding* of a
  Transformer encoder (instead of a linear layer), injecting temporal awareness.
- Paper A (TCN-Attention fusion): a dual-branch attention module reweights both
  time steps (temporal attention with temperature) and feature channels (squeeze
  and excitation), plus multi-scale fusion across depths.

Differences from the papers: behavior segmentation is offline, so the dilated
convolutions use centered (NON-causal) padding and the Transformer encoder is
fully bidirectional. A causal flag is kept for a future real-time mode.

The model is a backbone producing a per-frame embedding ``Z`` plus several heads:
- ``reconstruction``: masked-input reconstruction for self-supervised pretraining.
- ``fault``: a per-frame anomaly/novelty score (robustness + active learning).
- ``logits``: optional multi-label behavior head (sigmoid), for the supervised arm.

Tensor convention: input ``x`` is ``[B, D, T]`` (channels, time), matching the
existing pipeline. The embedding output is ``[B, E, T]``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EmbTCNConfig:
    num_features: int = 250          # D, input feature channels
    num_classes: int = 17            # K, behaviors for the optional supervised head
    d_model: int = 160
    embedding_dim: int = 96          # E
    tcn_dilations: tuple = (1, 2, 4, 8, 16, 32)
    kernel_size: int = 5
    num_encoder_layers: int = 4
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.15
    temperature: float = 0.5         # tau for temporal attention
    causal: bool = False             # offline -> non-causal (bidirectional)
    max_len: int = 4096
    use_supervised_head: bool = True
    use_fault_head: bool = True
    use_decoder: bool = True         # SSL reconstruction head
    multilabel: bool = True
    # MS-TCN-style iterative refinement of the per-frame logits (0 = single stage).
    # Each stage re-reads the previous stage's predictions through a dilated TCN and
    # outputs a refined logit map; all stages are deep-supervised. Reduces over-
    # segmentation and sharpens bouts.
    num_refinement_stages: int = 0
    refinement_hidden: int = 64
    refinement_dilations: tuple = (1, 2, 4, 8)

    @classmethod
    def from_dict(cls, payload: dict) -> "EmbTCNConfig":
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in payload.items() if key in fields}
        if "tcn_dilations" in values:
            values["tcn_dilations"] = tuple(values["tcn_dilations"])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EmbTCNOutput:
    embeddings: torch.Tensor                 # [B, E, T]
    reconstruction: torch.Tensor | None = None   # [B, D, T]
    fault: torch.Tensor | None = None            # [B, T]
    logits: torch.Tensor | None = None           # [B, K, T]
    probabilities: torch.Tensor | None = None    # [B, K, T] sigmoid(logits)
    temporal_weights: torch.Tensor | None = None # [B, T] interpretable
    channel_gate: torch.Tensor | None = None     # [B, D_model] interpretable
    stages: list | None = None                   # all refinement-stage logits (deep sup.)

    @property
    def stage_logits(self) -> list[torch.Tensor]:
        if self.stages:
            return self.stages
        return [self.logits] if self.logits is not None else []


def _num_groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class DilatedResidualBlock(nn.Module):
    """Non-causal (centered) dilated residual 1D conv block. [B, C, T]."""

    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.kernel_size = kernel_size
        pad = (kernel_size - 1) * dilation if causal else (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.causal:
            # remove the right padding so output depends only on past/current
            trim = (self.kernel_size - 1) * self.dilation
            if trim > 0:
                out = out[..., :-trim]
        out = self.drop(self.act(self.norm(out)))
        return x + self.res(out)


class EmbTCN(nn.Module):
    """TCN input-embedding: D -> d_model with stacked dilated residual blocks."""

    def __init__(self, cfg: EmbTCNConfig) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(cfg.num_features, cfg.d_model, 1)
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(cfg.d_model, cfg.kernel_size, d, cfg.dropout, cfg.causal)
            for d in cfg.tcn_dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, D, T] -> [B, d_model, T]
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return h


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, d_model]
        return x + self.pe[:, : x.shape[1]]


class DualBranchAttention(nn.Module):
    """Temporal attention (with temperature) + channel attention (SE), fused.

    Operates on ``[B, T, C]``. Paper A, Eqs. 15 + SE block.
    """

    def __init__(self, d_model: int, temperature: float, dropout: float) -> None:
        super().__init__()
        self.temperature = temperature
        self.w_h = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, 1, bias=False)
        se_hidden = max(d_model // 8, 8)
        self.se = nn.Sequential(
            nn.Linear(d_model, se_hidden), nn.ReLU(), nn.Linear(se_hidden, d_model)
        )
        self.proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ):  # [B, T, C]
        # temporal attention: per-frame scalar weight
        e = self.v(torch.tanh(self.w_h(h))).squeeze(-1)       # [B, T]
        if padding_mask is not None:
            e = e.masked_fill(padding_mask, -1e9)
        alpha = torch.softmax(e / self.temperature, dim=1)    # [B, T]
        if padding_mask is not None:
            alpha = alpha.masked_fill(padding_mask, 0.0)
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
        h_temp = h * alpha.unsqueeze(-1)                      # reweight time
        # channel attention (squeeze-excitation)
        if padding_mask is None:
            pooled = h.mean(dim=1)
        else:
            valid = (~padding_mask).float().unsqueeze(-1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        s = torch.sigmoid(self.se(pooled))                    # [B, C]
        h_chan = h * s.unsqueeze(1)                           # reweight channels
        fused = self.proj(torch.cat([h_temp, h_chan], dim=-1))
        out = self.norm(h + self.drop(fused))
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return out, alpha, s


class RefinementStage(nn.Module):
    """One MS-TCN refinement stage: K-channel prediction -> refined K-channel logits.

    Reads the (sigmoid for multilabel / softmax for multiclass) prediction of the
    previous stage and passes it through a stack of centered dilated residual blocks,
    so each stage cleans the temporal structure of the one before it.
    """

    def __init__(self, num_classes: int, hidden: int, dilations: tuple,
                 dropout: float, multilabel: bool) -> None:
        super().__init__()
        self.multilabel = multilabel
        self.in_proj = nn.Conv1d(num_classes, hidden, 1)
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(hidden, 3, d, dropout, causal=False) for d in dilations
        ])
        self.out = nn.Conv1d(hidden, num_classes, 1)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:  # [B, K, T] -> [B, K, T]
        prob = torch.sigmoid(logits) if self.multilabel else torch.softmax(logits, dim=1)
        h = self.in_proj(prob)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


class EmbTCNAttention(nn.Module):
    """Full backbone + heads."""

    def __init__(self, cfg: EmbTCNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_features = cfg.num_features
        self.num_classes = cfg.num_classes
        self.embedding_dim = cfg.embedding_dim
        self.multilabel = cfg.multilabel
        self.embtcn = EmbTCN(cfg)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        if cfg.num_encoder_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.num_heads,
                dim_feedforward=cfg.ffn_mult * cfg.d_model, dropout=cfg.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, cfg.num_encoder_layers)
        else:
            self.encoder = nn.Identity()
        self.dba = DualBranchAttention(cfg.d_model, cfg.temperature, cfg.dropout)
        # multi-scale fusion of {H_tcn, H_enc, H_dba}
        self.fuse = nn.Linear(3 * cfg.d_model, cfg.embedding_dim)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))

        if cfg.use_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(cfg.embedding_dim, cfg.d_model), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.num_features),
            )
        if cfg.use_fault_head:
            self.fault_head = nn.Linear(cfg.embedding_dim, 1)
        if cfg.use_supervised_head:
            self.cls_head = nn.Linear(cfg.embedding_dim, cfg.num_classes)
            self.refiners = nn.ModuleList([
                RefinementStage(cfg.num_classes, cfg.refinement_hidden,
                                tuple(cfg.refinement_dilations), cfg.dropout, cfg.multilabel)
                for _ in range(max(int(cfg.num_refinement_stages), 0))
            ])

    @classmethod
    def from_model_config(cls, payload: dict) -> "EmbTCNAttention":
        return cls(EmbTCNConfig.from_dict(payload))

    def encode(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-frame embedding Z as ``[B, E, T]``.

        ``mask`` (``[B, T]`` bool, True = masked) replaces masked frames with the
        learned mask token after the TCN embedding (for SSL).
        ``padding_mask`` (``[B, T]`` bool, True = padded) is passed to self-attention.
        """
        h_tcn = self.embtcn(x)                      # [B, d_model, T]
        h = h_tcn.transpose(1, 2)                   # [B, T, d_model]
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        h = self.pos(h)
        if isinstance(self.encoder, nn.Identity):
            h_enc = self.encoder(h)
        else:
            h_enc = self.encoder(h, src_key_padding_mask=padding_mask)  # [B, T, d_model]
        if padding_mask is not None:
            h_enc = h_enc.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        h_dba, alpha, gate = self.dba(h_enc, padding_mask)  # [B, T, d_model]
        fused = self.fuse(torch.cat([h_tcn.transpose(1, 2), h_enc, h_dba], dim=-1))
        z = fused.transpose(1, 2)                   # [B, E, T]
        if padding_mask is not None:
            z = z.masked_fill(padding_mask.unsqueeze(1), 0.0)
        self._last_alpha = alpha
        self._last_gate = gate
        return z

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> EmbTCNOutput:
        z = self.encode(x, mask, padding_mask)      # [B, E, T]
        zt = z.transpose(1, 2)                      # [B, T, E]
        out = EmbTCNOutput(
            embeddings=z,
            temporal_weights=getattr(self, "_last_alpha", None),
            channel_gate=getattr(self, "_last_gate", None),
        )
        if self.cfg.use_decoder:
            out.reconstruction = self.decoder(zt).transpose(1, 2)   # [B, D, T]
            if padding_mask is not None:
                out.reconstruction = out.reconstruction.masked_fill(
                    padding_mask.unsqueeze(1), 0.0
                )
        if self.cfg.use_fault_head:
            out.fault = self.fault_head(zt).squeeze(-1)             # [B, T]
            if padding_mask is not None:
                out.fault = out.fault.masked_fill(padding_mask, 0.0)
        if self.cfg.use_supervised_head:
            logits = self.cls_head(zt).transpose(1, 2)              # [B, K, T]
            if padding_mask is not None:
                logits = logits.masked_fill(padding_mask.unsqueeze(1), 0.0)
            stages = [logits]
            for refiner in self.refiners:
                refined = refiner(stages[-1])
                if padding_mask is not None:
                    refined = refined.masked_fill(padding_mask.unsqueeze(1), 0.0)
                stages.append(refined)
            out.logits = stages[-1]                                 # final refined map
            out.stages = stages                                     # deep supervision
            out.probabilities = torch.sigmoid(out.logits)
        return out

    def forward_time_major(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> EmbTCNOutput:
        """Convenience wrapper for toy/data batches shaped ``[B, T, D]``."""

        return self.forward(x.transpose(1, 2), mask=mask, padding_mask=padding_mask)


# --------------------------------------------------------------------------- #
# Self-supervised objective helpers
# --------------------------------------------------------------------------- #

def make_span_mask(batch: int, length: int, ratio: float, span: int,
                   device=None, generator=None) -> torch.Tensor:
    """Boolean ``[B, T]`` mask of contiguous spans covering ~``ratio`` of frames."""

    mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
    span = max(int(span), 1)
    n_spans = max(int(round(ratio * length / span)), 1)
    for b in range(batch):
        for _ in range(n_spans):
            start = int(torch.randint(0, max(length - span, 1), (1,), generator=generator))
            mask[b, start : start + span] = True
    return mask


def masked_reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor,
                               mask: torch.Tensor,
                               valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """MSE over masked frames only. ``x``/``x_hat`` ``[B, D, T]``, ``mask`` ``[B, T]``."""

    effective = mask if valid_mask is None else (mask & valid_mask)
    m = effective.unsqueeze(1).float()              # [B, 1, T]
    se = ((x - x_hat) ** 2) * m
    return se.sum() / m.sum().clamp_min(1.0) / x.shape[1]


class UncertaintyWeightedLoss(nn.Module):
    """Homoscedastic-uncertainty multi-task weighting (Kendall et al; Paper A Eq. 22)."""

    def __init__(self, num_tasks: int) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, loss in enumerate(losses):
            total = total + 0.5 * torch.exp(-2 * self.log_sigma[i]) * loss + self.log_sigma[i]
        return total


def apply_channel_dropout(
    x: torch.Tensor,
    probability: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop full feature channels in each sample, used for SSL robustness."""

    if probability <= 0:
        return x
    keep = torch.rand(
        x.shape[0],
        x.shape[1],
        1,
        device=x.device,
        generator=generator,
    ) > probability
    return x * keep.to(dtype=x.dtype)


def inject_synthetic_faults(
    x: torch.Tensor,
    probability: float = 0.15,
    noise_std: float = 0.25,
    channel_dropout: float = 0.35,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject noisy/channel-dropped frames and return corrupted input + frame targets."""

    if probability <= 0:
        return x, torch.zeros(x.shape[0], x.shape[-1], dtype=torch.float32, device=x.device)
    frame_mask = torch.rand(
        x.shape[0],
        x.shape[-1],
        device=x.device,
        generator=generator,
    ) < probability
    corrupted = x.clone()
    noise = noise_std * torch.randn(corrupted.shape, device=x.device, generator=generator)
    corrupted = torch.where(frame_mask.unsqueeze(1), corrupted + noise, corrupted)
    if channel_dropout > 0:
        drop = torch.rand(
            corrupted.shape,
            device=x.device,
            generator=generator,
        ) < (channel_dropout * frame_mask.unsqueeze(1).float())
        corrupted = corrupted.masked_fill(drop, 0.0)
    return corrupted, frame_mask.float()
