"""Self-contained loader + forward runtime for the free-interaction EmbTCN-Attention
behavior model, for real-time (closed-loop) inference.

A free-interaction checkpoint produced by ``behavior_segmentation`` is a plain
``torch.save`` dict carrying everything the runtime needs: the model config, the 432
ordered feature names, the per-class decision thresholds, the global z-score
normalizer (mean/std), the frame rate, the 16 s window, and the post-processing
constants. This module rebuilds the network and exposes the three primitives the
streaming engine needs:

    LiveModel.normalize(raw[T, D])  -> z-scored [T, D]   (plain global z-score)
    LiveModel.forward_window(win[T, D] or batch[B, T, D]) -> probs [.., K, T]
    plus all deploy constants (win, thresholds, labels, look-ahead defaults).

It depends only on the vendored ``behavior_segmentation.models.embtcn_attention``
module + torch + numpy. It does NOT import the training/inference stack.

The model as trained is BIDIRECTIONAL (``causal=False``): a forward pass over a
window lets frame t read frames after t. The streaming engine handles this with a
fixed look-ahead L (see live_engine.py); this module just runs the window.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from behavior_segmentation.models.embtcn_attention import EmbTCNAttention, EmbTCNConfig


def seconds_to_frames(seconds: float, frame_rate: float) -> int:
    """Exact reproduction of behavior_segmentation.windows.seconds_to_frames."""
    return max(int(round(seconds * frame_rate)), 1)


class LiveModel:
    """Loads a free-interaction EmbTCN-AT checkpoint and runs windowed forwards."""

    def __init__(self, ckpt_path: str, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if payload.get("architecture") != "embtcn_attention":
            raise ValueError(
                f"Unsupported architecture {payload.get('architecture')!r}; "
                "this runtime only serves the free-interaction embtcn_attention model."
            )

        self.cfg = EmbTCNConfig.from_dict(payload["model_config"])
        # The decoder/fault heads are SSL-only; skip the decoder to save a little
        # compute per frame. The cls path does not depend on it (verified).
        self.cfg.use_decoder = False
        self.cfg.use_fault_head = False
        self.model = EmbTCNAttention(self.cfg).to(device).eval()
        missing, unexpected = self.model.load_state_dict(payload["model_state"], strict=False)
        # Only decoder/fault tensors may be "missing" because we disabled them.
        bad_missing = [k for k in missing if not (k.startswith("decoder") or k.startswith("fault"))]
        if bad_missing:
            raise RuntimeError(f"checkpoint missing required weights: {bad_missing[:6]}")

        self.feature_names: list[str] = list(payload["feature_names"])
        self.labels: list[str] = list(payload["label_map"]["names"])
        self.background_label: str = payload["label_map"].get("background_label", "background")
        self.name_to_k = {n: i for i, n in enumerate(self.labels)}
        self.background_id = self.name_to_k.get(self.background_label, 0)

        self.thresholds = np.asarray(payload["thresholds"], dtype=np.float64)  # [K]

        norm = payload["normalizer"]
        self.mean = np.asarray(norm["mean"], dtype=np.float64)  # [D]
        self.std = np.asarray(norm["std"], dtype=np.float64)    # [D]
        # Streaming requires the plain global z-score. A checkpoint trained with
        # per-video CORAL / robust stats cannot be streamed (it needs whole-clip
        # statistics). Refuse it loudly rather than serve wrong features.
        if norm.get("video_median") or norm.get("video_mean"):
            raise RuntimeError(
                "This checkpoint's normalizer carries per-video statistics "
                "(CORAL/robust); it cannot be used for streaming inference. "
                "Retrain/deploy a checkpoint fitted with plain global z-score."
            )

        self.frame_rate = float(payload["frame_rate"])
        self.window_seconds = float(payload["window_seconds"])
        self.eval_stride_seconds = float(payload["eval_stride_seconds"])
        self.win = seconds_to_frames(self.window_seconds, self.frame_rate)
        self.eval_stride = seconds_to_frames(self.eval_stride_seconds, self.frame_rate)

        # Offline (non-causal) post-processing constants. The streaming engine
        # replaces these with causal equivalents but reuses the magnitudes.
        self.smooth_win = int(payload["smooth_win"])
        self.min_bout_frames = int(payload["min_bout_frames"])
        self.merge_gap_frames = int(payload["merge_gap_frames"])

        self.causal = bool(payload["model_config"].get("causal", False))
        self.num_classes = self.cfg.num_classes
        self.num_features = self.cfg.num_features
        self.metadata = payload.get("metadata", {})
        if self.num_features != len(self.feature_names):
            raise RuntimeError("model num_features != len(feature_names)")

    # ------------------------------------------------------------------ #
    def normalize(self, raw: np.ndarray) -> np.ndarray:
        """Plain global z-score, byte-equivalent to FeatureNormalizer.transform."""
        out = (raw.astype(np.float64) - self.mean) / self.std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @torch.no_grad()
    def forward_window(self, window: np.ndarray) -> np.ndarray:
        """Run the model on one window or a batch of windows.

        window: [T, D] (single) or [B, T, D] (batch). Already normalized.
        returns: probabilities [K, T] (single) or [B, K, T] (batch).
        """
        single = window.ndim == 2
        if single:
            window = window[None]
        # model expects [B, D, T]
        x = torch.from_numpy(np.ascontiguousarray(np.transpose(window, (0, 2, 1)), dtype=np.float32))
        x = x.to(self.device)
        out = self.model(x)
        probs = out.probabilities.cpu().numpy()  # [B, K, T]
        return probs[0] if single else probs

    def __repr__(self) -> str:
        return (f"LiveModel(features={self.num_features}, classes={self.num_classes}, "
                f"win={self.win}f@{self.frame_rate:.1f}fps, causal={self.causal}, "
                f"device={self.device})")
