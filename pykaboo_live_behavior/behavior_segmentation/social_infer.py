"""Save/load the social multi-label model and run inference + embeddings.

A social checkpoint is fully self-contained: model weights + architecture,
feature-column order, normalizer statistics, label map, per-class decision
thresholds, and the postprocessing parameters. That makes inference on a brand
new video a single call with no implicit global state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .labels import LabelMap
from .models.embtcn_attention import EmbTCNAttention, EmbTCNConfig
from .models.temporal_tcn import TemporalTcnModel
from .normalization import FeatureNormalizer
from .social_train import (
    enforce_min_bout,
    merge_short_gaps_binary,
    predict_track_probs,
    smooth_probs,
)
from .windows import seconds_to_frames


@dataclass
class SocialCheckpoint:
    model_config: dict[str, Any]
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
    architecture: str = "temporal_tcn"
    backbone_config: dict[str, Any] | None = None
    graph_adj: Any = None

    def build_model(self, device: str = "cpu"):
        """Rebuild the trained backbone. Self-describes its architecture so the
        same loader serves the MS-TCN social model, the EmbTCN free-interaction
        model, and the wrapped NEMBA backbones (``nemba:<name>``, e.g. the fused
        features+pose model) with no caller changes."""

        if self.architecture.startswith("nemba:"):
            from .free_backbones import NembaSupervised

            cfg = self.backbone_config
            m = NembaSupervised(
                cfg["backbone_name"], cfg["num_features"], cfg["num_classes"],
                embedding_dim=cfg["embedding_dim"], max_len=cfg["max_len"],
                graph_adj=self.graph_adj, channel_bottleneck=cfg.get("channel_bottleneck"),
                **cfg.get("overrides", {})).to(device)
        elif self.architecture == "embtcn_attention":
            m = EmbTCNAttention(EmbTCNConfig.from_dict(self.model_config)).to(device)
        else:
            m = TemporalTcnModel(**self.model_config).to(device)
        m.load_state_dict(self.model_state)
        m.eval()
        return m


def save_social_checkpoint(
    path: str | Path,
    model: TemporalTcnModel,
    feature_names: list[str],
    label_map: LabelMap,
    normalizer: FeatureNormalizer,
    thresholds: np.ndarray,
    frame_rate: float,
    tc,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": {
            "num_features": model.num_features,
            "num_classes": model.num_classes,
            "hidden_channels": tc.hidden_channels,
            "num_stages": tc.num_stages,
            "num_layers_per_stage": tc.num_layers_per_stage,
            "dropout": tc.dropout,
            "embedding_dim": tc.embedding_dim,
            "multilabel": True,
        },
        "feature_names": list(feature_names),
        "label_map": label_map.to_dict(),
        "normalizer": normalizer.to_dict(),
        "thresholds": np.asarray(thresholds, dtype=np.float64).tolist(),
        "frame_rate": float(frame_rate),
        "window_seconds": float(tc.window_seconds),
        "eval_stride_seconds": float(tc.eval_stride_seconds),
        "smooth_win": int(tc.smooth_win),
        "min_bout_frames": int(tc.min_bout_frames),
        "merge_gap_frames": int(getattr(tc, "merge_gap_frames", 0)),
        "metadata": metadata or {},
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, path)
    return path


def runtime_config_for_checkpoint(ckpt: "SocialCheckpoint", base_config):
    """Auto-derive the inference config a checkpoint needs, so a model can be loaded
    and run without the user matching settings by hand.

    Detects free vs aggression from the behaviors (sets the role/interaction mode),
    takes the frame rate from the checkpoint, and enables the feature families the
    checkpoint references in its column names. Returns a deep copy; the base config
    is untouched. Over-enabling a family is harmless (feature alignment drops extra
    columns); the danger is UNDER-building, which silently zero-pads.
    """
    cfg = base_config.model_copy(deep=True)
    label_map = getattr(ckpt, "label_map", None)
    names = [str(n).lower() for n in getattr(label_map, "names", [])]
    feature_names = getattr(ckpt, "feature_names", []) or []
    is_aggression = any(
        "attack" in n or n.startswith("bl6 ") or n.startswith("cd1 ")
        or "reciprocal" in n or "wrestling" in n
        for n in names
    )
    cfg.behavior_roles.interaction_mode = (
        "aggression_cd1_bl6" if is_aggression else "classic_free_interaction"
    )
    if getattr(ckpt, "frame_rate", 0):
        cfg.data.frame_rate = float(ckpt.frame_rate)
    cols = " ".join(str(c).lower() for c in feature_names)
    if "mc_" in cols:
        cfg.features.contact_geometry = True
    if "dy_" in cols or "pp_" in cols:
        cfg.features.dyadic = True
    if "mask_iou" in cols or "mask_overlap" in cols:
        cfg.features.mask_overlap = True
    return cfg


def run_social_inference(
    checkpoint_path,
    coco_json,
    config,
    pose_csv=None,
    device: str = "cpu",
    log=None,
    progress=None,
    should_stop=None,
):
    """Universal inference for ANY social / nemba / main-format checkpoint.

    Returns the same :class:`~behavior_segmentation.infer.InferenceOutputs` the
    Infer tab consumes, so a user can load any checkpoint from any folder and run
    it. Self-configures from the checkpoint (interaction mode / fps / feature
    families) and builds the exact feature surface it was trained on.
    """
    import numpy as np
    import pandas as pd
    import torch

    from .dataset import build_inference_tracks
    from .export import TrackPrediction
    from .free_infer import build_free_inference_tracks, is_fused_feature_names
    from .infer import InferenceOutputs
    from .social_pipeline import build_social_features

    def emit(msg):
        if log:
            log(str(msg))

    # Resolve the device: "auto"/None -> GPU when available, else CPU.
    if device in (None, "auto", ""):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    emit(f"Inference device: {device}")

    if progress:
        progress(0.05, "Loading checkpoint ...")
    ckpt = load_social_checkpoint(checkpoint_path)
    cfg = runtime_config_for_checkpoint(ckpt, config)
    emit(f"Auto-config: {cfg.behavior_roles.interaction_mode} "
         f"@ {cfg.data.frame_rate:.2f} fps, {len(ckpt.feature_names)} features")
    model = ckpt.build_model(device)

    if progress:
        progress(0.15, "Building features (mask + pose + wavelet) ...")
    if str(ckpt.architecture).startswith("nemba:") or is_fused_feature_names(ckpt.feature_names):
        feats, tracks = build_free_inference_tracks(
            coco_json, cfg, ckpt.feature_names, pose_path=pose_csv, log=log)
    else:
        feats = build_social_features(
            coco_json, cfg, pose_path=pose_csv, use_pose=True, use_wavelets=True, log=log)
        tracks = build_inference_tracks(
            feats.identity_df, feats.pair_df, ckpt.feature_names, is_pair=True)
        # safety align: reorder/pad to the checkpoint's exact column order
        for t in tracks:
            name_to_idx = {n: i for i, n in enumerate(t.feature_names)}
            new = np.zeros((t.features.shape[0], len(ckpt.feature_names)), dtype=np.float32)
            for j, name in enumerate(ckpt.feature_names):
                src = name_to_idx.get(name)
                if src is not None:
                    new[:, j] = t.features[:, src]
            t.features = new
            t.feature_names = list(ckpt.feature_names)

    # Raw contact-geometry channels (subject nose -> partner region) the model saw,
    # carried alongside the prediction so the contact-gate rule can suppress
    # impossible/premature contacts (e.g. anogenital while the mice are apart).
    geom_channels = ("pp_nose_tail", "pp_nose_nose", "pp_nose_body")
    fidx = {n: i for i, n in enumerate(ckpt.feature_names)}

    def _geometry_for(track):
        out = {}
        for ch in geom_channels:
            j = fidx.get(f"features:{ch}", fidx.get(ch))
            if j is not None and j < track.features.shape[1]:
                out[ch] = np.asarray(track.features[:, j], dtype=np.float32)
        return out or None

    preds = []
    n = max(len(tracks), 1)
    for i, t in enumerate(tracks):
        if should_stop and should_stop():
            break
        if progress:
            progress(0.3 + 0.6 * i / n, f"Inferring track {t.subject_id} ({i + 1}/{n}) ...")
        sp = infer_track(ckpt, t, model, device, want_embeddings=True)
        emb = (sp.embeddings if sp.embeddings is not None
               else np.zeros((1, len(sp.frame_indices)), dtype=np.float32))
        preds.append(TrackPrediction(
            sp.video_id, sp.subject_id, sp.object_id, sp.frame_indices,
            sp.labels, sp.probabilities, emb, geometry=_geometry_for(t)))

    frames = preds[0].frame_indices if preds else np.array([], dtype=int)
    framewise = pd.DataFrame({"frame_idx": np.asarray(frames)})
    if progress:
        progress(1.0, f"Inference complete ({len(preds)} tracks).")
    return InferenceOutputs(preds, framewise, pd.DataFrame(), pd.DataFrame())


def load_social_checkpoint(path: str | Path) -> SocialCheckpoint:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Tolerant loader: social-format checkpoints carry thresholds + post-proc /
    # windowing fields, but main-pipeline checkpoints (saved by train.py, e.g. the
    # aggression EmbTCN-AT models) do not. Default those so the same Analyze/Correct
    # workflow can run either format. NEMBA-wrapped backbones store a
    # ``backbone_config`` instead of a ``model_config``; everything else matches.
    model_config = payload.get("model_config", {}) or {}
    label_map = LabelMap.from_dict(payload["label_map"])
    thresholds = payload.get("thresholds")
    if thresholds is None:
        thresholds = np.full(label_map.num_classes, 0.5, dtype=np.float64)
    # architecture may live top-level (social) or inside model_config (main).
    architecture = payload.get("architecture") or model_config.get(
        "architecture", "temporal_tcn"
    )
    return SocialCheckpoint(
        model_config=model_config,
        feature_names=payload["feature_names"],
        label_map=label_map,
        normalizer=FeatureNormalizer.from_dict(payload["normalizer"]),
        thresholds=np.asarray(thresholds, dtype=np.float64),
        frame_rate=payload.get("frame_rate", 30.0),
        window_seconds=payload.get("window_seconds", 16.0),
        eval_stride_seconds=payload.get("eval_stride_seconds", 4.0),
        smooth_win=payload.get("smooth_win", 1),
        min_bout_frames=payload.get("min_bout_frames", 0),
        merge_gap_frames=int(payload.get("merge_gap_frames", 0)),
        metadata=payload.get("metadata", {}),
        model_state=payload["model_state"],
        architecture=architecture,
        backbone_config=payload.get("backbone_config"),
        graph_adj=payload.get("graph_adj"),
    )


@dataclass
class SocialPrediction:
    video_id: str
    subject_id: str
    object_id: str
    frame_indices: np.ndarray
    probabilities: np.ndarray  # [K, T]
    labels: np.ndarray  # [K, T] binary after threshold + postprocess
    embeddings: np.ndarray | None  # [E, T]


def infer_track(
    ckpt: SocialCheckpoint,
    track,
    model: TemporalTcnModel | None = None,
    device: str = "cpu",
    want_embeddings: bool = True,
) -> SocialPrediction:
    """Run the social model on one assembled track."""

    model = model or ckpt.build_model(device)
    win = seconds_to_frames(ckpt.window_seconds, ckpt.frame_rate)
    stride = seconds_to_frames(ckpt.eval_stride_seconds, ckpt.frame_rate)
    probs, emb = predict_track_probs(
        model, track, ckpt.normalizer, win, stride, device, want_embeddings
    )
    sp = smooth_probs(probs, ckpt.smooth_win)
    binary = (sp >= ckpt.thresholds[:, None]).astype(np.int8)
    binary = merge_short_gaps_binary(binary, ckpt.merge_gap_frames)
    binary = enforce_min_bout(binary, ckpt.min_bout_frames)
    return SocialPrediction(
        video_id=track.video_id,
        subject_id=track.subject_id,
        object_id=track.object_id,
        frame_indices=np.asarray(track.frame_indices),
        probabilities=probs,
        labels=binary,
        embeddings=emb,
    )


def project_embeddings(
    embeddings: np.ndarray, method: str = "umap", seed: int = 42
) -> np.ndarray:
    """Project ``[N, E]`` embeddings to 2-D for visualization (UMAP or PCA)."""

    X = np.asarray(embeddings, dtype=np.float32)
    if X.shape[0] < 5:
        return np.zeros((X.shape[0], 2), dtype=np.float32)
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed
            )
            return reducer.fit_transform(X).astype(np.float32)
        except Exception:
            method = "pca"
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed).fit_transform(X).astype(np.float32)
