"""Typed configuration model and YAML loading for the behavior segmentation pipeline.

The configuration is a single nested pydantic model. Every field has a default so
the GUI and the CLIs can construct a fully populated configuration even when the
user supplies only a partial YAML file. Loading supports a simple ``inherits``
key that merges a base YAML file underneath the current one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    seed: int = 42
    output_dir: str = "outputs/default_run"


class DataConfig(BaseModel):
    videos_dir: str = "data/videos"
    coco_jsons: list[str] = Field(default_factory=list)
    labels_csv: str | list[str] | None = None
    label_format: Literal["interval", "framewise"] = "interval"
    frame_rate: float = 30.0
    end_frame_inclusive: bool = True
    identity_field_priority: list[str] = Field(
        default_factory=lambda: [
            "track_id",
            "identity",
            "attributes.identity",
            "attributes.track_id",
            "category_id",
        ]
    )
    video_id_field: str = "video_id"
    frame_index_regex: str = r"frame(\d+)"


class FeaturesConfig(BaseModel):
    normalize_by_frame_size: bool = True
    normalize_by_body_size: bool = True
    egocentric: bool = True
    pose: bool = False                 # fuse single-animal pose features into identity rows
    lean: bool = False                # drop redundant mask rolling statistics
    rolling_stat_policy: Literal["full", "motion_only", "core", "none"] = "motion_only"
    dyadic: bool = False              # egocentric+relational pose features (cross-video)
    mask_overlap: bool = False        # inter-animal mask IoU/contact/overlap-velocity
    contact_geometry: bool = False    # mask contact-location-on-body-axis + occlusion (needs pose)
    parallel_extraction: bool = True  # extract videos' features in parallel (CPU)
    num_workers: int = 0              # 0 = auto (min(#videos, cpu_count))
    include_single_mouse_features: bool = True
    include_pairwise_features: bool = True
    include_arena_features: bool = False
    arena_config: dict[str, Any] | None = None
    rolling_windows_seconds: list[float] = Field(
        default_factory=lambda: [0.17, 0.5, 1.0, 2.0]
    )
    missing_mask_policy: Literal["interpolate_short_gaps", "zero", "keep_nan"] = (
        "interpolate_short_gaps"
    )
    max_interpolation_gap_frames: int = 5
    compute_mask_crop_embeddings: bool = False
    mask_crop_size: int = 96


class LabelsConfig(BaseModel):
    mode: Literal["multiclass", "multilabel"] = "multilabel"
    background_label: str = "background"
    behavior_priority: list[str] = Field(default_factory=list)
    min_bout_frames: int = 3


class BehaviorRolesConfig(BaseModel):
    interaction_mode: Literal[
        "classic_free_interaction",
        "aggression_cd1_bl6",
        "single_mouse",
    ] = "classic_free_interaction"
    bl6_identity: str = "1"
    cd1_identity: str = "2"
    primary_identity: str = "1"
    bl6_behavior_prefixes: list[str] = Field(default_factory=lambda: ["BL6"])
    cd1_behavior_prefixes: list[str] = Field(default_factory=lambda: ["CD1"])
    unprefixed_behaviors_allowed_for_all: bool = True


class ModelConfig(BaseModel):
    architecture: Literal[
        "temporal_tcn",
        "embtcn_attention",
        "embtcn_mask_video",
    ] = "embtcn_attention"
    input_mode: Literal["scalar_features", "scalar_plus_mask_crop"] = "scalar_features"
    hidden_channels: int = 128
    num_stages: int = 3
    num_layers_per_stage: int = 8
    kernel_size: int = 3
    dropout: float = 0.2
    embedding_dim: int = 128
    use_mask_crop_branch: bool = False


class MaskVideoModelConfig(BaseModel):
    in_channels: int = 3
    spatial_channels: list[int] = Field(default_factory=lambda: [16, 32, 64])
    temporal_channels: int = 96
    embedding_dim: int = 64
    temporal_dilations: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    kernel_size: int = 5
    dropout: float = 0.15
    clip_size: int = 48


class EmbTCNConfig(BaseModel):
    d_model: int = 160
    embedding_dim: int = 96
    tcn_dilations: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    kernel_size: int = 5
    num_encoder_layers: int = 4
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.15
    temperature: float = 0.5
    causal: bool = False
    max_len: int = 4096
    num_refinement_stages: int = 0
    refinement_hidden: int = 64
    refinement_dilations: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    mask_ratio: float = 0.15
    mask_span_seconds: float = 0.5
    channel_dropout: float = 0.15
    gaussian_noise_std: float = 0.02
    fault_probability: float = 0.15
    fault_noise_std: float = 0.25
    ssl_objective: Literal["recon", "contrastive", "jepa", "recon+contrastive"] = "recon"
    contrastive_temperature: float = 0.1
    contrastive_radius_frames: int = 2
    vicreg_variance_weight: float = 0.05
    vicreg_covariance_weight: float = 0.005
    jepa_ema_momentum: float = 0.996
    ssl_time_jitter_frames: int = 2
    supervised_ssl_aux_weight: float = 0.0
    ssl_learning_rate: float = 7e-4
    ssl_weight_decay: float = 1e-4
    ssl_max_epochs: int = 50
    ssl_batch_size: int = 8
    warmup_epochs: int = 10
    early_stopping_patience: int = 15
    per_video_robust_normalization: bool = True
    task_type: Literal["single", "pair"] = "single"
    pretrained_checkpoint: str | None = None
    freeze_encoder: bool = False


class TrainingConfig(BaseModel):
    split_by: Literal["video_id", "session", "animal", "experiment"] = "video_id"
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    window_seconds: float = 12.0
    stride_seconds: float = 3.0
    batch_size: int = 8
    max_epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    class_weighting: Literal[
        "none",
        "inverse_frequency",
        "inverse_sqrt_frequency",
        "effective_number",
    ] = "inverse_sqrt_frequency"
    loss: Literal[
        "cross_entropy",
        "focal",
        "bce",
        "soft_f1",
        "dice",
        "tversky",
    ] = "focal"
    focal_gamma: float = 2.0
    overlap_loss_weight: float = 0.0
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    effective_number_beta: float = 0.999
    ohem_fraction: float = 0.0
    support_balanced_sampler: bool = False  # quota+cap sampler (decouple from raw freq)
    support_cap_fraction: float = 0.40
    boundary_tolerance_frames: int = 0
    boundary_soft_target: float = 0.35
    grouped_attack_label: bool = False
    grouped_attack_name: str = "grouped_attack"
    grouped_attack_members: list[str] = Field(
        default_factory=lambda: ["attack", "attack_biting", "attack_wrestling"]
    )
    attack_parent_consistency_weight: float = 0.0
    attack_boundary_oversample_weight: float = 0.0
    threshold_tuning_metric: Literal["frame_f1", "bout_f1"] = "frame_f1"
    bout_tuning_iou_threshold: float = 0.25
    tune_postprocessing: bool = False
    postprocess_tuning_metric: Literal[
        "macro_f1",
        "weighted_f1",
        "bout_macro_f1",
        "bout_weighted_f1",
    ] = "macro_f1"
    validation_metric: Literal[
        "macro_f1",
        "weighted_f1",
        "bout_macro_f1",
        "bout_weighted_f1",
    ] = "weighted_f1"
    postprocess_smoothing_grid: list[float] = [0.0, 0.1, 0.2, 0.4]
    postprocess_min_bout_grid: list[int] = [1, 3, 5, 8]
    postprocess_merge_gap_grid: list[int] = [0, 2, 5, 8]
    postprocess_transition_grid: list[float] = [0.0]
    temporal_smoothing_weight: float = 0.15
    multilabel_temporal_smoothing_weight: float = 0.0
    per_video_robust_normalization: bool = False
    coral_normalization: bool = False
    feature_stat_augmentation: bool = False
    feature_scale_jitter: float = 0.08
    feature_offset_jitter: float = 0.05
    domain_adversary_weight: float = 0.0
    supervised_contrastive_weight: float = 0.0
    bout_transition_weight: float = 0.0
    input_feature_l1_weight: float = 0.0
    early_stopping_patience: int = 15
    gradient_clip_norm: float = 1.0
    device: Literal["auto", "cpu", "cuda"] = "auto"


class InferenceConfig(BaseModel):
    window_seconds: float = 12.0
    stride_seconds: float = 6.0
    probability_smoothing_seconds: float = 0.2
    min_bout_frames: int = 3
    merge_gap_frames: int = 2
    confidence_threshold: float = 0.5
    transition_penalty: float = 0.0
    prior_calibration_strength: float = 0.0
    prior_calibration_mode: Literal["dampen", "match"] = "dampen"
    prior_calibration_min_rate: float = 1e-4
    freezing_motion_refinement: bool = True
    freezing_stationary_seconds: float = 2.0
    freezing_speed_threshold: float = 0.2


class EmbeddingConfig(BaseModel):
    backbone: Literal["embtcn_at", "videoprism", "videoprism_mock"] = "embtcn_at"
    level: Literal["frame", "bout"] = "frame"
    pooling: Literal["none", "mean", "max"] = "none"
    export_format: Literal["parquet", "csv"] = "parquet"
    videoprism_model_name: str = "videoprism_lvt_public_v1_large"
    videoprism_checkpoint_path: str | None = None
    videoprism_frame_size: int = 288
    videoprism_num_frames: int = 8
    videoprism_stride_frames: int = 25
    videoprism_batch_size: int = 1
    videoprism_max_samples_per_video: int = 240
    videoprism_cache_dir: str = "outputs/videoprism_cache"
    videoprism_projection_dim: int = 128


class ClusteringConfig(BaseModel):
    method: Literal["hdbscan", "kmeans", "gmm", "dbscan", "agglomerative"] = "hdbscan"
    umap_components: int = 2
    min_cluster_size: int = 100
    n_clusters: int = 10


class GuiConfig(BaseModel):
    window_title: str = "Behavior Mask Segmentation Workbench"
    default_output_dir: str = "outputs/gui_session"
    overlay_opacity: float = 0.45
    show_identity_labels: bool = True
    show_behavior_labels: bool = True
    show_confidence: bool = True
    timeline_height: int = 220
    playback_fps: float = 30.0
    contour_thickness: int = 2
    font_scale: float = 0.6
    default_behaviors: list[str] = Field(
        default_factory=lambda: [
            "background",
            "locomotion",
            "rearing",
            "grooming",
            "social_contact",
        ]
    )


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    behavior_roles: BehaviorRolesConfig = Field(default_factory=BehaviorRolesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    embtcn: EmbTCNConfig = Field(default_factory=EmbTCNConfig)
    mask_video: MaskVideoModelConfig = Field(default_factory=MaskVideoModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    gui: GuiConfig = Field(default_factory=GuiConfig)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating inputs."""

    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_with_inherits(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, resolving a single optional ``inherits`` base path."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    inherits = payload.pop("inherits", None)
    payload.pop("annotation", None)
    if inherits:
        base_path = (path.parent / inherits).resolve()
        if not base_path.exists():
            base_path = Path(inherits)
        base = load_yaml_with_inherits(base_path)
        payload = deep_merge(base, payload)
    return payload


def load_config(path: str | Path | None = None) -> AppConfig:
    """Build an :class:`AppConfig`, optionally overlaying a YAML file on defaults."""

    if path is None:
        return AppConfig()
    payload = load_yaml_with_inherits(path)
    return AppConfig.model_validate(payload)


def save_config(config: AppConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
