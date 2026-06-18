"""Inference: turn a checkpoint + COCO masks into framewise predictions.

Provides :func:`run_inference`, used by both the CLI and the GUI. It extracts
features, validates them against the checkpoint's feature schema, runs
sliding-window inference, postprocesses probabilities into clean bouts, and
returns a list of :class:`~behavior_segmentation.export.TrackPrediction`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .checkpoint import Checkpoint, load_checkpoint
from .config import AppConfig, load_config
from .dataset import build_feature_column_list, build_inference_tracks
from .export import (
    TrackPrediction,
    bouts_dataframe,
    embeddings_dataframe,
    framewise_predictions_dataframe,
)
from .features import extract_video_features
from .coco_masks import load_coco_videos
from .pipeline import (
    _augment_pose_features,
    attach_mask_clips_to_tracks,
    logits_to_probabilities,
    resolve_device,
    sliding_window_inference,
)
from .postprocess import (
    calibrate_thresholds_to_priors,
    postprocess_predictions,
    refine_freezing_by_stationary_motion,
)
from .roles import constrain_probabilities_for_subject
from .storage import write_table

LogFn = Callable[[str], None]
StopFn = Callable[[], bool]


@dataclass
class InferenceOutputs:
    predictions: list[TrackPrediction]
    framewise: pd.DataFrame
    bouts: pd.DataFrame
    embeddings: pd.DataFrame


def run_inference(
    checkpoint: Checkpoint | str | Path,
    coco_json: str | Path,
    config: AppConfig | None = None,
    device: str | None = None,
    log: LogFn | None = None,
    progress: Callable[[float, str], None] | None = None,
    should_stop: StopFn | None = None,
) -> InferenceOutputs:
    """Run inference for every track in a COCO file against a checkpoint."""

    def check_stop() -> None:
        if should_stop and should_stop():
            raise InterruptedError("Inference stopped by user.")

    check_stop()
    if not isinstance(checkpoint, Checkpoint):
        checkpoint = load_checkpoint(checkpoint)
    if config is None:
        config = AppConfig()
    config.data.frame_rate = checkpoint.frame_rate or config.data.frame_rate
    metadata = checkpoint.training_metadata or {}
    for attr in ("lean", "pose", "egocentric", "dyadic", "mask_overlap"):
        key = f"features_{attr}"
        if key in metadata:
            setattr(config.features, attr, bool(metadata[key]))
    if "features_rolling_stat_policy" in metadata:
        config.features.rolling_stat_policy = str(metadata["features_rolling_stat_policy"])
    device = device or resolve_device(config.training.device)

    def emit(fraction: float, message: str) -> None:
        if log:
            log(message)
        if progress:
            progress(fraction, message)

    emit(0.05, f"Loading masks from {Path(coco_json).name} ...")
    check_stop()
    videos = load_coco_videos(coco_json, config.data)
    identity_frames = []
    pair_frames = []
    for video_id, video in videos.items():
        check_stop()
        identity_df, pair_df = extract_video_features(video, config)
        check_stop()
        identity_df, pair_df = _augment_pose_features(
            video, coco_json, identity_df, pair_df, config
        )
        rolling_policy = (
            "none"
            if getattr(config.features, "lean", False)
            else getattr(config.features, "rolling_stat_policy", "full")
        )
        if rolling_policy != "full":
            from .pipeline import apply_rolling_stat_policy

            identity_df = apply_rolling_stat_policy(identity_df, rolling_policy)
            pair_df = apply_rolling_stat_policy(pair_df, rolling_policy)
        identity_frames.append(identity_df)
        if not pair_df.empty:
            pair_frames.append(pair_df)
    check_stop()
    identity_all = pd.concat(identity_frames, ignore_index=True)
    pair_all = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()

    is_pair = checkpoint.task_type == "pair"
    feature_names = build_feature_column_list(identity_all, pair_all, is_pair)
    checkpoint.validate_features(feature_names)

    emit(0.3, "Building inference tracks ...")
    tracks = build_inference_tracks(identity_all, pair_all, feature_names, is_pair)
    if checkpoint.model_config.get("architecture") == "embtcn_mask_video":
        mask_cfg = checkpoint.model_config.get("mask_video") or {}
        attach_mask_clips_to_tracks(
            tracks,
            [coco_json],
            config.data,
            size=int(mask_cfg.get("clip_size", config.mask_video.clip_size)),
            log=lambda message: emit(0.32, message),
            should_stop=should_stop,
        )
    check_stop()
    normalizer = checkpoint.normalizer.with_video_stats_from_tracks(tracks)
    model = checkpoint.build_model(device)
    label_map = checkpoint.label_map
    multilabel = checkpoint.model_config.get("multilabel", False)
    thresholds = checkpoint.model_config.get("thresholds")
    if thresholds is None:
        thresholds = config.inference.confidence_threshold
    tuned_postprocess = checkpoint.model_config.get("postprocess")
    if isinstance(tuned_postprocess, dict):
        config.inference.probability_smoothing_seconds = float(
            tuned_postprocess.get(
                "probability_smoothing_seconds",
                config.inference.probability_smoothing_seconds,
            )
        )
        config.inference.min_bout_frames = int(
            tuned_postprocess.get("min_bout_frames", config.inference.min_bout_frames)
        )
        config.inference.merge_gap_frames = int(
            tuned_postprocess.get("merge_gap_frames", config.inference.merge_gap_frames)
        )
        config.inference.transition_penalty = float(
            tuned_postprocess.get("transition_penalty", config.inference.transition_penalty)
        )
    prior_calibration = checkpoint.model_config.get("prior_calibration")
    if isinstance(prior_calibration, dict):
        config.inference.prior_calibration_strength = float(
            prior_calibration.get(
                "strength",
                config.inference.prior_calibration_strength,
            )
        )
        config.inference.prior_calibration_mode = str(
            prior_calibration.get("mode", config.inference.prior_calibration_mode)
        )
        config.inference.prior_calibration_min_rate = float(
            prior_calibration.get(
                "min_rate",
                config.inference.prior_calibration_min_rate,
            )
        )
    class_priors = checkpoint.model_config.get("class_priors")

    predictions: list[TrackPrediction] = []
    for i, track in enumerate(tracks):
        check_stop()
        logits, embeddings = sliding_window_inference(
            model,
            track,
            normalizer,
            config.inference.window_seconds,
            config.inference.stride_seconds,
            checkpoint.frame_rate,
            device,
            should_stop=should_stop,
        )
        check_stop()
        probabilities = logits_to_probabilities(logits, multilabel)
        probabilities = constrain_probabilities_for_subject(
            probabilities,
            label_map,
            config.behavior_roles,
            track.subject_id,
            multilabel,
        )
        effective_thresholds = thresholds
        if (
            multilabel
            and class_priors is not None
            and config.inference.prior_calibration_strength > 0
        ):
            effective_thresholds = calibrate_thresholds_to_priors(
                probabilities,
                thresholds,
                class_priors,
                label_map.background_id,
                config.inference.prior_calibration_strength,
                config.inference.prior_calibration_mode,
                config.inference.prior_calibration_min_rate,
            )
        labels, smoothed = postprocess_predictions(
            probabilities,
            label_map,
            checkpoint.frame_rate,
            config.inference.probability_smoothing_seconds,
            config.inference.min_bout_frames,
            config.inference.merge_gap_frames,
            effective_thresholds,
            multilabel,
            config.inference.transition_penalty,
        )
        labels = refine_freezing_by_stationary_motion(
            labels,
            smoothed,
            label_map,
            track.features,
            track.feature_names,
            track.frame_indices,
            checkpoint.frame_rate,
            enabled=config.inference.freezing_motion_refinement,
            min_stationary_seconds=config.inference.freezing_stationary_seconds,
            speed_threshold=config.inference.freezing_speed_threshold,
        )
        predictions.append(
            TrackPrediction(
                video_id=track.video_id,
                subject_id=track.subject_id,
                object_id=track.object_id,
                frame_indices=track.frame_indices,
                labels=labels,
                probabilities=smoothed,
                embeddings=embeddings,
            )
        )
        emit(
            0.3 + 0.6 * (i + 1) / max(len(tracks), 1),
            f"Inferred track {i + 1}/{len(tracks)} "
            f"({track.subject_id}{'->' + track.object_id if track.object_id else ''})",
        )

    framewise = framewise_predictions_dataframe(predictions, label_map)
    bouts = bouts_dataframe(predictions, label_map, checkpoint.frame_rate)
    embeddings_df = embeddings_dataframe(predictions, label_map)
    emit(1.0, f"Inference complete: {len(predictions)} tracks.")
    return InferenceOutputs(predictions, framewise, bouts, embeddings_df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run behavior inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--video", default=None, help="Optional video for rendering.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--render", action="store_true", help="Render annotated MP4.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint)
    output_dir = Path(args.output_dir)

    outputs = run_inference(
        checkpoint, args.coco_json, config, log=lambda m: print(m, flush=True)
    )
    write_table(outputs.framewise, output_dir / "frame_predictions.parquet")
    write_table(outputs.bouts, output_dir / "bouts.csv")
    write_table(outputs.embeddings, output_dir / "embeddings.parquet")
    print(f"Wrote predictions to {output_dir}")

    if args.render and args.video:
        from .video_render import render_annotated_video

        render_annotated_video(
            args.video,
            args.coco_json,
            outputs.framewise,
            output_dir / "annotated.mp4",
            config,
        )
        print(f"Rendered annotated video to {output_dir / 'annotated.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
