"""VideoPrism embedding backend.

The real backend wraps google-deepmind/videoprism lazily so the rest of the
application can import without JAX/Flax installed. A deterministic mock backend is
kept for tests and GUI plumbing; it is always labelled as mock in outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from .config import AppConfig
from .storage import read_table, write_table

LogFn = Callable[[str], None]
ProgressFn = Callable[[float, str], None]


class VideoPrismUnavailableError(RuntimeError):
    """Raised when the optional VideoPrism/JAX stack is not importable."""


@dataclass
class VideoPrismStatus:
    available: bool
    message: str
    devices: list[str]
    model_name: str


@dataclass
class VideoPrismEmbeddingOutput:
    embeddings: pd.DataFrame
    label_names: list[str]
    labels: np.ndarray
    frames: np.ndarray
    method: str


def embedding_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).startswith("embedding_")]


def videoprism_status(model_name: str = "videoprism_lvt_public_v1_large") -> VideoPrismStatus:
    """Check whether the optional VideoPrism stack is importable."""

    try:
        import jax
        from videoprism import models as vp

        if not vp.has_model(model_name):
            return VideoPrismStatus(False, f"VideoPrism model not found: {model_name}", [], model_name)
        return VideoPrismStatus(
            True,
            f"VideoPrism ready: {model_name}",
            [str(device) for device in jax.devices()],
            model_name,
        )
    except Exception as exc:  # noqa: BLE001
        return VideoPrismStatus(False, str(exc), [], model_name)


def video_metadata(video_path: str | Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    return frames, fps


def sample_frame_indices(
    num_frames: int,
    stride_frames: int,
    max_samples: int,
) -> np.ndarray:
    stride = max(int(stride_frames), 1)
    centers = np.arange(0, max(num_frames, 1), stride, dtype=np.int64)
    if max_samples > 0 and len(centers) > max_samples:
        pick = np.linspace(0, len(centers) - 1, int(max_samples)).astype(np.int64)
        centers = centers[pick]
    return centers


def read_clip(
    capture: cv2.VideoCapture,
    center_frame: int,
    total_frames: int,
    num_clip_frames: int,
    frame_size: int,
) -> np.ndarray:
    half = max(num_clip_frames // 2, 1)
    indices = np.linspace(
        center_frame - half,
        center_frame + half,
        num_clip_frames,
    )
    indices = np.clip(np.rint(indices).astype(np.int64), 0, max(total_frames - 1, 0))
    frames = []
    last = None
    for idx in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = capture.read()
        if not ok or frame is None:
            if last is None:
                frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
            else:
                frame = last.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        last = frame
        frames.append(frame.astype(np.float32) / 255.0)
    return np.stack(frames, axis=0)


class MockVideoPrismEmbedder:
    """Deterministic visual summary backend used for tests and dry runs."""

    def __init__(self, projection_dim: int = 128, seed: int = 42) -> None:
        self.projection_dim = int(projection_dim)
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(0.0, 1.0 / np.sqrt(18), size=(18, self.projection_dim)).astype(np.float32)

    def embed_clips(self, clips: np.ndarray) -> np.ndarray:
        mean = clips.mean(axis=(1, 2, 3))
        std = clips.std(axis=(1, 2, 3))
        first = clips[:, 0].mean(axis=(1, 2))
        last = clips[:, -1].mean(axis=(1, 2))
        diff = last - first
        mid = clips[:, clips.shape[1] // 2]
        h, w = mid.shape[1:3]
        crop = mid[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean(axis=(1, 2))
        features = np.concatenate([mean, std, first, last, diff, crop], axis=1).astype(np.float32)
        out = features @ self.proj
        denom = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(denom, 1e-6, None)


class JaxVideoPrismEmbedder:
    """Thin wrapper around google-deepmind/videoprism."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | None = None,
        log: LogFn | None = None,
    ) -> None:
        try:
            import jax
            import jax.numpy as jnp
            from videoprism import models as vp
        except Exception as exc:  # noqa: BLE001
            raise VideoPrismUnavailableError(
                "VideoPrism is not installed in this Python environment. "
                "Install google-deepmind/videoprism plus jax/flax."
            ) from exc

        self.jax = jax
        self.jnp = jnp
        self.model_name = model_name
        if log:
            log(f"Loading VideoPrism model: {model_name}")
        self.model = vp.get_model(model_name)
        if log:
            log("Loading VideoPrism pretrained weights (first run may download from Hugging Face) ...")
        self.state = vp.load_pretrained_weights(model_name, checkpoint_path=checkpoint_path)

        def forward(inputs):
            if "lvt" in model_name:
                video_embeddings, _text_embeddings, _aux = self.model.apply(
                    self.state,
                    inputs,
                    None,
                    None,
                    train=False,
                )
                return video_embeddings
            outputs, _aux = self.model.apply(self.state, inputs, train=False)
            if outputs.ndim == 3:
                outputs = outputs.mean(axis=1)
            return outputs

        self.forward = jax.jit(forward)

    def embed_clips(self, clips: np.ndarray) -> np.ndarray:
        outputs = self.forward(self.jnp.asarray(clips, dtype=self.jnp.float32))
        out = np.asarray(outputs, dtype=np.float32)
        denom = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(denom, 1e-6, None)


def make_embedder(config: AppConfig, log: LogFn | None = None):
    if config.embedding.backbone == "videoprism_mock":
        return MockVideoPrismEmbedder(
            projection_dim=config.embedding.videoprism_projection_dim,
            seed=config.project.seed,
        )
    return JaxVideoPrismEmbedder(
        config.embedding.videoprism_model_name,
        checkpoint_path=config.embedding.videoprism_checkpoint_path,
        log=log,
    )


def cache_key(video_path: str | Path, config: AppConfig) -> str:
    safe_name = Path(video_path).stem.replace(" ", "_")
    model = (
        "mock"
        if config.embedding.backbone == "videoprism_mock"
        else config.embedding.videoprism_model_name
    )
    model = model.replace("/", "_")
    return (
        f"{safe_name}_{config.embedding.backbone}_{model}"
        f"_f{config.embedding.videoprism_num_frames}"
        f"_s{config.embedding.videoprism_stride_frames}"
        f"_n{config.embedding.videoprism_max_samples_per_video}"
    )


def export_video_embeddings(
    video_path: str | Path,
    config: AppConfig,
    output_dir: str | Path | None = None,
    embedder=None,
    force: bool = False,
    log: LogFn | None = None,
    progress: ProgressFn | None = None,
) -> pd.DataFrame:
    """Export VideoPrism clip embeddings for sampled frames of one video."""

    output_dir = Path(output_dir or config.embedding.videoprism_cache_dir)
    output_path = output_dir / f"{cache_key(video_path, config)}.parquet"
    if not force and (output_path.exists() or output_path.with_suffix(".csv").exists()):
        if log:
            log(f"Loading cached VideoPrism embeddings: {output_path}")
        return read_table(output_path)

    total_frames, fps = video_metadata(video_path)
    centers = sample_frame_indices(
        total_frames,
        config.embedding.videoprism_stride_frames,
        config.embedding.videoprism_max_samples_per_video,
    )
    if log:
        log(f"VideoPrism sampling {len(centers)} clips from {Path(video_path).name}")
    embedder = embedder or make_embedder(config, log=log)
    capture = cv2.VideoCapture(str(video_path))
    rows = []
    batch_clips: list[np.ndarray] = []
    batch_frames: list[int] = []
    batch_size = max(int(config.embedding.videoprism_batch_size), 1)
    for idx, frame_idx in enumerate(centers):
        clip = read_clip(
            capture,
            int(frame_idx),
            total_frames,
            config.embedding.videoprism_num_frames,
            config.embedding.videoprism_frame_size,
        )
        batch_clips.append(clip)
        batch_frames.append(int(frame_idx))
        if len(batch_clips) >= batch_size or idx == len(centers) - 1:
            embs = embedder.embed_clips(np.stack(batch_clips, axis=0))
            for local_frame, emb in zip(batch_frames, embs):
                row = {
                    "video_id": Path(video_path).stem,
                    "video_path": str(video_path),
                    "frame_idx": int(local_frame),
                    "time_s": float(local_frame / fps) if fps > 0 else 0.0,
                    "backbone": config.embedding.backbone,
                    "model_name": (
                        "mock"
                        if config.embedding.backbone == "videoprism_mock"
                        else config.embedding.videoprism_model_name
                    ),
                }
                row.update({f"embedding_{j:04d}": float(v) for j, v in enumerate(emb)})
                rows.append(row)
            batch_clips.clear()
            batch_frames.clear()
            if progress:
                progress(0.95 * (idx + 1) / max(len(centers), 1), f"Embedded {idx + 1}/{len(centers)} clips")
    capture.release()
    df = pd.DataFrame(rows)
    write_table(df, output_path)
    if progress:
        progress(1.0, f"Wrote VideoPrism embeddings: {output_path}")
    return df


def scene_labels_for_frames(
    coco_json: str | Path,
    labels_csv: str | Path | None,
    frames: np.ndarray,
    config: AppConfig,
) -> tuple[list[str], np.ndarray, str]:
    """Align wide manual labels to sampled frame indices as scene-level labels."""

    if not labels_csv:
        return ["background"], np.zeros((len(frames), 1), dtype=np.int8), Path(coco_json).stem
    from .coco_masks import load_coco_videos
    from .labels import build_label_map, load_labels_dataframe
    from .social_pipeline import align_wide_labels_per_identity

    videos = load_coco_videos(str(coco_json), config.data)
    video = max(videos.values(), key=lambda item: item.num_frames)
    labels_df = load_labels_dataframe(
        str(labels_csv),
        config.data.label_format,
        frame_rate=video.fps or config.data.frame_rate,
    )
    label_map = build_label_map(labels_df, config.labels)
    gt_tracks = align_wide_labels_per_identity(
        str(labels_csv),
        video.video_id,
        list(video.frame_indices),
        sorted({record.identity for record in video.records}),
        label_map,
        frame_rate=video.fps or config.data.frame_rate,
        frame_times=np.asarray(video.timestamps) if video.timestamps else None,
    )
    scene = np.zeros((label_map.num_classes, len(video.frame_indices)), dtype=np.int8)
    for arr in gt_tracks.values():
        n = min(scene.shape[1], arr.shape[1])
        scene[:, :n] |= arr[:, :n].astype(np.int8)
    frame_to_pos = {int(frame): pos for pos, frame in enumerate(video.frame_indices)}
    labels = np.zeros((len(frames), label_map.num_classes), dtype=np.int8)
    for idx, frame in enumerate(frames.astype(int)):
        pos = frame_to_pos.get(int(frame))
        if pos is not None and pos < scene.shape[1]:
            labels[idx] = scene[:, pos]
    return label_map.names, labels, video.video_id
