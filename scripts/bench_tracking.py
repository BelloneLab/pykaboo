"""
Benchmark the tracking pipeline: mask (RF-DETR seg) + pose (YOLO) combined.

Measures mask-only, pose-only, sequential (legacy crop path), and parallel
(tracking mode) throughput on real video frames using the production
LiveInferenceWorker code paths.

Usage:
    python scripts/bench_tracking.py --video <mp4> \
        --mask-checkpoint <pth> --pose-checkpoint <pt> [--frames 60] [--width 960]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch  # noqa: F401  (DLL order: torch before PySpin)
except Exception:
    pass

import cv2
import numpy as np


def load_frames(video_path: str, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise SystemExit(f"No frames decoded from {video_path}")
    # Cycle if the clip is shorter than requested.
    while len(frames) < count:
        frames.append(frames[len(frames) % max(1, len(frames))].copy())
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pose-threshold", type=float, default=0.25)
    parser.add_argument("--acceleration", default="balanced")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication([])
    _ = app
    from live_inference_worker import LiveInferenceConfig, LiveInferenceWorker

    worker = LiveInferenceWorker()
    worker.status_changed.connect(lambda message: print(f"  [status] {message}"))
    worker.error_occurred.connect(lambda message: print(f"  [error] {message}"))

    frames = load_frames(args.video, args.frames)
    print(f"Loaded {len(frames)} frames {frames[0].shape} from {args.video}")

    config = LiveInferenceConfig(
        model_key="rfdetr-seg-large",
        checkpoint_path=args.mask_checkpoint,
        threshold=args.threshold,
        pose_checkpoint_path=args.pose_checkpoint,
        pose_threshold=args.pose_threshold,
        inference_max_width=args.width,
        acceleration_mode=args.acceleration,
    ).normalized()

    print("Loading mask model...")
    t0 = time.perf_counter()
    mask_model = worker._load_model(
        config.model_key, config.checkpoint_path, acceleration_mode=config.acceleration_mode
    )
    worker._model = mask_model
    print(f"  mask model loaded in {time.perf_counter() - t0:.1f}s")

    print("Loading pose model...")
    t0 = time.perf_counter()
    worker._pose_model = worker._load_pose_model(
        config.pose_checkpoint_path, acceleration_mode=config.acceleration_mode
    )
    print(f"  pose model loaded in {time.perf_counter() - t0:.1f}s")

    def prepare(frame):
        return worker._prepare_inference_frame(frame, config.inference_max_width)

    def run_mask(inference_frame):
        detections = worker._predict(mask_model, config.model_key, inference_frame, config.threshold)
        return worker._normalize_detections(detections)

    results: dict = {"detections_seen": 0, "keypoints_seen": 0}

    def bench(label, fn, warmup=5):
        for index in range(warmup):
            fn(frames[index % len(frames)])
        durations = []
        for frame in frames:
            start = time.perf_counter()
            fn(frame)
            durations.append(time.perf_counter() - start)
        mean_s = statistics.fmean(durations)
        p95_s = sorted(durations)[int(0.95 * (len(durations) - 1))]
        fps = 1.0 / mean_s if mean_s > 0 else 0.0
        print(f"{label:34s} mean {mean_s*1000:7.1f} ms | p95 {p95_s*1000:7.1f} ms | {fps:6.1f} fps")
        results[label] = {"mean_ms": mean_s * 1000.0, "p95_ms": p95_s * 1000.0, "fps": fps}

    # --- mask only -----------------------------------------------------------
    def mask_only(frame):
        inference_frame, _, _ = prepare(frame)
        run_mask(inference_frame)

    bench("mask only", mask_only)

    # --- pose only (full frame) ----------------------------------------------
    def pose_only(frame):
        inference_frame, _, _ = prepare(frame)
        worker._predict_pose_fullframe(inference_frame, config.pose_threshold)

    bench("pose only (full frame)", pose_only)

    # --- sequential legacy: mask then pose-on-crops ---------------------------
    def sequential(frame):
        inference_frame, scale_x, scale_y = prepare(frame)
        normalized = run_mask(inference_frame)
        if scale_x != 1.0 or scale_y != 1.0:
            normalized = worker._rescale_detections(
                normalized, frame_shape=frame.shape[:2], scale_x=scale_x, scale_y=scale_y
            )
        records = worker._build_detection_records(normalized, config)
        if records:
            worker._attach_pose_keypoints_in_bboxes(
                frame, records, pose_threshold=config.pose_threshold, min_confident_kp=0
            )
        return records

    bench("sequential (mask + crop pose)", sequential)

    # --- parallel tracking mode ------------------------------------------------
    def parallel(frame):
        inference_frame, scale_x, scale_y = prepare(frame)
        pose_future = worker._pose_executor().submit(
            worker._predict_pose_fullframe, inference_frame, config.pose_threshold
        )
        normalized = run_mask(inference_frame)
        if scale_x != 1.0 or scale_y != 1.0:
            normalized = worker._rescale_detections(
                normalized, frame_shape=frame.shape[:2], scale_x=scale_x, scale_y=scale_y
            )
        records = worker._build_detection_records(normalized, config)
        pose_result = pose_future.result(timeout=10.0)
        if records and pose_result is not None:
            worker._attach_pose_keypoints_fullframe(
                records,
                pose_result,
                scale_x=scale_x,
                scale_y=scale_y,
                pose_threshold=config.pose_threshold,
                min_confident_kp=0,
            )
        results["detections_seen"] += len(records)
        results["keypoints_seen"] += sum(1 for record in records if record.get("keypoints") is not None)
        return records

    bench("parallel (tracking mode)", parallel)

    print(
        f"\nDetections across parallel pass: {results['detections_seen']} "
        f"({results['keypoints_seen']} with keypoints)"
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
