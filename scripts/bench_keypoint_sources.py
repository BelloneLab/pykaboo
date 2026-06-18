"""Compare live keypoint sources: YOLO pose (+mask) vs mask geometry (+mask).

Runs the production LiveInferenceWorker code paths on real frames and reports,
for each mode, the full per-frame latency plus a breakdown (mask forward, pose
wait / geometry). Saves a JSON summary and a bar-chart PNG.

Usage:
    python scripts/bench_keypoint_sources.py --video <mp4> \
        --mask-checkpoint <pth> --pose-checkpoint <pt> \
        [--model-key rfdetr-seg-medium] [--frames 60] [--width 512] \
        [--acceleration max_gpu] [--out dev_screenshots/keypoint_bench]
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
    while len(frames) < count:
        frames.append(frames[len(frames) % max(1, len(frames))].copy())
    return frames


def summarize(durations: list[float]) -> dict:
    mean_s = statistics.fmean(durations)
    p95_s = sorted(durations)[int(0.95 * (len(durations) - 1))]
    return {
        "mean_ms": mean_s * 1000.0,
        "p95_ms": p95_s * 1000.0,
        "fps": (1.0 / mean_s) if mean_s > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--mask-checkpoint", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--model-key", default="rfdetr-seg-medium")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--pose-threshold", type=float, default=0.25)
    parser.add_argument("--pose-imgsz-cap", type=int, default=640)
    parser.add_argument("--expected-mice", type=int, default=2)
    parser.add_argument("--acceleration", default="max_gpu")
    parser.add_argument("--out", default=str(REPO_ROOT / "dev_screenshots" / "keypoint_bench"))
    args = parser.parse_args()

    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication.instance() or QCoreApplication([])
    _ = app
    from live_inference_worker import LiveInferenceConfig, LiveInferenceWorker

    worker = LiveInferenceWorker()
    worker.status_changed.connect(lambda m: print(f"  [status] {m}"))
    worker.error_occurred.connect(lambda m: print(f"  [error] {m}"))

    frames = load_frames(args.video, args.frames)
    print(f"Loaded {len(frames)} frames {frames[0].shape} from {args.video}")

    def make_config(keypoint_source: str, *, output_masks: bool = True) -> LiveInferenceConfig:
        return LiveInferenceConfig(
            model_key=args.model_key,
            checkpoint_path=args.mask_checkpoint,
            threshold=args.threshold,
            keypoint_source=keypoint_source,
            tracking_mode=True,
            pose_checkpoint_path=args.pose_checkpoint,
            pose_threshold=args.pose_threshold,
            pose_imgsz_cap=args.pose_imgsz_cap,
            inference_max_width=args.width,
            acceleration_mode=args.acceleration,
            expected_mouse_count=args.expected_mice,
            output_masks=output_masks,
        ).normalized()

    config_pose = make_config("yolo_pose")
    config_geom = make_config("mask_geometry")
    config_geom_fast = make_config("mask_geometry", output_masks=False)

    print("Loading mask model...")
    t0 = time.perf_counter()
    worker._model = worker._load_model(
        config_pose.model_key, config_pose.checkpoint_path,
        acceleration_mode=config_pose.acceleration_mode,
    )
    print(f"  mask model loaded in {time.perf_counter() - t0:.1f}s")

    print("Loading pose model...")
    t0 = time.perf_counter()
    worker._pose_model = worker._load_pose_model(
        config_pose.pose_checkpoint_path, acceleration_mode=config_pose.acceleration_mode
    )
    print(f"  pose model loaded in {time.perf_counter() - t0:.1f}s")

    def run_mask(frame, config):
        """Mask forward + normalize + rescale + records + clean. Returns (records, scale_x, scale_y, frame_rgb, mask_ms)."""
        frame_rgb = worker._ensure_rgb(frame)
        inference_frame, scale_x, scale_y = worker._prepare_inference_frame(
            frame_rgb, config.inference_max_width
        )
        t = time.perf_counter()
        detections = worker._predict(worker._model, config.model_key, inference_frame, config.threshold)
        mask_ms = (time.perf_counter() - t) * 1000.0
        normalized = worker._normalize_detections(detections)
        if scale_x != 1.0 or scale_y != 1.0:
            normalized = worker._rescale_detections(
                normalized, frame_shape=frame_rgb.shape[:2], scale_x=scale_x, scale_y=scale_y
            )
        records = worker._build_detection_records(normalized, config)
        worker._clean_record_masks(records, config)
        return records, scale_x, scale_y, frame_rgb, inference_frame, mask_ms

    breakdown: dict = {}

    def bench(label, fn, config, warmup=5):
        worker._tracker.reset(expected_mice=config.expected_mouse_count)
        worker._reset_mask_skeleton_extractor()
        for i in range(warmup):
            fn(frames[i % len(frames)])
        durations, parts = [], []
        det_total, kp_total = 0, 0
        for frame in frames:
            start = time.perf_counter()
            info = fn(frame)
            durations.append(time.perf_counter() - start)
            parts.append(info["parts"])
            det_total += info["dets"]
            kp_total += info["kps"]
        stats = summarize(durations)
        stats["detections"] = det_total
        stats["keypoints"] = kp_total
        # average each breakdown component
        keys = parts[0].keys() if parts else []
        stats["breakdown_ms"] = {k: statistics.fmean([p[k] for p in parts]) for k in keys}
        breakdown[label] = stats
        bd = "  ".join(f"{k} {v:.1f}" for k, v in stats["breakdown_ms"].items())
        print(f"{label:26s} mean {stats['mean_ms']:7.1f} ms | p95 {stats['p95_ms']:7.1f} ms | "
              f"{stats['fps']:6.1f} fps | dets {det_total} kp {kp_total}  [{bd}]")
        return stats

    # ---- mask only ----------------------------------------------------------
    def mask_only(frame):
        records, *_rest, mask_ms = run_mask(frame, config_geom)
        return {"parts": {"mask": mask_ms}, "dets": len(records),
                "kps": sum(1 for r in records if r.get("keypoints") is not None)}
    bench("mask only", mask_only, config_geom)

    # ---- YOLO pose + mask (parallel tracking mode) --------------------------
    def yolo_pose(frame):
        frame_rgb = worker._ensure_rgb(frame)
        inference_frame, scale_x, scale_y = worker._prepare_inference_frame(
            frame_rgb, config_pose.inference_max_width
        )
        pose_future = worker._pose_executor().submit(
            worker._predict_pose_fullframe, inference_frame,
            config_pose.pose_threshold, config_pose.pose_imgsz_cap,
        )
        t = time.perf_counter()
        detections = worker._predict(worker._model, config_pose.model_key, inference_frame, config_pose.threshold)
        mask_ms = (time.perf_counter() - t) * 1000.0
        normalized = worker._normalize_detections(detections)
        if scale_x != 1.0 or scale_y != 1.0:
            normalized = worker._rescale_detections(
                normalized, frame_shape=frame_rgb.shape[:2], scale_x=scale_x, scale_y=scale_y
            )
        records = worker._build_detection_records(normalized, config_pose)
        worker._clean_record_masks(records, config_pose)
        t = time.perf_counter()
        pose_result = pose_future.result(timeout=10.0)
        pose_wait_ms = (time.perf_counter() - t) * 1000.0
        if records and pose_result is not None:
            worker._attach_pose_keypoints_fullframe(
                records, pose_result, scale_x=scale_x, scale_y=scale_y,
                pose_threshold=config_pose.pose_threshold, min_confident_kp=config_pose.min_pose_keypoints,
            )
        worker._clamp_record_keypoints(records, config_pose)
        tracked = worker._tracker.update(records)
        return {"parts": {"mask": mask_ms, "pose_wait": pose_wait_ms},
                "dets": len(tracked),
                "kps": sum(1 for m in tracked if getattr(m, "keypoints", None) is not None)}
    bench("yolo_pose + mask", yolo_pose, config_pose)

    # ---- mask geometry + mask ----------------------------------------------
    def mask_geom(frame):
        frame_rgb = worker._ensure_rgb(frame)
        inference_frame, scale_x, scale_y = worker._prepare_inference_frame(
            frame_rgb, config_geom.inference_max_width
        )
        t = time.perf_counter()
        detections = worker._predict(worker._model, config_geom.model_key, inference_frame, config_geom.threshold)
        mask_ms = (time.perf_counter() - t) * 1000.0
        normalized = worker._normalize_detections(detections)
        records = worker._build_detection_records(normalized, config_geom)
        worker._clean_record_masks(records, config_geom)
        tracked = worker._tracker.update(records)
        t = time.perf_counter()
        worker._attach_mask_skeleton_keypoints(tracked, config_geom)
        geom_ms = (time.perf_counter() - t) * 1000.0
        worker._scale_tracked_states(
            tracked,
            scale_x=scale_x,
            scale_y=scale_y,
            keep_masks=True,
            output_shape=frame_rgb.shape[:2],
        )
        return {"parts": {"mask": mask_ms, "geometry": geom_ms},
                "dets": len(tracked),
                "kps": sum(1 for m in tracked if getattr(m, "keypoints", None) is not None)}
    bench("mask_geometry + mask", mask_geom, config_geom)

    # ---- closed-loop mask geometry -----------------------------------------
    # This is the path that matters for real-time optogenetics: do not emit
    # full-resolution masks when masks are not displayed or exported. Geometry
    # is computed on inference-resolution masks, then centers, boxes, and
    # keypoints are scaled back to the camera coordinate frame.
    def mask_geom_closed_loop(frame):
        frame_rgb = worker._ensure_rgb(frame)
        inference_frame, scale_x, scale_y = worker._prepare_inference_frame(
            frame_rgb, config_geom_fast.inference_max_width
        )
        t = time.perf_counter()
        detections = worker._predict(
            worker._model,
            config_geom_fast.model_key,
            inference_frame,
            config_geom_fast.threshold,
        )
        mask_ms = (time.perf_counter() - t) * 1000.0
        normalized = worker._normalize_detections(detections)
        records = worker._build_detection_records(normalized, config_geom_fast)
        worker._clean_record_masks(records, config_geom_fast)
        tracked = worker._tracker.update(records)
        t = time.perf_counter()
        worker._attach_mask_skeleton_keypoints(tracked, config_geom_fast)
        geom_ms = (time.perf_counter() - t) * 1000.0
        worker._scale_tracked_states(tracked, scale_x=scale_x, scale_y=scale_y, keep_masks=False)
        return {"parts": {"mask": mask_ms, "geometry": geom_ms},
                "dets": len(tracked),
                "kps": sum(1 for m in tracked if getattr(m, "keypoints", None) is not None)}
    bench("mask_geometry closed-loop", mask_geom_closed_loop, config_geom_fast)

    # ---- save outputs -------------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(breakdown, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["mask only", "yolo_pose + mask", "mask_geometry + mask", "mask_geometry closed-loop"]
        labels = [l for l in labels if l in breakdown]
        fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=130)
        fig.patch.set_facecolor("#0d1626")
        ax.set_facecolor("#0d1626")
        colors = {"mask": "#3aa0ff", "pose_wait": "#ff7a45", "geometry": "#6fe06e"}
        bottoms = [0.0] * len(labels)
        for comp in ("mask", "pose_wait", "geometry"):
            vals = [breakdown[l]["breakdown_ms"].get(comp, 0.0) for l in labels]
            if not any(vals):
                continue
            ax.bar(labels, vals, bottom=bottoms, label=comp, color=colors[comp], edgecolor="#0d1626")
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        for i, l in enumerate(labels):
            total = breakdown[l]["mean_ms"]
            ax.text(i, bottoms[i] + 1.0, f"{total:.1f} ms\n{breakdown[l]['fps']:.0f} fps",
                    ha="center", va="bottom", color="#e6eef8", fontsize=10, fontweight="bold")
        ax.set_ylabel("latency per frame (ms)", color="#cfe0f5")
        ax.set_title(f"Live keypoint source latency  ·  {args.model_key} @ {args.width}px  ·  {args.acceleration}",
                     color="#e6eef8", fontsize=11)
        ax.tick_params(colors="#cfe0f5")
        for spine in ax.spines.values():
            spine.set_color("#26344a")
        ax.legend(facecolor="#0b1626", edgecolor="#26344a", labelcolor="#cfe0f5")
        ax.set_ylim(0, max(bottoms) * 1.25 + 5)
        fig.tight_layout()
        chart = out_dir / "keypoint_latency.png"
        fig.savefig(str(chart))
        print(f"\nChart saved: {chart}")
    except Exception as exc:
        print(f"chart skipped: {exc}")

    print(f"JSON saved: {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
