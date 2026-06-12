"""Sweep seg model / inference-width / compile combinations at the Full HD target.

Frames are resized to 1920x1080 first (the real acquisition target) so the mask
rescale + postprocess cost matches production. Reports end-to-end tracking-mode
fps (mask + pose, parallel) which is what the live overlay actually runs.
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
    import torch  # noqa: F401
except Exception:
    pass

import cv2
import numpy as np


def load_frames(video_path: str, count: int, target_wh: tuple[int, int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, target_wh, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit("no frames")
    while len(frames) < count:
        frames.append(frames[len(frames) % len(frames)].copy())
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pose-checkpoint", required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance() or QCoreApplication([])
    _ = app
    from live_inference_worker import LiveInferenceConfig, LiveInferenceWorker

    frames = load_frames(args.video, args.frames, (1920, 1080))
    print(f"Loaded {len(frames)} frames {frames[0].shape}")

    seg_models = {
        "medium": "D:/Models/Segmentation_models/rfdetr_v3_weights_20260427/medium/checkpoint_best_total.pth",
        "nano": "D:/Models/sam3/pykaboo_project/rf-detr-seg-nano.pt",
        "small": "D:/Models/sam3/pykaboo_project/rf-detr-seg-small.pt",
    }

    results: dict = {}

    def bench(label, fn, warmup=5):
        for i in range(warmup):
            fn(frames[i % len(frames)])
        durs = []
        for fr in frames:
            t = time.perf_counter()
            fn(fr)
            durs.append(time.perf_counter() - t)
        mean_s = statistics.fmean(durs)
        fps = 1.0 / mean_s if mean_s > 0 else 0.0
        print(f"{label:42s} mean {mean_s*1000:7.1f} ms | {fps:6.1f} fps")
        results[label] = {"mean_ms": mean_s * 1000.0, "fps": fps}

    for seg_name, seg_path in seg_models.items():
        if not Path(seg_path).is_file():
            print(f"skip {seg_name}: missing {seg_path}")
            continue
        for width in (640, 768, 960):
            for accel in ("balanced", "max_gpu"):
                worker = LiveInferenceWorker()
                worker.status_changed.connect(lambda m: None)
                worker.error_occurred.connect(lambda m: print("  [err]", m))
                config = LiveInferenceConfig(
                    model_key="rfdetr-seg-medium",
                    checkpoint_path=seg_path,
                    threshold=0.5,
                    pose_checkpoint_path=args.pose_checkpoint,
                    pose_threshold=0.5,
                    inference_max_width=width,
                    tracking_mode=True,
                    acceleration_mode=accel,
                ).normalized()
                try:
                    mask_model = worker._load_model(config.model_key, config.checkpoint_path, acceleration_mode=accel)
                    worker._model = mask_model
                    worker._pose_model = worker._load_pose_model(config.pose_checkpoint_path, acceleration_mode=accel)
                except Exception as exc:
                    print(f"  load failed {seg_name} w{width} {accel}: {exc}")
                    continue

                def parallel(frame, _w=worker, _c=config):
                    inf, sx, sy = _w._prepare_inference_frame(frame, _c.inference_max_width)
                    pf = _w._pose_executor().submit(_w._predict_pose_fullframe, inf, _c.pose_threshold)
                    det = _w._predict(_w._model, _c.model_key, inf, _c.threshold)
                    norm = _w._normalize_detections(det)
                    if sx != 1.0 or sy != 1.0:
                        norm = _w._rescale_detections(norm, frame_shape=frame.shape[:2], scale_x=sx, scale_y=sy)
                    recs = _w._build_detection_records(norm, _c)
                    pr = pf.result(timeout=10.0)
                    if recs and pr is not None:
                        _w._attach_pose_keypoints_fullframe(recs, pr, scale_x=sx, scale_y=sy,
                                                            pose_threshold=_c.pose_threshold, min_confident_kp=0)
                    return recs

                bench(f"{seg_name:6s} w{width} {accel:9s} track", parallel)
                worker.shutdown()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
    # Print a sorted leaderboard.
    print("\n=== leaderboard (fps) ===")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["fps"]):
        print(f"{k:42s} {v['fps']:6.1f} fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
