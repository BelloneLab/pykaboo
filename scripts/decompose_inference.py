"""Decompose the live tracking pipeline cost on the RTX 3060.

Isolates: mask network only, pose at several imgsz, postprocess/rescale, and
whether torch.compile is usable here. Goal: find the path to >30fps.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except Exception:
    torch = None

import cv2
import numpy as np


def load_frames(video, n, wh):
    cap = cv2.VideoCapture(video)
    fr = []
    while len(fr) < n:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, wh, interpolation=cv2.INTER_AREA)
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    while len(fr) < n:
        fr.append(fr[len(fr) % len(fr)].copy())
    return fr


def timeit(label, fn, frames, warmup=8):
    for i in range(warmup):
        fn(frames[i % len(frames)])
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    durs = []
    for f in frames:
        t = time.perf_counter()
        fn(f)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        durs.append(time.perf_counter() - t)
    m = statistics.fmean(durs)
    print(f"{label:38s} {m*1000:7.1f} ms | {1.0/m:6.1f} fps")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--frames", type=int, default=50)
    args = ap.parse_args()

    from PySide6.QtCore import QCoreApplication
    QCoreApplication.instance() or QCoreApplication([])
    from live_inference_worker import LiveInferenceConfig, LiveInferenceWorker

    frames = load_frames(args.video, args.frames, (1920, 1080))
    print(f"frames {frames[0].shape}, cuda={torch.cuda.is_available() if torch else False}")

    w = LiveInferenceWorker()
    w.status_changed.connect(lambda m: None)
    w.error_occurred.connect(lambda m: print("  [err]", m))
    cfg = LiveInferenceConfig(model_key="rfdetr-seg-medium", checkpoint_path=args.seg,
                              threshold=0.5, pose_checkpoint_path=args.pose, pose_threshold=0.5,
                              inference_max_width=768, tracking_mode=True,
                              acceleration_mode="max_gpu").normalized()
    w._model = w._load_model(cfg.model_key, cfg.checkpoint_path, acceleration_mode="max_gpu")
    w._pose_model = w._load_pose_model(cfg.pose_checkpoint_path, acceleration_mode="max_gpu")
    print("models loaded\n--- decomposition (768 inference width, 1080p frames) ---")

    def prep(f):
        return w._prepare_inference_frame(f, cfg.inference_max_width)

    timeit("mask network only", lambda f: w._predict(w._model, cfg.model_key, prep(f)[0], 0.5), frames)

    # Pose full-frame at several imgsz by monkeypatching the imgsz selection.
    for target_imgsz in (960, 768, 640, 512, 416):
        def pose_fn(f, _imgsz=target_imgsz):
            inf = prep(f)[0]
            import numpy as _np
            use_half = bool(torch is not None and torch.cuda.is_available())
            res = w._pose_model.predict(inf, conf=0.5, imgsz=_imgsz, half=use_half, verbose=False)
            return res
        timeit(f"pose full-frame imgsz={target_imgsz}", pose_fn, frames)

    # Full parallel tracking (reference).
    def parallel(f):
        inf, sx, sy = prep(f)
        pf = w._pose_executor().submit(w._predict_pose_fullframe, inf, 0.5)
        det = w._predict(w._model, cfg.model_key, inf, 0.5)
        norm = w._normalize_detections(det)
        if sx != 1.0 or sy != 1.0:
            norm = w._rescale_detections(norm, frame_shape=f.shape[:2], scale_x=sx, scale_y=sy)
        recs = w._build_detection_records(norm, cfg)
        pr = pf.result(timeout=10.0)
        if recs and pr is not None:
            w._attach_pose_keypoints_fullframe(recs, pr, scale_x=sx, scale_y=sy, pose_threshold=0.5, min_confident_kp=0)
        return recs
    timeit("parallel tracking (current)", parallel, frames)

    # torch.compile probe on the seg inference module.
    from live_inference_worker import _triton_nvidia_driver_source_available
    print(f"\ntriton source available: {_triton_nvidia_driver_source_available()}")
    if torch is not None:
        try:
            print("torch version:", torch.__version__, "| cuda:", torch.version.cuda)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
