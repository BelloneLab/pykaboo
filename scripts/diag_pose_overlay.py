"""Diagnose why YOLO-pose keypoints don't appear in the overlay.

Replicates LiveInferenceWorker's non-tracking yolo_pose path on a real frame and
prints keypoint presence/coords/scores after attach, after mask-clamp, and after
tracking, so we can see exactly which stage drops them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from live_detection_types import PreviewFramePacket
from live_inference_worker import LiveInferenceConfig, LiveInferenceWorker


def _kp_stats(records):
    out = []
    for i, r in enumerate(records):
        kp = r.get("keypoints")
        sc = r.get("keypoint_scores")
        if kp is None:
            out.append(f"rec{i}: NO keypoints")
            continue
        kp = np.asarray(kp, dtype=float).reshape(-1, 2)
        finite = np.isfinite(kp).all(axis=1)
        xr = (np.nanmin(kp[:, 0]), np.nanmax(kp[:, 0])) if finite.any() else (np.nan, np.nan)
        yr = (np.nanmin(kp[:, 1]), np.nanmax(kp[:, 1])) if finite.any() else (np.nan, np.nan)
        scr = "none" if sc is None else f"[{np.nanmin(sc):.2f},{np.nanmax(sc):.2f}]"
        out.append(
            f"rec{i}: {int(finite.sum())}/{len(kp)} finite kp, x{tuple(round(v,1) for v in xr)} "
            f"y{tuple(round(v,1) for v in yr)} scores={scr} bbox={tuple(round(v,1) for v in r.get('bbox',()))}"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=150)
    ap.add_argument("--max-width", type=int, default=512)
    ap.add_argument("--accel", default="max_gpu_trt")
    ap.add_argument("--tracking", action="store_true", help="Test the tracking-mode fullframe pose path")
    ap.add_argument("--source-scale", type=float, default=1.0,
                    help="Simulate camera downscaling: source frame = frame * scale (frame_rgb < output)")
    args = ap.parse_args()

    from PySide6.QtCore import QCoreApplication

    QCoreApplication.instance() or QCoreApplication([])

    w = LiveInferenceWorker()
    w.status_changed.connect(lambda m: print("  status:", m, flush=True))
    w.error_occurred.connect(lambda m: print("  ERROR:", m, flush=True))

    w._model = w._load_model("rfdetr-seg-medium", str(Path(args.checkpoint)), acceleration_mode=args.accel)
    w._pose_model = w._load_pose_model(str(Path(args.pose)), acceleration_mode=args.accel)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("could not read frame")
        return 1

    config = LiveInferenceConfig(
        model_key="rfdetr-seg-medium",
        threshold=0.25,
        keypoint_source="yolo_pose",
        pose_threshold=0.25,
        inference_max_width=args.max_width,
        acceleration_mode=args.accel,
    ).normalized()

    meta = {}
    if args.source_scale and args.source_scale != 1.0:
        meta["source_frame_width"] = int(round(frame.shape[1] * args.source_scale))
        meta["source_frame_height"] = int(round(frame.shape[0] * args.source_scale))
    packet = PreviewFramePacket(
        frame=frame, frame_index=0, timestamp_s=0.0,
        width=frame.shape[1], height=frame.shape[0], metadata=meta,
    )

    frame_rgb = w._ensure_rgb(packet.frame)
    inference_frame, sx, sy = w._prepare_inference_frame(frame_rgb, config.inference_max_width)
    ow, oh, osx, osy = w._output_geometry(packet, frame_rgb.shape, sx, sy)
    print(f"frame_rgb={frame_rgb.shape} inference={inference_frame.shape} output=({oh},{ow}) "
          f"scale=({sx:.3f},{sy:.3f}) out_scale=({osx:.3f},{osy:.3f})", flush=True)

    detections = w._predict(w._model, config.model_key, inference_frame, config.threshold)
    normalized = w._normalize_detections(detections)
    print(f"seg detections: {len(normalized.get('xyxy', []))}", flush=True)

    if osx != 1.0 or osy != 1.0:
        normalized = w._rescale_detections(normalized, frame_shape=(oh, ow), scale_x=osx, scale_y=osy)
    records = w._build_detection_records(normalized, config)
    w._clean_record_masks(records, config)
    print(f"records: {len(records)}", flush=True)

    if args.tracking:
        print("PATH: tracking-mode fullframe pose", flush=True)
        pose_result = w._predict_pose_fullframe(inference_frame, config.pose_threshold, config.pose_imgsz_cap)
        print(f"  pose_result is None? {pose_result is None}", flush=True)
        if pose_result is not None:
            kpobj = getattr(pose_result, "keypoints", None)
            kxy = None if kpobj is None else getattr(kpobj, "xy", None)
            print(f"  raw pose keypoints tensor shape: {None if kxy is None else tuple(kxy.shape)}", flush=True)
        w._attach_pose_keypoints_fullframe(
            records, pose_result, scale_x=osx, scale_y=osy,
            pose_threshold=config.pose_threshold, min_confident_kp=config.min_pose_keypoints,
        )
    else:
        print("PATH: non-tracking per-bbox pose", flush=True)
        fh, fw = frame_rgb.shape[:2]
        rtf = (float(fw) / float(max(1, ow)), float(fh) / float(max(1, oh)))
        print(f"  record_to_frame_scale={tuple(round(v,3) for v in rtf)}", flush=True)
        w._attach_pose_keypoints_in_bboxes(
            frame_rgb, records, pose_threshold=config.pose_threshold,
            min_confident_kp=config.min_pose_keypoints, record_to_frame_scale=rtf,
        )
    print("AFTER ATTACH:", flush=True)
    for line in _kp_stats(records):
        print("  ", line, flush=True)

    # mask coordinate space sanity: do record masks match frame_rgb?
    for i, r in enumerate(records):
        m = r.get("mask")
        if m is not None:
            print(f"  rec{i} mask shape={np.asarray(m).shape} frame_rgb={frame_rgb.shape[:2]} "
                  f"match={np.asarray(m).shape[:2] == frame_rgb.shape[:2]}", flush=True)

    w._clamp_record_keypoints(records, config)
    print("AFTER CLAMP (clamp_pose_to_mask=%s):" % config.clamp_pose_to_mask, flush=True)
    for line in _kp_stats(records):
        print("  ", line, flush=True)

    # Match the app: tracker reset to the expected mouse count, smoothing on.
    w._tracker.reset(expected_mice=2)
    w._tracker.smooth_keypoints_enabled = bool(config.smooth_keypoints)
    tracked = w._tracker.update(records)
    print(f"AFTER TRACK (expected_mice=2, smooth={config.smooth_keypoints}): {len(tracked)} mice", flush=True)
    for mtr in tracked:
        kp = getattr(mtr, "keypoints", None)
        n = 0 if kp is None else int(np.isfinite(np.asarray(kp, dtype=float).reshape(-1, 2)).all(axis=1).sum())
        print(f"   mouse {mtr.mouse_id}: bbox={tuple(round(v,1) for v in getattr(mtr,'bbox',()))} keypoints finite={n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
