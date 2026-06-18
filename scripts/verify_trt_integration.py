"""Verify LiveInferenceWorker drives the TensorRT engine end-to-end.

Loads the model in max_gpu_trt mode through the worker, confirms the engine was
attached, and runs the real _predict_rfdetr_direct path on a video frame.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-key", default="rfdetr-seg-medium")
    ap.add_argument("--video", default="")
    ap.add_argument("--frame", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()

    from PySide6.QtCore import QCoreApplication

    QCoreApplication.instance() or QCoreApplication([])
    from live_inference_worker import LiveInferenceWorker
    from rfdetr_trt import RFDETRTensorRTModule

    w = LiveInferenceWorker()
    w.status_changed.connect(lambda m: print("  status:", m, flush=True))
    w.error_occurred.connect(lambda m: print("  ERROR:", m, flush=True))

    model = w._load_model(args.model_key, str(Path(args.checkpoint)), acceleration_mode="max_gpu_trt")

    using_trt = bool(getattr(model, "_using_tensorrt", False))
    inf = getattr(getattr(model, "model", None), "inference_model", None)
    is_engine = isinstance(inf, RFDETRTensorRTModule)
    print(f"_using_tensorrt={using_trt}  inference_model is engine={is_engine}", flush=True)
    print(f"_optimized_resolution={getattr(model, '_optimized_resolution', None)}  "
          f"_optimized_dtype={getattr(model, '_optimized_dtype', None)}", flush=True)
    if not (using_trt and is_engine):
        print("FAIL: engine was not attached", flush=True)
        return 1

    if args.video:
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print("FAIL: could not read frame", flush=True)
            return 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    det = w._predict_rfdetr_direct(model, frame_rgb, args.threshold, torch)
    n = len(det.get("xyxy", []))
    print(f"detections={n}", flush=True)
    if "xyxy" in det and n:
        for i in range(min(n, 5)):
            box = det["xyxy"][i]
            conf = det["confidence"][i]
            has_mask = "mask" in det and det["mask"] is not None and i < len(det["mask"])
            print(f"  det {i}: box={np.round(box, 1).tolist()} conf={conf:.3f} mask={has_mask}", flush=True)
    print("PASS: worker ran inference through the TensorRT engine", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
