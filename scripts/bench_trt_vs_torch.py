"""Benchmark the RF-DETR TensorRT engine against the torch inference paths.

Times the network forward (inference_model) only, on the engine's native
resolution, with proper CUDA synchronization. Compares:
  - TensorRT FP16 engine (sidecar .engine next to the checkpoint)
  - torch max_gpu (fp16, the app's current best path)
  - torch compatibility (fp32, reference)

Example:
  python scripts/bench_trt_vs_torch.py \
      --checkpoint "D:/Models/.../checkpoint_best_total.pth" --iters 200
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from rfdetr_trt import RFDETRTensorRTModule, engine_path_for_checkpoint


def _worker():
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.instance() or QCoreApplication([])
    from live_inference_worker import LiveInferenceWorker

    w = LiveInferenceWorker()
    w.status_changed.connect(lambda m: None)
    w.error_occurred.connect(lambda m: print("  err:", m, flush=True))
    return w


def _time(fn, x, iters, warmup=20):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    ms = statistics.fmean(samples) * 1000.0
    p50 = statistics.median(samples) * 1000.0
    return ms, p50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-key", default="rfdetr-seg-medium")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", flush=True)
        return 2

    checkpoint = str(Path(args.checkpoint))
    engine_path = engine_path_for_checkpoint(checkpoint)
    device = "cuda:0"

    results = {}

    # TensorRT engine
    if engine_path.is_file():
        eng = RFDETRTensorRTModule(engine_path, device=device)
        res = eng.resolution
        x = torch.randn(1, 3, res, res, device=device, dtype=torch.float32)
        with torch.no_grad():
            results["TensorRT fp16"] = _time(lambda t: eng(t), x, args.iters)
        del eng
        torch.cuda.empty_cache()
    else:
        print(f"No engine at {engine_path}; build it first.", flush=True)
        res = 432

    # torch paths
    for mode, label, dtype in (("max_gpu", "torch fp16 (max_gpu)", torch.float16),
                               ("compatibility", "torch fp32 (compat)", torch.float32)):
        w = _worker()
        model = w._load_model(args.model_key, checkpoint, acceleration_mode=mode)
        inf = model.model.inference_model
        r = int(getattr(model, "_optimized_resolution", res) or res)
        x = torch.randn(1, 3, r, r, device=device, dtype=dtype)
        with torch.no_grad():
            results[label] = _time(lambda t: inf(t), x, args.iters)
        del model, inf, w
        torch.cuda.empty_cache()

    print("\n=== RF-DETR inference_model latency (batch 1, native res) ===", flush=True)
    base = results.get("torch fp16 (max_gpu)", (None, None))[0]
    for label, (ms, p50) in results.items():
        speed = f" | {base / ms:.2f}x vs torch fp16" if base and ms else ""
        print(f"  {label:24s} mean={ms:6.2f} ms  p50={p50:6.2f} ms  ({1000.0 / ms:5.1f} fps){speed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
