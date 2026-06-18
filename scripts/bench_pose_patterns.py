"""Benchmark the YOLO pose model in the two ways the worker actually calls it:
  A) full-frame, fixed shape  (tracking mode)
  B) per-detection crop batch, shape changes every frame  (non-tracking mode)

Compares .pt vs .engine to expose dynamic-engine shape-reconfiguration cost.
"""
from __future__ import annotations
import argparse, statistics, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
import torch


def timeit(fn, iters, warm=12):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        s.append(time.perf_counter() - t0)
    return statistics.fmean(s) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--full", type=int, default=512, help="full-frame imgsz (tracking)")
    a = ap.parse_args()

    from ultralytics import YOLO

    pt = Path(a.pose)
    eng = pt.with_suffix(".engine")
    models = [("pt", YOLO(str(pt)), {"half": True})]
    if eng.is_file():
        models.append(("engine", YOLO(str(eng), task="pose"), {}))

    full = np.random.randint(0, 256, (a.full, a.full, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)

    def make_crops():
        n = int(rng.integers(2, 5))
        return [
            np.random.randint(0, 256, (int(rng.integers(90, 300)), int(rng.integers(90, 300)), 3), dtype=np.uint8)
            for _ in range(n)
        ]

    for name, m, extra in models:
        a_ms = timeit(lambda: m.predict(full, imgsz=a.full, conf=0.25, verbose=False, **extra), a.iters)

        def runB():
            crops = make_crops()
            md = max(max(c.shape[0], c.shape[1]) for c in crops)
            imgsz = int(min(640, max(256, int(np.ceil(md / 32.0) * 32))))
            m.predict(crops, imgsz=imgsz, conf=0.25, verbose=False, **extra)

        b_ms = timeit(runB, a.iters)
        # Fixed-size per-crop batch (constant imgsz=640) to isolate shape churn.
        def runC():
            crops = make_crops()
            m.predict(crops, imgsz=640, conf=0.25, verbose=False, **extra)
        c_ms = timeit(runC, a.iters)
        print(f"{name:7s} | full@{a.full}: {a_ms:6.2f} ms | per-crop varying: {b_ms:6.2f} ms | per-crop fixed@640: {c_ms:6.2f} ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
