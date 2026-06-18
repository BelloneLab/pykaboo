"""Compile a YOLO pose .pt model to a TensorRT .engine via ultralytics.

ultralytics builds and loads engines natively, so the output .engine sits next
to the .pt and is picked up automatically by LiveInferenceWorker in the
"Max GPU (TensorRT)" mode (it prefers a sibling .engine for the pose model).

Dynamic shapes are used by default so the engine accepts the variable batch
(per-detection crops) and image sizes the live worker feeds it.

Example:
  python scripts/build_yolo_pose_engine.py --pose "D:/.../poseModel_largebest.pt"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True, help="Path to the YOLO pose .pt checkpoint")
    ap.add_argument("--imgsz", type=int, default=640, help="Engine image size (max for dynamic)")
    ap.add_argument("--batch", type=int, default=8, help="Max batch size (for dynamic batching of crops)")
    ap.add_argument("--no-half", action="store_true", help="Build FP32 instead of FP16")
    ap.add_argument("--static", action="store_true", help="Static shapes (no dynamic batch/imgsz)")
    ap.add_argument("--opt-level", type=int, default=1,
                    help="TensorRT builder_optimization_level (default 1; levels 3/5 crash TRT 10.13's builder here)")
    ap.add_argument("--device", default="0", help="CUDA device index")
    args = ap.parse_args()

    pose_path = Path(args.pose)
    if not pose_path.is_file():
        print(f"ERROR: pose checkpoint not found: {pose_path}", flush=True)
        return 2

    # Force a low builder optimization level. ultralytics builds engines at the
    # default (max) level, which hard-crashes TensorRT 10.13's builder on this
    # machine (access violation). Level 1 builds reliably. We patch the build call
    # so ultralytics still writes its own engine + metadata header unchanged.
    if args.opt_level >= 0:
        import tensorrt as trt

        _orig_build = trt.Builder.build_serialized_network

        def _build_low_opt(self, network, config, _orig=_orig_build, _lvl=int(args.opt_level)):
            try:
                config.builder_optimization_level = _lvl
            except Exception:
                pass
            return _orig(self, network, config)

        trt.Builder.build_serialized_network = _build_low_opt
        print(f"Patched TensorRT builder to optimization_level={args.opt_level}", flush=True)

    from ultralytics import YOLO

    model = YOLO(str(pose_path))
    print(f"Exporting {pose_path.name} to TensorRT engine "
          f"(imgsz={args.imgsz}, half={not args.no_half}, dynamic={not args.static}, batch={args.batch})...",
          flush=True)
    out = model.export(
        format="engine",
        half=not args.no_half,
        dynamic=not args.static,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    out_path = Path(out)
    expected = pose_path.with_suffix(".engine")
    if out_path.resolve() != expected.resolve() and out_path.is_file():
        # ultralytics names it <stem>.engine next to the .pt; normalize if needed.
        out_path.replace(expected)
        out_path = expected
    print(f"ENGINE: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
