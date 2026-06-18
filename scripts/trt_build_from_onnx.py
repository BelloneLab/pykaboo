"""Build a TensorRT engine from an ONNX file. Pure TensorRT, no torch import.

Building in a torch-free process avoids CUDA library (cuDNN/cuBLAS) conflicts
between PyTorch's bundled libs and TensorRT's during tactic profiling, which can
hard-crash the interpreter on Windows.

Usage:
  python trt_build_from_onnx.py --onnx model.onnx --engine model.engine [--fp32] [--workspace-gb 4]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import tensorrt as trt


def build(
    onnx_path: str,
    engine_path: str,
    *,
    fp16: bool,
    workspace_gb: float,
    verbose: bool,
    no_tf32: bool = False,
    opt_level: int = -1,
    tactics: str = "",
) -> int:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    print(f"[trt_build] Parsing ONNX: {onnx_path}", flush=True)
    with open(onnx_path, "rb") as fh:
        if not parser.parse(fh.read()):
            for i in range(parser.num_errors):
                print(f"[trt_build] PARSE ERROR: {parser.get_error(i)}", flush=True)
            return 3

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[trt_build] FP16 enabled", flush=True)
        else:
            print("[trt_build] WARNING: no fast FP16; building FP32", flush=True)
    if no_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
        print("[trt_build] TF32 disabled (true FP32 accumulation)", flush=True)
    if opt_level >= 0:
        config.builder_optimization_level = int(opt_level)
        print(f"[trt_build] builder_optimization_level={opt_level}", flush=True)
    if tactics:
        sources = 0
        names = {t.strip().upper() for t in tactics.split(",") if t.strip()}
        for name in names:
            src = getattr(trt.TacticSource, name, None)
            if src is not None:
                sources |= 1 << int(src)
        config.set_tactic_sources(sources)
        print(f"[trt_build] tactic sources restricted to: {sorted(names)}", flush=True)

    print("[trt_build] Building serialized engine...", flush=True)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[trt_build] build_serialized_network returned None", flush=True)
        return 4
    Path(engine_path).write_bytes(serialized)
    print(
        f"[trt_build] OK: {Path(engine_path).stat().st_size / 1e6:.1f} MB in "
        f"{time.perf_counter() - t0:.1f}s -> {engine_path}",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--workspace-gb", type=float, default=4.0)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--no-tf32", action="store_true", help="Disable TF32 (true FP32 accumulation)")
    ap.add_argument("--opt-level", type=int, default=-1, help="builder_optimization_level 0-5 (-1=default)")
    ap.add_argument("--tactics", default="", help="Comma list of TacticSource to allow, e.g. CUBLAS,CUBLAS_LT,EDGE_MASK_CONVOLUTIONS")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    return build(
        args.onnx,
        args.engine,
        fp16=not args.fp32,
        workspace_gb=args.workspace_gb,
        verbose=args.verbose,
        no_tf32=args.no_tf32,
        opt_level=args.opt_level,
        tactics=args.tactics,
    )


if __name__ == "__main__":
    raise SystemExit(main())
