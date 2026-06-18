"""TensorRT runtime for RF-DETR live inference.

This module loads a serialized TensorRT ``.engine`` built from rfdetr's ONNX
export and exposes a callable that matches the contract of rfdetr's deploy-mode
``inference_model``:

    module(x: torch.Tensor[B, 3, res, res]) -> tuple(dets, labels[, masks])

Because that tuple is exactly what ``LiveInferenceWorker._predict_rfdetr_direct``
already consumes (``predictions[0]`` = boxes, ``[1]`` = logits, ``[2]`` = masks),
the engine drops straight into the existing pre/post-processing path: only the
network forward is replaced, every line of normalization, top-k selection and
mask decoding stays untouched.

Device buffers are plain CUDA torch tensors whose ``data_ptr()`` is handed to
TensorRT, so no pycuda / cuda-python dependency is needed. Inference is enqueued
on torch's current CUDA stream, which keeps it correctly ordered with the
postprocessing kernels that read the outputs.

Engines are hardware-, driver- and TensorRT-version specific. They are NOT
portable: build the engine on the machine that will run it (see
``scripts/build_rfdetr_engine.py``).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import torch

# Output tensor names produced by rfdetr's ONNX export. Detection models export
# (dets, labels); segmentation models add masks. We map by name so the order is
# robust regardless of how TensorRT enumerates the bindings.
CANONICAL_OUTPUT_ORDER = ("dets", "labels", "masks")

ENGINE_SUFFIX = ".engine"
ENGINE_METADATA_SUFFIX = ".engine.json"


def engine_path_for_checkpoint(checkpoint_path: str | Path) -> Path:
    """Return the sidecar ``.engine`` path that pairs with a ``.pth`` checkpoint."""
    path = Path(str(checkpoint_path))
    return path.with_suffix(ENGINE_SUFFIX)


def metadata_path_for_engine(engine_path: str | Path) -> Path:
    """Return the JSON metadata path that pairs with an ``.engine`` file."""
    return Path(str(engine_path)).with_suffix(ENGINE_METADATA_SUFFIX)


def tensorrt_version() -> Optional[str]:
    """Return the installed TensorRT version string, or None if unavailable."""
    try:
        import tensorrt as trt
    except Exception:
        return None
    return str(getattr(trt, "__version__", "")) or None


def _trt_to_torch_dtype(trt, trt_dtype):
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.INT8: torch.int8,
    }
    if hasattr(trt.DataType, "BF16"):
        mapping[trt.DataType.BF16] = torch.bfloat16
    return mapping.get(trt_dtype, torch.float32)


class RFDETRTensorRTModule:
    """Callable wrapper around a serialized RF-DETR TensorRT engine.

    Mimics rfdetr's deploy-mode ``inference_model`` so it can be assigned to
    ``rfdetr_model.model.inference_model`` and driven by the existing direct
    predict path.
    """

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        verbose: bool = False,
    ) -> None:
        import tensorrt as trt

        self._trt = trt
        self.engine_path = str(engine_path)
        if not Path(self.engine_path).is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT inference requires a CUDA device")

        self.device = torch.device(device)
        severity = trt.Logger.VERBOSE if verbose else trt.Logger.ERROR
        self._logger = trt.Logger(severity)
        self._runtime = trt.Runtime(self._logger)

        engine_bytes = Path(self.engine_path).read_bytes()
        self.engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine (version mismatch or corrupt file): {self.engine_path}"
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")

        # Serialize concurrent calls: a single execution context is not reentrant.
        self._lock = threading.Lock()

        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

        if len(self._input_names) != 1:
            raise RuntimeError(
                f"Expected exactly one engine input, got {self._input_names!r} in {self.engine_path}"
            )
        self.input_name = self._input_names[0]

        input_shape = tuple(int(d) for d in self.engine.get_tensor_shape(self.input_name))
        if len(input_shape) != 4 or input_shape[1] != 3 or input_shape[2] != input_shape[3]:
            raise RuntimeError(
                f"Unexpected engine input shape {input_shape!r}; expected [B, 3, res, res] in {self.engine_path}"
            )
        self.input_shape = input_shape
        self.batch_size = int(input_shape[0])
        self.resolution = int(input_shape[-1])
        self.input_dtype = _trt_to_torch_dtype(trt, self.engine.get_tensor_dtype(self.input_name))

        self.context.set_input_shape(self.input_name, input_shape)

        # Order outputs canonically (dets, labels, masks), falling back to binding order.
        ordered = [n for n in CANONICAL_OUTPUT_ORDER if n in self._output_names]
        ordered += [n for n in self._output_names if n not in ordered]
        self.ordered_output_names = ordered
        self.has_masks = "masks" in self._output_names

        self._output_buffers: dict[str, torch.Tensor] = {}
        for name in self._output_names:
            shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
            dtype = _trt_to_torch_dtype(trt, self.engine.get_tensor_dtype(name))
            self._output_buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)

    @property
    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(buf.shape) for name, buf in self._output_buffers.items()}

    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor input, got {type(x)!r}")
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input [B, 3, H, W], got {tuple(x.shape)!r}")
        if x.shape[0] != self.batch_size:
            raise ValueError(
                f"Engine was built for batch size {self.batch_size}, got {x.shape[0]}"
            )
        if x.shape[-2] != self.resolution or x.shape[-1] != self.resolution:
            raise ValueError(
                f"Engine resolution is {self.resolution}x{self.resolution}, got "
                f"{x.shape[-2]}x{x.shape[-1]}"
            )

        x = x.to(device=self.device, dtype=self.input_dtype).contiguous()

        with self._lock:
            self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
            for name, buffer in self._output_buffers.items():
                self.context.set_tensor_address(name, int(buffer.data_ptr()))
            stream = torch.cuda.current_stream(self.device)
            ok = self.context.execute_async_v3(stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned False")

        # Outputs share the same CUDA stream as the postprocessing that consumes
        # them, so stream ordering guarantees the data is ready without a sync.
        return tuple(self._output_buffers[name] for name in self.ordered_output_names)

    # The torch jit-traced inference_model exposes ``.eval``/``.to`` because it is
    # an nn.Module. rfdetr never calls these on the optimized model during the
    # direct predict path, but provide no-op shims so any defensive ``hasattr``
    # checks behave.
    def eval(self):  # noqa: D401 - shim
        return self

    def to(self, *args, **kwargs):  # noqa: D401 - shim
        return self


class RFDETRSegEngineModel:
    """Standalone engine-only RF-DETR segmentation model (no .pth weights needed).

    Runs the serialized ``.engine`` with rfdetr's weightless ``PostProcess`` and
    exposes the exact attribute surface that
    ``LiveInferenceWorker._predict_rfdetr_direct`` reads, so it drops into the
    existing predict path. This lets the app ship/run from just the ``.engine``
    (e.g. when the large ``.pth`` is not present), while still falling back to the
    torch ``.pth`` path elsewhere when that file exists.
    """

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    def __init__(self, engine_path: str | Path, *, device: str | torch.device = "cuda:0", num_select: int | None = None) -> None:
        from types import SimpleNamespace

        engine = RFDETRTensorRTModule(engine_path, device=device)
        try:
            from rfdetr.models import PostProcess
        except Exception:  # pragma: no cover - older rfdetr layout
            from rfdetr.models.postprocess import PostProcess

        dets_shape = engine.output_shapes.get("dets") or next(iter(engine.output_shapes.values()))
        num_queries = int(dets_shape[1]) if len(dets_shape) >= 2 else 300
        if not num_select or int(num_select) <= 0:
            num_select = num_queries
        # Cap candidates: PostProcess upsamples one mask each to full resolution, and
        # a few-animal tracker never needs hundreds. Matches _RFDETR_MAX_NUM_SELECT.
        num_select = min(int(num_select), num_queries, 100)

        dev = torch.device(device)
        self.model = SimpleNamespace(
            inference_model=engine,
            model=None,
            postprocess=PostProcess(num_select=num_select),
            resolution=engine.resolution,
            device=dev,
        )
        self.model_config = SimpleNamespace(resolution=engine.resolution, num_queries=num_queries)
        self.class_names: list[str] = []
        self._is_optimized_for_inference = True
        self._optimized_resolution = engine.resolution
        self._optimized_dtype = engine.input_dtype
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = engine.batch_size
        self._using_tensorrt = True

    def predict(self, *args, **kwargs):  # pragma: no cover - direct path is used
        raise RuntimeError("RFDETRSegEngineModel supports only the direct predict path")

    def eval(self):  # noqa: D401 - shim
        return self


def write_engine_metadata(
    engine_path: str | Path,
    *,
    checkpoint_path: str,
    model_key: str,
    resolution: int,
    batch_size: int,
    fp16: bool,
    output_names: list[str],
    output_shapes: dict[str, tuple[int, ...]],
) -> Path:
    """Write a JSON sidecar describing how the engine was built (for provenance)."""
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "model_key": str(model_key),
        "resolution": int(resolution),
        "batch_size": int(batch_size),
        "fp16": bool(fp16),
        "output_names": list(output_names),
        "output_shapes": {k: list(v) for k, v in output_shapes.items()},
        "tensorrt_version": tensorrt_version(),
        "torch_version": str(torch.__version__),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    meta_path = metadata_path_for_engine(engine_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return meta_path


def read_engine_metadata(engine_path: str | Path) -> Optional[dict]:
    """Read the JSON sidecar for an engine, or None if absent/unreadable."""
    meta_path = metadata_path_for_engine(engine_path)
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
