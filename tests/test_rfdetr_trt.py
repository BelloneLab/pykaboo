"""Tests for the RF-DETR TensorRT engine path.

The unit tests run anywhere (no GPU/TensorRT needed). The end-to-end test is
gated on PYKABOO_TRT_CHECKPOINT pointing at a .pth whose sidecar .engine exists,
plus CUDA + the tensorrt package, and is skipped otherwise.
"""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

import rfdetr_trt
from live_inference_worker import LiveInferenceConfig, LiveInferenceWorker


def _tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
    except Exception:
        return False
    return True


class RFDETRTrtUnitTests(unittest.TestCase):
    def test_engine_path_for_checkpoint(self):
        self.assertEqual(
            rfdetr_trt.engine_path_for_checkpoint("foo/bar/model.pth"),
            Path("foo/bar/model.engine"),
        )

    def test_metadata_path_for_engine(self):
        self.assertEqual(
            rfdetr_trt.metadata_path_for_engine("foo/bar/model.engine"),
            Path("foo/bar/model.engine.json"),
        )

    def test_metadata_roundtrip(self):
        with TemporaryDirectory() as tmp:
            engine_path = Path(tmp) / "model.engine"
            engine_path.write_bytes(b"not-a-real-engine")
            meta_path = rfdetr_trt.write_engine_metadata(
                engine_path,
                checkpoint_path="model.pth",
                model_key="rfdetr-seg-medium",
                resolution=432,
                batch_size=1,
                fp16=True,
                output_names=["dets", "labels", "masks"],
                output_shapes={"dets": (1, 200, 4)},
            )
            self.assertTrue(meta_path.is_file())
            data = rfdetr_trt.read_engine_metadata(engine_path)
            self.assertEqual(data["resolution"], 432)
            self.assertEqual(data["model_key"], "rfdetr-seg-medium")
            self.assertEqual(data["output_names"], ["dets", "labels", "masks"])
            # The on-disk JSON is well formed.
            self.assertEqual(json.loads(meta_path.read_text())["batch_size"], 1)

    def test_read_metadata_missing_returns_none(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(rfdetr_trt.read_engine_metadata(Path(tmp) / "absent.engine"))

    def test_config_accepts_max_gpu_trt(self):
        self.assertEqual(
            LiveInferenceConfig(acceleration_mode="max_gpu_trt").normalized().acceleration_mode,
            "max_gpu_trt",
        )
        # Hyphen/space variants normalize too.
        self.assertEqual(
            LiveInferenceConfig(acceleration_mode="max-gpu-trt").normalized().acceleration_mode,
            "max_gpu_trt",
        )

    def test_attach_trt_engine_without_engine_returns_false(self):
        worker = LiveInferenceWorker()
        model = SimpleNamespace(
            optimize_for_inference=lambda **kwargs: None,
            model=SimpleNamespace(device="cuda:0", inference_model=None),
        )
        with TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / "model.pth")  # no sibling .engine
            self.assertFalse(worker._attach_trt_engine(model, "rfdetr-seg-medium", checkpoint, torch))

    def test_attach_trt_engine_empty_checkpoint_returns_false(self):
        worker = LiveInferenceWorker()
        model = SimpleNamespace(optimize_for_inference=lambda **kwargs: None)
        self.assertFalse(worker._attach_trt_engine(model, "rfdetr-seg-medium", "", torch))

    def test_describe_model_reports_tensorrt(self):
        worker = LiveInferenceWorker()
        model = SimpleNamespace(
            model_config=SimpleNamespace(resolution=432, num_queries=200),
            _using_tensorrt=True,
        )
        desc = worker._describe_loaded_model(model, "rfdetr-seg-medium", "max_gpu_trt")
        self.assertIn("TensorRT", desc)

    def test_resolve_engine_only_seg(self):
        worker = LiveInferenceWorker()
        with TemporaryDirectory() as tmp:
            engine = Path(tmp) / "m.engine"
            checkpoint = Path(tmp) / "m.pth"
            # Nothing present yet.
            self.assertIsNone(worker._resolve_engine_only_seg(str(checkpoint)))
            self.assertIsNone(worker._resolve_engine_only_seg(""))
            # An explicit .engine path that exists is used directly.
            engine.write_bytes(b"x")
            self.assertEqual(worker._resolve_engine_only_seg(str(engine)), str(engine))
            # A missing .pth with a sibling .engine falls back to the engine.
            self.assertEqual(worker._resolve_engine_only_seg(str(checkpoint)), str(engine))
            # When the .pth exists, the normal torch path is used (None).
            checkpoint.write_bytes(b"y")
            self.assertIsNone(worker._resolve_engine_only_seg(str(checkpoint)))

    def test_default_models_resolves_to_engine_when_weights_absent(self):
        import default_models

        seg = default_models.default_seg_checkpoint()
        pose = default_models.default_pose_checkpoint()
        # Bundled folder ships only engines; weights are gitignored.
        if seg:
            self.assertTrue(seg.endswith((".pth", ".engine")))
        if pose:
            self.assertTrue(pose.endswith((".pt", ".engine")))


@unittest.skipUnless(
    os.environ.get("PYKABOO_TRT_CHECKPOINT")
    and torch.cuda.is_available()
    and _tensorrt_available(),
    "Set PYKABOO_TRT_CHECKPOINT to a checkpoint with a sidecar .engine (needs CUDA + tensorrt)",
)
class RFDETRTrtEngineTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = os.environ["PYKABOO_TRT_CHECKPOINT"]
        self.engine_path = rfdetr_trt.engine_path_for_checkpoint(self.checkpoint)
        if not self.engine_path.is_file():
            self.skipTest(f"No engine at {self.engine_path}")

    def test_engine_module_runs_and_shapes_match(self):
        eng = rfdetr_trt.RFDETRTensorRTModule(self.engine_path, device="cuda:0")
        self.assertEqual(eng.input_shape[1], 3)
        self.assertEqual(eng.input_shape[2], eng.input_shape[3])
        x = torch.randn(1, 3, eng.resolution, eng.resolution, device="cuda:0")
        out = eng(x)
        # Seg engines emit (dets, labels, masks).
        self.assertGreaterEqual(len(out), 2)
        self.assertEqual(tuple(out[0].shape)[0], 1)
        self.assertEqual(tuple(out[0].shape)[-1], 4)  # boxes are xywh/cxcywh -> 4
        if len(out) >= 3:
            self.assertEqual(out[2].dim(), 4)  # masks [B, Q, Hm, Wm]

    def test_worker_attaches_engine_in_max_gpu_trt(self):
        worker = LiveInferenceWorker()
        model = worker._load_model("rfdetr-seg-medium", self.checkpoint, acceleration_mode="max_gpu_trt")
        self.assertTrue(getattr(model, "_using_tensorrt", False))
        self.assertIsInstance(model.model.inference_model, rfdetr_trt.RFDETRTensorRTModule)
        self.assertEqual(int(model._optimized_resolution), int(model.model.inference_model.resolution))


if __name__ == "__main__":
    unittest.main()
