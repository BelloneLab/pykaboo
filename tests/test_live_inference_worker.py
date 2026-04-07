import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from live_inference_worker import LiveInferenceConfig, LiveInferenceWorker


class LiveInferenceWorkerTests(unittest.TestCase):
    def test_config_normalization_clamps_inference_width(self):
        config = LiveInferenceConfig(inference_max_width=-10, expected_mouse_count=0).normalized()

        self.assertEqual(config.inference_max_width, 0)
        self.assertEqual(config.expected_mouse_count, 1)

    def test_prepare_inference_frame_downscales_when_width_exceeds_limit(self):
        worker = LiveInferenceWorker()
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        resized, scale_x, scale_y = worker._prepare_inference_frame(frame, 100)

        self.assertEqual(resized.shape[:2], (50, 100))
        self.assertAlmostEqual(scale_x, 4.0)
        self.assertAlmostEqual(scale_y, 4.0)

    def test_rescale_detections_restores_boxes_and_masks_to_preview_shape(self):
        worker = LiveInferenceWorker()
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:6, 4:10] = True
        detections = {
            "xyxy": np.asarray([[4.0, 2.0, 10.0, 6.0]], dtype=float),
            "confidence": np.asarray([0.9], dtype=float),
            "class_id": np.asarray([0], dtype=int),
            "mask": np.asarray([mask], dtype=bool),
        }

        result = worker._rescale_detections(
            detections,
            frame_shape=(20, 40),
            scale_x=2.0,
            scale_y=2.0,
        )

        self.assertEqual(result["mask"].shape, (1, 20, 40))
        self.assertTrue(bool(result["mask"][0, 5, 9]))
        self.assertEqual(result["xyxy"].tolist(), [[8.0, 4.0, 20.0, 12.0]])

    def test_optimize_loaded_model_calls_rfdetr_optimization_once(self):
        worker = LiveInferenceWorker()

        class DummyModel:
            def __init__(self):
                self.model = SimpleNamespace(device="cpu")
                self.calls = []

            def optimize_for_inference(self, **kwargs):
                self.calls.append(dict(kwargs))

        model = DummyModel()
        with patch("torch.cuda.is_available", return_value=False):
            optimized = worker._optimize_loaded_model(model, "rfdetr-seg-large")

        self.assertIs(optimized, model)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["batch_size"], 1)
        self.assertFalse(model.calls[0]["compile"])
        self.assertEqual(model.calls[0]["dtype"], torch.float32)


if __name__ == "__main__":
    unittest.main()
