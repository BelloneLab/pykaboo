import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from live_inference_worker import LiveInferenceConfig, LiveInferenceWorker


class LiveInferenceWorkerTests(unittest.TestCase):
    def test_config_normalization_clamps_inference_width(self):
        config = LiveInferenceConfig(
            inference_max_width=-10,
            expected_mouse_count=0,
            acceleration_mode="not-a-mode",
        ).normalized()

        self.assertEqual(config.inference_max_width, 0)
        self.assertEqual(config.expected_mouse_count, 1)
        self.assertEqual(config.acceleration_mode, "balanced")

    def test_config_normalization_accepts_quoted_windows_checkpoint_paths(self):
        config = LiveInferenceConfig(
            checkpoint_path='"C:\\Users\\bellone\\Videos\\ultimate_rfdetr_medium_seg\\rfdetr_medium_seg_best_ema.pth"',
            pose_checkpoint_path="'D:\\Models\\pose\\best.pt'",
        ).normalized()

        self.assertEqual(
            config.checkpoint_path,
            "C:\\Users\\bellone\\Videos\\ultimate_rfdetr_medium_seg\\rfdetr_medium_seg_best_ema.pth",
        )
        self.assertEqual(config.pose_checkpoint_path, "D:\\Models\\pose\\best.pt")

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

    def test_normalize_detections_prefers_yolo_polygons_in_original_shape(self):
        worker = LiveInferenceWorker()
        detections = SimpleNamespace(
            orig_shape=(20, 20),
            boxes=SimpleNamespace(
                xyxy=np.asarray([[5.0, 5.0, 10.0, 10.0]]),
                conf=np.asarray([0.9]),
                cls=np.asarray([0]),
            ),
            masks=SimpleNamespace(
                xy=[np.asarray([[5.0, 5.0], [10.0, 5.0], [10.0, 10.0], [5.0, 10.0]])],
                data=np.zeros((1, 4, 4), dtype=bool),
            ),
        )

        normalized = worker._normalize_detections(detections)

        self.assertEqual(normalized["mask"].shape, (1, 20, 20))
        self.assertTrue(bool(normalized["mask"][0, 7, 7]))
        self.assertFalse(bool(normalized["mask"][0, 2, 2]))

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

    def test_optimize_loaded_model_max_gpu_enables_cuda_fast_paths(self):
        worker = LiveInferenceWorker()

        class DummyModel:
            def __init__(self):
                self.model = SimpleNamespace(device="cuda:0")
                self.calls = []

            def optimize_for_inference(self, **kwargs):
                self.calls.append(dict(kwargs))

        model = DummyModel()
        original_benchmark = torch.backends.cudnn.benchmark
        original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            with patch("torch.cuda.is_available", return_value=True):
                optimized = worker._optimize_loaded_model(
                    model,
                    "rfdetr-seg-medium",
                    acceleration_mode="max_gpu",
                )

            self.assertIs(optimized, model)
            self.assertEqual(len(model.calls), 1)
            self.assertTrue(model.calls[0]["compile"])
            self.assertEqual(model.calls[0]["dtype"], torch.float16)
            self.assertTrue(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cudnn.benchmark = original_benchmark
            torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
            torch.backends.cudnn.allow_tf32 = original_cudnn_tf32

    def test_load_model_supports_rfdetr_seg_small_and_nano(self):
        worker = LiveInferenceWorker()

        class DummyBaseModel:
            def __init__(self, **kwargs):
                self.kwargs = dict(kwargs)
                self.eval_called = False
                self.model_config = SimpleNamespace(num_select=100)
                self.model = SimpleNamespace(
                    args=SimpleNamespace(num_queries=100, num_select=300),
                    postprocess=SimpleNamespace(num_select=300),
                )

            def eval(self):
                self.eval_called = True

        class DummyNano(DummyBaseModel):
            pass

        class DummySmall(DummyBaseModel):
            pass

        fake_rfdetr = SimpleNamespace(
            RFDETRSegNano=DummyNano,
            RFDETRSegSmall=DummySmall,
            RFDETRSegMedium=DummyBaseModel,
            RFDETRSegLarge=DummyBaseModel,
        )

        with patch.dict(sys.modules, {"rfdetr": fake_rfdetr}):
            with patch.object(worker, "_optimize_loaded_model", side_effect=lambda model, _key, **_kwargs: model):
                nano_model = worker._load_model("rfdetr-seg-nano", "")
                small_model = worker._load_model("rfdetr-seg-small", "")

        self.assertIsInstance(nano_model, DummyNano)
        self.assertTrue(nano_model.eval_called)
        self.assertIsInstance(small_model, DummySmall)
        self.assertTrue(small_model.eval_called)
        self.assertEqual(nano_model.model.postprocess.num_select, 100)
        self.assertEqual(nano_model.model.args.num_select, 100)
        self.assertEqual(small_model.model.postprocess.num_select, 100)
        self.assertEqual(small_model.model.args.num_select, 100)

    def test_predict_rfdetr_direct_returns_filtered_numpy_outputs(self):
        worker = LiveInferenceWorker()

        class DummyInferenceModel:
            def __init__(self):
                self.last_shape = None
                self.last_dtype = None

            def __call__(self, batch_tensor):
                self.last_shape = tuple(batch_tensor.shape)
                self.last_dtype = batch_tensor.dtype
                return {
                    "pred_logits": torch.zeros((1, 1, 1), dtype=batch_tensor.dtype),
                    "pred_boxes": torch.zeros((1, 1, 4), dtype=batch_tensor.dtype),
                    "pred_masks": torch.zeros((1, 1, 2, 2), dtype=batch_tensor.dtype),
                }

        class DummyPostprocess:
            def __call__(self, predictions, target_sizes):
                self.predictions = predictions
                self.target_sizes = target_sizes
                return [
                    {
                        "scores": torch.tensor([0.9, 0.2], dtype=torch.float32),
                        "labels": torch.tensor([1, 2], dtype=torch.int64),
                        "boxes": torch.tensor(
                            [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]],
                            dtype=torch.float32,
                        ),
                        "masks": torch.tensor(
                            [
                                [[[1.0, 1.0], [1.0, 1.0]]],
                                [[[0.0, 0.0], [0.0, 0.0]]],
                            ],
                            dtype=torch.float32,
                        ),
                    }
                ]

        inference_model = DummyInferenceModel()
        postprocess = DummyPostprocess()
        model = SimpleNamespace(
            model=SimpleNamespace(
                device=torch.device("cpu"),
                resolution=4,
                postprocess=postprocess,
                inference_model=inference_model,
                model=None,
            ),
            _is_optimized_for_inference=True,
            _optimized_dtype=torch.float32,
            _optimized_resolution=4,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
        )

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        detections = worker._predict_rfdetr_direct(model, frame, 0.5, torch)

        self.assertEqual(inference_model.last_shape, (1, 3, 4, 4))
        self.assertEqual(inference_model.last_dtype, torch.float32)
        self.assertEqual(postprocess.target_sizes.tolist(), [[8, 8]])
        self.assertEqual(detections["xyxy"].tolist(), [[1.0, 2.0, 5.0, 6.0]])
        self.assertAlmostEqual(float(detections["confidence"][0]), 0.9, places=5)
        self.assertEqual(detections["class_id"].tolist(), [1])
        self.assertEqual(detections["mask"].shape, (1, 2, 2))
        self.assertTrue(bool(detections["mask"][0, 0, 0]))

    def test_attach_pose_keypoints_batches_crops_and_offsets_to_image_space(self):
        worker = LiveInferenceWorker()
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        records = [
            {"bbox": (10.0, 15.0, 40.0, 55.0), "center": (25.0, 35.0), "mask": None},
            {"bbox": (60.0, 20.0, 95.0, 70.0), "center": (77.5, 45.0), "mask": None},
        ]

        calls = []

        class DummyPoseModel:
            def predict(self, inputs, **kwargs):
                calls.append((inputs, kwargs))
                return [
                    SimpleNamespace(
                        keypoints=SimpleNamespace(
                            xy=np.asarray([[[5.0, 6.0], [10.0, 14.0]]], dtype=float),
                            conf=np.asarray([[0.9, 0.8]], dtype=float),
                        ),
                        boxes=SimpleNamespace(
                            conf=np.asarray([0.9], dtype=float),
                            xyxy=np.asarray([[2.0, 3.0, 16.0, 20.0]], dtype=float),
                        ),
                    ),
                    SimpleNamespace(
                        keypoints=SimpleNamespace(
                            xy=np.asarray([[[7.0, 8.0], [12.0, 18.0]]], dtype=float),
                            conf=np.asarray([[0.95, 0.9]], dtype=float),
                        ),
                        boxes=SimpleNamespace(
                            conf=np.asarray([0.95], dtype=float),
                            xyxy=np.asarray([[4.0, 5.0, 18.0, 24.0]], dtype=float),
                        ),
                    ),
                ]

        worker._pose_model = DummyPoseModel()
        worker._attach_pose_keypoints_in_bboxes(frame, records, pose_threshold=0.25, min_confident_kp=0)

        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0][0], list)
        self.assertEqual(len(calls[0][0]), 2)
        self.assertIn("imgsz", calls[0][1])
        self.assertEqual(np.asarray(records[0]["keypoints"]).tolist(), [[5.0, 9.0], [10.0, 17.0]])
        self.assertEqual(np.asarray(records[1]["keypoints"]).tolist(), [[55.0, 16.0], [60.0, 26.0]])

    def test_attach_pose_keypoints_prefers_candidate_matching_seg_geometry(self):
        worker = LiveInferenceWorker()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        records = [
            {"bbox": (20.0, 20.0, 80.0, 80.0), "center": (50.0, 50.0), "mask": None},
        ]

        class DummyPoseModel:
            def predict(self, inputs, **kwargs):
                return [
                    SimpleNamespace(
                        keypoints=SimpleNamespace(
                            xy=np.asarray(
                                [
                                    [[2.0, 2.0], [8.0, 8.0]],
                                    [[16.0, 18.0], [58.0, 60.0]],
                                ],
                                dtype=float,
                            ),
                            conf=np.asarray(
                                [
                                    [0.99, 0.99],
                                    [0.55, 0.55],
                                ],
                                dtype=float,
                            ),
                        ),
                        boxes=SimpleNamespace(
                            conf=np.asarray([0.99, 0.55], dtype=float),
                            xyxy=np.asarray(
                                [
                                    [0.0, 0.0, 10.0, 10.0],
                                    [14.0, 16.0, 62.0, 64.0],
                                ],
                                dtype=float,
                            ),
                        ),
                    )
                ]

        worker._pose_model = DummyPoseModel()
        worker._attach_pose_keypoints_in_bboxes(frame, records, pose_threshold=0.25, min_confident_kp=0)

        self.assertEqual(np.asarray(records[0]["keypoints"]).tolist(), [[24.0, 26.0], [66.0, 68.0]])


if __name__ == "__main__":
    unittest.main()
