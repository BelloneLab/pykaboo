"""The manual head/tail correction must survive a model reload and toggle cleanly.

A user who clicks "Fix head/tail" on a motionless subject should not lose that
correction when the mask-skeleton extractor is rebuilt (a settings change / model
reload), and a second click must undo it.
"""
import numpy as np
import cv2

import live_inference_worker as liw
from live_detection_types import TrackedMouseState


def _tadpole():
    img = np.zeros((200, 420), np.uint8)
    cv2.ellipse(img, (210, 100), (70, 28), 0, 0, 360, 255, -1)
    cv2.fillConvexPoly(img, np.array([[270, 82], [270, 118], [300, 100]], np.int32), 255)
    cv2.line(img, (140, 100), (70, 100), 255, 5)
    return img.astype(bool)


def _mouse():
    return TrackedMouseState(
        mouse_id=1, class_id=0, confidence=0.9, center=(210.0, 100.0),
        bbox=(70.0, 60.0, 300.0, 140.0), mask=_tadpole(), label="m1",
    )


def _nose_right(mouse) -> bool:
    kp = np.asarray(mouse.keypoints).reshape(-1, 2)
    return float(kp[0][0]) > float(kp[7][0])


def _run(worker, cfg, n=3):
    last = None
    for _ in range(n):
        last = _mouse()
        worker._attach_mask_skeleton_keypoints([last], cfg)
    return last


def test_manual_flip_persists_across_extractor_rebuild_and_toggles():
    cfg = liw.LiveInferenceConfig()
    try:
        cfg = cfg.normalized()
    except Exception:
        pass
    worker = liw.LiveInferenceWorker()

    auto = _nose_right(_run(worker, cfg))

    worker.request_flip_orientation(1)
    assert worker._manual_orientation_flips.get(1) == 1
    after_flip = _nose_right(_run(worker, cfg))
    assert after_flip != auto  # the flip took effect

    # Simulate a model reload: the extractor is dropped and rebuilt on next use.
    worker._reset_mask_skeleton_extractor()
    after_reload = _nose_right(_run(worker, cfg))
    assert after_reload == after_flip  # correction persisted across the rebuild

    # A second click toggles the correction back off.
    worker.request_flip_orientation(1)
    assert worker._manual_orientation_flips.get(1) == 0
    after_toggle = _nose_right(_run(worker, cfg))
    assert after_toggle == auto
