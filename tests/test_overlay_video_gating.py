"""Regression: the overlay (Rec MP4) sidecar video must survive the fast mask-geometry path.

The bug: enabling Tracking with keypoint_source=='mask_geometry' + closed_loop_fast (the
default) force-unchecked Rec MP4 and forced output_masks off, so the overlay recorder was
never started (or was starved), producing no file while the UI reported "saved". These
tests pin that:
  * output_masks is True whenever a mask-bearing output (show masks / COCO / overlay video)
    is requested, and False only on the pure-speed path;
  * enabling Tracking no longer un-checks Rec MP4 or hides masks when the user asked to
    record the overlay video, while still suppressing masks when nothing needs them.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYKABOO_SHOW_SIMULATED_CAMERAS", "1")

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    pass

QApplication = pytest.importorskip("PySide6.QtWidgets").QApplication


@pytest.fixture(scope="module")
def window():
    app = QApplication.instance() or QApplication([])
    try:
        import main_window_enhanced as mwe
        w = mwe.MainWindow()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"MainWindow could not be constructed headless: {exc}")
    w._startup_autoconnect_done = True
    yield w
    try:
        w.close()
    except Exception:
        pass


def _set_source(panel, source, fast):
    panel.combo_keypoint_source.setCurrentIndex(panel.combo_keypoint_source.findData(source))
    panel.check_closed_loop_fast.setChecked(fast)


def _overlay(panel, *, overlay_video=False, masks=False, coco=False):
    panel.set_overlay_options(
        save_overlay_video=overlay_video, show_masks=masks, show_boxes=True,
        show_keypoints=True, save_tracking_csv=True, save_masks_coco=coco, mask_opacity=0.18,
    )


def test_output_masks_forced_on_when_overlay_video_requested(window):
    _set_source(window.live_detection_panel, "mask_geometry", True)
    _overlay(window.live_detection_panel, overlay_video=True, masks=True)
    assert window._build_live_inference_config().output_masks is True


def test_output_masks_forced_on_when_coco_requested(window):
    _set_source(window.live_detection_panel, "mask_geometry", True)
    _overlay(window.live_detection_panel, overlay_video=False, masks=False, coco=True)
    assert window._build_live_inference_config().output_masks is True


def test_output_masks_off_on_pure_speed_path(window):
    _set_source(window.live_detection_panel, "mask_geometry", True)
    _overlay(window.live_detection_panel, overlay_video=False, masks=False, coco=False)
    assert window._build_live_inference_config().output_masks is False


def test_tracking_keeps_rec_mp4_and_masks_when_overlay_requested(window):
    w = window
    _set_source(w.live_detection_panel, "mask_geometry", True)
    _overlay(w.live_detection_panel, overlay_video=True, masks=True)
    w.live_detection_panel.edit_checkpoint.setText("models/checkpoint_best_total.engine")

    class _StubWorker:
        def start_inference(self, *a, **k):
            pass

    w.live_inference_worker = _StubWorker()
    w.live_detection_enabled = True  # take the no-preview/no-model branch
    w.live_tracking_mode_active = False
    w._on_tracking_mode_toggled(True)

    cfg = w.live_detection_panel.detection_config()
    assert cfg.get("save_overlay_video") is True   # Rec MP4 not silently un-checked
    assert cfg.get("show_masks") is True            # masks stay visible to paint the overlay
    assert w._should_save_live_overlay_video() is True


def test_tracking_still_suppresses_masks_on_pure_speed_path(window):
    w = window
    _set_source(w.live_detection_panel, "mask_geometry", True)
    _overlay(w.live_detection_panel, overlay_video=False, masks=True, coco=False)
    w.live_detection_panel.edit_checkpoint.setText("models/checkpoint_best_total.engine")

    class _StubWorker:
        def start_inference(self, *a, **k):
            pass

    w.live_inference_worker = _StubWorker()
    w.live_detection_enabled = True
    w.live_tracking_mode_active = False
    w._on_tracking_mode_toggled(True)
    # No overlay video and no COCO requested -> the fast path may still drop masks for speed.
    assert w.live_detection_panel.detection_config().get("show_masks") is False
