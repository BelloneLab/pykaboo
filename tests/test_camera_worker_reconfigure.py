"""FLIR reconfigure regression: setting OffsetX/resolution must NOT raise the
"Timed out while pausing FLIR acquisition" error when the camera is not actively
streaming (startup / between acquisitions). The pause only waits when streaming."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QApplication = pytest.importorskip("PySide6.QtWidgets").QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _spinnaker_worker(app):
    import camera_worker as cw

    w = cw.CameraWorker()
    w.is_spinnaker_camera = lambda: True
    w.isRunning = lambda: True
    return w


def test_pause_returns_immediately_when_not_streaming(app):
    w = _spinnaker_worker(app)
    w.spinnaker_streaming = False
    t0 = time.time()
    assert w._pause_spinnaker_acquisition_for_reconfigure(timeout_s=3.0) is True
    assert time.time() - t0 < 0.2  # did NOT block for ~3s waiting for an ack


def test_pause_waits_only_when_streaming(app):
    w = _spinnaker_worker(app)
    w.spinnaker_streaming = True
    w.spinnaker_paused = False
    # No acquisition loop to acknowledge -> times out, but ONLY because it is streaming.
    assert w._pause_spinnaker_acquisition_for_reconfigure(timeout_s=0.15) is False
    # Once the loop signals it has paused, the pause succeeds.
    w.spinnaker_paused = True
    assert w._pause_spinnaker_acquisition_for_reconfigure(timeout_s=0.15) is True


def test_pause_noop_when_thread_not_running(app):
    w = _spinnaker_worker(app)
    w.isRunning = lambda: False
    w.spinnaker_streaming = True  # even if a stale flag says streaming
    assert w._pause_spinnaker_acquisition_for_reconfigure(timeout_s=3.0) is True
