"""Recording-fluidity regression: when recording, a saturated processing queue must
drop the oldest frame instead of BLOCKING the capture thread, so a momentary
processing stall never stutters acquisition. Guarded by drop_oldest_on_record."""
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QApplication = pytest.importorskip("PySide6.QtWidgets").QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _worker(app):
    import camera_worker as cw

    w = cw.CameraWorker()
    w.running = True
    w.is_recording = True
    w.camera_type = "virtual"  # non-GigE
    return w, cw.FramePacket


def _tag(FramePacket, idx):
    return FramePacket(backend="virtual", frame=np.zeros((4, 4), np.uint8), metadata={"idx": idx})


def test_drop_oldest_during_record_does_not_block(app):
    w, FramePacket = _worker(app)
    w.drop_oldest_on_record = True
    cap = int(w._get_effective_processing_queue_capacity())
    for i in range(cap):
        w.processing_queue.append(_tag(FramePacket, i))

    t0 = time.time()
    ok = w._enqueue_frame_packet(_tag(FramePacket, 999))
    elapsed = time.time() - t0

    assert ok is True
    assert elapsed < 0.5  # returned promptly (a brief grace wait, then drop) â€” no block
    assert w.acquisition_drop_count == 1
    assert len(w.processing_queue) == cap
    assert w.processing_queue[-1].metadata["idx"] == 999  # newest frame is kept
    assert w.processing_queue[0].metadata["idx"] == 1      # oldest (0) was dropped


def test_gige_path_unchanged_drops_via_processing_counter(app):
    w, FramePacket = _worker(app)
    w.drop_oldest_on_record = True
    # Force the GigE branch: it must keep using processing_queue_drop_count, not the
    # new acquisition counter, so existing GigE behavior is untouched.
    w._is_basler_gige_camera = lambda: True
    cap = int(w._get_effective_processing_queue_capacity())
    for i in range(cap):
        w.processing_queue.append(_tag(FramePacket, i))
    ok = w._enqueue_frame_packet(_tag(FramePacket, 999))
    assert ok is True
    assert w.processing_queue_drop_count == 1
    assert w.acquisition_drop_count == 0


def test_not_recording_still_drops_oldest(app):
    w, FramePacket = _worker(app)
    w.is_recording = False
    cap = int(w._get_effective_processing_queue_capacity())
    for i in range(cap):
        w.processing_queue.append(_tag(FramePacket, i))
    ok = w._enqueue_frame_packet(_tag(FramePacket, 999))
    assert ok is True
    assert w.processing_queue_drop_count == 1  # preview path unchanged
    assert w.acquisition_drop_count == 0
