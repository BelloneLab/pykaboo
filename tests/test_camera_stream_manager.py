import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera_stream_manager import camera_identity_key, slugify_stream_suffix


def test_identity_prefers_serial():
    info = {"type": "flir", "backend": "spinnaker", "serial": "23456", "index": 0}
    assert camera_identity_key(info) == "flir:spinnaker:serial=23456"


def test_identity_falls_back_to_video_index_then_index():
    boson = {"type": "flir", "backend": "boson", "video_index": 2, "index": 2}
    assert camera_identity_key(boson) == "flir:boson:video=2"
    usb = {"type": "usb", "index": 1}
    assert camera_identity_key(usb) == "usb::index=1"


def test_identity_distinguishes_two_usb_cameras():
    usb0 = {"type": "usb", "index": 0}
    usb1 = {"type": "usb", "index": 1}
    assert camera_identity_key(usb0) != camera_identity_key(usb1)


def test_identity_handles_virtual_cameras():
    virtual = {"type": "virtual", "backend": "simulated", "serial": "sim-basler-0", "index": 0}
    assert camera_identity_key(virtual) == "virtual:simulated:serial=sim-basler-0"


def test_identity_empty_info():
    assert camera_identity_key(None) == ""
    assert camera_identity_key({}) == ""


def test_slugify_stream_suffix():
    assert slugify_stream_suffix("FLIR Spinnaker: BFS (123)", "cam2") == "flir_spinnaker_bfs_123"
    assert slugify_stream_suffix("", "cam2") == "cam2"
    assert slugify_stream_suffix("___", "cam3") == "cam3"


@pytest.fixture()
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_manager_stream_lifecycle(qapp):
    from camera_stream_manager import CameraStreamManager

    manager = CameraStreamManager()
    assert manager.can_add_stream()
    assert manager.connected_streams() == []
    assert not manager.any_recording()

    streams = []
    for _ in range(CameraStreamManager.MAX_STREAMS):
        stream = manager.create_stream()
        assert stream is not None
        streams.append(stream)

    assert not manager.can_add_stream()
    errors = []
    manager.error_message.connect(errors.append)
    assert manager.create_stream() is None
    assert errors, "creating beyond the limit must report a friendly error"

    # Stream ids are unique and display names are 1-based and human friendly.
    ids = [stream.stream_id for stream in streams]
    assert len(set(ids)) == len(ids)
    assert streams[0].display_name == "Camera 2"  # primary camera is Camera 1

    manager.remove_stream(streams[0])
    assert manager.can_add_stream()
    manager.shutdown()


def test_manager_used_camera_keys_includes_primary(qapp):
    from camera_stream_manager import CameraStreamManager

    manager = CameraStreamManager()
    manager.primary_camera_info_provider = lambda: {"type": "usb", "index": 0}
    assert "usb::index=0" in manager.used_camera_keys()


def test_disconnected_stream_does_not_start_recording(qapp, tmp_path):
    from camera_stream_manager import CameraStreamManager

    manager = CameraStreamManager()
    stream = manager.create_stream()
    assert stream.start_recording(str(tmp_path / "session")) is None
    assert manager.start_recording_all(str(tmp_path / "session")) == []
    manager.shutdown()


def test_stream_filename_suffix_is_filesystem_safe(qapp):
    from camera_stream_manager import CameraStreamManager

    manager = CameraStreamManager()
    stream = manager.create_stream()
    stream.camera_info = {"type": "flir", "backend": "spinnaker", "serial": "AB 12.3"}
    suffix = stream.filename_suffix()
    assert " " not in suffix and "." not in suffix
    assert suffix.startswith("spinnaker")
    manager.shutdown()
