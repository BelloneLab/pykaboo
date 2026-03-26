"""
Camera backend discovery helpers.

This module keeps optional camera stack imports isolated so the application can
start even when vendor SDKs are not installed.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import cv2

try:
    from pypylon import pylon as _pylon
except Exception as exc:  # pragma: no cover - depends on local SDK install
    _pylon = None
    PYPYLON_IMPORT_ERROR = exc
else:
    PYPYLON_IMPORT_ERROR = None

pylon = _pylon
PYPYLON_AVAILABLE = pylon is not None

try:
    from flirpy.camera.boson import Boson as _Boson
except Exception as exc:  # pragma: no cover - optional dependency
    _Boson = None
    FLIRPY_BOSON_IMPORT_ERROR = exc
else:
    FLIRPY_BOSON_IMPORT_ERROR = None

try:
    from flirpy.camera.lepton import Lepton as _Lepton
except Exception as exc:  # pragma: no cover - optional dependency
    _Lepton = None
    FLIRPY_LEPTON_IMPORT_ERROR = exc
else:
    FLIRPY_LEPTON_IMPORT_ERROR = None

try:
    from flirpy.camera.tau import TeaxGrabber as _TeaxGrabber
except Exception as exc:  # pragma: no cover - optional dependency
    _TeaxGrabber = None
    FLIRPY_TEAX_IMPORT_ERROR = exc
else:
    FLIRPY_TEAX_IMPORT_ERROR = None

try:
    import usb.core as _usb_core
except Exception as exc:  # pragma: no cover - optional dependency
    _usb_core = None
    USB_IMPORT_ERROR = exc
else:
    USB_IMPORT_ERROR = None

Boson = _Boson
Lepton = _Lepton
TeaxGrabber = _TeaxGrabber
usb_core = _usb_core


def discover_basler_cameras() -> List[Dict]:
    """Enumerate Basler cameras through Pylon when available."""
    if not PYPYLON_AVAILABLE:
        return []

    cameras: List[Dict] = []
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    for index, dev in enumerate(devices):
        model = dev.GetModelName()
        serial = dev.GetSerialNumber()
        cameras.append(
            {
                "label": f"Basler: {model} ({serial})",
                "type": "basler",
                "index": index,
                "serial": serial,
                "model": model,
            }
        )
    return cameras


def discover_flir_cameras() -> Tuple[List[Dict], Set[int]]:
    """
    Enumerate FLIR cameras available through flirpy.

    Returns both the discovered camera descriptors and any USB video indices
    already claimed by FLIR-specific backends so the generic OpenCV USB scan can
    skip duplicates.
    """

    cameras: List[Dict] = []
    reserved_usb_indices: Set[int] = set()

    if Boson is not None:
        try:
            video_index = Boson.find_video_device()
            serial_port = Boson.find_serial_device()
            if video_index is not None or serial_port is not None:
                if video_index is not None:
                    reserved_usb_indices.add(int(video_index))
                location = _format_flir_location(video_index, serial_port)
                cameras.append(
                    {
                        "label": f"FLIR Boson{location}",
                        "type": "flir",
                        "backend": "boson",
                        "index": int(video_index) if video_index is not None else 0,
                        "video_index": video_index,
                        "serial_port": serial_port,
                    }
                )
        except Exception:
            pass

    if Lepton is not None:
        try:
            video_index = Lepton.find_video_device()
            if video_index is not None:
                reserved_usb_indices.add(int(video_index))
                cameras.append(
                    {
                        "label": f"FLIR Lepton (video {video_index})",
                        "type": "flir",
                        "backend": "lepton",
                        "index": int(video_index),
                        "video_index": int(video_index),
                        "serial_port": None,
                    }
                )
        except Exception:
            pass

    if TeaxGrabber is not None and usb_core is not None:
        try:
            device = usb_core.find(idVendor=0x0403, idProduct=0x6010)
            if device is not None:
                cameras.append(
                    {
                        "label": "FLIR Tau / TeAx Grabber",
                        "type": "flir",
                        "backend": "teax",
                        "index": 0,
                        "video_index": None,
                        "serial_port": None,
                    }
                )
        except Exception:
            pass

    return cameras, reserved_usb_indices


def discover_usb_cameras(
    skip_indices: Optional[Set[int]] = None,
    max_devices: int = 10,
) -> List[Dict]:
    """Enumerate generic OpenCV USB cameras, excluding reserved indices."""
    cameras: List[Dict] = []
    skip = {int(idx) for idx in (skip_indices or set())}
    backend = cv2.CAP_MSMF if os.name == "nt" else cv2.CAP_V4L2

    for index in range(max_devices):
        if index in skip:
            continue
        cap = cv2.VideoCapture(index, backend)
        try:
            if cap.isOpened():
                cameras.append(
                    {
                        "label": f"USB: Device {index}",
                        "type": "usb",
                        "index": index,
                    }
                )
        finally:
            cap.release()

    return cameras


def _format_flir_location(video_index: Optional[int], serial_port: Optional[str]) -> str:
    parts: List[str] = []
    if video_index is not None:
        parts.append(f"video {video_index}")
    if serial_port:
        parts.append(str(serial_port))
    if not parts:
        return ""
    return " (" + ", ".join(parts) + ")"
