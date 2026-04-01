"""
Camera backend discovery helpers.

This module keeps optional camera stack imports isolated so the application can
start even when vendor SDKs are not installed.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
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
PYPYLON_IMPORT_DIAGNOSTIC = ""

_PYSPIN_DLL_DIR_HANDLES: List[object] = []
PYSPIN_PACKAGE_DIR = ""


def _is_real_pyspin_package_dir(package_dir: Optional[Path]) -> bool:
    """True when package_dir looks like an installed PySpin package."""
    if package_dir is None or not package_dir.is_dir():
        return False
    return (package_dir / "__init__.py").is_file() and (package_dir / "PySpin.py").is_file()


def _resolve_pyspin_package_dir() -> Optional[Path]:
    """Locate the real installed PySpin package, not the repo's wheel cache folder."""
    try:
        spec = importlib.util.find_spec("PySpin")
    except Exception:
        return None
    if spec is None:
        return None

    origin = getattr(spec, "origin", None)
    if origin and origin not in {"built-in", "namespace"}:
        package_dir = Path(origin).resolve().parent
        if _is_real_pyspin_package_dir(package_dir):
            return package_dir

    for search_location in getattr(spec, "submodule_search_locations", []) or []:
        package_dir = Path(search_location).resolve()
        if _is_real_pyspin_package_dir(package_dir):
            return package_dir

    return None


def _append_path_env(env_var: str, directories: List[Path]) -> None:
    """Append unique directories to a PATH-like environment variable."""
    current_entries = [
        os.path.normcase(os.path.abspath(entry))
        for entry in os.environ.get(env_var, "").split(os.pathsep)
        if entry
    ]
    updated_entries = list(os.environ.get(env_var, "").split(os.pathsep)) if os.environ.get(env_var) else []
    for directory in directories:
        normalized = os.path.normcase(os.path.abspath(str(directory)))
        if normalized in current_entries:
            continue
        current_entries.append(normalized)
        updated_entries.append(str(directory))
    if updated_entries:
        os.environ[env_var] = os.pathsep.join(updated_entries)


def _iter_spinnaker_runtime_dirs(pyspin_package_dir: Optional[Path]) -> List[Path]:
    """Collect candidate DLL / CTI directories for Spinnaker."""
    candidates: List[Path] = []
    if pyspin_package_dir is not None:
        candidates.append(pyspin_package_dir)

    if os.name == "nt":
        root_candidates: List[Path] = []
        for env_var in ("SPINNAKER_ROOT", "SPINNAKER_HOME"):
            raw_path = os.environ.get(env_var, "").strip()
            if raw_path:
                root_candidates.append(Path(raw_path))
        root_candidates.extend(
            [
                Path(r"C:\Program Files\FLIR Systems\Spinnaker"),
                Path(r"C:\Program Files\Teledyne FLIR\Spinnaker"),
                Path(r"C:\Program Files\TeledyneFLIR\Spinnaker"),
                Path(r"C:\Program Files\Teledyne FLIR IIS\Spinnaker"),
            ]
        )
        for root_dir in root_candidates:
            candidates.extend(
                [
                    root_dir / "bin64" / "vs2015",
                    root_dir / "bin" / "vs2015",
                    root_dir / "dependencies" / "GenICam_v3_0" / "bin" / "Win64_x64" / "msvc2015",
                    root_dir / "cti64" / "vs2015",
                    root_dir / "cti" / "vs2015",
                ]
            )

    unique_candidates: List[Path] = []
    seen: Set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if not resolved.is_dir():
            continue
        normalized = os.path.normcase(str(resolved))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(resolved)
    return unique_candidates


def _configure_pyspin_runtime(pyspin_package_dir: Optional[Path]) -> None:
    """Prime DLL and GenTL search paths before importing PySpin."""
    runtime_dirs = _iter_spinnaker_runtime_dirs(pyspin_package_dir)
    if not runtime_dirs:
        return

    _append_path_env("PATH", runtime_dirs)

    cti_dirs = [directory for directory in runtime_dirs if any(directory.glob("*.cti"))]
    if cti_dirs:
        _append_path_env("GENICAM_GENTL64_PATH", cti_dirs)

    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    for runtime_dir in runtime_dirs:
        try:
            handle = os.add_dll_directory(str(runtime_dir))
        except (FileNotFoundError, OSError):
            continue
        _PYSPIN_DLL_DIR_HANDLES.append(handle)


def _import_pyspin() -> Tuple[Optional[object], Optional[Exception], Optional[Path]]:
    """Import the real PySpin API, not a namespace-only placeholder."""
    pyspin_package_dir = _resolve_pyspin_package_dir()
    _configure_pyspin_runtime(pyspin_package_dir)

    last_error: Optional[Exception] = None
    for module_name in ("PySpin", "PySpin.PySpin"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on local SDK install
            last_error = exc
            continue
        if hasattr(module, "System"):
            if pyspin_package_dir is None:
                module_file = getattr(module, "__file__", None)
                if module_file:
                    candidate_dir = Path(module_file).resolve().parent
                    if _is_real_pyspin_package_dir(candidate_dir):
                        pyspin_package_dir = candidate_dir
            return module, None, pyspin_package_dir
        last_error = RuntimeError(
            f"{module_name} loaded from "
            f"'{getattr(module, '__file__', None) or 'namespace package'}' "
            "but does not expose the Spinnaker API."
        )
    return None, last_error, pyspin_package_dir


_PySpin, PYSPIN_IMPORT_ERROR, _PYSPIN_PACKAGE_DIR = _import_pyspin()
if _PYSPIN_PACKAGE_DIR is not None:
    PYSPIN_PACKAGE_DIR = str(_PYSPIN_PACKAGE_DIR)

PySpin = _PySpin
PYSPIN_AVAILABLE = PySpin is not None
PYSPIN_IMPORT_DIAGNOSTIC = ""

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

    cameras.extend(discover_flir_spinnaker_cameras())

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


def discover_flir_spinnaker_cameras() -> List[Dict]:
    """Enumerate FLIR machine-vision cameras through Spinnaker/PySpin."""
    if not PYSPIN_AVAILABLE or PySpin is None:
        return []

    cameras: List[Dict] = []
    system = None
    cam_list = None

    try:
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        for index in range(cam_list.GetSize()):
            camera = cam_list.GetByIndex(index)
            try:
                tl_map = camera.GetTLDeviceNodeMap()
                model = _read_pyspin_string_node(tl_map, "DeviceModelName") or "Unknown Model"
                serial = _read_pyspin_string_node(tl_map, "DeviceSerialNumber") or f"index-{index}"
                vendor = _read_pyspin_string_node(tl_map, "DeviceVendorName") or "FLIR"
                cameras.append(
                    {
                        "label": f"FLIR Spinnaker: {model} ({serial})",
                        "type": "flir",
                        "backend": "spinnaker",
                        "index": index,
                        "serial": serial,
                        "model": model,
                        "vendor": vendor,
                    }
                )
            finally:
                del camera
    except Exception:
        return cameras
    finally:
        if cam_list is not None:
            try:
                cam_list.Clear()
            except Exception:
                pass
        if system is not None:
            try:
                system.ReleaseInstance()
            except Exception:
                pass

    return cameras


def get_camera_backend_diagnostics() -> Dict[str, str]:
    """Return backend import diagnostics that are useful to surface in the UI."""
    diagnostics: Dict[str, str] = {}
    if PYPYLON_IMPORT_DIAGNOSTIC:
        diagnostics["pypylon"] = PYPYLON_IMPORT_DIAGNOSTIC
    if PYSPIN_IMPORT_DIAGNOSTIC:
        diagnostics["pyspin"] = PYSPIN_IMPORT_DIAGNOSTIC
    return diagnostics


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


def _read_pyspin_string_node(node_map, node_name: str) -> str:
    """Safely read a string node from a PySpin transport-layer node map."""
    if not PYSPIN_AVAILABLE or PySpin is None or node_map is None:
        return ""
    try:
        node = PySpin.CStringPtr(node_map.GetNode(node_name))
    except Exception:
        return ""
    try:
        if not (PySpin.IsAvailable(node) and PySpin.IsReadable(node)):
            return ""
    except Exception:
        return ""
    try:
        return str(node.GetValue()).strip()
    except Exception:
        return ""


def _build_pyspin_import_diagnostic(import_error: Optional[Exception]) -> str:
    """Explain why PySpin is unavailable in the current interpreter."""
    import_error_text = str(import_error or "")
    lowered_error = import_error_text.lower()
    if "does not expose the spinnaker api" in lowered_error or "cannot import name 'pyspin'" in lowered_error:
        return (
            "CamApp Live Detection found a local 'PySpin' folder, but not the installed Spinnaker "
            "Python package. Install the vendor PySpin wheel into the Python "
            "environment used to launch or build CamApp Live Detection."
        )
    if "dll load failed" in lowered_error or "the specified module could not be found" in lowered_error:
        package_hint = f" Detected PySpin package: {PYSPIN_PACKAGE_DIR}." if PYSPIN_PACKAGE_DIR else ""
        return (
            "PySpin was found, but its native Spinnaker DLLs did not load. Reinstall "
            "the matching Spinnaker SDK / PySpin wheel or rebuild CamApp Live Detection from the same "
            "environment that can import PySpin successfully."
            f"{package_hint}"
        )
    if (
        "_array_api" in lowered_error
        or "numpy.core.multiarray failed to import" in lowered_error
        or "multiarray failed to import" in lowered_error
    ):
        return (
            "PySpin is installed but cannot load against NumPy 2.x. "
        "Use numpy<2 in the CamApp Live Detection environment for Spinnaker support."
        )

    current_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    candidates = _find_local_pyspin_wheels()
    if not candidates:
        if import_error is None:
            return ""
        return (
            "PySpin is not importable in this interpreter. Install the Spinnaker "
            f"SDK wheel that matches Python {sys.version_info.major}.{sys.version_info.minor}."
        )

    messages: List[str] = []
    for wheel_path, wheel_tag in candidates:
        if wheel_tag and wheel_tag != current_tag:
            messages.append(
                f"Found local PySpin wheel '{wheel_path.name}' for {wheel_tag}, "
                f"but CamApp Live Detection is running on {current_tag}."
            )
        else:
            messages.append(
                f"Found local PySpin wheel '{wheel_path.name}', but it is not installed "
                "into the active Python environment."
            )

    messages.append(
        "Use a matching Python runtime or install the correct PySpin wheel into "
        "the environment that launches CamApp Live Detection."
    )
    return " ".join(messages)


def _build_pypylon_import_diagnostic(import_error: Optional[Exception]) -> str:
    """Explain why pypylon is unavailable in the current interpreter."""
    if import_error is None:
        return ""
    lowered_error = str(import_error).lower()
    if "no module named 'pypylon'" in lowered_error:
        return (
            "Basler support is unavailable because pypylon is not installed in the "
            "current Python environment. Install pypylon into the environment that "
            "launches CamApp Live Detection."
        )
    if "dll load failed" in lowered_error or "the specified module could not be found" in lowered_error:
        return (
            "pypylon is installed but the Basler Pylon runtime DLLs did not load. "
            "Install or repair the Basler Pylon runtime in the same environment that "
            "launches CamApp Live Detection."
        )
    return f"Basler support is unavailable: {import_error}"


def _find_local_pyspin_wheels() -> List[Tuple[Path, str]]:
    """Find PySpin wheels stored inside the repository."""
    repo_root = Path(__file__).resolve().parent
    pyspin_root = repo_root / "PySpin"
    if not pyspin_root.is_dir():
        return []

    candidates: List[Tuple[Path, str]] = []
    for wheel_path in pyspin_root.rglob("*.whl"):
        tag = _extract_python_tag_from_wheel_name(wheel_path.name)
        candidates.append((wheel_path, tag))
    return candidates


def _extract_python_tag_from_wheel_name(filename: str) -> str:
    """Extract cpXY from a wheel filename when present."""
    match = re.search(r"-(cp\d{2,3})-(cp\d{2,3})-", filename)
    if not match:
        return ""
    return match.group(1)


if not PYSPIN_AVAILABLE:
    PYSPIN_IMPORT_DIAGNOSTIC = _build_pyspin_import_diagnostic(PYSPIN_IMPORT_ERROR)
if not PYPYLON_AVAILABLE:
    PYPYLON_IMPORT_DIAGNOSTIC = _build_pypylon_import_diagnostic(PYPYLON_IMPORT_ERROR)
