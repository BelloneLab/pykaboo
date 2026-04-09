"""Branding helpers for PyKaboo runtime assets."""
import ctypes
import os
import sys
from pathlib import Path

from PySide6.QtGui import QFont, QIcon, QPixmap


def resource_path(*parts: str) -> Path:
    """Resolve bundled assets in source and frozen PyInstaller layouts."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_dir = Path(sys._MEIPASS)
    else:
        base_dir = Path(__file__).resolve().parent
    return base_dir.joinpath(*parts)


def set_windows_app_id(app_id: str = "PyKaboo.Desktop") -> None:
    """Set an explicit Windows AppUserModelID so taskbar icons group correctly."""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def preferred_app_font() -> QFont:
    """Return the default UI font requested for the application."""
    return QFont("Arial Narrow", 10)


def _first_existing_asset(*filenames: str) -> Path | None:
    """Return the first existing asset path from the provided candidates."""
    for filename in filenames:
        candidate = resource_path("assets", filename)
        if candidate.exists():
            return candidate
    return None


def load_app_icon() -> QIcon:
    """Load the branded application icon if available."""
    icon_path = _first_existing_asset(
        "pykaboo.ico",
        "pykaboo_icon.png",
        "camapp-live-detection.ico",
        "camapp_live_detection.ico",
        "camapp.ico",
        "camApp.ico",
        "camapp-live-detection_icon.png",
        "camapp_live_detection_icon.png",
        "camapp_icon.png",
        "camApp_icon.png",
    )
    if icon_path is None:
        return QIcon()
    return QIcon(str(icon_path))


def load_splash_pixmap() -> QPixmap:
    """Load the branded splash screen if available."""
    splash_path = _first_existing_asset(
        "pykaboo_splash.png",
        "camapp-live-detection_splash.png",
        "camapp_live_detection_splash.png",
        "camapp_splash.png",
        "camApp_splash.png",
    )
    if splash_path is None:
        return QPixmap()
    return QPixmap(str(splash_path))
