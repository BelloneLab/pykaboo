from __future__ import annotations

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
        "pykaboo_small.png",
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


def _generate_splash_pixmap() -> QPixmap:
    """Generate a clean, minimal splash screen programmatically."""
    from PySide6.QtCore import QPoint, QRect, Qt
    from PySide6.QtGui import (
        QColor,
        QLinearGradient,
        QPainter,
        QPen,
        QRadialGradient,
    )

    W, H = 520, 280
    pixmap = QPixmap(W, H)
    pixmap.fill(QColor(0, 0, 0, 0))

    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Background gradient
    bg = QLinearGradient(0, 0, W, H)
    bg.setColorAt(0.0, QColor(8, 14, 24))
    bg.setColorAt(1.0, QColor(12, 22, 36))
    p.setBrush(bg)
    p.setPen(QPen(QColor(30, 58, 90), 1.5))
    p.drawRoundedRect(0, 0, W, H, 18, 18)

    # Subtle accent glow (bottom-right)
    glow = QRadialGradient(W * 0.75, H * 0.65, 160)
    glow.setColorAt(0.0, QColor(30, 80, 140, 40))
    glow.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setBrush(glow)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QPoint(int(W * 0.75), int(H * 0.65)), 160, 160)

    # Minimal lens icon
    cx, cy = 80, H // 2
    accent = QColor(70, 150, 230)
    p.setPen(QPen(accent, 2.0))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPoint(cx, cy), 22, 22)
    p.setBrush(accent)
    p.drawEllipse(QPoint(cx, cy), 7, 7)

    # Text
    tx = 130
    title_font = QFont("Arial Narrow", 22, QFont.Weight.Bold)
    p.setFont(title_font)
    p.setPen(QColor(120, 195, 255))
    p.drawText(tx, H // 2 - 18, "PyKaboo")

    sub_font = QFont("Arial Narrow", 10)
    p.setFont(sub_font)
    p.setPen(QColor(140, 165, 195))
    p.drawText(tx, H // 2 + 8, "Camera Acquisition & Live Detection")

    # Accent bar
    p.setPen(QPen(QColor(50, 130, 220), 2))
    p.drawLine(tx, H // 2 + 28, tx + 160, H // 2 + 28)

    ver_font = QFont("Arial Narrow", 9)
    p.setFont(ver_font)
    p.setPen(QColor(90, 115, 145))
    p.drawText(tx, H // 2 + 50, "Loading workspace...")

    p.end()
    return pixmap


def load_splash_pixmap() -> QPixmap:
    """Load the branded splash screen if available, or generate one."""
    splash_path = _first_existing_asset(
        "pykaboo_big.png",
        "pykaboo_splash.png",
        "camapp-live-detection_splash.png",
        "camapp_live_detection_splash.png",
        "camapp_splash.png",
        "camApp_splash.png",
    )
    if splash_path is not None:
        pm = QPixmap(str(splash_path))
        if not pm.isNull():
            if pm.width() > 720:
                from PySide6.QtCore import Qt

                pm = pm.scaledToWidth(720, Qt.TransformationMode.SmoothTransformation)
            return pm
    return _generate_splash_pixmap()
