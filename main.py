"""
Camera Control Application
Main entry point for the application.

Professional desktop application for multi-camera control with:
- Real-time live view
- GPU-accelerated video recording (FFmpeg)
- Synchronized metadata logging (timestamps, exposure, GPIO status)
- ROI support
- Thread-safe architecture using QThread
- Arduino TTL interface for gated acquisition
"""
import os
import sys
from PySide6.QtWidgets import QApplication
from main_window_enhanced import MainWindow


def main():
    """Application entry point."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        qt_base = os.path.join(sys._MEIPASS, "PySide6", "plugins")
        os.environ.setdefault("QT_PLUGIN_PATH", qt_base)
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.path.join(qt_base, "platforms"))

    app = QApplication(sys.argv)
    app.setApplicationName("Camera Control")
    app.setOrganizationName("Professional Vision Systems")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
