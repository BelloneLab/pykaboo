"""
PyKaboo
Main entry point for the application.

Desktop application for camera acquisition with:
- Real-time live view
- GPU-accelerated video recording (FFmpeg)
- Synchronized metadata logging (timestamps, exposure, GPIO status)
- ROI support
- Thread-safe architecture using QThread
- Arduino TTL interface for gated acquisition
"""
import os
import site
import sys
from pathlib import Path


def _is_existing_dir(path_value: str | None) -> bool:
    """True when path_value points to an existing directory."""
    if not path_value or not path_value.strip():
        return False
    return Path(path_value).is_dir()


def _prefer_environment_site_packages() -> None:
    """
    In conda/venv, prefer environment packages over user-site packages.
    This avoids loading mismatched Qt wheels from %APPDATA%.
    """
    in_managed_env = bool(os.environ.get("CONDA_PREFIX")) or sys.prefix != sys.base_prefix
    if not in_managed_env:
        return

    user_sites: list[str] = []
    try:
        site_paths = site.getusersitepackages()
        if isinstance(site_paths, str):
            user_sites = [site_paths]
        else:
            user_sites = list(site_paths)
    except Exception:
        return

    if not user_sites:
        return

    normalized_user_sites = {os.path.normcase(os.path.abspath(p)) for p in user_sites}
    sys.path[:] = [
        p
        for p in sys.path
        if os.path.normcase(os.path.abspath(p)) not in normalized_user_sites
    ]
    os.environ.setdefault("PYTHONNOUSERSITE", "1")


def _find_frozen_qt_plugins_dir() -> Path | None:
    """Locate Qt plugin folder inside frozen (PyInstaller) layouts."""
    if not getattr(sys, "frozen", False):
        return None

    candidate_roots = []
    if hasattr(sys, "_MEIPASS"):
        meipass = Path(sys._MEIPASS)
        candidate_roots.extend(
            [
                meipass / "PySide6" / "plugins",
                meipass / "PySide6" / "Qt" / "plugins",
                meipass / "plugins",
            ]
        )

    exe_dir = Path(sys.executable).resolve().parent
    candidate_roots.extend(
        [
            exe_dir / "PySide6" / "plugins",
            exe_dir / "PySide6" / "Qt" / "plugins",
            exe_dir / "plugins",
        ]
    )

    for plugins_dir in candidate_roots:
        platforms_dir = plugins_dir / "platforms"
        if (platforms_dir / "qwindows.dll").exists() or platforms_dir.is_dir():
            return plugins_dir

    return None


def _configure_qt_plugin_environment() -> Path | None:
    """
    Ensure Qt platform plugins are discoverable before QApplication is created.
    """
    for env_var in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
        if not _is_existing_dir(os.environ.get(env_var)):
            os.environ.pop(env_var, None)

    plugins_dir = _find_frozen_qt_plugins_dir()
    if plugins_dir is not None:
        os.environ["QT_PLUGIN_PATH"] = str(plugins_dir)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins_dir / "platforms")

    return plugins_dir


def _configure_qt_runtime_plugin_paths(plugins_dir: Path | None) -> None:
    """Populate Qt plugin search paths in source/IDE and frozen runs."""
    from PySide6.QtCore import QCoreApplication, QLibraryInfo

    candidate_dirs: list[Path] = []
    if plugins_dir is not None:
        candidate_dirs.append(plugins_dir)

    qt_plugins_raw = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    if qt_plugins_raw:
        candidate_dirs.append(Path(qt_plugins_raw))

    for candidate in candidate_dirs:
        if not candidate.is_dir():
            continue

        if not _is_existing_dir(os.environ.get("QT_PLUGIN_PATH")):
            os.environ["QT_PLUGIN_PATH"] = str(candidate)

        platform_dir = candidate / "platforms"
        if platform_dir.is_dir() and not _is_existing_dir(os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_dir)

        QCoreApplication.addLibraryPath(str(candidate))
        return


def main():
    """Application entry point."""
    _prefer_environment_site_packages()
    qt_plugins_dir = _configure_qt_plugin_environment()

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QSplashScreen
    from branding import load_app_icon, load_splash_pixmap, preferred_app_font, set_windows_app_id
    from main_window_enhanced import MainWindow

    _configure_qt_runtime_plugin_paths(qt_plugins_dir)

    set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("PyKaboo")
    app.setOrganizationName("PyKaboo")
    app.setFont(preferred_app_font())

    app_icon = load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    splash = None
    splash_pixmap = load_splash_pixmap()
    if not splash_pixmap.isNull():
        splash = QSplashScreen(
            splash_pixmap,
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint,
        )
        splash.setMask(splash_pixmap.mask())
        splash.show()
        app.processEvents()

    window = MainWindow()
    if not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()
    if splash is not None:
        splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
