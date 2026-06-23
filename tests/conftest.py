"""Shared test bootstrap.

Import torch before any module that pulls in camera_backends/PySpin.
On Windows the reverse order breaks torch's DLL loading (WinError 127);
main.py applies the same workaround at app startup.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
except Exception:
    pass

# Isolate QSettings to a throwaway INI file so tests that construct the real MainWindow
# (which persists live-detection settings to QSettings("PyKaboo","PyKaboo")) can NEVER
# clobber a developer's real registry settings (checkpoint paths, ROIs, rules, output
# map). The two-arg QSettings(org, app) constructor forces NativeFormat (the Windows
# registry) and ignores setDefaultFormat/setPath, so we monkeypatch the class to reroute
# that form to an IniFormat file under a temp dir. Patching PySide6.QtCore.QSettings here
# (before any app module does `from PySide6.QtCore import QSettings`) makes every later
# import pick up the isolated subclass.
try:
    from PySide6 import QtCore

    _TEST_SETTINGS_DIR = tempfile.mkdtemp(prefix="pykaboo_test_qsettings_")
    _RealQSettings = QtCore.QSettings
    _RealQSettings.setPath(_RealQSettings.IniFormat, _RealQSettings.UserScope, _TEST_SETTINGS_DIR)
    _RealQSettings.setPath(_RealQSettings.IniFormat, _RealQSettings.SystemScope, _TEST_SETTINGS_DIR)

    class _IsolatedQSettings(_RealQSettings):
        def __init__(self, *args, **kwargs):
            if len(args) == 2 and all(isinstance(a, str) for a in args) and not kwargs:
                super().__init__(
                    _RealQSettings.IniFormat, _RealQSettings.UserScope, args[0], args[1]
                )
            else:
                super().__init__(*args, **kwargs)

    QtCore.QSettings = _IsolatedQSettings
except Exception:
    pass
