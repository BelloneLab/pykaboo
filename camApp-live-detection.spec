# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

base_prefix = sys.base_prefix
dll_dirs = [
    os.path.join(base_prefix, "DLLs"),
    os.path.join(base_prefix, "Library", "bin"),
]
needed_dlls = (
    "ffi.dll",
    "ffi-8.dll",
    "ffi-7.dll",
    "libffi-8.dll",
    "libffi-7.dll",
    "libexpat.dll",
    "libbz2.dll",
    "liblzma.dll",
    "sqlite3.dll",
)

extra_binaries = []
for dll_dir in dll_dirs:
    for dll_name in needed_dlls:
        dll_path = os.path.join(dll_dir, dll_name)
        if os.path.exists(dll_path):
            extra_binaries.append((dll_path, "."))

qt_datas = collect_data_files(
    "PySide6",
    includes=[
        "plugins/platforms/*",
        "plugins/styles/*",
        "plugins/imageformats/*",
        "plugins/iconengines/*",
        "plugins/platformthemes/*",
    ],
)
branding_datas = [
    ("assets/camapp_icon.png", "assets"),
    ("assets/camapp_splash.png", "assets"),
]
pyspin_datas = []
pyspin_binaries = []
pyspin_hiddenimports = []


def _resolve_installed_pyspin_package_dir():
    """Find the real installed PySpin package, not the repo's wheel cache folder."""
    try:
        spec = importlib.util.find_spec("PySpin")
    except Exception:
        return None
    if spec is None:
        return None

    origin = getattr(spec, "origin", None)
    if origin and origin not in {"built-in", "namespace"}:
        package_dir = Path(origin).resolve().parent
        if (package_dir / "__init__.py").is_file() and (package_dir / "PySpin.py").is_file():
            return package_dir

    for search_location in getattr(spec, "submodule_search_locations", []) or []:
        package_dir = Path(search_location).resolve()
        if (package_dir / "__init__.py").is_file() and (package_dir / "PySpin.py").is_file():
            return package_dir

    return None


installed_pyspin_dir = _resolve_installed_pyspin_package_dir()
if installed_pyspin_dir is not None:
    pyspin_datas, pyspin_binaries, pyspin_hiddenimports = collect_all("PySpin", include_py_files=True)

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=extra_binaries + pyspin_binaries,
    datas=qt_datas + branding_datas + pyspin_datas,
    hiddenimports=[
        "PySide6.QtOpenGL",
    ] + pyspin_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='CamAppLiveDetection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join("assets", "camapp.ico"),
)
