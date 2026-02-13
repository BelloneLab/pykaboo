# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files

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

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=extra_binaries,
    datas=qt_datas,
    hiddenimports=[
        "PySide6.QtOpenGL",
    ],
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
    name='camApp',
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
)
