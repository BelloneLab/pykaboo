"""Shared test bootstrap.

Import torch before any module that pulls in camera_backends/PySpin.
On Windows the reverse order breaks torch's DLL loading (WinError 127);
main.py applies the same workaround at app startup.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
except Exception:
    pass
