"""Helpers for importing torch reliably on Windows runtimes."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
import sysconfig
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TORCH_IMPORT_LOCK = threading.Lock()


class _NullDllDirectoryHandle:
    """Fallback handle when add_dll_directory cannot be registered."""

    def close(self) -> None:
        return None


def _normalize_path_entry(path_entry: str | os.PathLike[str] | None) -> str:
    raw = str(path_entry or "").strip()
    if not raw:
        return ""
    return raw.strip('"')


def _dedupe_path_env() -> None:
    raw_path = os.environ.get("PATH", "")
    if not raw_path:
        return

    seen: set[str] = set()
    deduped_entries: list[str] = []
    for entry in raw_path.split(os.pathsep):
        cleaned = _normalize_path_entry(entry)
        if not cleaned:
            continue
        normalized = os.path.normcase(os.path.abspath(cleaned))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_entries.append(cleaned)

    os.environ["PATH"] = os.pathsep.join(deduped_entries)


def _prepend_unique_path_entries(entries: list[Path]) -> None:
    current_raw_entries = [
        _normalize_path_entry(entry)
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if _normalize_path_entry(entry)
    ]
    seen = {
        os.path.normcase(os.path.abspath(entry))
        for entry in current_raw_entries
    }

    prepended: list[str] = []
    for entry in entries:
        cleaned = _normalize_path_entry(entry)
        if not cleaned:
            continue
        if not Path(cleaned).is_dir():
            continue
        normalized = os.path.normcase(os.path.abspath(cleaned))
        if normalized in seen:
            continue
        seen.add(normalized)
        prepended.append(cleaned)

    if prepended:
        os.environ["PATH"] = os.pathsep.join(prepended + current_raw_entries)


def _torch_package_dir() -> Optional[Path]:
    try:
        spec = importlib.util.find_spec("torch")
    except Exception:
        return None
    if spec is None:
        return None

    origin = getattr(spec, "origin", None)
    if origin and origin not in {"built-in", "namespace"}:
        try:
            return Path(origin).resolve().parent
        except Exception:
            return Path(origin).parent
    return None


def _build_torch_dll_directories() -> list[Path]:
    candidates: list[Path] = []

    exec_prefix = _normalize_path_entry(sys.exec_prefix)
    if exec_prefix:
        candidates.extend(
            [
                Path(exec_prefix) / "Library" / "bin",
                Path(exec_prefix) / "bin",
            ]
        )

    base_exec_prefix = _normalize_path_entry(getattr(sys, "base_exec_prefix", ""))
    if base_exec_prefix and base_exec_prefix != exec_prefix:
        candidates.append(Path(base_exec_prefix) / "Library" / "bin")

    try:
        userbase = _normalize_path_entry(sysconfig.get_config_var("userbase"))
    except Exception:
        userbase = ""
    if userbase:
        candidates.append(Path(userbase) / "Library" / "bin")

    torch_package_dir = _torch_package_dir()
    if torch_package_dir is not None:
        candidates.append(torch_package_dir / "lib")

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        normalized = os.path.normcase(str(resolved))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(resolved)
    return unique_candidates


def _is_windows_path_limit_error(exc: BaseException) -> bool:
    return getattr(exc, "winerror", None) == 206 or getattr(exc, "errno", None) == 206


def import_torch(*, required: bool = True):
    """Import torch, using a Windows fallback when AddDllDirectory fails."""
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        return torch_module

    try:
        if os.name != "nt" or not hasattr(os, "add_dll_directory"):
            return importlib.import_module("torch")

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        with _TORCH_IMPORT_LOCK:
            torch_module = sys.modules.get("torch")
            if torch_module is not None:
                return torch_module

            _dedupe_path_env()
            torch_dll_dirs = _build_torch_dll_directories()
            _prepend_unique_path_entries(torch_dll_dirs)

            original_add_dll_directory = os.add_dll_directory

            def _safe_add_dll_directory(path: str | os.PathLike[str]):
                try:
                    return original_add_dll_directory(path)
                except OSError as exc:
                    normalized_path = _normalize_path_entry(path)
                    if (
                        normalized_path
                        and Path(normalized_path).is_dir()
                        and _is_windows_path_limit_error(exc)
                    ):
                        _prepend_unique_path_entries([Path(normalized_path)])
                        logger.warning(
                            "Torch DLL directory fallback to PATH after WinError 206: %s",
                            normalized_path,
                        )
                        return _NullDllDirectoryHandle()
                    raise

            os.add_dll_directory = _safe_add_dll_directory
            try:
                return importlib.import_module("torch")
            finally:
                os.add_dll_directory = original_add_dll_directory
    except Exception:
        if required:
            raise
        return None
