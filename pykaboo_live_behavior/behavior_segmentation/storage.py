"""Tabular and array storage helpers with graceful format fallbacks.

Parquet is preferred for feature tables and embeddings because it is compact and
typed. When ``pyarrow`` is unavailable the helpers transparently fall back to
CSV so the whole pipeline still runs. JSON helpers are used for label maps and
checkpoints metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


def resolve_table_path(path: str | Path) -> Path:
    """Return a writable table path, swapping ``.parquet`` for ``.csv`` if needed."""

    path = Path(path)
    if path.suffix == ".parquet" and not parquet_available():
        return path.with_suffix(".csv")
    return path


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a dataframe to parquet (preferred) or CSV, creating parent dirs."""

    path = resolve_table_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a dataframe from parquet or CSV based on the file suffix."""

    path = Path(path)
    if not path.exists():
        alt = path.with_suffix(".csv")
        if path.suffix == ".parquet" and alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Table not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
