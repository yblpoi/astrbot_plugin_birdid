"""Read process and host load figures for the status reply and dashboard.

Both readers degrade to an empty mapping instead of raising: a host without
``psutil`` should still answer the administrator command, just without the
memory line.
"""

from __future__ import annotations

import os
from typing import Any, Dict

__all__ = ["cpu_snapshot", "memory_snapshot"]

_MB = 1024.0 * 1024.0


def _psutil() -> Any:
    """Return the psutil module, or None when it is unavailable.

    Returns:
        The imported module, or None.
    """
    try:
        import psutil
    except Exception:  # noqa: BLE001 - a missing optional dependency
        return None
    return psutil


def memory_snapshot() -> Dict[str, float]:
    """Return process and host memory in mebibytes.

    Returns:
        A mapping with rss_mb, used_mb, total_mb and percent, or an empty
        mapping when the figures cannot be read.
    """
    psutil = _psutil()
    if psutil is None:
        return {}
    try:
        rss = float(psutil.Process().memory_info().rss)
        virtual = psutil.virtual_memory()
        total = float(virtual.total)
        available = float(getattr(virtual, "available", 0.0) or 0.0)
        percent = float(getattr(virtual, "percent", 0.0) or 0.0)
    except Exception:  # noqa: BLE001 - any psutil failure means "unknown"
        return {}
    return {
        "rss_mb": rss / _MB,
        "used_mb": max(0.0, (total - available)) / _MB,
        "total_mb": total / _MB,
        "percent": percent,
    }


def cpu_snapshot() -> Dict[str, float]:
    """Return the load average and the CPU share used by this process.

    Returns:
        A mapping with load1, load5, load15, cores and process_percent; an
        empty mapping when nothing can be read.
    """
    psutil = _psutil()
    if psutil is None:
        return {}
    try:
        load1, load5, load15 = os.getloadavg()
    except (AttributeError, OSError):
        load1 = load5 = load15 = 0.0
    try:
        cores = float(psutil.cpu_count() or 0)
    except Exception:  # noqa: BLE001 - cores only affects the display
        cores = 0.0
    try:
        process_percent = float(psutil.Process().cpu_percent(interval=None))
    except Exception:  # noqa: BLE001 - a reading failure is not fatal
        process_percent = 0.0
    return {
        "load1": float(load1),
        "load5": float(load5),
        "load15": float(load15),
        "cores": cores,
        "process_percent": process_percent,
    }
