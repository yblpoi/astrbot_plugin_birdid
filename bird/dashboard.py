"""Assemble the documents the plugin dashboard renders.

Everything here is pure: the caller passes live objects in and gets plain
JSON-friendly dictionaries back. Keeping it free of AstrBot imports means the
shapes can be unit tested without a running bot.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .settings import Settings

__all__ = [
    "FIELD_SPECS",
    "chat_candidates_payload",
    "overview_payload",
    "settings_payload",
]

#: Fields the dashboard form renders, in display order.
#:
#: Every entry is either a switch or a choice: the page never asks the user to
#: type a value, so a setting that only makes sense as free text (the region
#: code, the device id, the model names) stays in the configuration file and
#: is left out here on purpose.
FIELD_SPECS: List[Dict[str, Any]] = [
    {
        "key": "auto_reply",
        "type": "bool",
        "label": "自动回复",
        "hint": "打开后，聊天里直接发来的图片或录音会自动识别；关闭后只有 LLM 工具调用才会识别。",
        "group": "行为",
    },
    {
        "key": "archive_enabled",
        "type": "bool",
        "label": "归档数据",
        "hint": "关闭后识别记录只在内存里计数，不写盘（阅后即焚）。",
        "group": "行为",
    },
    {
        "key": "image_mode",
        "type": "select",
        "label": "识别后发送图片",
        "hint": "关闭只回文字；开启发送全部图片；仅照片只发裁剪图；仅频谱图只发鸟鸣频谱。",
        "group": "行为",
        "options": [
            {"value": "all", "label": "开启"},
            {"value": "crop", "label": "仅照片"},
            {"value": "spectrogram", "label": "仅频谱图"},
            {"value": "off", "label": "关闭"},
        ],
    },
    {
        "key": "group_whitelist_enabled",
        "type": "bool",
        "label": "群聊白名单",
        "hint": "打开后只有名单里的群聊可以识别；名单为空表示谁都不识别。",
        "group": "范围",
    },
    {
        "key": "group_whitelist",
        "type": "multi",
        "label": "群聊名单",
        "hint": "点一下加入名单，再点一下移出；候选来自机器人已加入的群聊。",
        "group": "范围",
    },
    {
        "key": "private_whitelist_enabled",
        "type": "bool",
        "label": "私聊白名单",
        "hint": "打开后只有名单里的私聊可以识别；名单为空表示谁都不识别。",
        "group": "范围",
    },
    {
        "key": "private_whitelist",
        "type": "multi",
        "label": "私聊名单",
        "hint": "点一下加入名单，再点一下移出；候选来自已有的私聊会话。",
        "group": "范围",
    },
]


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce one value to an integer without raising.

    Args:
        value: Raw value.
        default: Value used when the input is unusable.

    Returns:
        The integer, or the default.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce one value to a float without raising.

    Args:
        value: Raw value.
        default: Value used when the input is unusable.

    Returns:
        The float, or the default.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def settings_payload(settings: Settings) -> Dict[str, Any]:
    """Return the editable settings document.

    Args:
        settings: Current snapshot.

    Returns:
        A mapping with the values and the form description.
    """
    return {
        "values": settings.to_public_dict(),
        "fields": FIELD_SPECS,
    }


def overview_payload(
    *,
    settings: Settings,
    stats: Mapping[str, Any],
    health: Mapping[str, Any],
    queue: int,
    running: int,
    max_in_flight: int,
    archive: Mapping[str, Any],
    names: Mapping[str, Any],
    memory: Mapping[str, Any],
    cpu: Mapping[str, Any],
    models: Sequence[str] = (),
    version: str = "",
    backend_kind: str = "",
) -> Dict[str, Any]:
    """Build the whole dashboard document in one round trip.

    Args:
        settings: Current settings snapshot.
        stats: Counters from :class:`bird.stats.StatsCollector`.
        health: Raw health payload of the backend.
        queue: Requests waiting for a worker slot.
        running: Requests being processed right now.
        max_in_flight: Configured concurrency.
        archive: Archive counters with an ``enabled`` flag.
        names: Name table counters.
        memory: Process and host memory figures.
        cpu: Load figures.
        models: Model names to advertise.
        version: Plugin version.

    Returns:
        A mapping the page can render directly.
    """
    online = bool(health)
    health_status = str(health.get("status") or "") if health else ""
    kind = str(backend_kind or ("local" if online else "offline"))
    return {
        "plugin": {
            "version": version,
            "uptime_s": _safe_float(stats.get("uptime_s", 0.0)),
            "started_at": _safe_float(stats.get("started_at", 0.0)),
        },
        "backend": {
            "online": online,
            "kind": kind,
            "status": health_status or ("ok" if online else kind),
            "version": str(health.get("version") or "") if health else "",
            "pending": _safe_int(health.get("pending", 0)) if health else 0,
            "max_queue": _safe_int(health.get("max_queue", 0)) if health else 0,
        },
        "service": {
            "queue": _safe_int(queue),
            "running": _safe_int(running),
            "max_in_flight": max(1, _safe_int(max_in_flight, 1)),
        },
        "archive": {
            "enabled": bool(archive.get("enabled", True)),
            "pending_mb": _safe_float(archive.get("pending_mb", 0.0)),
            "archive_mb": _safe_float(archive.get("archive_mb", 0.0)),
            "batches": _safe_int(archive.get("batches", 0)),
        },
        "names": {
            "count": _safe_int(names.get("count", 0)),
            "loaded": bool(names.get("loaded", False)),
        },
        "memory": dict(memory or {}),
        "cpu": dict(cpu or {}),
        "models": [str(item) for item in models if str(item).strip()],
        "stats": dict(stats),
        "settings": settings.to_public_dict(),
        "server_time": _safe_float(stats.get("now", 0.0)),
    }


def chat_candidates_payload(
    groups: Sequence[Mapping[str, Any]] = (),
    friends: Sequence[Mapping[str, Any]] = (),
    *,
    errors: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Return the whitelist pickers.

    Args:
        groups: Rows of ``{id, name, platform}``.
        friends: Rows of ``{id, name, platform}``.
        errors: Optional notes about platforms that could not be queried.

    Returns:
        A mapping with the two lists and the collected errors.
    """

    def _rows(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
        """Normalise one candidate list."""
        rows: List[Dict[str, str]] = []
        for item in items:
            identifier = str(item.get("id") or "").strip()
            if not identifier:
                continue
            rows.append(
                {
                    "id": identifier,
                    "name": str(item.get("name") or "").strip(),
                    "platform": str(item.get("platform") or "").strip(),
                }
            )
        return rows

    return {
        "groups": _rows(groups),
        "friends": _rows(friends),
        "errors": [str(item) for item in (errors or ()) if str(item).strip()],
    }
