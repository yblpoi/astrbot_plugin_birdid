"""Collect the counters the dashboard shows.

The figures live in process memory only: the page is a window on the running
plugin, not a replacement for the on-disk archive. Everything is therefore
bounded, so a long running bot cannot grow this module without limit.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

__all__ = ["StatsCollector", "KIND_IMAGE", "KIND_AUDIO"]

KIND_IMAGE = "image"
KIND_AUDIO = "audio"

#: Outcomes a request can end with.
OUTCOME_OK = "ok"
OUTCOME_EMPTY = "empty"
OUTCOME_BUSY = "busy"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_UNAVAILABLE = "unavailable"
OUTCOME_INVALID = "invalid"
OUTCOME_ERROR = "error"

_RECENT_LIMIT = 24
_LATENCY_LIMIT = 200
_SPECIES_LIMIT = 40
_DAILY_LIMIT = 14
_HOURLY_LIMIT = 24


def _today() -> str:
    """Return the current UTC date as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _hour_key() -> str:
    """Return the current UTC hour as YYYY-MM-DDTHH."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")


class StatsCollector:
    """Bounded counters describing what the plugin has been doing.

    Only the event loop touches the collector, so no lock is needed: the
    identifying requests and the web handlers all run on the same loop.
    """

    def __init__(self) -> None:
        """Create an empty collector stamped with the start time."""
        self.started_at = time.time()
        self.total = 0
        self.by_outcome: Dict[str, int] = {}
        self.by_kind: Dict[str, int] = {KIND_IMAGE: 0, KIND_AUDIO: 0}
        self.by_trigger: Dict[str, int] = {"auto": 0, "command": 0, "tool": 0}
        self.rejected = {"whitelist": 0, "disabled": 0}
        self._latency: Deque[float] = deque(maxlen=_LATENCY_LIMIT)
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=_RECENT_LIMIT)
        self._species: Dict[str, int] = {}
        self._daily: Dict[str, Dict[str, int]] = {}
        self._hourly: Dict[str, int] = {}

    def record(
        self,
        *,
        kind: str,
        outcome: str,
        elapsed_ms: float = 0.0,
        trigger: str = "auto",
        species: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record one finished request.

        Args:
            kind: "image" or "audio"; anything else is counted as image.
            outcome: Terminal outcome of the request.
            elapsed_ms: Measured processing time, ignored when not positive.
            trigger: What started the request: auto, command or tool.
            species: Best species name, kept for the popularity ranking.
            error: Optional short reason, stored with the recent event.

        Returns:
            None
        """
        kind = KIND_AUDIO if kind == KIND_AUDIO else KIND_IMAGE
        outcome = outcome or OUTCOME_OK
        self.total += 1
        self.by_outcome[outcome] = self.by_outcome.get(outcome, 0) + 1
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
        if trigger in self.by_trigger:
            self.by_trigger[trigger] += 1
        if elapsed_ms and elapsed_ms > 0:
            self._latency.append(float(elapsed_ms))
        label = " ".join(str(species).split()) if species else ""
        if label and outcome in (OUTCOME_OK, OUTCOME_EMPTY):
            self._species[label] = self._species.get(label, 0) + 1
        day = _today()
        bucket = self._daily.get(day)
        if bucket is None:
            bucket = {"total": 0, "ok": 0, "empty": 0, "failed": 0}
            self._daily[day] = bucket
            if len(self._daily) > _DAILY_LIMIT:
                oldest = sorted(self._daily)[0]
                self._daily.pop(oldest, None)
        bucket["total"] += 1
        if outcome == OUTCOME_OK:
            bucket["ok"] += 1
        elif outcome == OUTCOME_EMPTY:
            bucket["empty"] += 1
        else:
            bucket["failed"] += 1
        hour = _hour_key()
        self._hourly[hour] = self._hourly.get(hour, 0) + 1
        if len(self._hourly) > _HOURLY_LIMIT:
            for stale in sorted(self._hourly)[: len(self._hourly) - _HOURLY_LIMIT]:
                self._hourly.pop(stale, None)
        self._recent.appendleft(
            {
                "ts": time.time(),
                "kind": kind,
                "outcome": outcome,
                "trigger": trigger,
                "ms": int(elapsed_ms) if elapsed_ms and elapsed_ms > 0 else 0,
                "species": label,
                "error": (str(error)[:120] if error else ""),
            }
        )

    def record_rejected(self, reason: str) -> None:
        """Count a message the plugin refused before calling the backend.

        Args:
            reason: Either "whitelist" or "disabled".

        Returns:
            None
        """
        key = reason if reason in self.rejected else "disabled"
        self.rejected[key] = self.rejected.get(key, 0) + 1

    def _latency_summary(self) -> Dict[str, float]:
        """Return average, maximum and p95 latency in milliseconds."""
        if not self._latency:
            return {"avg": 0.0, "max": 0.0, "p95": 0.0, "count": 0}
        values = sorted(self._latency)
        count = len(values)
        index = min(count - 1, max(0, int(round(count * 0.95)) - 1))
        return {
            "avg": sum(values) / count,
            "max": values[-1],
            "p95": values[index],
            "count": count,
        }

    def top_species(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Return the most frequently identified species.

        Args:
            limit: Maximum number of rows.

        Returns:
            Rows of {name, count}, most frequent first.
        """
        if limit <= 0:
            return []
        rows = sorted(self._species.items(), key=lambda item: (-item[1], item[0]))
        return [{"name": name, "count": count} for name, count in rows[:limit]]

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON friendly view of the counters.

        Returns:
            A mapping the dashboard can render directly.
        """
        success = self.by_outcome.get(OUTCOME_OK, 0) + self.by_outcome.get(
            OUTCOME_EMPTY, 0
        )
        rate = (success / self.total) if self.total else 0.0
        daily = [{"date": day, **counts} for day, counts in sorted(self._daily.items())]
        hourly = [
            {"hour": hour, "count": count}
            for hour, count in sorted(self._hourly.items())
        ]
        return {
            "started_at": self.started_at,
            "uptime_s": max(0.0, time.time() - self.started_at),
            "total": self.total,
            "success_rate": rate,
            "by_outcome": dict(self.by_outcome),
            "by_kind": dict(self.by_kind),
            "by_trigger": dict(self.by_trigger),
            "rejected": dict(self.rejected),
            "latency": self._latency_summary(),
            "top_species": self.top_species(),
            "daily": daily,
            "hourly": hourly,
            "recent": list(self._recent),
        }
