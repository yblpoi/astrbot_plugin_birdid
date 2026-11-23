"""Archive storage for bird identification observations.

This module keeps the plugin's disk usage bounded while preserving the media
that later training runs need.  One record consists of a single JSONL metadata
line plus zero or more media files:

* record() writes media into  <base_dir>/staging/<YYYY-MM-DD>/  and appends one
  JSON object per line to records.jsonl in the same folder.
* flush() packs a day folder into  <base_dir>/archive/<YYYY-MM-DD>/  as a
  tar.gz archive, keeps a plain index.jsonl beside the tarball and then removes
  the staging folder.
* rotate() drops the oldest archives until both the retention window and the
  total size budget are satisfied.
* run_forever() runs flush() and rotate() periodically on the event loop and
  exits cleanly when the caller cancels it or sets a stop event.

Only the Python standard library is used and every path is derived from the
base_dir argument, so nothing here depends on the host layout.  Absolute paths,
control characters and secret-like keys are scrubbed before anything reaches
disk.

Typical use::

    store = ArchiveStore(Path("data/bird_archive"))
    store.record(
        source="image",
        species={"zh": "Egret", "en": "Little Egret",
                 "sci": "Egretta garzetta"},
        confidence=0.93,
        gate=[0.1, 0.2, 0.3, 0.4],
        media=[("crop.jpg", jpeg_bytes)],
    )
    await store.run_forever(3600, stop_event)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import tarfile
import threading
import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_LOGGER = logging.getLogger(__name__)

#: Default number of days an archive folder is kept.
DEFAULT_RETENTION_DAYS = 14
#: Default upper bound for everything below archive/ (2 GiB).
DEFAULT_MAX_TOTAL_BYTES = 2 * 1024**3

#: Values accepted for the source field of a record.
ALLOWED_SOURCES = ("image", "audio")

_STAGING_DIRNAME = "staging"
_ARCHIVE_DIRNAME = "archive"
_MEDIA_DIRNAME = "media"
_RECORDS_FILENAME = "records.jsonl"
_INDEX_FILENAME = "index.jsonl"

_DATE_FORMAT = "%Y-%m-%d"
_BATCH_FORMAT = "%Y%m%dT%H%M%SZ"

_MAX_TEXT_LENGTH = 200
_MAX_EXTRA_KEYS = 24
_MAX_MEDIA_ITEMS = 16

_RESERVED_FIELDS = frozenset(
    {
        "ts",
        "source",
        "species",
        "confidence",
        "gate",
        "location",
        "elapsed_ms",
        "request",
        "media",
    }
)

_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SECRET_KEY_RE = re.compile(
    r"(?:api[_-]?key|secret|token|passw|credential|auth|signature)",
    re.IGNORECASE,
)
_SPECIES_ALIASES = {
    "cn": "zh",
    "zh_cn": "zh",
    "zh-cn": "zh",
    "en_us": "en",
    "en-us": "en",
    "scientific": "sci",
    "latin": "sci",
}
_SPECIES_KEYS = ("zh", "en", "sci")


def _coerce_now(moment: Optional[datetime]) -> datetime:
    """Return moment as an aware UTC datetime.

    Args:
        moment: Caller supplied timestamp, possibly naive or None.

    Returns:
        An aware UTC datetime; the current time when moment is None.
    """
    if moment is None:
        return datetime.now(timezone.utc)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _iso_timestamp(moment: datetime) -> str:
    """Format moment as an ISO-8601 UTC string with millisecond precision."""
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _looks_absolute(text: str) -> bool:
    """Return True when text looks like an absolute path or a file URL."""
    stripped = text.strip()
    if not stripped:
        return False
    if _WINDOWS_ABSOLUTE_RE.match(stripped):
        return True
    if stripped.startswith(("/", "\\", "~")):
        return True
    return stripped.lower().startswith("file://")


def _scrub_text(value: Any) -> Optional[str]:
    """Return a log-safe, path-free copy of a string field.

    Control characters are removed, absolute paths are reduced to their final
    component and long values are truncated.

    Args:
        value: Raw field value taken from a record.

    Returns:
        The scrubbed text, or None when the value is not a usable string.
    """
    if not isinstance(value, str):
        return None
    text = "".join(ch for ch in value if ch >= " " and ch != "\x7f").strip()
    if not text:
        return None
    if _looks_absolute(text):
        text = Path(text.replace("\\", "/")).name or "redacted"
    return text[:_MAX_TEXT_LENGTH]


def _safe_filename(name: Any, fallback: str = "media.bin") -> str:
    """Return a basename-only, filesystem-safe filename for a media blob.

    Args:
        name: Candidate filename supplied by the caller.
        fallback: Name used when nothing usable is left after sanitizing.

    Returns:
        A filename without directory separators, leading dots or unsupported
        characters.
    """
    raw = name if isinstance(name, str) else fallback
    candidate = Path(raw.replace("\\", "/")).name
    candidate = _UNSAFE_FILENAME_RE.sub("_", candidate).strip("._")
    if not candidate:
        return fallback
    return candidate[:120]


def _coerce_float(value: Any, *, minimum: Optional[float] = None) -> Optional[float]:
    """Return value as a finite float, or None when it is unusable.

    Args:
        value: Raw numeric value.
        minimum: Optional inclusive lower bound.

    Returns:
        The converted float, or None for booleans, non-numbers, infinities and
        values below minimum.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if minimum is not None and number < minimum:
        return None
    return number


def _normalize_gate(gate: Any) -> Optional[List[float]]:
    """Normalize a gate box to four floats in [x, y, w, h] order.

    Args:
        gate: Sequence of four numbers, or a mapping with x/y/w/h keys.

    Returns:
        A list of four floats, or None when the box cannot be normalized.
    """
    if gate is None:
        return None
    if isinstance(gate, Mapping):
        values = [_coerce_float(gate.get(key)) for key in ("x", "y", "w", "h")]
    else:
        try:
            values = [_coerce_float(item) for item in gate]
        except TypeError:
            return None
    if len(values) != 4 or any(value is None for value in values):
        return None
    return [float(value) for value in values]


def _normalize_species(species: Any) -> Optional[Dict[str, str]]:
    """Normalize a species description to the zh/en/sci keys.

    Args:
        species: Mapping with Chinese, English and scientific names, or a
            single string that is treated as the Chinese name.

    Returns:
        A mapping limited to the three known keys, or None when empty.
    """
    if species is None:
        return None
    if isinstance(species, str):
        text = _scrub_text(species)
        return {"zh": text} if text else None
    if not isinstance(species, Mapping):
        return None
    normalized: Dict[str, str] = {}
    for key, value in species.items():
        name = str(key).strip().lower()
        name = _SPECIES_ALIASES.get(name, name)
        if name not in _SPECIES_KEYS:
            continue
        text = _scrub_text(value)
        if text:
            normalized[name] = text
    return normalized or None


def _normalize_extra(extra: Any) -> Dict[str, Any]:
    """Return flat, scrubbed extra fields without secret-like or reserved keys.

    Args:
        extra: Optional mapping of additional metadata.

    Returns:
        A mapping with at most _MAX_EXTRA_KEYS scalar entries.
    """
    if not isinstance(extra, Mapping):
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in extra.items():
        if len(cleaned) >= _MAX_EXTRA_KEYS:
            break
        name = _scrub_text(str(key))
        if not name or name in _RESERVED_FIELDS or _SECRET_KEY_RE.search(name):
            continue
        if value is None or isinstance(value, bool):
            cleaned[name] = value
        elif isinstance(value, (int, float)):
            cleaned[name] = _coerce_float(value)
        elif isinstance(value, str):
            cleaned[name] = _scrub_text(value)
    return cleaned


def _dir_size(root: Path) -> int:
    """Return the total size in bytes of the regular files below root."""
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _archive_moment(path: Path) -> datetime:
    """Return the age reference of an archive folder.

    The YYYY-MM-DD folder name is authoritative; folders that do not follow the
    convention fall back to their modification time.

    Args:
        path: Archive folder to inspect.

    Returns:
        An aware UTC datetime used for age comparison.
    """
    try:
        return datetime.strptime(path.name, _DATE_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return datetime.now(timezone.utc)


class ArchiveStore:
    """Persist bird observations as JSONL metadata plus packed media archives.

    The store is deliberately fail-soft: every public method logs its own
    errors and reports them through its return value, so a broken disk never
    breaks the identification pipeline.

    Attributes:
        base_dir: Root folder that owns staging/ and archive/.
        retention_days: Archives older than this many days are rotated away;
            zero disables the age based rotation.
        max_total_bytes: Soft budget for archive/ in bytes; None or a value
            below one disables the size based rotation.
        enabled: When false, observations are dropped instead of staged, so
            nothing reaches the disk.
    """

    def __init__(
        self,
        base_dir: Path,
        *,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        max_total_bytes: Optional[int] = DEFAULT_MAX_TOTAL_BYTES,
        enabled: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the store without touching the filesystem.

        Args:
            base_dir: Root folder for staging/ and archive/.
            retention_days: Days to keep packed archives.
            max_total_bytes: Soft size budget for the archive folder.
            enabled: Whether observations are kept at all.
            logger: Optional logger; the module logger is used by default.
        """
        self.base_dir = Path(base_dir)
        self.enabled = bool(enabled)
        #: Delay used by the running maintenance loop; readable so the owner can
        #: retune it without cancelling the task.
        self.interval_s = 3600.0
        self.retention_days = max(0, int(retention_days))
        self.max_total_bytes = (
            None if max_total_bytes is None else max(0, int(max_total_bytes))
        )
        self._logger = logger if logger is not None else _LOGGER
        self._lock = threading.Lock()

    @property
    def staging_root(self) -> Path:
        """Return the folder that holds not-yet-packed day folders."""
        return self.base_dir / _STAGING_DIRNAME

    @property
    def archive_root(self) -> Path:
        """Return the folder that holds the packed archives."""
        return self.base_dir / _ARCHIVE_DIRNAME

    def record(
        self,
        *,
        source: str,
        species: Optional[Mapping[str, Any]] = None,
        confidence: Optional[float] = None,
        gate: Optional[Sequence[float]] = None,
        location: Optional[str] = None,
        elapsed_ms: Optional[float] = None,
        request: Optional[str] = None,
        media: Optional[Iterable[Tuple[str, bytes]]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """Persist one observation and its media files.

        Media is staged under staging/<day>/media/ and the metadata line is
        appended to staging/<day>/records.jsonl.  Nothing is written when the
        record is rejected.

        Args:
            source: "image" or "audio".
            species: Mapping with zh/en/sci names; aliases such as cn or
                scientific are accepted.
            confidence: Detection confidence, clamped to [0, 1].
            gate: Gate box as four numbers or an x/y/w/h mapping.
            location: Optional human readable site name.
            elapsed_ms: Optional processing duration in milliseconds.
            request: Optional anonymous request identifier; generated when
                omitted.
            media: Optional iterable of (filename, bytes) pairs.
            extra: Optional extra scalar metadata; secret-like keys are dropped.
            now: Optional timestamp override, defaults to the current UTC time.

        Returns:
            The stored payload, or None when the record was rejected or could
            not be written.  All failures are logged, never raised.
        """
        if not self.enabled:
            # Archiving is turned off, so the observation is discarded instead
            # of staged: nothing about the request survives on disk.
            return None
        try:
            with self._lock:
                if source not in ALLOWED_SOURCES:
                    raise ValueError("unsupported source: %r" % (source,))
                moment = _coerce_now(now)
                day_dir = self.staging_root / moment.strftime(_DATE_FORMAT)
                payload = self._build_payload(
                    source=source,
                    moment=moment,
                    species=species,
                    confidence=confidence,
                    gate=gate,
                    location=location,
                    elapsed_ms=elapsed_ms,
                    request=request,
                    extra=extra,
                )
                plans = self._media_plan(media, moment)
                payload["media"] = [
                    "%s/%s" % (_MEDIA_DIRNAME, name) for name, _ in plans
                ]
                line = json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                if plans:
                    media_dir = day_dir / _MEDIA_DIRNAME
                    for name, blob in plans:
                        self._atomic_write(media_dir / name, blob)
                self._append_line(day_dir / _RECORDS_FILENAME, line)
                return payload
        except Exception:
            self._logger.exception("Failed to record a bird observation")
            return None

    def flush(self, *, now: Optional[datetime] = None) -> List[Path]:
        """Pack every staging day folder into archive/<YYYY-MM-DD>/.

        Each packed folder becomes <batch>.tar.gz next to an index.jsonl that
        accumulates the metadata of that day.  A staging folder is removed only
        after its archive was written successfully.

        Args:
            now: Timestamp used to name the batch; defaults to the current UTC
                time.

        Returns:
            The archive paths created by this call, oldest day first.
        """
        created: List[Path] = []
        try:
            with self._lock:
                staging_root = self.staging_root
                if not staging_root.is_dir():
                    return created
                moment = _coerce_now(now)
                day_dirs = sorted(
                    path for path in staging_root.iterdir() if path.is_dir()
                )
                for day_dir in day_dirs:
                    try:
                        archive_path = self._pack_day(day_dir, moment)
                    except Exception:
                        self._logger.exception(
                            "Failed to pack the staging folder %s", day_dir.name
                        )
                        continue
                    if archive_path is not None:
                        created.append(archive_path)
                self._prune_empty_staging()
        except Exception:
            self._logger.exception("Failed to flush the bird archive")
        return created

    def rotate(self, *, now: Optional[datetime] = None) -> List[Path]:
        """Remove the oldest archives until both limits are satisfied.

        Args:
            now: Reference timestamp for the retention window; defaults to the
                current UTC time.

        Returns:
            The archive folders removed by this call, oldest first.
        """
        removed: List[Path] = []
        try:
            with self._lock:
                archive_root = self.archive_root
                if not archive_root.is_dir():
                    return removed
                moment = _coerce_now(now)
                entries = sorted(
                    (path for path in archive_root.iterdir() if path.is_dir()),
                    key=lambda path: (_archive_moment(path), path.name),
                )
                if self.retention_days > 0:
                    cutoff = moment - timedelta(days=self.retention_days)
                    for path in list(entries):
                        if _archive_moment(path).date() < cutoff.date():
                            if self._remove_tree(path):
                                removed.append(path)
                                entries.remove(path)
                budget = self.max_total_bytes
                if budget is None or budget <= 0 or not entries:
                    return removed
                sizes = {path: _dir_size(path) for path in entries}
                total = sum(sizes.values())
                for path in entries:
                    if total <= budget:
                        break
                    if self._remove_tree(path):
                        removed.append(path)
                        total -= sizes[path]
        except Exception:
            self._logger.exception("Failed to rotate the bird archive")
        return removed

    async def run_forever(
        self,
        interval_seconds: float = 3600.0,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Flush and rotate periodically until cancelled or asked to stop.

        Args:
            interval_seconds: Delay between maintenance passes.  Zero or a
                negative value performs a single pass and returns, which keeps
                tests fast and deterministic.
            stop_event: Optional event that ends the loop once it is set.

        Returns:
            None when the loop ends because stop_event was set or the
            surrounding task was cancelled.
        """
        event = stop_event if stop_event is not None else asyncio.Event()
        delay = _coerce_float(interval_seconds, minimum=0.0)
        if delay is None:
            delay = 3600.0
        self.interval_s = delay
        try:
            while not event.is_set():
                # Packing can be slow, so keep it off the event loop thread.
                await asyncio.to_thread(self.maintain)
                # Read the attribute every pass so a new interval applies.
                if self.interval_s <= 0:
                    return None
                await asyncio.sleep(self.interval_s)
        except asyncio.CancelledError:
            self._logger.debug("Bird archive loop cancelled")
        return None

    def maintain(self, *, now: Optional[datetime] = None) -> None:
        """Run one flush plus rotate pass without raising.

        Args:
            now: Optional timestamp forwarded to both steps.

        Returns:
            None; failures are logged by the underlying methods.
        """
        self.flush(now=now)
        self.rotate(now=now)

    def pending_bytes(self) -> int:
        """Return the size in bytes of the not-yet-packed staging area."""
        try:
            root = self.staging_root
            return _dir_size(root) if root.is_dir() else 0
        except Exception:
            self._logger.exception("Failed to measure the staging area")
            return 0

    def archive_bytes(self) -> int:
        """Return the size in bytes of everything below archive/."""
        try:
            root = self.archive_root
            return _dir_size(root) if root.is_dir() else 0
        except Exception:
            self._logger.exception("Failed to measure the archive area")
            return 0

    @staticmethod
    def _build_payload(
        *,
        source: str,
        moment: datetime,
        species: Any,
        confidence: Any,
        gate: Any,
        location: Any,
        elapsed_ms: Any,
        request: Any,
        extra: Any,
    ) -> Dict[str, Any]:
        """Return the JSON-ready metadata payload for one observation."""
        payload: Dict[str, Any] = {
            "ts": _iso_timestamp(moment),
            "source": source,
        }
        normalized_species = _normalize_species(species)
        if normalized_species:
            payload["species"] = normalized_species
        score = _coerce_float(confidence, minimum=0.0)
        if score is not None:
            payload["confidence"] = round(min(score, 1.0), 4)
        box = _normalize_gate(gate)
        if box is not None:
            payload["gate"] = [round(item, 6) for item in box]
        text = _scrub_text(location)
        if text:
            payload["location"] = text
        elapsed = _coerce_float(elapsed_ms, minimum=0.0)
        if elapsed is not None:
            payload["elapsed_ms"] = round(elapsed, 3)
        payload["request"] = _scrub_text(request) or "req_%s" % uuid.uuid4().hex[:12]
        payload["media"] = []
        payload.update(_normalize_extra(extra))
        return payload

    @staticmethod
    def _media_plan(
        media: Optional[Iterable[Tuple[str, bytes]]], moment: datetime
    ) -> List[Tuple[str, bytes]]:
        """Return unique (filename, blob) pairs for the media argument."""
        plans: List[Tuple[str, bytes]] = []
        if media is None:
            return plans
        try:
            items = list(media)[:_MAX_MEDIA_ITEMS]
        except TypeError:
            return plans
        prefix = moment.strftime("%Y%m%dT%H%M%S")
        for index, item in enumerate(items):
            try:
                raw_name, blob = item
            except (TypeError, ValueError):
                continue
            if not isinstance(blob, (bytes, bytearray)) or not blob:
                continue
            name = _safe_filename(raw_name)
            # A short random suffix keeps names unique when several records
            # share the same timestamp, so media is never silently overwritten.
            suffix = uuid.uuid4().hex[:6]
            plans.append(("%s_%02d_%s_%s" % (prefix, index, suffix, name), bytes(blob)))
        return plans

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        """Write data to path through a temporary file and os.replace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(".%s.%s.tmp" % (path.name, uuid.uuid4().hex[:8]))
        try:
            with open(temp_path, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        """Append one JSON line to path with a single O_APPEND write."""
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = (line + "\n").encode("utf-8")
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(descriptor, blob)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _pack_day(self, day_dir: Path, moment: datetime) -> Optional[Path]:
        """Pack one staging day folder, write its index and drop the folder.

        Args:
            day_dir: Staging folder named after the observation day.
            moment: Timestamp used to name the batch.

        Returns:
            The created tar.gz path, or None when the folder held no packable
            file (an empty folder is still cleaned up).
        """
        files = sorted(
            path
            for path in day_dir.rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and not path.name.endswith(".tmp")
        )
        if not files:
            shutil.rmtree(day_dir, ignore_errors=True)
            return None
        archive_dir = self.archive_root / day_dir.name
        archive_dir.mkdir(parents=True, exist_ok=True)
        batch = self._next_batch_name(archive_dir, moment)
        target = archive_dir / ("%s.tar.gz" % batch)
        temp_target = archive_dir / ("." + batch + ".tar.gz.tmp")
        try:
            with tarfile.open(temp_target, mode="w:gz", compresslevel=6) as tar:
                for path in files:
                    tar.add(
                        str(path),
                        arcname=path.relative_to(day_dir).as_posix(),
                        recursive=False,
                    )
            os.replace(temp_target, target)
        finally:
            if temp_target.exists():
                try:
                    temp_target.unlink()
                except OSError:
                    pass
        records_path = day_dir / _RECORDS_FILENAME
        if records_path.is_file():
            lines = records_path.read_text(encoding="utf-8").splitlines()
            self._merge_index(archive_dir, lines)
        shutil.rmtree(day_dir, ignore_errors=True)
        return target

    @staticmethod
    def _next_batch_name(archive_dir: Path, moment: datetime) -> str:
        """Return a batch identifier that is unique inside archive_dir."""
        base = moment.strftime(_BATCH_FORMAT)
        candidate = base
        counter = 1
        while (archive_dir / ("%s.tar.gz" % candidate)).exists():
            candidate = "%s-%02d" % (base, counter)
            counter += 1
        return candidate

    def _merge_index(self, archive_dir: Path, lines: List[str]) -> None:
        """Append lines to the day index while keeping it a single file."""
        index_path = archive_dir / _INDEX_FILENAME
        existing = ""
        if index_path.is_file():
            existing = index_path.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                existing += "\n"
        payload = existing + "".join("%s\n" % line for line in lines if line.strip())
        if not payload:
            return
        self._atomic_write(index_path, payload.encode("utf-8"))

    def _prune_empty_staging(self) -> None:
        """Remove the staging root once no day folder is left."""
        staging_root = self.staging_root
        if not staging_root.is_dir():
            return
        try:
            if next(staging_root.iterdir(), None) is None:
                staging_root.rmdir()
        except OSError:
            self._logger.debug("Staging folder is still in use", exc_info=True)

    def _remove_tree(self, path: Path) -> bool:
        """Delete path recursively, reporting success instead of raising."""
        try:
            shutil.rmtree(path)
            return True
        except OSError:
            self._logger.warning(
                "Failed to remove the archive folder %s", path.name, exc_info=True
            )
            return False
