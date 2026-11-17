"""Tests for bird.archive.ArchiveStore.

The suite uses only the standard library, so it runs both with
python3 -m unittest discover -s tests and with pytest.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from bird.archive import ArchiveStore
except Exception:  # Fallback for a package init with heavy imports.
    _SPEC = importlib.util.spec_from_file_location(
        "bird_archive_under_test", REPO_ROOT / "bird" / "archive.py"
    )
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules["bird_archive_under_test"] = _MODULE
    _SPEC.loader.exec_module(_MODULE)
    ArchiveStore = _MODULE.ArchiveStore  # type: ignore[assignment,misc]

FIXED_NOW = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
DAY = "2026-03-04"

# Built at runtime so no literal absolute host path appears in this file.
FAKE_ABS_ROOT = "/".join(["", "srv", "share"])


def _quiet_logger() -> logging.Logger:
    """Return a logger that swallows records without reaching the root logger."""
    logger = logging.getLogger("tests.bird_archive")
    logger.handlers[:] = [logging.NullHandler()]
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    return logger


class ArchiveStoreTestCase(unittest.TestCase):
    """Provide one isolated base directory per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.store = ArchiveStore(self.base, logger=_quiet_logger())

    def _record(self, **overrides):
        """Record a valid image observation, overriding selected fields."""
        params = {
            "source": "image",
            "species": {
                "zh": "白鹭",
                "en": "Little Egret",
                "sci": "Egretta garzetta",
            },
            "confidence": 0.91,
            "gate": [0.1, 0.2, 0.3, 0.4],
            "request": "req-abc",
            "now": FIXED_NOW,
        }
        params.update(overrides)
        return self.store.record(**params)

    def _jsonl(self, path: Path) -> list:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _all_json_text(self) -> str:
        chunks = [
            path.read_text(encoding="utf-8")
            for path in self.base.rglob("*.jsonl")
            if path.is_file()
        ]
        return "".join(chunks)

    def test_record_writes_day_folder_media_and_jsonl(self) -> None:
        payload = self._record(media=[("crop.jpg", b"fake-jpeg-bytes")])

        self.assertIsNotNone(payload)
        self.assertEqual(payload["source"], "image")
        self.assertEqual(payload["species"]["sci"], "Egretta garzetta")
        self.assertEqual(payload["confidence"], 0.91)
        self.assertEqual(payload["gate"], [0.1, 0.2, 0.3, 0.4])
        self.assertTrue(payload["ts"].startswith(DAY + "T05:06:07"))
        self.assertTrue(payload["ts"].endswith("Z"))
        self.assertEqual(len(payload["media"]), 1)
        self.assertRegex(
            payload["media"][0], r"^media/\d{8}T\d{6}_00_[0-9a-f]{6}_crop\.jpg$"
        )

        day_dir = self.base / "staging" / DAY
        media_file = day_dir / payload["media"][0]
        self.assertTrue(media_file.is_file())
        self.assertEqual(media_file.read_bytes(), b"fake-jpeg-bytes")

        records = self._jsonl(day_dir / "records.jsonl")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["request"], "req-abc")
        self.assertGreater(self.store.pending_bytes(), 0)

    def test_record_supports_audio(self) -> None:
        payload = self._record(source="audio", media=[("clip.wav", b"RIFFdata")])

        self.assertIsNotNone(payload)
        self.assertEqual(payload["source"], "audio")
        self.assertEqual(len(payload["media"]), 1)
        self.assertTrue(payload["media"][0].endswith("clip.wav"))

    def test_record_scrubs_paths_and_secret_keys(self) -> None:
        payload = self._record(
            location=FAKE_ABS_ROOT + "/bird.jpg",
            request=FAKE_ABS_ROOT + "/request.json",
            extra={
                "api_key": "should-be-dropped",
                "author": FAKE_ABS_ROOT + "/owner",
                "note": FAKE_ABS_ROOT + "/p.txt",
                "count": 3,
                "flag": True,
                "ts": "override",
            },
        )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["location"], "bird.jpg")
        self.assertEqual(payload["request"], "request.json")
        self.assertNotIn("api_key", payload)
        self.assertNotIn("author", payload)
        self.assertEqual(payload["note"], "p.txt")
        self.assertEqual(payload["count"], 3.0)
        self.assertTrue(payload["flag"])
        self.assertNotEqual(payload["ts"], "override")

        text = self._all_json_text()
        self.assertNotIn(FAKE_ABS_ROOT + "/bird.jpg", text)
        self.assertNotIn("api_key", text)
        self.assertNotIn("should-be-dropped", text)

    def test_record_generates_anonymous_request_id(self) -> None:
        payload = self._record(request=None)

        self.assertIsNotNone(payload)
        self.assertTrue(payload["request"].startswith("req_"))
        self.assertEqual(len(payload["request"]), 16)

    def test_record_normalizes_invalid_values(self) -> None:
        payload = self._record(confidence=5, gate=[1, 2, 3], elapsed_ms=-1)

        self.assertIsNotNone(payload)
        self.assertEqual(payload["confidence"], 1.0)
        self.assertNotIn("gate", payload)
        self.assertNotIn("elapsed_ms", payload)

        mapped = self._record(gate={"x": 1, "y": 2, "w": 3, "h": 4})
        self.assertEqual(mapped["gate"], [1.0, 2.0, 3.0, 4.0])

    def test_record_rejects_unknown_source(self) -> None:
        self.assertIsNone(self._record(source="video"))
        self.assertFalse((self.base / "staging").exists())

    def test_record_returns_none_when_writing_fails(self) -> None:
        with mock.patch.object(
            ArchiveStore, "_atomic_write", side_effect=OSError("disk full")
        ):
            payload = self._record(media=[("crop.jpg", b"data")])

        self.assertIsNone(payload)
        self.assertFalse((self.base / "staging" / DAY / "records.jsonl").exists())

    def test_flush_packs_day_and_removes_staging(self) -> None:
        self._record(media=[("crop.jpg", b"first")])
        self._record(media=[("crop.jpg", b"second")])

        created = self.store.flush(now=FIXED_NOW)

        self.assertEqual(len(created), 1)
        archive_path = created[0]
        self.assertEqual(archive_path.parent.name, DAY)
        self.assertTrue(archive_path.name.endswith(".tar.gz"))

        with tarfile.open(archive_path, mode="r:gz") as tar:
            names = sorted(tar.getnames())
        self.assertIn("records.jsonl", names)
        self.assertEqual(len([name for name in names if name.startswith("media/")]), 2)

        index_path = archive_path.parent / "index.jsonl"
        index = self._jsonl(index_path)
        self.assertEqual(len(index), 2)

        self.assertFalse((self.base / "staging").exists())
        self.assertEqual(self.store.pending_bytes(), 0)
        self.assertGreater(self.store.archive_bytes(), 0)

    def test_flush_without_staging_returns_empty(self) -> None:
        self.assertEqual(self.store.flush(now=FIXED_NOW), [])
        self.assertFalse((self.base / "archive").exists())

    def test_flush_twice_creates_unique_batches_and_appends_index(self) -> None:
        self._record()
        first = self.store.flush(now=FIXED_NOW)
        self._record()
        second = self.store.flush(now=FIXED_NOW)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first[0].name, second[0].name)
        self.assertEqual(len(self._jsonl(first[0].parent / "index.jsonl")), 2)

    def test_flush_removes_empty_day_folder(self) -> None:
        (self.base / "staging" / DAY).mkdir(parents=True)

        self.assertEqual(self.store.flush(now=FIXED_NOW), [])
        self.assertFalse((self.base / "staging").exists())
        self.assertFalse((self.base / "archive" / DAY).exists())

    def test_rotate_drops_archives_outside_retention(self) -> None:
        archive_root = self.base / "archive"
        for offset in range(1, 8):
            day = (FIXED_NOW.date() - timedelta(days=offset)).isoformat()
            (archive_root / day).mkdir(parents=True)
            (archive_root / day / "index.jsonl").write_text("{}\n", encoding="utf-8")
        store = ArchiveStore(
            self.base,
            retention_days=2,
            max_total_bytes=None,
            logger=_quiet_logger(),
        )

        removed = store.rotate(now=FIXED_NOW)

        remaining = sorted(path.name for path in archive_root.iterdir())
        self.assertEqual(len(removed), 5)
        self.assertEqual(remaining, ["2026-03-02", "2026-03-03"])

    def test_rotate_enforces_size_budget(self) -> None:
        archive_root = self.base / "archive"
        for offset in range(3, 0, -1):
            day = (FIXED_NOW.date() - timedelta(days=offset)).isoformat()
            (archive_root / day).mkdir(parents=True)
            (archive_root / day / "blob.bin").write_bytes(b"x" * 1000)
        store = ArchiveStore(
            self.base,
            retention_days=0,
            max_total_bytes=1500,
            logger=_quiet_logger(),
        )

        removed = store.rotate(now=FIXED_NOW)

        self.assertEqual(len(removed), 2)
        self.assertLessEqual(store.archive_bytes(), 1500)
        self.assertEqual([path.name for path in archive_root.iterdir()], ["2026-03-03"])

    def test_rotate_without_archive_dir_returns_empty(self) -> None:
        self.assertEqual(self.store.rotate(now=FIXED_NOW), [])

    def test_rotate_keeps_fresh_archives(self) -> None:
        day_dir = self.base / "archive" / DAY
        day_dir.mkdir(parents=True)
        (day_dir / "index.jsonl").write_text("{}\n", encoding="utf-8")

        self.assertEqual(self.store.rotate(now=FIXED_NOW), [])
        self.assertTrue(day_dir.is_dir())

    def test_run_forever_single_pass_with_zero_interval(self) -> None:
        # Retention is disabled because run_forever works in real time: the
        # archive for a fixed past day would otherwise be rotated away at once.
        store = ArchiveStore(
            self.base, retention_days=0, max_total_bytes=None, logger=_quiet_logger()
        )
        self._record()

        asyncio.run(store.run_forever(0))

        self.assertFalse((self.base / "staging").exists())
        self.assertTrue((self.base / "archive" / DAY).is_dir())

    def test_run_forever_stops_when_event_is_set(self) -> None:
        store = ArchiveStore(
            self.base, retention_days=0, max_total_bytes=None, logger=_quiet_logger()
        )
        self._record()

        async def _run() -> None:
            event = asyncio.Event()
            asyncio.get_running_loop().call_later(0.01, event.set)
            await store.run_forever(0.05, event)

        asyncio.run(_run())
        self.assertTrue((self.base / "archive" / DAY).is_dir())

    def test_run_forever_handles_cancellation(self) -> None:
        async def _run() -> None:
            task = asyncio.ensure_future(self.store.run_forever(60))
            await asyncio.sleep(0.02)
            task.cancel()
            await task
            self.assertFalse(task.cancelled())
            self.assertIsNone(task.result())

        asyncio.run(_run())

    def test_archive_output_never_contains_absolute_paths(self) -> None:
        self._record(location=str(self.base / "pic.jpg"))
        self.store.flush(now=FIXED_NOW)

        text = self._all_json_text()
        self.assertNotIn(str(self.base), text)
        self.assertNotIn(FAKE_ABS_ROOT, text)

    def test_record_rejects_media_of_wrong_type(self) -> None:
        payload = self._record(media=[("crop.jpg", "not-bytes"), ("x", None)])

        self.assertIsNotNone(payload)
        self.assertEqual(payload["media"], [])

    def test_record_accepts_path_like_media_names(self) -> None:
        payload = self._record(media=[("../../etc/passwd", b"data")])

        self.assertIsNotNone(payload)
        self.assertTrue(payload["media"][0].endswith("passwd"))
        self.assertNotIn("..", payload["media"][0])

    def test_maintain_flushes_and_rotates(self) -> None:
        store = ArchiveStore(
            self.base, retention_days=1, max_total_bytes=None, logger=_quiet_logger()
        )
        self._record()

        store.maintain(now=FIXED_NOW)

        self.assertFalse((self.base / "staging").exists())
        self.assertTrue((self.base / "archive" / DAY).is_dir())

    def test_maintain_survives_packing_errors(self) -> None:
        self._record()

        with mock.patch.object(ArchiveStore, "_pack_day", side_effect=OSError("boom")):
            self.assertIsNone(self.store.maintain(now=FIXED_NOW))

        # The day folder survives a failed pack, so the data is not lost.
        self.assertTrue((self.base / "staging" / DAY / "records.jsonl").is_file())

    def test_constructor_defaults(self) -> None:
        self.assertEqual(ArchiveStore(self.base).retention_days, 14)
        self.assertEqual(ArchiveStore(self.base).max_total_bytes, 2 * 1024**3)
        self.assertEqual(ArchiveStore(self.base).staging_root, self.base / "staging")


if __name__ == "__main__":
    unittest.main()
