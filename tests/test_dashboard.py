"""Tests for the dashboard documents."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bird.dashboard import (  # noqa: E402  (path bootstrap)
    FIELD_SPECS,
    chat_candidates_payload,
    overview_payload,
    settings_payload,
)
from bird.settings import Settings  # noqa: E402


def test_settings_payload_describes_every_field():
    payload = settings_payload(Settings.from_config({}))
    keys = [field["key"] for field in payload["fields"]]
    assert keys == [field["key"] for field in FIELD_SPECS]
    assert "auto_reply" in keys
    assert "group_whitelist" in keys
    assert payload["values"]["auto_reply"] is True


def test_every_field_spec_is_serialisable_and_typed():
    for field in FIELD_SPECS:
        assert field["type"] in {"bool", "select", "multi"}
        assert field["label"]
        assert field["group"]
        json.dumps(field, ensure_ascii=False)


def test_the_form_only_offers_switches_and_choices():
    by_key = {field["key"]: field for field in FIELD_SPECS}
    assert list(by_key) == [
        "auto_reply",
        "archive_enabled",
        "image_mode",
        "group_whitelist_enabled",
        "group_whitelist",
        "private_whitelist_enabled",
        "private_whitelist",
    ]
    for field in FIELD_SPECS:
        assert field["type"] != "text"
        if field["type"] == "select":
            assert len(field["options"]) >= 2
        if field["type"] == "multi":
            assert "options" not in field


def test_overview_payload_marks_an_unreachable_backend_offline():
    payload = overview_payload(
        settings=Settings.from_config({}),
        stats={"uptime_s": 5.0, "started_at": 1.0, "now": 6.0},
        health={},
        queue=0,
        running=0,
        max_in_flight=2,
        archive={"enabled": True},
        names={"count": 10, "loaded": True},
        memory={},
        cpu={},
    )
    assert payload["backend"]["online"] is False
    assert payload["backend"]["status"] == "offline"
    assert payload["backend"]["version"] == ""
    assert payload["models"] == []
    assert payload["server_time"] == 6.0


def test_overview_payload_reads_the_health_document():
    payload = overview_payload(
        settings=Settings.from_config({"model_names": "birdnet、fastvit"}),
        stats={"now": 1.0},
        health={"status": "ok", "version": "0.1.0", "pending": 3, "max_queue": 32},
        queue=1,
        running=2,
        max_in_flight=4,
        archive={"enabled": False, "pending_mb": 1.5, "archive_mb": 2.0, "batches": 3},
        names={"count": 10480, "loaded": True},
        memory={"rss_mb": 100.0},
        cpu={"load1": 0.5},
        models=["birdnet"],
        version="v0.2.0",
    )
    assert payload["backend"] == {
        "online": True,
        "kind": "local",
        "status": "ok",
        "version": "0.1.0",
        "pending": 3,
        "max_queue": 32,
    }
    assert payload["service"] == {"queue": 1, "running": 2, "max_in_flight": 4}
    assert payload["archive"] == {
        "enabled": False,
        "pending_mb": 1.5,
        "archive_mb": 2.0,
        "batches": 3,
    }
    assert payload["models"] == ["birdnet"]
    assert payload["plugin"]["version"] == "v0.2.0"
    assert payload["settings"]["model_names"] == "birdnet、fastvit"


def test_overview_payload_survives_missing_sections():
    payload = overview_payload(
        settings=Settings.from_config({}),
        stats={},
        health={},
        queue=0,
        running=0,
        max_in_flight=0,
        archive={},
        names={},
        memory={},
        cpu={},
    )
    assert payload["service"]["max_in_flight"] == 1
    assert payload["archive"]["enabled"] is True
    assert payload["names"] == {"count": 0, "loaded": False}


def test_overview_payload_rejects_unusable_numbers():
    payload = overview_payload(
        settings=Settings.from_config({}),
        stats={"now": "not-a-number"},
        health={"pending": "x", "max_queue": None},
        queue="2",
        running=None,
        max_in_flight=2,
        archive={"pending_mb": "big"},
        names={"count": "12", "loaded": 1},
        memory={},
        cpu={},
    )
    assert payload["backend"]["pending"] == 0
    assert payload["backend"]["max_queue"] == 0
    assert payload["service"]["queue"] == 2
    assert payload["service"]["running"] == 0
    assert payload["archive"]["pending_mb"] == 0.0
    assert payload["names"] == {"count": 12, "loaded": True}
    assert payload["server_time"] == 0.0


def test_chat_candidates_drop_blank_identifiers():
    payload = chat_candidates_payload(
        [{"id": "1", "name": "群一", "platform": "aiocqhttp"}, {"id": ""}],
        [{"id": "42", "name": "小明"}],
    )
    assert payload["groups"] == [{"id": "1", "name": "群一", "platform": "aiocqhttp"}]
    assert payload["friends"] == [{"id": "42", "name": "小明", "platform": ""}]
    assert payload["errors"] == []


def test_chat_candidates_are_empty_by_default():
    payload = chat_candidates_payload()
    assert payload == {"groups": [], "friends": [], "errors": []}


def test_chat_candidates_keep_the_platform_notes():
    payload = chat_candidates_payload(errors=["QQ 无法读取群列表", ""])
    assert payload["errors"] == ["QQ 无法读取群列表"]
