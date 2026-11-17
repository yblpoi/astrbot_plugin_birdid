"""Tests for the plugin entry point.

The module imports AstrBot, which is only present inside a running
installation, so the whole file is skipped where the framework is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("astrbot")

# The plugin is imported as a package, so the parent directory must be on the
# path even though pytest runs inside the test folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from astrbot_plugin_birdid.main import (  # noqa: E402 - path set above
    MEDIA_HINT,
    UNAVAILABLE_HINT,
    BirdIdPlugin,
)


class FakeEvent:
    """Stand in for one message event with the parts the plugin touches."""

    def __init__(self, components=None, *, admin=False):
        self._components = list(components or [])
        self._admin = admin
        self.results = []
        self.stopped = False

    def get_messages(self):
        return list(self._components)

    def get_message_str(self):
        return "birds"

    def is_admin(self):
        return self._admin

    def chain_result(self, chain):
        return chain

    async def send(self, chain):
        self.results.append(chain)

    def stop_event(self):
        self.stopped = True


def _text(event):
    return "".join(part.text for part in event.results[0] if hasattr(part, "text"))


async def test_reply_sends_a_plain_text_and_stops_the_event():
    event = FakeEvent()
    await BirdIdPlugin._reply(event, UNAVAILABLE_HINT)
    assert event.stopped is True
    assert _text(event) == UNAVAILABLE_HINT


async def test_reply_keeps_the_component_order():
    event = FakeEvent()
    marker = object()
    await BirdIdPlugin._reply(event, [marker])
    assert event.results[0][0] is marker


async def test_status_refuses_a_message_from_a_normal_user():
    plugin = BirdIdPlugin(context=None, config={})
    event = FakeEvent(admin=False)
    await plugin.birdstatus(event)
    assert _text(event) == "该指令仅限管理员使用。"
    assert event.stopped is True


async def test_birds_reports_an_unavailable_service(monkeypatch):
    plugin = BirdIdPlugin(context=None, config={})

    async def _no_service():
        return None

    monkeypatch.setattr(plugin, "_ensure_service", _no_service)
    event = FakeEvent()
    await plugin.birds(event, location="")
    assert _text(event) == UNAVAILABLE_HINT


async def test_birds_asks_for_media_when_the_message_has_none(monkeypatch):
    plugin = BirdIdPlugin(context=None, config={})

    async def _service():
        return object()

    monkeypatch.setattr(plugin, "_ensure_service", _service)
    event = FakeEvent()
    await plugin.birds(event, location="云南")
    assert _text(event) == MEDIA_HINT
    assert event.stopped is True
