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
    WHITELIST_HINT,
    BirdIdPlugin,
)


class Image:
    """Stand in for the platform image component.

    Only the class name is read, so this stub is enough to drive the probe.
    """


class FakeEvent:
    """Stand in for one message event with the parts the plugin touches."""

    def __init__(self, components=None, *, admin=False, origin="aiocqhttp:group:1"):
        self._components = list(components or [])
        self._admin = admin
        self._origin = origin
        self.results = []
        self.stopped = False

    def get_messages(self):
        return list(self._components)

    def get_message_str(self):
        return "识鸟状态"

    def get_message_type(self):
        return "GroupMessage" if ":group:" in self._origin else "FriendMessage"

    def get_group_id(self):
        parts = self._origin.split(":")
        return parts[-1] if parts[-2] == "group" else ""

    def get_session_id(self):
        return self._origin.split(":")[-1]

    @property
    def unified_msg_origin(self):
        return self._origin

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


async def test_status_reports_missing_parts_before_the_service_is_ready(monkeypatch):
    plugin = BirdIdPlugin(context=None, config={})
    event = FakeEvent(admin=True)

    async def _no_service():
        return None

    monkeypatch.setattr(plugin, "_ensure_service", _no_service)
    await plugin.birdstatus(event)
    assert _text(event) == "插件尚未就绪。"


async def test_identify_for_llm_reports_an_unavailable_service(monkeypatch):
    plugin = BirdIdPlugin(context=None, config={})

    async def _no_service():
        return None

    monkeypatch.setattr(plugin, "_ensure_service", _no_service)
    event = FakeEvent([Image()])
    result = await plugin.identify_for_llm(event, location="")
    assert _text(event) == UNAVAILABLE_HINT
    assert event.stopped is False
    # A terminating tool returns None so the agent loop stops there.
    assert result is None


async def test_identify_for_llm_asks_for_media_when_the_message_has_none(monkeypatch):
    plugin = BirdIdPlugin(context=None, config={})

    async def _service():
        return object()

    monkeypatch.setattr(plugin, "_ensure_service", _service)
    event = FakeEvent()
    result = await plugin.identify_for_llm(event, location="云南")
    assert _text(event) == MEDIA_HINT
    assert event.stopped is False
    assert result is None


async def test_identify_for_llm_refuses_a_chat_outside_the_whitelist():
    plugin = BirdIdPlugin(
        context=None,
        config={"group_whitelist_enabled": True, "group_whitelist": ["999"]},
    )
    event = FakeEvent(origin="aiocqhttp:group:1")
    result = await plugin.identify_for_llm(event, location="")
    assert _text(event) == WHITELIST_HINT
    assert plugin._stats.rejected["whitelist"] == 1
    assert result is None


async def test_identify_for_llm_passes_the_location_through(monkeypatch):
    plugin = BirdIdPlugin(context=None, config={})
    seen = []

    async def _capture(event, location, *, trigger="", stop=True):
        seen.append((location, trigger, stop))

    monkeypatch.setattr(plugin, "_identify_and_reply", _capture)
    event = FakeEvent([Image()])
    result = await plugin.identify_for_llm(event, location="云南、四川")
    assert seen == [("云南、四川", "tool", False)]
    assert result is None
    assert event.stopped is False


async def test_auto_image_leaves_a_message_without_an_attachment_alone():
    plugin = BirdIdPlugin(context=None, config={})
    event = FakeEvent()
    await plugin.auto_image(event)
    assert event.results == []
    assert event.stopped is False


async def test_auto_image_answers_an_attachment(monkeypatch):
    plugin = BirdIdPlugin(context=None, config={})
    seen = []

    async def _capture(event, location):
        seen.append(location)

    monkeypatch.setattr(plugin, "_identify_and_reply", _capture)
    event = FakeEvent([Image()])
    await plugin.auto_image(event)
    assert seen == [""]


async def test_auto_image_stays_quiet_when_automatic_replies_are_off(monkeypatch):
    plugin = BirdIdPlugin(context=None, config={"auto_reply": False})
    seen = []

    async def _capture(event, location):
        seen.append(location)

    monkeypatch.setattr(plugin, "_identify_and_reply", _capture)
    event = FakeEvent([Image()])
    await plugin.auto_image(event)
    assert seen == []
    assert plugin._stats.rejected["disabled"] == 1


async def test_auto_image_stays_quiet_outside_the_whitelist(monkeypatch):
    plugin = BirdIdPlugin(
        context=None,
        config={"private_whitelist_enabled": True, "private_whitelist": ["42"]},
    )
    seen = []

    async def _capture(event, location):
        seen.append(location)

    monkeypatch.setattr(plugin, "_identify_and_reply", _capture)
    event = FakeEvent([Image()], origin="aiocqhttp:private:1")
    await plugin.auto_image(event)
    assert seen == []
    assert plugin._stats.rejected["whitelist"] == 1


def test_chat_kind_reads_the_message_type_segment():
    from astrbot_plugin_birdid.main import _chat_kind

    assert _chat_kind("qq_official:GroupMessage:ABC") == "groupmessage"
    assert _chat_kind("aiocqhttp:FriendMessage:1") == "friendmessage"
    assert _chat_kind("plain") == ""
    assert _chat_kind("") == ""


def test_umo_platform_reads_the_leading_segment():
    from astrbot_plugin_birdid.main import _umo_platform

    assert _umo_platform("qq_official:GroupMessage:ABC") == "qq_official"
    assert _umo_platform("plain") == ""
    assert _umo_platform("") == ""


async def test_db_candidates_sorts_groups_and_friends(monkeypatch):
    captured = {}

    class _Session:
        pass

    class _Scope:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, *exc_info):
            return False

    class _Db:
        def get_db(self):
            return _Scope()

    async def _groups(session, add):
        captured["session"] = session
        add("qq:GroupMessage:2", "群 B", "qq")
        add("qq:GroupMessage:2", "重复", "qq")
        add("qq:GroupMessage:1", "群 A", "qq")

    async def _friends(session, add):
        add("qq:FriendMessage:9", "好友", "qq")

    async def _platforms(session, add):
        add("webchat-1", "网页", "webchat", False)

    class _Ctx:
        def get_db(self):
            return _Db()

    plugin = BirdIdPlugin(context=_Ctx(), config={})
    monkeypatch.setattr(plugin, "_read_umo_aliases", _groups)
    monkeypatch.setattr(plugin, "_read_conversations", _friends)
    monkeypatch.setattr(plugin, "_read_platform_sessions", _platforms)

    groups, friends, notes = await plugin._db_candidates()
    assert [row["id"] for row in groups] == ["qq:GroupMessage:1", "qq:GroupMessage:2"]
    assert [row["id"] for row in friends] == ["qq:FriendMessage:9", "webchat-1"]
    assert notes == []
    assert isinstance(captured["session"], _Session)


def test_group_base_collapses_per_speaker_sessions():
    from astrbot_plugin_birdid.main import _group_base

    group = "wechat_native:GroupMessage:main|group|43484362802%40chatroom"
    speaker = group + "|wxid_kdjpu8ru8i0612"
    assert _group_base(speaker) == group
    assert _group_base("qq_official:GroupMessage:ABC") == ""
    assert _group_base("wechat_native:FriendMessage:main|private|wxid") == ""
    assert _group_base("plain") == ""


def test_wechat_thread_decodes_the_room_id():
    from astrbot_plugin_birdid.main import _wechat_thread

    group = "wechat_native:GroupMessage:main|group|43484362802%40chatroom"
    assert _wechat_thread(group) == "43484362802@chatroom"
    assert _wechat_thread(group + "|wxid_a") == "43484362802@chatroom"
    assert _wechat_thread("qq_official:GroupMessage:ABC") == ""
    assert _wechat_thread("") == ""


async def test_db_candidates_offers_one_row_per_wechat_group(monkeypatch):
    class _Session:
        pass

    class _Scope:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, *exc_info):
            return False

    class _Db:
        def get_db(self):
            return _Scope()

    group = "wechat_native:GroupMessage:main|group|43484362802%40chatroom"

    async def _groups(session, add):
        add(group + "|wxid_a", "", "wechat_native")
        add(group + "|wxid_b", "", "wechat_native")
        add(group, "", "wechat_native")

    async def _nothing(session, add):
        return None

    class _Ctx:
        def get_db(self):
            return _Db()

    plugin = BirdIdPlugin(context=_Ctx(), config={})
    monkeypatch.setattr(plugin, "_read_umo_aliases", _groups)
    monkeypatch.setattr(plugin, "_read_conversations", _nothing)
    monkeypatch.setattr(plugin, "_read_platform_sessions", _nothing)

    groups, friends, notes = await plugin._db_candidates()
    assert [row["id"] for row in groups] == [group]
    assert friends == []
    assert notes == []


async def test_fill_wechat_names_reads_the_room_document():
    class _Client:
        async def group(self, thread):
            return {"displayName": "家庭群", "name": "ignored"}

    class _Meta:
        name = "wechat_native"

    class _Platform:
        def meta(self):
            return _Meta()

        def get_client(self):
            return _Client()

    class _Ctx:
        pass

    plugin = BirdIdPlugin(context=_Ctx(), config={})
    plugin._platforms = lambda: [_Platform()]
    rows = [
        {
            "id": "wechat_native:GroupMessage:main|group|room%40chatroom",
            "name": "",
            "platform": "wechat_native",
        },
        {"id": "qq:GroupMessage:1", "name": "", "platform": "qq"},
    ]
    await plugin._fill_wechat_names(rows)
    assert rows[0]["name"] == "家庭群"
    assert rows[1]["name"] == ""


class _Service:
    """Minimal service double that records the budgets pushed into it."""

    def __init__(self):
        self.class_code = ""
        self.area_code = ""
        self.did = ""
        self.image_mode = ""
        self.max_in_flight = 0
        self.request_timeout_s = 0.0
        self.queue_timeout_s = 0.0

    def set_max_in_flight(self, limit):
        self.max_in_flight = max(1, int(limit))


class _Archive:
    """Minimal archive double receiving the retention switches."""

    def __init__(self):
        self.enabled = True
        self.retention_days = 0
        self.max_total_bytes = 0
        self.interval_s = 0.0


class _Client:
    """Minimal client double receiving the network budgets."""

    def __init__(self):
        self.timeout_s = 0.0
        self.max_concurrency = 0

    def set_max_concurrency(self, limit):
        self.max_concurrency = max(1, int(limit))


def test_settings_read_pushes_the_snapshot_into_live_objects():
    config = {
        "auto_reply": False,
        "class_code": "m",
        "archive_enabled": False,
        "retention_days": "3",
        "archive_max_mb": "512",
        "archive_interval_s": "120",
        "max_in_flight": "5",
        "queue_timeout_s": "7.5",
        "request_timeout_s": "45",
        "backend_concurrency": "6",
    }
    plugin = BirdIdPlugin(context=None, config=config)
    plugin._service = _Service()
    plugin._archive = _Archive()
    plugin._client = _Client()

    settings = plugin._settings()

    assert settings.class_code == "M"
    assert plugin._service.class_code == "M"
    assert plugin._service.max_in_flight == 5
    assert plugin._service.queue_timeout_s == 7.5
    assert plugin._service.request_timeout_s == 45.0
    assert plugin._archive.enabled is False
    assert plugin._archive.retention_days == 3
    assert plugin._archive.max_total_bytes == 512 * 1024 * 1024
    assert plugin._archive.interval_s == 120.0
    assert plugin._client.timeout_s == 45.0
    assert plugin._client.max_concurrency == 6


def test_settings_read_survives_unusable_numbers():
    config = {"max_in_flight": "many", "retention_days": None, "archive_max_mb": ""}
    plugin = BirdIdPlugin(context=None, config=config)
    plugin._service = _Service()
    plugin._archive = _Archive()

    plugin._settings()

    assert plugin._service.max_in_flight == 2
    assert plugin._archive.retention_days == 7
    assert plugin._archive.max_total_bytes == 2048 * 1024 * 1024


async def test_db_candidates_reports_an_unreadable_table():
    class _Ctx:
        def get_db(self):
            raise RuntimeError("no database")

    plugin = BirdIdPlugin(context=_Ctx(), config={})
    groups, friends, notes = await plugin._db_candidates()
    assert groups == [] and friends == []
    assert notes == ["会话列表读取不完整，可手动填写会话标识。"]


async def test_the_identify_command_is_registered_with_its_aliases():
    from astrbot.core.star.star_handler import star_handlers_registry

    handler = None
    for item in star_handlers_registry:
        if getattr(item.handler, "__name__", "") == "bird_command":
            handler = item
            break
    assert handler is not None
    command = handler.event_filters[0]
    assert command.command_name == "识鸟"
    assert "这是什么鸟" in command.alias
    assert "啥鸟" in command.alias


async def test_command_identifies_with_a_location_hint(monkeypatch):
    plugin = BirdIdPlugin(context=None, config={})
    seen = []

    async def _capture(event, location, *, trigger="", stop=True):
        seen.append((location, trigger, stop))

    monkeypatch.setattr(plugin, "_identify_and_reply", _capture)
    event = FakeEvent([Image()])
    await plugin.bird_command(event, " 云南  西双版纳 ")
    # A plain command stops the event and sends its own answer.
    assert seen == [("云南 西双版纳", "command", True)]


async def test_command_asks_for_media_when_the_message_has_none():
    plugin = BirdIdPlugin(context=None, config={})
    event = FakeEvent()
    await plugin.bird_command(event, "")
    assert _text(event) == MEDIA_HINT
    assert event.stopped is True


async def test_command_refuses_a_chat_outside_the_whitelist():
    plugin = BirdIdPlugin(
        context=None,
        config={"group_whitelist_enabled": True, "group_whitelist": ["999"]},
    )
    event = FakeEvent([Image()], origin="aiocqhttp:group:1")
    await plugin.bird_command(event, "")
    assert _text(event) == WHITELIST_HINT
    assert event.stopped is True
    assert plugin._stats.rejected["whitelist"] == 1
