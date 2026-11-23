"""Bird identification plugin entry point.

The plugin exposes three chat surfaces. Common phrasings ("识鸟", "这是什么鸟"
and friends) are handled by a command filter, so the usual request never wakes
the model at all. Anything else falls through to the ``identify_bird`` LLM tool:
the model reads the unusual wording, calls this plugin, and the answer is
delivered straight to the chat without the model ever looking at the picture.
Administrators get one more command, ``识鸟状态``, plus a dashboard page that
renders the same figures as charts and lets them change the switches without
editing JSON.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import unquote

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.star.filter.command import GreedyStr

from .bird import hostinfo
from .bird import media as media_tools
from .bird.archive import ArchiveStore
from .bird.client import BirdClient, BusyError, ClientError
from .bird.dashboard import (
    chat_candidates_payload,
    overview_payload,
    settings_payload,
)
from .bird.format import format_reply, format_status
from .bird.names import NameResolver
from .bird.service import BirdService, ServiceBusy
from .bird.settings import Settings, SettingsError, normalize_patch
from .bird.stats import (
    KIND_AUDIO,
    KIND_IMAGE,
    OUTCOME_BUSY,
    OUTCOME_EMPTY,
    OUTCOME_ERROR,
    OUTCOME_INVALID,
    OUTCOME_OK,
    OUTCOME_TIMEOUT,
    OUTCOME_UNAVAILABLE,
    StatsCollector,
)

PLUGIN_NAME = "astrbot_plugin_birdid"
PLUGIN_VERSION = "v0.2.0"
MEDIA_HINT = "请附上一张图片或一段录音，也可以引用一条包含图片或录音的消息。"
UNAVAILABLE_HINT = "鸟类识别暂时不可用，请联系管理员。"
WHITELIST_HINT = "这个会话没有开启鸟类识别。"

#: How long a quota document is reused before asking the backend again.
_QUOTA_TTL_S = 60.0

#: How long the local/official classification is trusted.
_BACKEND_TTL_S = 30.0

#: Routes the dashboard page calls through the plugin bridge.
_ROUTE_PREFIX = "/" + PLUGIN_NAME

#: The command that identifies the media of one message without the model.
IDENTIFY_COMMAND = "识鸟"

#: Everyday phrasings of the same request. The command filter only matches a
#: whole message or a message that continues with a space, so these stay
#: narrow: "这是什么鸟这么吵" is left to the model on purpose.
IDENTIFY_ALIASES = (
    "认鸟",
    "辨鸟",
    "识别鸟",
    "识别一下鸟",
    "帮我认鸟",
    "帮我识别鸟",
    "鸟种识别",
    "识别鸟类",
    "这是什么鸟",
    "这什么鸟",
    "这是啥鸟",
    "什么鸟",
    "啥鸟",
    "看看这是什么鸟",
)


def _tool_event(context: Any) -> Optional[AstrMessageEvent]:
    """Return the chat event carried by an LLM tool context.

    Args:
        context: The ``ContextWrapper`` handed to ``FunctionTool.call``.

    Returns:
        The originating message event, or None when the call has no chat.
    """
    event = getattr(getattr(context, "context", None), "event", None)
    return event if isinstance(event, AstrMessageEvent) else None


def _tool_error(message: str) -> str:
    """Render a tool failure the model can relay.

    Args:
        message: Human readable reason.

    Returns:
        A JSON document describing the failure.
    """
    return json.dumps(
        {"status": "error", "message": message},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _chat_kind(identifier: str) -> str:
    """Return the lower-cased chat kind carried by a unified origin.

    Args:
        identifier: A unified message origin such as
            ``platform:GroupMessage:id``.

    Returns:
        The message type segment, or an empty string when absent.
    """
    parts = (identifier or "").split(":")
    if len(parts) < 2:
        return ""
    return parts[1].casefold()


def _umo_platform(umo: str) -> str:
    """Return the platform segment of a unified message origin.

    Args:
        umo: A unified message origin.

    Returns:
        The platform name, or an empty string when absent.
    """
    head = (umo or "").split(":", 1)[0]
    return head if head != umo else ""


def _group_base(umo: str) -> str:
    """Drop the speaker segment a group session id may carry.

    ``wechat_native`` keeps one session per speaker, so one group appears as
    ``platform:GroupMessage:acc|group|room|speaker`` once per member. The group
    itself is every segment up to and including the room, which is the
    identifier an administrator can actually whitelist.

    Args:
        umo: Unified message origin of a chat.

    Returns:
        The group-wide origin, or an empty string for other chats.
    """
    text = (umo or "").strip()
    if ":" not in text:
        return ""
    platform, kind, rest = text.split(":", 2)
    if "group" not in kind.casefold():
        return ""
    segments = rest.split("|")
    if len(segments) < 4 or segments[1].casefold() != "group":
        return ""
    return ":".join((platform, kind, "|".join(segments[:3])))


def _wechat_thread(identifier: str) -> str:
    """Return the raw room id a wechat_native group origin names.

    The adapter quotes the thread, so ``43484362802%40chatroom`` must be
    decoded before it can be handed back to the native client.

    Args:
        identifier: A ``wechat_native`` group origin.

    Returns:
        The decoded thread id, or an empty string for other chats.
    """
    text = (identifier or "").strip()
    if ":" not in text:
        return ""
    platform, _kind, rest = text.split(":", 2)
    if platform.casefold() != "wechat_native":
        return ""
    segments = rest.split("|")
    if len(segments) < 3 or segments[1].casefold() != "group":
        return ""
    return unquote(segments[2])


def _as_int(value: Any, default: int = 0) -> int:
    """Read one optional configuration value as an integer.

    Args:
        value: Raw configuration value.
        default: Fallback used when the value is missing or unusable.

    Returns:
        The integer value, or the fallback.
    """
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    """Read one optional configuration value as a float.

    Args:
        value: Raw configuration value.
        default: Fallback used when the value is missing or unusable.

    Returns:
        The float value, or the fallback.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_text(value: Any) -> str:
    """Trim one optional string value.

    Args:
        value: Raw value from the model or the configuration.

    Returns:
        The trimmed text, or an empty string.
    """
    if value is None:
        return ""
    return " ".join(str(value).split())


def _split_models(value: Any) -> List[str]:
    """Split the configured model names into individual labels.

    Args:
        value: Raw configuration value, usually a comma separated string.

    Returns:
        The non empty labels in their original order.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        raw: Sequence[Any] = list(value)
    else:
        text = str(value)
        for separator in ("、", "，", "\n"):
            text = text.replace(separator, ",")
        raw = text.split(",")
    labels: List[str] = []
    for item in raw:
        label = _as_text(item)
        if label and label not in labels:
            labels.append(label)
    return labels


class IdentifyBirdTool(FunctionTool):
    """Let the model hand a bird question to the Dongniao recogniser.

    The answer is sent to the chat by the plugin itself, so the model never
    receives the picture and never paraphrases species names.
    """

    def __init__(self, plugin: "BirdIdPlugin") -> None:
        """Declare the tool schema.

        Args:
            plugin: Owning plugin instance.
        """
        super().__init__(
            name="identify_bird",
            description=(
                "识别用户发来的鸟类图片或鸟鸣录音。当用户想识别鸟、问这是什么鸟、"
                "或发来图片/语音并希望知道鸟的种类时使用。工具会直接调用识别接口并把"
                "结果发给用户，不要自己看图或猜物种。这是终止型工具：调用前后不要输出"
                "过程、解释或确认文字。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "maxLength": 60,
                        "description": (
                            "可选的地点先验，用户明确说了拍摄地点时才填，"
                            "例如“云南”“北京”。不要编造。"
                        ),
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        )
        self._plugin = plugin

    async def call(self, context: Any, **kwargs: Any) -> Optional[str]:
        """Run one identification for the current chat.

        Args:
            context: Wrapper carrying the originating chat event.
            **kwargs: ``location`` from the model.

        Returns:
            None once the reply reached the chat, or an error document when
            the call has no chat to answer in.
        """
        event = _tool_event(context)
        if event is None:
            return _tool_error("无法获取当前聊天会话。")
        return await self._plugin.identify_for_llm(
            event, _as_text(kwargs.get("location"))
        )


class BirdIdPlugin(Star):
    """Identify birds from pictures and recordings."""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        """Store the framework handles without starting any work.

        Args:
            context: AstrBot plugin context.
            config: Flat plugin configuration.
        """
        super().__init__(context)
        self.config = config or {}
        self._service: Optional[BirdService] = None
        self._client: Optional[BirdClient] = None
        self._archive: Optional[ArchiveStore] = None
        self._names = NameResolver()
        self._archive_task: Optional[asyncio.Task] = None
        self._stats = StatsCollector()
        self._quota_cache: tuple = (0.0, {})
        self._backend_kind: tuple = (0.0, "")
        self._routes_registered = False

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    def _setting(self, key: str, default: Any) -> Any:
        """Read one configuration value with the documented fallback.

        Args:
            key: Configuration key.
            default: Value used when the entry is missing or blank.

        Returns:
            The configured value or the default.
        """
        value = (
            self.config.get(key, default) if hasattr(self.config, "get") else default
        )
        if value is None or value == "":
            return default
        return value

    def _settings(self) -> Settings:
        """Build a settings snapshot from the current configuration.

        The long lived objects cache a few of these values, so every read also
        pushes the snapshot back into them. That way a change made outside the
        dashboard - the plugin configuration page, or a hand edited file -
        takes effect on the next message instead of needing a restart.

        Returns:
            The normalised settings.
        """
        settings = Settings.from_config(self.config)
        self._sync_runtime(settings)
        return settings

    def _sync_runtime(self, settings: Settings) -> None:
        """Push a settings snapshot into the long lived objects.

        The archive switch, the worker budgets and the recognition hints live
        on objects created once at startup, so a change made in the dashboard -
        or in the plugin configuration page - has to be copied over instead of
        waiting for a restart.

        Args:
            settings: Snapshot to apply.

        Returns:
            None
        """
        if self._archive is not None:
            self._archive.enabled = settings.archive_enabled
            self._archive.retention_days = _as_int(
                self._setting("retention_days", 7), 7
            )
            self._archive.max_total_bytes = (
                _as_int(self._setting("archive_max_mb", 2048), 2048) * 1024 * 1024
            )
            self._archive.interval_s = _as_float(
                self._setting("archive_interval_s", 900), 900.0
            )
        if self._service is not None:
            self._service.class_code = settings.class_code or "B"
            self._service.area_code = settings.api_area
            self._service.did = settings.did
            self._service.image_mode = settings.image_mode
            self._service.set_max_in_flight(
                _as_int(self._setting("max_in_flight", 2), 2)
            )
            self._service.request_timeout_s = _as_float(
                self._setting("request_timeout_s", 30), 30.0
            )
            self._service.queue_timeout_s = _as_float(
                self._setting("queue_timeout_s", 20), 20.0
            )
        if self._client is not None:
            self._client.timeout_s = _as_float(
                self._setting("request_timeout_s", 30), 30.0
            )
            self._client.set_max_concurrency(
                _as_int(self._setting("backend_concurrency", 4), 4)
            )

    async def _apply_settings(self, values: Mapping[str, Any]) -> Settings:
        """Persist a settings patch and refresh what depends on it.

        Args:
            values: Normalised values to merge into the configuration.

        Returns:
            The new snapshot.

        Raises:
            SettingsError: The configuration object cannot be written.
        """
        merged: Dict[str, Any] = dict(self._settings().to_public_dict())
        merged.update(dict(values))
        saved = False
        saver = getattr(self.config, "save_config_async", None)
        if callable(saver):
            saved = bool(await saver(merged))
        else:
            fallback = getattr(self.config, "save_config", None)
            if callable(fallback):
                fallback(merged)
                saved = True
        if not saved:
            try:
                self.config.update(merged)  # type: ignore[attr-defined]
            except (AttributeError, TypeError) as exc:
                raise SettingsError("当前配置无法保存") from exc
        settings = self._settings()
        self._sync_runtime(settings)
        return settings

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        """Create the backend client, the service and the archive worker.

        Returns:
            None
        """
        self._names.load()
        settings = self._settings()
        data_dir = Path(StarTools.get_data_dir(PLUGIN_NAME))
        self._archive = ArchiveStore(
            data_dir / "bird_archive",
            retention_days=int(self._setting("retention_days", 7)),
            max_total_bytes=int(self._setting("archive_max_mb", 2048)) * 1024 * 1024,
            enabled=settings.archive_enabled,
            logger=self.logger,
        )
        self._client = BirdClient(
            str(self._setting("backend_url", "http://127.0.0.1:8080")),
            api_key=str(self._setting("api_key", "")),
            api_key_header=str(self._setting("api_key_header", "api_key")),
            max_concurrency=int(self._setting("backend_concurrency", 4)),
            timeout_s=float(self._setting("request_timeout_s", 30)),
        )
        self._service = BirdService(
            self._client,
            self._names,
            self._archive,
            max_in_flight=int(self._setting("max_in_flight", 2)),
            request_timeout_s=float(self._setting("request_timeout_s", 30)),
            queue_timeout_s=float(self._setting("queue_timeout_s", 20)),
            ffmpeg=str(self._setting("ffmpeg_path", "")),
            class_code=settings.class_code,
            area_code=settings.api_area,
            did=settings.did,
            image_mode=settings.image_mode,
            logger=self.logger,
        )
        self._sync_runtime(settings)
        self._archive_task = asyncio.create_task(
            self._archive.run_forever(
                interval_seconds=float(self._setting("archive_interval_s", 900))
            )
        )
        self._register_web_apis()
        self._register_llm_tools()
        logger.info("Bird identification plugin ready with %d names.", len(self._names))

    async def terminate(self) -> None:
        """Stop the background worker and release the HTTP session.

        Returns:
            None
        """
        if self._archive_task is not None:
            self._archive_task.cancel()
            try:
                await self._archive_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._archive_task = None
        if self._archive is not None:
            try:
                self._archive.flush()
            except Exception as exc:  # noqa: BLE001 - shutdown is best effort
                logger.warning("Bird archive flush failed: %s", exc)
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._service = None
        self._routes_registered = False
        logger.info("Bird identification plugin stopped.")

    async def _ensure_service(self) -> Optional[BirdService]:
        """Return a ready service, starting the plugin on first use.

        Returns:
            The service, or None when initialisation failed.
        """
        if self._service is not None:
            return self._service
        try:
            await self.initialize()
        except Exception as exc:  # noqa: BLE001 - never break the pipeline
            logger.error("Bird plugin initialisation failed: %s", exc)
            return None
        return self._service

    def _register_llm_tools(self) -> None:
        """Publish the identification tool to the model.

        Returns:
            None
        """
        add = getattr(self.context, "add_llm_tools", None)
        if not callable(add):
            return
        try:
            add(IdentifyBirdTool(self))
        except Exception as exc:  # noqa: BLE001 - a missing tool is not fatal
            logger.warning("Registering the bird LLM tool failed: %s", exc)

    def _register_web_apis(self) -> None:
        """Expose the dashboard endpoints through the plugin bridge.

        Returns:
            None
        """
        if self._routes_registered:
            return
        register = getattr(self.context, "register_web_api", None)
        if not callable(register):
            return
        routes = (
            (
                _ROUTE_PREFIX + "/overview",
                self._api_overview,
                ["GET"],
                "识鸟看板数据",
            ),
            (
                _ROUTE_PREFIX + "/settings",
                self._api_settings,
                ["GET", "POST"],
                "识鸟设置",
            ),
            (
                _ROUTE_PREFIX + "/chats",
                self._api_chats,
                ["GET"],
                "白名单候选会话",
            ),
        )
        for route, handler, methods, desc in routes:
            try:
                register(route, handler, methods, desc)
            except Exception as exc:  # noqa: BLE001 - old cores lack the API
                logger.warning("Registering %s failed: %s", route, exc)
        self._routes_registered = True

    # ------------------------------------------------------------------
    # access policy
    # ------------------------------------------------------------------
    @staticmethod
    def _chat_scope(event: Any) -> Dict[str, Any]:
        """Read the identifiers a whitelist may match against.

        Every attribute is optional so a partially featured platform still
        yields a usable scope instead of raising.

        Args:
            event: Message event.

        Returns:
            A mapping with is_group, chat_key, session_id and group_id.
        """

        def _call(name: str) -> str:
            """Return one string attribute of the event, or a blank string."""
            try:
                value = getattr(event, name, None)
                if callable(value):
                    value = value()
                if isinstance(value, str):
                    return value.strip()
            except Exception:  # noqa: BLE001 - a probe must never raise
                return ""
            return ""

        message_type = _call("get_message_type")
        group_id = _call("get_group_id")
        is_group = bool(group_id) or "group" in message_type.casefold()
        chat_key = _call("unified_msg_origin")
        session_id = _call("get_session_id")
        return {
            "is_group": is_group,
            "chat_key": chat_key,
            "session_id": session_id,
            "group_id": group_id,
            "chat_alias": _group_base(chat_key) or _group_base(session_id),
        }

    def _allowed(self, event: Any, settings: Settings) -> tuple:
        """Decide whether this chat may use the recogniser.

        Args:
            event: Message event.
            settings: Current settings snapshot.

        Returns:
            An (allowed, reason) tuple.
        """
        scope = self._chat_scope(event)
        return settings.allows(**scope)

    # ------------------------------------------------------------------
    # identification
    # ------------------------------------------------------------------
    @filter.event_message_type(
        filter.EventMessageType.PRIVATE_MESSAGE | filter.EventMessageType.GROUP_MESSAGE,
        priority=-1,
    )
    async def auto_image(self, event: AstrMessageEvent) -> None:
        """React to a message that carries an attachment.

        Only runs when automatic replies are enabled and the chat is allowed;
        otherwise the message is left for the model to handle.

        Args:
            event: Message event.

        Returns:
            None
        """
        settings = self._settings()
        if not media_tools.has_media(event):
            return
        if not settings.auto_reply:
            self._stats.record_rejected("disabled")
            return
        if not self._allowed(event, settings)[0]:
            self._stats.record_rejected("whitelist")
            return
        await self._identify_and_reply(event, "")

    @filter.command(IDENTIFY_COMMAND, alias=set(IDENTIFY_ALIASES))
    async def bird_command(self, event: AstrMessageEvent, location: GreedyStr) -> None:
        """Identify the media of the current message without the model.

        The everyday phrasings listed in IDENTIFY_ALIASES land here, so the
        usual request costs no tokens: the event never reaches the model. The
        rest of the message is a free-form location hint, which the recogniser
        uses to narrow the candidate species.

        Args:
            event: Message event.
            location: Everything typed after the command, used as a hint.

        Returns:
            None
        """
        settings = self._settings()
        if not self._allowed(event, settings)[0]:
            self._stats.record_rejected("whitelist")
            await self._reply(event, WHITELIST_HINT)
            return
        if not media_tools.has_media(event):
            await self._reply(event, MEDIA_HINT)
            return
        await self._identify_and_reply(event, _as_text(location), trigger="command")

    async def identify_for_llm(
        self, event: AstrMessageEvent, location: str = ""
    ) -> None:
        """Identify the media of one message on behalf of the model.

        The answer is sent here, so the tool returns None on purpose: the
        agent loop ends instead of asking the model to repeat what the user
        has already received.

        Args:
            event: Message event holding the media.
            location: Optional location hint.

        Returns:
            None, which terminates the tool round.
        """
        settings = self._settings()
        allowed, _reason = self._allowed(event, settings)
        if not allowed:
            self._stats.record_rejected("whitelist")
            await self._respond(event, WHITELIST_HINT, stop=False)
            return None
        if not media_tools.has_media(event):
            await self._respond(event, MEDIA_HINT, stop=False)
            return None
        await self._identify_and_reply(event, location, trigger="tool", stop=False)
        return None

    async def _identify_and_reply(
        self,
        event: AstrMessageEvent,
        location: str = "",
        *,
        trigger: str = "",
        stop: bool = True,
    ) -> None:
        """Run one identification and answer in the chat.

        Args:
            event: Message event.
            location: Optional location hint.
            trigger: What started the request; inferred when blank.
            stop: Whether to stop the event afterwards.

        Returns:
            None
        """
        trigger = trigger or ("auto" if not location else "command")
        service = await self._ensure_service()
        if service is None:
            self._stats.record(
                kind=KIND_IMAGE,
                outcome=OUTCOME_UNAVAILABLE,
                trigger=trigger,
            )
            await self._respond(event, UNAVAILABLE_HINT, stop=stop)
            return
        started = time.monotonic()
        workdir = Path(tempfile.mkdtemp(prefix="birdid-"))
        kind = KIND_IMAGE
        try:
            try:
                items = await media_tools.collect_media(event, workdir)
            except Exception as exc:  # noqa: BLE001 - a bad file is normal
                logger.warning("Reading attachments failed: %s", exc)
                items = []
            if not items:
                await self._respond(event, MEDIA_HINT, stop=stop)
                return
            try:
                reply = await service.identify(
                    items,
                    workdir,
                    location=location,
                    request_id=uuid.uuid4().hex[:12],
                )
            except ServiceBusy:
                self._stats.record(kind=kind, outcome=OUTCOME_BUSY, trigger=trigger)
                await self._respond(event, "现在有点忙，请稍后再试。", stop=stop)
                return
            except ClientError:
                self._stats.record(
                    kind=kind, outcome=OUTCOME_UNAVAILABLE, trigger=trigger
                )
                await self._respond(
                    event, "识别服务暂时不可用，请稍后再试。", stop=stop
                )
                return
            except asyncio.TimeoutError:
                self._stats.record(kind=kind, outcome=OUTCOME_TIMEOUT, trigger=trigger)
                await self._respond(event, "这次识别超时了，请稍后再试。", stop=stop)
                return
            except ValueError:
                self._stats.record(kind=kind, outcome=OUTCOME_INVALID, trigger=trigger)
                await self._respond(
                    event,
                    "这段附件无法读取，请换一张图片或一段录音。",
                    stop=stop,
                )
                return
            except Exception as exc:  # noqa: BLE001 - one reply must not crash
                logger.exception("Identification failed unexpectedly: %s", exc)
                self._stats.record(kind=kind, outcome=OUTCOME_ERROR, trigger=trigger)
                await self._respond(event, "识别过程出错了，请稍后再试。", stop=stop)
                return
            kind = KIND_AUDIO if reply.kind == "audio" else KIND_IMAGE
            elapsed_ms = max(reply.elapsed_ms, (time.monotonic() - started) * 1000.0)
            top = reply.observations[0] if reply.observations else None
            self._stats.record(
                kind=kind,
                outcome=OUTCOME_EMPTY if not reply.observations else OUTCOME_OK,
                elapsed_ms=elapsed_ms,
                trigger=trigger,
                species=top.label() if top is not None else None,
            )
            text = format_reply(
                reply.observations,
                kind=reply.kind,
                elapsed_ms=elapsed_ms,
                location=reply.location,
            )
            chain: List[Any] = [Plain(text)]
            for payload in reply.images:
                chain.append(Image.fromBytes(payload))
            await self._respond(event, chain, stop=stop)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # dashboard endpoints
    # ------------------------------------------------------------------
    async def _archive_stats(self) -> Dict[str, Any]:
        """Collect the archive counters shown by the dashboard.

        Returns:
            A mapping with the switch state and the size figures.
        """
        archive = self._archive
        if archive is None:
            return {
                "enabled": False,
                "pending_mb": 0.0,
                "archive_mb": 0.0,
                "batches": 0,
            }
        pending = 0.0
        archived = 0.0
        batches = 0
        try:
            pending = archive.pending_bytes() / (1024.0 * 1024.0)
            archived = archive.archive_bytes() / (1024.0 * 1024.0)
            batches = len(list(archive.archive_root.glob("*/*.tar.gz")))
        except Exception as exc:  # noqa: BLE001 - the figures are advisory
            logger.warning("Reading the archive size failed: %s", exc)
        return {
            "enabled": bool(archive.enabled),
            "pending_mb": pending,
            "archive_mb": archived,
            "batches": batches,
        }

    async def _quota(self) -> Dict[str, Any]:
        """Return the backend quota document, cached for a minute.

        Returns:
            The quota document, or an empty mapping when unavailable.
        """
        now = time.monotonic()
        cached_at, cached = self._quota_cache
        if cached and now - cached_at < _QUOTA_TTL_S:
            return dict(cached)
        client = self._client
        if client is None:
            return {}
        try:
            payload = await asyncio.wait_for(client.quota(), timeout=5.0)
        except (asyncio.TimeoutError, ClientError, BusyError) as exc:
            logger.debug("Quota request failed: %s", exc)
            return {}
        except Exception as exc:  # noqa: BLE001 - a probe must not break the page
            logger.debug("Quota request failed: %s", exc)
            return {}
        if not isinstance(payload, dict):
            return {}
        self._quota_cache = (now, dict(payload))
        return payload

    async def _backend_kind_now(self) -> str:
        """Classify the configured endpoint, cached for a short while.

        The official service has no health document, so the status reply has
        to say so explicitly instead of claiming the backend is offline.

        Returns:
            local, official or offline.
        """
        now = time.monotonic()
        cached_at, cached = self._backend_kind
        if cached and now - cached_at < _BACKEND_TTL_S:
            return cached
        client = self._client
        if client is None:
            return ""
        try:
            kind = await asyncio.wait_for(client.probe(), timeout=8.0)
        except Exception as exc:  # noqa: BLE001 - a probe must not fail the reply
            logger.debug("Backend probe failed: %s", exc)
            kind = ""
        if kind:
            self._backend_kind = (now, kind)
        return kind

    def _model_labels(self, health: Mapping[str, Any]) -> List[str]:
        """Return the model names to advertise in the status reply.

        Args:
            health: Raw backend health payload.

        Returns:
            The configured names, then any name the backend reports.
        """
        labels = _split_models(self._settings().model_names)
        if health:
            for item in _split_models(health.get("models")):
                if item not in labels:
                    labels.append(item)
        return labels

    async def _api_overview(self) -> Dict[str, Any]:
        """Return the whole dashboard document in one round trip.

        Returns:
            A JSON friendly mapping.
        """
        settings = self._settings()
        health: Dict[str, Any] = {}
        if self._client is not None:
            health = await self._client.health()
        archive = await self._archive_stats()
        quota = await self._quota()
        stats = self._stats.snapshot()
        stats["now"] = time.time()
        payload = overview_payload(
            settings=settings,
            stats=stats,
            health=health,
            queue=self._service.waiting if self._service is not None else 0,
            running=self._service.running if self._service is not None else 0,
            max_in_flight=(
                self._service.max_in_flight
                if self._service is not None
                else int(self._setting("max_in_flight", 2))
            ),
            archive=archive,
            names={"count": len(self._names), "loaded": self._names.loaded},
            memory=hostinfo.memory_snapshot(),
            cpu=hostinfo.cpu_snapshot(),
            models=self._model_labels(health),
            version=PLUGIN_VERSION,
            backend_kind=await self._backend_kind_now(),
        )
        payload["quota"] = quota
        return payload

    async def _api_settings(self) -> Any:
        """Read or update the settings from the dashboard.

        Returns:
            The settings document, or an error mapping the page can show.
        """
        from astrbot.api.web import request

        method = str(getattr(request, "method", "GET")).upper()
        if method != "POST":
            return settings_payload(self._settings())
        body = await request.json(default={})
        if not isinstance(body, Mapping):
            return {"status": "error", "message": "请求内容格式不正确"}
        patch = body.get("values", body)
        try:
            values = normalize_patch(patch)
            settings = await self._apply_settings(values)
        except SettingsError as exc:
            return {"status": "error", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001 - report, never a 500 page
            logger.warning("Saving the bird settings failed: %s", exc)
            return {"status": "error", "message": "保存失败，请稍后重试"}
        return settings_payload(settings)

    def _platforms(self) -> Sequence[Any]:
        """Return the live platform instances, empty when unavailable.

        Returns:
            A sequence of platform objects.
        """
        manager = getattr(self.context, "platform_manager", None)
        insts = getattr(manager, "platform_insts", None)
        if isinstance(insts, (list, tuple)):
            return list(insts)
        return []

    @staticmethod
    async def _call_action(platform: Any, action: str) -> Any:
        """Call one OneBot action on a platform, tolerating every failure.

        Args:
            platform: Platform instance.
            action: OneBot action name.

        Returns:
            The decoded action result, or None.
        """
        getter = getattr(platform, "get_client", None)
        if not callable(getter):
            return None
        try:
            client = getter()
        except Exception:  # noqa: BLE001 - a platform without a client is normal
            return None
        caller = getattr(client, "call_action", None)
        if not callable(caller):
            return None
        try:
            return await caller(action)
        except Exception:  # noqa: BLE001 - several adapters refuse friend lists
            return None

    async def _db_candidates(self) -> tuple:
        """Read chat candidates from the AstrBot conversation history.

        The plugin has no protocol-specific friend list on platforms such as
        ``qq_official`` or ``wechat_native``, so the sessions the bot has
        already talked to are the only reliable source. ``umo_aliases``
        carries the name the user gave a chat, ``conversations`` carries
        every unified message origin that ever produced a message; both are
        read so a whitelist entry can be picked instead of typed.

        Returns:
            Two lists of rows plus a list of notes for the page.
        """
        groups: List[Dict[str, Any]] = []
        friends: List[Dict[str, Any]] = []
        notes: List[str] = []
        database = getattr(self.context, "get_db", None)
        if not callable(database):
            return groups, friends, notes
        seen: set = set()

        def add(
            identifier: str, name: str, platform: str, is_group: Any = None
        ) -> None:
            """Append one candidate row, skipping blanks and repeats.

            Args:
                identifier: Whitelist-ready chat identifier.
                name: Display name shown in the picker.
                platform: Platform the chat belongs to.
                is_group: Known chat kind; inferred from the identifier
                    when left as None.
            """
            identifier = (identifier or "").strip()
            if not identifier:
                return
            # A group with per-speaker sessions is offered once, under the
            # identifier that also matches the live event. Normalise before the
            # duplicate check, otherwise every speaker adds the group again.
            identifier = _group_base(identifier) or identifier
            if identifier in seen:
                return
            seen.add(identifier)
            if is_group is None:
                is_group = "group" in _chat_kind(identifier)
            bucket = groups if is_group else friends
            bucket.append(
                {"id": identifier, "name": (name or "").strip(), "platform": platform}
            )

        try:
            db = database()
            if db is None:
                return groups, friends, notes
            # get_db is an async context manager, not an async iterator.
            async with db.get_db() as session:
                await self._read_umo_aliases(session, add)
                await self._read_conversations(session, add)
                await self._read_platform_sessions(session, add)
        except Exception as exc:  # noqa: BLE001 - the tables may not exist
            logger.debug("Reading chat candidates from the DB failed: %s", exc)
            notes.append("会话列表读取不完整，可手动填写会话标识。")
        groups.sort(key=lambda row: row.get("name") or row.get("id") or "")
        friends.sort(key=lambda row: row.get("name") or row.get("id") or "")
        return groups, friends, notes

    @staticmethod
    async def _read_umo_aliases(session: Any, add: Any) -> None:
        """Feed chats the user has renamed into the candidate collector."""
        try:
            from astrbot.core.db.po import UmoAlias
            from sqlalchemy import select

            result = await session.execute(select(UmoAlias))
            for row in result.scalars():
                add(
                    str(getattr(row, "umo", "") or ""),
                    str(
                        getattr(row, "user_alias", "")
                        or getattr(row, "auto_name", "")
                        or ""
                    ),
                    _umo_platform(str(getattr(row, "umo", "") or "")),
                )
        except Exception as exc:  # noqa: BLE001 - the table is optional
            logger.debug("Reading umo_aliases failed: %s", exc)

    @staticmethod
    async def _read_conversations(session: Any, add: Any) -> None:
        """Feed every unified message origin that produced a message."""
        try:
            from astrbot.core.db import po
            from sqlalchemy import select
        except Exception as exc:  # noqa: BLE001 - the package is optional
            logger.debug("Importing the conversation models failed: %s", exc)
            return
        for name in ("ConversationV3", "ConversationV2"):
            model = getattr(po, name, None)
            if model is None:
                continue
            try:
                result = await session.execute(select(model))
                for row in result.scalars():
                    umo = str(
                        getattr(row, "umo", "") or getattr(row, "user_id", "") or ""
                    )
                    add(umo, str(getattr(row, "title", "") or ""), _umo_platform(umo))
            except Exception as exc:  # noqa: BLE001 - the table is optional
                logger.debug("Reading %s failed: %s", name, exc)

    @staticmethod
    async def _read_platform_sessions(session: Any, add: Any) -> None:
        """Feed the web-chat style sessions kept by the platform table."""
        try:
            from astrbot.core.db.po import PlatformSession
            from sqlalchemy import select

            result = await session.execute(select(PlatformSession))
            for row in result.scalars():
                add(
                    str(getattr(row, "session_id", "") or ""),
                    str(getattr(row, "display_name", "") or ""),
                    str(getattr(row, "platform_id", "") or ""),
                    bool(int(getattr(row, "is_group", 0) or 0)),
                )
        except Exception as exc:  # noqa: BLE001 - the table is optional
            logger.debug("Reading platform_sessions failed: %s", exc)

    @staticmethod
    def _native_client(platform: Any) -> Any:
        """Return a platform client object, or None when it has none."""
        getter = getattr(platform, "get_client", None)
        if not callable(getter):
            return None
        try:
            return getter()
        except Exception:  # noqa: BLE001 - a missing client is normal
            return None

    async def _fill_wechat_names(self, groups: List[Dict[str, Any]]) -> None:
        """Give every ``wechat_native`` group the name the service knows.

        The adapter keeps one session per speaker and the conversation table
        stores no title, so those rows arrive unnamed. The native client can
        still describe a room; each lookup is best effort and bounded, and a
        failure simply leaves the row as it was.

        Args:
            groups: Candidate rows, updated in place.

        Returns:
            None
        """
        pending = [
            row
            for row in groups
            if not str(row.get("name") or "").strip()
            and _wechat_thread(str(row.get("id") or ""))
        ]
        if not pending:
            return
        client = None
        for platform in self._platforms():
            meta = getattr(platform, "meta", None)
            metadata = meta() if callable(meta) else None
            name = str(getattr(metadata, "name", "") or "")
            if name.casefold() == "wechat_native":
                client = self._native_client(platform)
                break
        if client is None:
            return
        lookup = getattr(client, "group", None)
        if not callable(lookup):
            return
        for row in pending[:12]:
            thread = _wechat_thread(str(row.get("id") or ""))
            try:
                document = await asyncio.wait_for(lookup(thread), timeout=5.0)
            except Exception as exc:  # noqa: BLE001 - the name is optional
                logger.debug("Reading the WeChat group name failed: %s", exc)
                continue
            if not isinstance(document, Mapping):
                continue
            label = str(
                document.get("displayName") or document.get("name") or ""
            ).strip()
            if label:
                row["name"] = label

    async def _api_chats(self) -> Dict[str, Any]:
        """Return the chats a whitelist can point at.

        Returns:
            A mapping with ``groups``, ``friends`` and ``errors``.
        """
        groups: List[Dict[str, Any]] = []
        friends: List[Dict[str, Any]] = []
        errors: List[str] = []
        for platform in self._platforms():
            kind = ""
            label = ""
            try:
                meta = getattr(platform, "meta", None)
                metadata = meta() if callable(meta) else None
                kind = str(getattr(metadata, "name", "") or "")
                label = str(
                    getattr(metadata, "id", "")
                    or getattr(metadata, "adapter_display_name", "")
                    or kind
                )
            except Exception:  # noqa: BLE001 - metadata is decorative
                kind = ""
                label = ""
            raw_groups = await self._call_action(platform, "get_group_list")
            if isinstance(raw_groups, list):
                for row in raw_groups:
                    if not isinstance(row, Mapping):
                        continue
                    identifier = str(
                        row.get("group_id") or row.get("group") or ""
                    ).strip()
                    if not identifier:
                        continue
                    groups.append(
                        {
                            "id": identifier,
                            "name": str(row.get("group_name") or "").strip(),
                            "platform": label,
                        }
                    )
            raw_friends = await self._call_action(platform, "get_friend_list")
            if isinstance(raw_friends, list):
                for row in raw_friends:
                    if not isinstance(row, Mapping):
                        continue
                    identifier = str(row.get("user_id") or row.get("uin") or "").strip()
                    if not identifier:
                        continue
                    friends.append(
                        {
                            "id": identifier,
                            "name": str(
                                row.get("remark")
                                or row.get("nickname")
                                or row.get("name")
                                or ""
                            ).strip(),
                            "platform": label,
                        }
                    )
        if not groups and not friends:
            db_groups, db_friends, db_notes = await self._db_candidates()
            groups.extend(db_groups)
            friends.extend(db_friends)
            errors.extend(db_notes)
        await self._fill_wechat_names(groups)
        return chat_candidates_payload(groups, friends, errors=errors)

    # ------------------------------------------------------------------
    # admin command
    # ------------------------------------------------------------------
    @filter.command("识鸟状态")
    @filter.permission_type(filter.PermissionType.ADMIN)
    async def birdstatus(self, event: AstrMessageEvent) -> None:
        """Report the plugin and backend health to an administrator.

        Args:
            event: Message event.

        Returns:
            None
        """
        if not event.is_admin():
            await self._reply(event, "该指令仅限管理员使用。")
            return
        service = await self._ensure_service()
        if service is None or self._client is None or self._archive is None:
            await self._reply(event, "插件尚未就绪。")
            return
        health: Dict[str, Any] = {}
        try:
            health = await self._client.health()
        except Exception as exc:  # noqa: BLE001 - health never fails the reply
            logger.warning("Backend health check failed: %s", exc)
        online = bool(health)
        settings = self._settings()
        backend_kind = await self._backend_kind_now()
        text = format_status(
            online=online,
            health=health,
            queue=service.waiting,
            running=service.running,
            archive=await self._archive_stats(),
            names={"count": len(self._names), "loaded": self._names.loaded},
            memory=hostinfo.memory_snapshot(),
            models={
                "names": self._model_labels(health),
                "status": str(health.get("status") or "") if health else "",
            },
            toggles={
                "auto_reply": settings.auto_reply,
                "group_whitelist": settings.group_whitelist_enabled,
                "private_whitelist": settings.private_whitelist_enabled,
            },
            backend_kind=backend_kind,
        )
        await self._reply(event, text)

    @staticmethod
    async def _respond(event: Any, content: Any, *, stop: bool) -> None:
        """Send one reply and optionally stop the event.

        Args:
            event: Message event.
            content: A string or a list of message components.
            stop: Whether to stop further event propagation.

        Returns:
            None
        """
        chain = [Plain(content)] if isinstance(content, str) else list(content)
        try:
            await event.send(event.chain_result(chain))
        except Exception as exc:  # noqa: BLE001 - a send failure is reported
            logger.warning("Sending the bird reply failed: %s", exc)
        finally:
            if stop:
                event.stop_event()

    @staticmethod
    async def _reply(event: Any, content: Any) -> None:
        """Send one reply and always stop the event.

        Args:
            event: Message event.
            content: A string or a list of message components.

        Returns:
            None
        """
        await BirdIdPlugin._respond(event, content, stop=True)
