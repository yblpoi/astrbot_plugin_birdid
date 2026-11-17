"""AstrBot entry point for the local bird identification plugin.

The plugin answers two commands: `/birds` identifies an attached or quoted
image or recording through a local identification service, and `/birdstatus`
reports backend health to administrators only.

Design notes:

* A plain image message activates no handler, so the waking stage stops it and
  no language model is called. Replies are sent explicitly and the event is
  then stopped, which keeps the language model out of the loop even when the
  command itself is a wake message.
* Every file this plugin creates for one request lives in a temporary directory
  that is removed before the handler returns; the archive keeps its own copies.
* Nothing in a reply names a model, a device or a path on disk.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, StarTools

from .bird import media as media_tools
from .bird.archive import ArchiveStore
from .bird.client import BirdClient, ClientError
from .bird.format import format_reply, format_status
from .bird.names import NameResolver
from .bird.service import BirdService, ServiceBusy

PLUGIN_NAME = "astrbot_plugin_birdid"

#: Shown when a message carries no usable attachment.
MEDIA_HINT = "请附上一张图片或一段录音，也可以引用一条包含图片或录音的消息。"

#: Shown when the plugin cannot be started at all.
UNAVAILABLE_HINT = "鸟类识别暂时不可用，请联系管理员。"


class BirdIdPlugin(Star):
    """Identify birds from photos and recordings with the local service."""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        """Read the plugin configuration.

        Args:
            context: AstrBot context.
            config: Plugin configuration from the dashboard.
        """
        super().__init__(context)
        self.config = config or {}
        self._service: Optional[BirdService] = None
        self._client: Optional[BirdClient] = None
        self._archive: Optional[ArchiveStore] = None
        self._names = NameResolver()
        self._archive_task: Optional[asyncio.Task] = None

    def _setting(self, key: str, default):
        """Read one configuration value.

        Args:
            key: Configuration key.
            default: Value used when the key is absent or empty.

        Returns:
            The configured value, or the default.
        """
        value = (
            self.config.get(key, default) if hasattr(self.config, "get") else default
        )
        if value is None or value == "":
            return default
        return value

    async def initialize(self) -> None:
        """Create the clients and start the archive maintenance loop."""
        self._names.load()
        data_dir = Path(StarTools.get_data_dir(PLUGIN_NAME))
        self._archive = ArchiveStore(
            data_dir / "bird_archive",
            retention_days=int(self._setting("retention_days", 7)),
            max_total_bytes=int(self._setting("archive_max_mb", 2048)) * 1024 * 1024,
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
            ffmpeg=str(self._setting("ffmpeg_path", "")),
            logger=self.logger,
        )
        self._archive_task = asyncio.create_task(
            self._archive.run_forever(
                interval_seconds=float(self._setting("archive_interval_s", 900)),
            )
        )
        logger.info("Bird identification plugin ready with %d names.", len(self._names))

    async def terminate(self) -> None:
        """Stop background work and release the HTTP session."""
        if self._archive_task is not None:
            self._archive_task.cancel()
            try:
                await self._archive_task
            except (asyncio.CancelledError, Exception):
                pass
            self._archive_task = None
        if self._archive is not None:
            try:
                self._archive.flush()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.warning("Bird archive flush failed: %s", exc)
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._service = None
        logger.info("Bird identification plugin stopped.")

    async def _ensure_service(self) -> Optional[BirdService]:
        """Return the service, creating it when the plugin was not initialised.

        Returns:
            The service, or None when it cannot be created.
        """
        if self._service is not None:
            return self._service
        try:
            await self.initialize()
        except Exception as exc:  # noqa: BLE001 - report instead of failing the event
            logger.error("Bird plugin initialisation failed: %s", exc)
            return None
        return self._service

    @filter.command("birds")
    async def birds(
        self,
        event: AstrMessageEvent,
        location: str = "",
    ) -> None:
        """Identify the bird in an attached or quoted photo or recording.

        Args:
            event: Message event that triggered the command.
            location: Optional province or country used as a range hint.

        Returns:
            None; the reply is sent from inside the handler.
        """
        service = await self._ensure_service()
        if service is None:
            await self._reply(event, UNAVAILABLE_HINT)
            return

        started = time.monotonic()
        workdir = Path(tempfile.mkdtemp(prefix="birdid-"))
        try:
            try:
                items = await media_tools.collect_media(event, workdir)
            except Exception as exc:  # noqa: BLE001 - a bad attachment is not fatal
                logger.warning("Reading attachments failed: %s", exc)
                items = []
            if not items:
                await self._reply(event, MEDIA_HINT)
                return

            try:
                reply = await service.identify(
                    items,
                    workdir,
                    location=location,
                    request_id=uuid.uuid4().hex[:12],
                )
            except ServiceBusy:
                logger.info("Rejecting request because every worker slot is busy.")
                await self._reply(event, "现在有点忙，请稍后再试。")
                return
            except ClientError as exc:
                logger.warning("Backend rejected the request: %s", exc)
                await self._reply(event, "识别服务暂时不可用，请稍后再试。")
                return
            except ValueError as exc:
                logger.warning("Preparing media failed: %s", exc)
                await self._reply(event, "这段附件无法读取，请换一张图片或一段录音。")
                return

            elapsed_ms = max(reply.elapsed_ms, (time.monotonic() - started) * 1000.0)
            text = format_reply(
                reply.observations,
                kind=reply.kind,
                elapsed_ms=elapsed_ms,
                location=reply.location,
            )
            chain = [Plain(text)]
            for payload in reply.images:
                chain.append(Image.fromBytes(payload))
            await self._reply(event, chain)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @filter.command("birdstatus")
    async def birdstatus(self, event: AstrMessageEvent) -> None:
        """Report backend health and archive usage to administrators.

        Args:
            event: Message event that triggered the command.

        Returns:
            None; the report is sent from inside the handler.
        """
        if not event.is_admin():
            await self._reply(event, "该指令仅限管理员使用。")
            return
        service = await self._ensure_service()
        if service is None or self._client is None or self._archive is None:
            await self._reply(event, "插件尚未就绪。")
            return
        health = {}
        online = False
        try:
            health = await self._client.health()
            online = True
        except Exception as exc:  # noqa: BLE001 - an offline backend is a fact
            logger.warning("Backend health check failed: %s", exc)
        archive_stats = {
            "pending_mb": self._archive.pending_bytes() / (1024 * 1024),
            "archive_mb": self._archive.archive_bytes() / (1024 * 1024),
            "batches": len(list(self._archive.archive_root.glob("*/"))),
        }
        text = format_status(
            online=online,
            health=health,
            queue=service.waiting,
            running=service.running,
            archive=archive_stats,
            names={"count": len(self._names), "loaded": self._names.loaded},
        )
        await self._reply(event, text)

    @staticmethod
    async def _reply(event: AstrMessageEvent, content) -> None:
        """Send one reply and stop the event before the language model runs.

        Sending explicitly sets the event send flag, which keeps the default
        language model request away; stopping afterwards ends the pipeline.

        Args:
            event: Message event to answer.
            content: Either a text string or a list of message components.

        Returns:
            None
        """
        chain = [Plain(content)] if isinstance(content, str) else list(content)
        try:
            await event.send(event.chain_result(chain))
        except Exception as exc:  # noqa: BLE001 - never raise into the pipeline
            logger.warning("Sending the bird reply failed: %s", exc)
        finally:
            event.stop_event()
