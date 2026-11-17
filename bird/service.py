"""Coordinate one identification request from chat to reply.

The orchestrator owns the parts that must not be spread across a chat handler:
a bounded in-flight budget so a flood of messages cannot pile up behind the
slow backend, the media preparation steps, the name lookup and the archive
hand-off.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from . import media as media_tools
from .archive import ArchiveStore
from .client import BirdClient, BirdResult
from .format import Observation
from .names import NameResolver
from .regions import resolve_location
from .render import crop_image, image_to_jpeg_bytes, spectrogram_image

__all__ = ["BirdService", "PreparedReply", "ServiceBusy"]

_LOGGER = logging.getLogger(__name__)

#: Candidates shown per detection.
MAX_CANDIDATES = 3

#: Detections rendered per request; the rest only appear as text.
MAX_RENDERED = 3

#: Spectrogram geometry used for audio replies.
SPECTROGRAM_WIDTH = 900
SPECTROGRAM_HEIGHT = 420


class ServiceBusy(RuntimeError):
    """Raised when this plugin already handles as many requests as it allows."""


@dataclass
class PreparedReply:
    """Everything needed to answer one request.

    Attributes:
        text: Reply text.
        images: JPEG payloads to attach, in display order.
        observations: Species shown in the text.
        elapsed_ms: Measured processing time.
        kind: "image" or "audio".
    """

    text: str
    images: List[bytes] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    elapsed_ms: float = 0.0
    kind: str = "image"
    location: str = ""


class BirdService:
    """Run identification requests against the local backend.

    Attributes:
        client: Backend client.
        names: Bilingual name table.
        archive: Observation archive.
    """

    def __init__(
        self,
        client: BirdClient,
        names: NameResolver,
        archive: ArchiveStore,
        *,
        max_in_flight: int = 2,
        request_timeout_s: float = 30.0,
        ffmpeg: str = "",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Prepare the service without starting background work.

        Args:
            client: Backend client.
            names: Bilingual name table.
            archive: Observation archive.
            max_in_flight: Requests this plugin processes at the same time.
            request_timeout_s: Deadline handed to the backend client.
            ffmpeg: Optional ffmpeg command used to decode recordings.
            logger: Optional logger override.
        """
        self.client = client
        self.names = names
        self.archive = archive
        self.max_in_flight = max(1, int(max_in_flight))
        self.request_timeout_s = max(1.0, float(request_timeout_s))
        self._ffmpeg = ffmpeg
        self._logger = logger or _LOGGER
        self._slots = asyncio.Semaphore(self.max_in_flight)
        self._waiting = 0
        self._running = 0
        self._lock = asyncio.Lock()

    @property
    def waiting(self) -> int:
        """Number of requests waiting for a worker slot."""
        return self._waiting

    @property
    def running(self) -> int:
        """Number of requests currently being processed."""
        return self._running

    async def _enter(self) -> None:
        """Take a worker slot, counting the wait.

        Returns:
            None
        """
        self._waiting += 1
        try:
            await self._slots.acquire()
        finally:
            self._waiting -= 1
        async with self._lock:
            self._running += 1

    async def _leave(self) -> None:
        """Release a worker slot."""
        async with self._lock:
            self._running = max(0, self._running - 1)
        self._slots.release()

    async def identify(
        self,
        items: Sequence[media_tools.MediaItem],
        workdir: Path,
        *,
        location: str = "",
        request_id: str = "",
    ) -> PreparedReply:
        """Identify one message worth of media.

        Args:
            items: Media extracted from the message.
            workdir: Temporary directory owned by this request.
            location: Optional location text from the command arguments.
            request_id: Optional anonymous id used by the archive.

        Returns:
            The prepared reply.

        Raises:
            ServiceBusy: Every worker slot is already busy.
            ClientError: The backend rejected the request.
            asyncio.TimeoutError: The backend missed its deadline.
            ValueError: The media could not be prepared or was not found.
        """
        started = time.monotonic()
        if any(item.kind == "image" for item in items):
            kind = "image"
        elif items:
            kind = "audio"
        else:
            raise ValueError("no media")
        if self._slots.locked():
            raise ServiceBusy("every worker slot is busy")
        await self._enter()
        try:
            area, place = resolve_location(location) if location else (None, "")
            if kind == "image":
                return await self._identify_image(
                    items, workdir, area, place, started, request_id
                )
            return await self._identify_audio(
                items, workdir, area, place, started, request_id
            )
        finally:
            await self._leave()

    async def _identify_image(
        self,
        items: Sequence[media_tools.MediaItem],
        workdir: Path,
        area: Optional[str],
        place: str,
        started: float,
        request_id: str,
    ) -> PreparedReply:
        """Identify the first image of a message.

        Args:
            items: Media extracted from the message.
            workdir: Temporary directory owned by this request.
            area: Optional region code.
            place: Display name of the applied location.
            started: Monotonic start time of the request.
            request_id: Anonymous request id.

        Returns:
            The prepared reply.
        """
        source = next(item.path for item in items if item.kind == "image")
        image_path = media_tools.normalize_image(source, workdir / "upload.jpg")
        payload = image_path.read_bytes()
        result = await self.client.identify_image(
            payload,
            filename="upload.jpg",
            area=area or "",
            deadline_s=self.request_timeout_s,
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        observations = self._observations(result)
        images: List[bytes] = []
        for detection in result.detections[:MAX_RENDERED]:
            crop = crop_image(image_path, detection.box, margin_ratio=0.15)
            if crop is None:
                continue
            try:
                images.append(image_to_jpeg_bytes(crop))
            except Exception as exc:  # noqa: BLE001 - a crop is optional
                self._logger.debug("crop encoding failed: %s", exc)
        self._store(
            source="image",
            detection=result.top,
            place=place,
            elapsed_ms=elapsed_ms,
            request_id=request_id,
            media=[("frame.jpg", payload)],
        )
        return PreparedReply(
            text="",
            images=images,
            observations=observations,
            elapsed_ms=elapsed_ms,
            kind="image",
            location=place,
        )

    async def _identify_audio(
        self,
        items: Sequence[media_tools.MediaItem],
        workdir: Path,
        area: Optional[str],
        place: str,
        started: float,
        request_id: str,
    ) -> PreparedReply:
        """Identify the first recording of a message.

        Args:
            items: Media extracted from the message.
            workdir: Temporary directory owned by this request.
            area: Optional region code.
            place: Display name of the applied location.
            started: Monotonic start time of the request.
            request_id: Anonymous request id.

        Returns:
            The prepared reply.
        """
        source = next(item.path for item in items if item.kind == "audio")
        wav_path = await media_tools.transcode_audio(
            source, workdir / "upload.wav", ffmpeg=self._ffmpeg
        )
        if media_tools.trim_to_limit(wav_path, media_tools.MAX_AUDIO_BYTES) is None:
            raise ValueError("the recording stays above the upload limit")
        payload = wav_path.read_bytes()
        result = await self.client.identify_sound(
            payload,
            filename="upload.wav",
            area=area or "",
            deadline_s=self.request_timeout_s,
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        observations = self._observations(result)
        images: List[bytes] = []
        spectrogram = self._spectrogram(wav_path)
        if spectrogram is not None:
            images.append(spectrogram)
        self._store(
            source="audio",
            detection=result.top if result.detections else None,
            place=place,
            elapsed_ms=elapsed_ms,
            request_id=request_id,
            media=[("clip.wav", payload)],
        )
        return PreparedReply(
            text="",
            images=images,
            observations=observations,
            elapsed_ms=elapsed_ms,
            kind="audio",
            location=place,
        )

    def _spectrogram(self, wav_path: Path) -> Optional[bytes]:
        """Render the spectrogram of a prepared recording.

        Args:
            wav_path: Mono 32 kHz WAV produced for the upload.

        Returns:
            JPEG bytes, or None when the recording cannot be rendered.
        """
        loaded = media_tools.read_pcm_samples(wav_path)
        if loaded is None:
            return None
        samples, rate = loaded
        duration = media_tools.audio_duration_seconds(wav_path)
        title = None
        if duration:
            title = "%.1f s @ %d Hz" % (duration, rate)
        try:
            image = spectrogram_image(
                samples,
                rate,
                width=SPECTROGRAM_WIDTH,
                height=SPECTROGRAM_HEIGHT,
                title=title,
            )
            return image_to_jpeg_bytes(image)
        except Exception as exc:  # noqa: BLE001 - the plot is optional
            self._logger.debug("spectrogram rendering failed: %s", exc)
            return None

    def _observations(self, result: BirdResult) -> List[Observation]:
        """Turn backend detections into display rows.

        Args:
            result: Parsed backend reply.

        Returns:
            Observations ordered by the backend ranking.
        """
        rows: List[Observation] = []
        for detection in result.detections:
            for species in detection.species[:MAX_CANDIDATES]:
                if species.animal_class == "B":
                    zh, en = self.names.resolve(
                        species.scientific,
                        fallback_en=species.english,
                    )
                    # The image classifier puts an English common name in the
                    # scientific slot, so only recordings carry a real one.
                    scientific = species.scientific if result.kind == "audio" else ""
                else:
                    zh, en = species.common, species.english
                    scientific = species.scientific
                rows.append(
                    Observation(
                        zh=zh,
                        en=en,
                        scientific=scientific,
                        confidence=species.confidence,
                        gate_box=detection.box,
                        gate_confidence=detection.gate_confidence,
                        animal_class=species.animal_class,
                    )
                )
        return rows

    def _store(
        self,
        *,
        source: str,
        detection,
        place: str,
        elapsed_ms: float,
        request_id: str,
        media: Optional[list] = None,
    ) -> None:
        """Write one observation into the archive.

        Args:
            source: "image" or "audio".
            detection: Top detection, or None when nothing was found.
            place: Display name of the applied location.
            elapsed_ms: Processing time for the request.
            request_id: Anonymous request id.
            media: Optional (filename, bytes) pairs stored next to the record.

        Returns:
            None
        """
        species = {}
        confidence = None
        gate = None
        if detection is not None and detection.species:
            top = detection.species[0]
            if top.animal_class == "B":
                zh, en = self.names.resolve(top.scientific, fallback_en=top.english)
            else:
                zh, en = top.common, top.english
            # Only recordings carry a real scientific name; the image
            # classifier reports an English common name in that slot.
            scientific = top.scientific if source == "audio" else ""
            species = {"zh": zh, "en": en, "sci": scientific}
            confidence = top.confidence
            gate = list(detection.box)
        try:
            self.archive.record(
                source=source,
                species=species,
                confidence=confidence,
                gate=gate,
                location=place or None,
                elapsed_ms=elapsed_ms,
                request=request_id or None,
                media=media,
            )
        except Exception as exc:  # noqa: BLE001 - archiving must never fail a reply
            self._logger.warning("archive write failed: %s", exc)
