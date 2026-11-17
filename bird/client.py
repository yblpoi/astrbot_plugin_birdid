"""Async client for the local bird identification service.

The service speaks the Dongniao v2 protocol over a single POST endpoint. A
request that carries media is accepted asynchronously: the first response
returns code 1001 together with a result id, and the caller polls with that id
until the payload is ready.

Busy states are surfaced as exceptions instead of being retried silently, so
the plugin can tell the user to try again instead of making them wait behind an
unbounded queue.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

__all__ = [
    "BirdClient",
    "BirdResult",
    "BusyError",
    "ClientError",
    "Detection",
    "Species",
]

_LOGGER = logging.getLogger(__name__)

#: Protocol codes defined by the Dongniao v2 specification.
CODE_OK = 1000
CODE_PROCESSING = 1001
CODE_SOUND_ERROR = 1002
CODE_QUOTA = 1007
CODE_NO_ANIMALS = 1010

#: The local deployment reports an exhausted queue with this literal message.
BUSY_MESSAGE = "server busy"

DEFAULT_POLL_INTERVAL_S = 0.15
DEFAULT_TIMEOUT_S = 30.0


class ClientError(RuntimeError):
    """Raised when the service answers with a business level failure.

    Attributes:
        code: Protocol code that came back from the service.
        message: Human readable reason reported by the service.
    """

    def __init__(self, code: int, message: str) -> None:
        """Store the protocol code alongside the service message.

        Args:
            code: Numeric protocol code.
            message: Message reported by the service.
        """
        super().__init__("[%d] %s" % (code, message))
        self.code = code
        self.message = message


class BusyError(ClientError):
    """Raised when the service refused the request because it is saturated."""

    def __init__(self, message: str = BUSY_MESSAGE) -> None:
        """Build a busy error carrying the saturation code.

        Args:
            message: Message reported by the service.
        """
        super().__init__(CODE_QUOTA, message)


@dataclass(frozen=True)
class Species:
    """One candidate name triple reported by the service.

    Attributes:
        confidence: Confidence in percent, 0 to 100.
        scientific: Scientific (Latin) name.
        english: English name as reported by the backend, may be empty.
        animal_class: One letter class code, B for birds.
        common: First label segment, which the image backend fills with the
            Latin name and the audio backend fills with the English name. For
            non-bird detections it carries the Chinese name.
    """

    confidence: float
    scientific: str
    english: str
    animal_class: str = "B"
    common: str = ""


@dataclass(frozen=True)
class Detection:
    """One detection region and its ranked candidates.

    Attributes:
        box: Detection box as (x1, y1, x2, y2) in source pixels.
        species: Candidates ordered by descending confidence.
    """

    box: tuple
    species: List[Species] = field(default_factory=list)

    @property
    def gate_confidence(self) -> float:
        """Return the confidence of the detection gate.

        The gate model reports its own score in the same candidate entry that
        carries the crop, so the best candidate doubles as the gate score.

        Returns:
            Confidence in percent, or 0.0 when the detection has no candidate.
        """
        return self.species[0].confidence if self.species else 0.0


@dataclass(frozen=True)
class BirdResult:
    """Parsed service reply.

    Attributes:
        detections: Detection regions in the order the service returned them.
        kind: Either "image" or "audio", reflecting the producing request.
        elapsed_ms: Wall clock time spent talking to the service, in ms.
    """

    detections: List[Detection] = field(default_factory=list)
    kind: str = "image"
    elapsed_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Whether the service found nothing usable in the request."""
        return not self.detections

    @property
    def top(self) -> Optional[Detection]:
        """Return the highest ranked detection, or None when empty."""
        return self.detections[0] if self.detections else None


def _parse_species(item: Any) -> Optional[Species]:
    """Parse one [confidence, label, animal_id, class] candidate.

    The label is a pipe separated triple of common, English and scientific
    names. The local deployment does not always fill every segment for birds,
    so the scientific name falls back to the first segment.

    Args:
        item: Raw candidate entry from the service.

    Returns:
        A Species instance, or None when the entry is malformed.
    """
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    try:
        confidence = float(item[0])
    except (TypeError, ValueError):
        return None
    parts = [part.strip() for part in str(item[1]).split("|")]
    while len(parts) < 3:
        parts.append("")
    animal_class = str(item[3]) if len(item) > 3 else "B"
    return Species(
        confidence=confidence,
        scientific=parts[2] or parts[0],
        english=parts[1] or parts[0],
        animal_class=animal_class,
        common=parts[0],
    )


def _decode_payload(payload: Any, kind: str, elapsed_ms: float) -> BirdResult:
    """Turn the raw JSON payload into a BirdResult.

    Args:
        payload: Raw payload from the service.
        kind: Request kind used to annotate the result.
        elapsed_ms: Measured duration for the result.

    Returns:
        A BirdResult holding every usable detection in the payload.
    """
    detections: List[Detection] = []
    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            box = entry.get("box")
            if isinstance(box, (list, tuple)) and not box:
                # Recordings carry no gate box; the candidates still count.
                coords = ()
            elif not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            else:
                try:
                    coords = tuple(float(value) for value in box)
                except (TypeError, ValueError):
                    continue
            species = [
                parsed
                for parsed in (_parse_species(item) for item in entry.get("list") or [])
                if parsed is not None
            ]
            if species:
                detections.append(Detection(box=coords, species=species))
    return BirdResult(detections=detections, kind=kind, elapsed_ms=elapsed_ms)


def _error_or_result(
    code: Any, payload: Any, kind: str, elapsed_ms: float
) -> BirdResult:
    """Map a protocol reply onto either a result or an exception.

    Args:
        code: Protocol code from the service.
        payload: Payload that came with the code.
        kind: Request kind used to annotate the result.
        elapsed_ms: Measured duration.

    Returns:
        A BirdResult for success and for the "nothing found" code.

    Raises:
        BusyError: The service queue is full.
        ClientError: Any other business failure reported by the service.
    """
    if code == CODE_OK:
        return _decode_payload(payload, kind, elapsed_ms)
    if code == CODE_NO_ANIMALS:
        return BirdResult(kind=kind, elapsed_ms=elapsed_ms)
    message = payload if isinstance(payload, str) else str(payload)
    if code == CODE_QUOTA and message.strip().lower() == BUSY_MESSAGE:
        raise BusyError(message)
    raise ClientError(int(code) if isinstance(code, int) else 0, message)


class BirdClient:
    """Client for the local Dongniao compatible endpoint.

    The client keeps one aiohttp session per instance and never retries a busy
    answer: saturation is reported immediately so the chat layer can respond,
    rather than letting requests pile up behind the service queue.

    Attributes:
        base_url: Service base URL without a trailing slash.
        api_key: Optional key sent in the configured header.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        api_key_header: str = "api_key",
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_concurrency: int = 4,
    ) -> None:
        """Prepare the client without opening a connection.

        Args:
            base_url: Service base URL, for example http://127.0.0.1:8080.
            api_key: Optional key forwarded to the service.
            api_key_header: Header used to carry the key.
            poll_interval_s: Delay between polls while a task is running.
            timeout_s: Overall deadline for one submission and its polls.
            max_concurrency: Local in-flight request budget.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.api_key_header = api_key_header
        self.poll_interval_s = max(0.02, float(poll_interval_s))
        self.timeout_s = max(1.0, float(timeout_s))
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return the shared session, creating it on first use.

        Returns:
            An open aiohttp session.
        """
        if self._session is not None and not self._session.closed:
            return self._session
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout_s)
                )
        return self._session

    async def close(self) -> None:
        """Close the shared session; safe to call more than once."""
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()

    def _headers(self) -> Dict[str, str]:
        """Build request headers, including the optional service key.

        Returns:
            A header mapping for one request.
        """
        if not self.api_key:
            return {}
        return {self.api_key_header: self.api_key}

    async def health(self) -> Dict[str, Any]:
        """Read the service health document.

        Returns:
            The decoded health payload; an empty mapping when unavailable.
        """
        try:
            session = await self._get_session()
            async with session.get(self.base_url + "/healthz") as response:
                if response.status != 200:
                    return {}
                payload = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError) as exc:
            _LOGGER.debug("health request failed: %s", exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    async def _submit(
        self,
        data: aiohttp.FormData,
        kind: str,
        deadline_s: Optional[float],
    ) -> BirdResult:
        """Submit one job and poll it until the payload is ready.

        Args:
            data: Form payload describing the job.
            kind: Either "image" or "audio".
            deadline_s: Optional per-request deadline in seconds.

        Returns:
            The parsed identification result.

        Raises:
            BusyError: The service is saturated.
            ClientError: Any other protocol level failure.
            asyncio.TimeoutError: The job did not finish within the deadline.
            aiohttp.ClientError: The transport itself failed.
        """
        limit = self.timeout_s if deadline_s is None else max(1.0, float(deadline_s))
        loop = asyncio.get_running_loop()
        started = loop.time()
        async with self._semaphore:
            session = await self._get_session()
            async with session.post(
                self.base_url + "/dongniao", data=data, headers=self._headers()
            ) as response:
                if response.status != 200:
                    raise ClientError(0, "http %d" % response.status)
                reply = await response.json(content_type=None)
            code, payload = _split_reply(reply)
            if code != CODE_PROCESSING or not isinstance(payload, str):
                return _error_or_result(
                    code, payload, kind, (loop.time() - started) * 1000.0
                )
            result_id = payload
            while loop.time() - started < limit:
                await asyncio.sleep(self.poll_interval_s)
                async with session.post(
                    self.base_url + "/dongniao",
                    data={"resultid": result_id},
                    headers=self._headers(),
                ) as response:
                    if response.status != 200:
                        raise ClientError(0, "http %d" % response.status)
                    reply = await response.json(content_type=None)
                code, payload = _split_reply(reply)
                if code != CODE_PROCESSING:
                    return _error_or_result(
                        code, payload, kind, (loop.time() - started) * 1000.0
                    )
        raise asyncio.TimeoutError("identification did not finish in time")

    async def identify_image(
        self,
        image_bytes: bytes,
        *,
        filename: str = "image.jpg",
        area: str = "",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        deadline_s: Optional[float] = None,
    ) -> BirdResult:
        """Identify the birds in one image.

        Args:
            image_bytes: Encoded image payload.
            filename: Filename advertised in the multipart body.
            area: Optional Dongniao region code such as CN-BJ.
            latitude: Optional latitude for the geographic prior.
            longitude: Optional longitude for the geographic prior.
            deadline_s: Optional per-request deadline in seconds.

        Returns:
            The parsed identification result.
        """
        form = aiohttp.FormData()
        form.add_field("image", image_bytes, filename=filename)
        _add_geo(form, area, latitude, longitude)
        return await self._submit(form, "image", deadline_s)

    async def identify_sound(
        self,
        sound_bytes: bytes,
        *,
        filename: str = "sound.wav",
        area: str = "",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        deadline_s: Optional[float] = None,
    ) -> BirdResult:
        """Identify the birds in one recording.

        The service only decodes 16 bit PCM WAV, so callers must transcode
        other containers before reaching this method.

        Args:
            sound_bytes: Encoded audio payload.
            filename: Filename advertised in the multipart body.
            area: Optional Dongniao region code such as CN-BJ.
            latitude: Optional latitude for the geographic prior.
            longitude: Optional longitude for the geographic prior.
            deadline_s: Optional per-request deadline in seconds.

        Returns:
            The parsed identification result.
        """
        form = aiohttp.FormData()
        form.add_field("sound", sound_bytes, filename=filename)
        _add_geo(form, area, latitude, longitude)
        return await self._submit(form, "audio", deadline_s)


def _split_reply(reply: Any) -> tuple:
    """Split the [code, payload] envelope the service always returns.

    Args:
        reply: Decoded JSON value.

    Returns:
        A (code, payload) tuple; (0, reply) when the envelope is malformed.
    """
    if isinstance(reply, (list, tuple)) and len(reply) >= 2:
        return reply[0], reply[1]
    return 0, "unexpected reply"


def _add_geo(
    form: aiohttp.FormData,
    area: str,
    latitude: Optional[float],
    longitude: Optional[float],
) -> None:
    """Attach the optional geographic hints to a form body.

    Args:
        form: Multipart body under construction.
        area: Optional region code.
        latitude: Optional latitude.
        longitude: Optional longitude.

    Returns:
        None
    """
    if latitude is not None and longitude is not None:
        form.add_field("latitude", "%.6f" % float(latitude))
        form.add_field("longitude", "%.6f" % float(longitude))
    elif area:
        form.add_field("area", area)
