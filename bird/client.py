"""Async client for the Dongniao v2 bird identification service.

The service speaks the Dongniao v2 protocol over one POST endpoint and always
answers with a `[code, payload]` envelope. A request that carries media is
accepted asynchronously: the first reply returns code 1001 with a result id and
the caller polls with that id until the payload is ready.

Every documented field of the protocol is supported so the plugin can be pointed
at the official endpoint as well as at a local deployment:

* `image` + `upload=1`, or `url` for a remote photo;
* `sound` + `upload=1`, or `sound_url` for a remote recording;
* `resultid` to collect a running job;
* `animalid` (+ `class`) for the encyclopedia entry;
* `quota=1` for the remaining allowance.

Busy states are surfaced as exceptions instead of being retried silently, so the
plugin can tell the user to try again instead of making them wait behind an
unbounded queue.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

__all__ = [
    "BACKEND_LOCAL",
    "BACKEND_OFFICIAL",
    "BACKEND_OFFLINE",
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
CODE_DID_ERROR = 1003
CODE_CLASS_ERROR = 1004
CODE_AREA_ERROR = 1005
CODE_IP_ERROR = 1006
CODE_QUOTA = 1007
CODE_NO_TARGET = 1008
CODE_AREA_FORMAT_ERROR = 1009
CODE_NO_ANIMALS = 1010
CODE_OUT_OF_RANGE = 1011

#: Codes that mean "the request worked but the answer is empty".
EMPTY_CODES = (CODE_NO_TARGET, CODE_NO_ANIMALS, CODE_OUT_OF_RANGE)

#: Local deployments expose a health document; the official service does not.
BACKEND_LOCAL = "local"
BACKEND_OFFICIAL = "official"
BACKEND_OFFLINE = "offline"

#: Element codes the protocol accepts in the `class` field.
VALID_CLASS_CODES = ("B", "M", "A", "R", "F")

#: The local deployment reports an exhausted queue with this literal message.
BUSY_MESSAGE = "server busy"

DEFAULT_POLL_INTERVAL_S = 0.15
DEFAULT_TIMEOUT_S = 30.0

#: Human readable fallback for codes whose payload carries no message.
_CODE_MESSAGES = {
    CODE_DID_ERROR: "invalid did",
    CODE_CLASS_ERROR: "invalid class",
    CODE_AREA_ERROR: "invalid area",
    CODE_IP_ERROR: "ip is not allowed",
    CODE_QUOTA: "quota exhausted",
    CODE_NO_TARGET: "no target found",
    CODE_AREA_FORMAT_ERROR: "invalid area code",
    CODE_NO_ANIMALS: "no animal recognised",
    CODE_OUT_OF_RANGE: "species out of the area range",
}


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


def normalize_class(value: Any) -> str:
    """Reduce a caller supplied element selector to a protocol class code.

    The protocol accepts any combination of B, M, A, R and F. Unknown letters
    are dropped instead of being rejected, so a typo degrades to a wider search
    rather than to a failed request.

    Args:
        value: Raw selector from the configuration or a caller.

    Returns:
        The sanitised code, or an empty string when nothing valid remains.
    """
    if not isinstance(value, str):
        return ""
    seen: List[str] = []
    for char in value.upper():
        if char in VALID_CLASS_CODES and char not in seen:
            seen.append(char)
    return "".join(seen)


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


def _sound_species(entries: Any) -> List[Species]:
    """Parse the summary rows of a recording reply.

    The official protocol reports a recording as
    `[species_id, confidence, scientific, chinese]` with a confidence between
    0 and 1. Both a ratio and a percentage are accepted so a deployment that
    already scaled the score keeps working.

    Args:
        entries: The `summary` (or one `splits` element) of the payload.

    Returns:
        Species in descending confidence order.
    """
    rows: List[Species] = []
    if not isinstance(entries, (list, tuple)):
        return rows
    for item in entries:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            score = float(item[1])
        except (TypeError, ValueError):
            continue
        if 0.0 <= score <= 1.0:
            score *= 100.0
        scientific = str(item[2]).strip() if len(item) > 2 and item[2] else ""
        chinese = str(item[3]).strip() if len(item) > 3 and item[3] else ""
        if not scientific and not chinese:
            continue
        rows.append(
            Species(
                confidence=score,
                scientific=scientific or chinese,
                english=scientific or chinese,
                animal_class="B",
                common=chinese or scientific,
            )
        )
    rows.sort(key=lambda row: row.confidence, reverse=True)
    return rows


def _decode_sound_payload(payload: dict, elapsed_ms: float) -> BirdResult:
    """Turn the dictionary shaped recording payload into a result.

    Args:
        payload: Decoded reply body holding `duration`, `summary` and
            `splits`.
        elapsed_ms: Measured duration for the result.

    Returns:
        A BirdResult with one box-less detection, matching the audio reply
        layout of the local deployment.
    """
    species = _sound_species(payload.get("summary"))
    if not species:
        # Older replies only carry the per window split list; fold it back into
        # a single ranking instead of dropping the answer.
        for entry in payload.get("splits") or []:
            if isinstance(entry, (list, tuple)) and len(entry) > 2:
                species.extend(_sound_species(entry[2]))
        species.sort(key=lambda row: row.confidence, reverse=True)
    detections = [Detection(box=(), species=species)] if species else []
    return BirdResult(detections=detections, kind="audio", elapsed_ms=elapsed_ms)


def _decode_payload(payload: Any, kind: str, elapsed_ms: float) -> BirdResult:
    """Turn the raw JSON payload into a BirdResult.

    Photographs come back as a list of boxed regions while recordings use a
    dictionary holding a ranked summary, so both shapes are accepted for any
    request kind.

    Args:
        payload: Raw payload from the service.
        kind: Request kind used to annotate the result.
        elapsed_ms: Measured duration for the result.

    Returns:
        A BirdResult holding every usable detection in the payload.
    """
    if isinstance(payload, dict):
        return _decode_sound_payload(payload, elapsed_ms)

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


def _message_of(payload: Any) -> str:
    """Return the readable part of an error payload."""
    if isinstance(payload, str):
        return payload
    if payload is None:
        return ""
    return str(payload)


def _raise_for_code(code: Any, payload: Any) -> None:
    """Raise the exception matching a failure code.

    Args:
        code: Protocol code from the service.
        payload: Payload that came with the code.

    Returns:
        None; the function only returns for the success code.

    Raises:
        BusyError: The service queue is full.
        ClientError: Any other business failure reported by the service.
    """
    message = _message_of(payload)
    if code == CODE_QUOTA and message.strip().lower() == BUSY_MESSAGE:
        raise BusyError(message)
    numeric = int(code) if isinstance(code, int) else 0
    if not message:
        message = _CODE_MESSAGES.get(numeric, "unexpected reply")
    raise ClientError(numeric, message)


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
        A BirdResult for success and for the "nothing found" codes.

    Raises:
        BusyError: The service queue is full.
        ClientError: Any other business failure reported by the service.
    """
    if code == CODE_OK:
        return _decode_payload(payload, kind, elapsed_ms)
    if code in EMPTY_CODES:
        return BirdResult(kind=kind, elapsed_ms=elapsed_ms)
    _raise_for_code(code, payload)
    # Unreachable: `_raise_for_code` always raises.
    return BirdResult(kind=kind, elapsed_ms=elapsed_ms)


class BirdClient:
    """Client for the Dongniao compatible endpoint.

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
        self.api_key_header = api_key_header or "api_key"
        self.poll_interval_s = max(0.02, float(poll_interval_s))
        self.timeout_s = max(1.0, float(timeout_s))
        self.max_concurrency = max(1, int(max_concurrency))
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    def set_max_concurrency(self, limit: int) -> None:
        """Change the local in-flight request budget.

        Args:
            limit: New budget; values below one are treated as one.

        Returns:
            None. In-flight requests keep their slot; later ones use the new
            budget.
        """
        value = max(1, int(limit))
        if value == self.max_concurrency:
            return
        self.max_concurrency = value
        self._semaphore = asyncio.Semaphore(value)

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
            session = self._session
        return session

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

    async def probe(self) -> str:
        """Classify the configured endpoint.

        The official service has no health document, so an empty health reply
        says nothing on its own. A quota request is free of charge and answers
        with a protocol envelope whenever a Dongniao server is listening, which
        is what separates the official endpoint from an unreachable one.

        Returns:
            One of BACKEND_LOCAL, BACKEND_OFFICIAL or BACKEND_OFFLINE.
        """
        if await self.health():
            return BACKEND_LOCAL
        try:
            await self._call({"quota": "1"})
        except ClientError as exc:
            # A protocol level answer (bad key, throttling, busy) still proves
            # that a Dongniao service is there; code 0 is a transport failure.
            return BACKEND_OFFICIAL if exc.code else BACKEND_OFFLINE
        except (asyncio.TimeoutError, OSError, ValueError) as exc:
            _LOGGER.debug("quota probe failed: %s", exc)
            return BACKEND_OFFLINE
        return BACKEND_OFFICIAL

    async def _call(self, fields: Dict[str, str]) -> Any:
        """Post one plain field map and return the success payload.

        Args:
            fields: Form fields of the request.

        Returns:
            The payload of a successful reply.

        Raises:
            BusyError: The service is saturated.
            ClientError: Any protocol level failure.
            asyncio.TimeoutError: The request missed its deadline.
        """
        session = await self._get_session()
        async with self._semaphore:
            async with session.post(
                self.base_url + "/dongniao",
                data=fields,
                headers=self._headers(),
            ) as response:
                if response.status != 200:
                    raise ClientError(0, "http %d" % response.status)
                reply = await response.json(content_type=None)
        code, payload = _split_reply(reply)
        if code == CODE_OK:
            return payload
        _raise_for_code(code, payload)
        return None

    async def quota(self) -> Dict[str, Any]:
        """Read the remaining allowance of this key.

        Returns:
            The quota document; empty when the reply is not a mapping.
        """
        payload = await self._call({"quota": "1"})
        return payload if isinstance(payload, dict) else {}

    async def encyclopedia(
        self, animal_id: Any, class_code: str = "B"
    ) -> Dict[str, Any]:
        """Read the encyclopedia entry of one animal id.

        Args:
            animal_id: Numeric id reported next to a candidate.
            class_code: Element selector, B for birds.

        Returns:
            The encyclopedia document; empty when the reply is not a mapping.
        """
        fields = {"animalid": str(animal_id)}
        selector = normalize_class(class_code) or "B"
        fields["class"] = selector[0]
        payload = await self._call(fields)
        return payload if isinstance(payload, dict) else {}

    async def _submit(
        self,
        data: Any,
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
        try:
            return await self._submit_loop(data, kind, limit, started)
        except asyncio.TimeoutError:
            # TimeoutError is an OSError subclass, so this branch has to come
            # first or a deadline would be reported as a dead connection.
            raise
        except (aiohttp.ClientError, OSError):
            # A refused connection or a dropped socket must reach the caller as
            # a plain protocol failure, so the handler never sees a transport
            # exception with the backend address inside it.
            raise ClientError(0, "transport failed") from None

    async def _submit_loop(
        self,
        data: Any,
        kind: str,
        limit: float,
        started: float,
    ) -> BirdResult:
        """Post one job and poll it until the payload is ready.

        Args:
            data: Form payload describing the job.
            kind: Either "image" or "audio".
            limit: Wall clock budget in seconds.
            started: Loop time when the request began.

        Returns:
            The parsed identification result.

        Raises:
            BusyError: The service is saturated.
            ClientError: Any other protocol level failure.
            asyncio.TimeoutError: The job did not finish within the deadline.
        """
        loop = asyncio.get_running_loop()
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
        class_code: str = "",
        did: str = "",
        deadline_s: Optional[float] = None,
    ) -> BirdResult:
        """Identify the birds in one image.

        Args:
            image_bytes: Encoded image payload.
            filename: Filename advertised in the multipart body.
            area: Optional Dongniao region code such as CN-BJ.
            latitude: Optional latitude for the geographic prior.
            longitude: Optional longitude for the geographic prior.
            class_code: Optional element selector, B for birds only.
            did: Optional device id of the caller.
            deadline_s: Optional per-request deadline in seconds.

        Returns:
            The parsed identification result.
        """
        form = aiohttp.FormData()
        form.add_field("image", image_bytes, filename=filename)
        # The official endpoint refuses an upload without this flag; the local
        # deployment ignores it.
        form.add_field("upload", "1")
        _add_class(form, class_code)
        _add_geo(form, area, latitude, longitude)
        _add_did(form, did)
        return await self._submit(form, "image", deadline_s)

    async def identify_image_url(
        self,
        url: str,
        *,
        area: str = "",
        class_code: str = "",
        did: str = "",
        deadline_s: Optional[float] = None,
    ) -> BirdResult:
        """Identify the birds in a photo the service downloads itself.

        Args:
            url: Public http(s) address of the photo.
            area: Optional region code.
            class_code: Optional element selector.
            did: Optional device id of the caller.
            deadline_s: Optional per-request deadline in seconds.

        Returns:
            The parsed identification result.
        """
        fields = {"url": url}
        _add_class(fields, class_code)
        if area:
            fields["area"] = area
        _add_did(fields, did)
        form = aiohttp.FormData()
        for key, value in fields.items():
            form.add_field(key, value)
        return await self._submit(form, "image", deadline_s)

    async def identify_sound(
        self,
        sound_bytes: bytes,
        *,
        filename: str = "sound.wav",
        area: str = "",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        did: str = "",
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
            did: Optional device id of the caller.
            deadline_s: Optional per-request deadline in seconds.

        Returns:
            The parsed identification result.
        """
        form = aiohttp.FormData()
        form.add_field("sound", sound_bytes, filename=filename)
        form.add_field("upload", "1")
        _add_geo(form, area, latitude, longitude)
        _add_did(form, did)
        return await self._submit(form, "audio", deadline_s)

    async def identify_sound_url(
        self,
        url: str,
        *,
        area: str = "",
        did: str = "",
        deadline_s: Optional[float] = None,
    ) -> BirdResult:
        """Identify the birds in a recording the service downloads itself.

        Args:
            url: Public http(s) address of the recording.
            area: Optional region code.
            did: Optional device id of the caller.
            deadline_s: Optional per-request deadline in seconds.

        Returns:
            The parsed identification result.
        """
        form = aiohttp.FormData()
        form.add_field("sound_url", url)
        if area:
            form.add_field("area", area)
        _add_did(form, did)
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


def _add_class(target: Any, class_code: str) -> None:
    """Attach the optional element selector to a form or field map.

    Args:
        target: Multipart body or plain field mapping under construction.
        class_code: Raw selector; only the first letter is sent.

    Returns:
        None
    """
    selector = normalize_class(class_code)
    if not selector:
        return
    if isinstance(target, dict):
        target["class"] = selector
    else:
        target.add_field("class", selector)


def _add_did(target: Any, did: str) -> None:
    """Attach the optional device id to a form or field map.

    Args:
        target: Multipart body or plain field mapping under construction.
        did: Device id; blank values are omitted.

    Returns:
        None
    """
    value = (did or "").strip()
    if not value:
        return
    if isinstance(target, dict):
        target["did"] = value
    else:
        target.add_field("did", value)


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
