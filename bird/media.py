"""Pull images and recordings out of chat messages.

Messages arrive in many shapes: an image may be a local path, a URL or a base64
payload, a voice message may be a recorded audio file in any container, and the
media the user means may sit inside the message being quoted instead of the one
being sent. This module hides all of that behind one extraction call and hands
back files that the identification service can accept.

Everything is written under a caller supplied temporary directory. Nothing is
kept in the plugin directory and every file is removed once the reply is sent.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image, ImageOps

__all__ = [
    "MAX_AUDIO_BYTES",
    "MAX_IMAGE_BYTES",
    "MediaItem",
    "collect_media",
    "normalize_image",
    "probe_kind",
    "audio_duration_seconds",
    "read_pcm_samples",
    "transcode_audio",
    "trim_to_limit",
]

_LOGGER = logging.getLogger(__name__)

#: Service limits: image uploads are capped at 2 MiB, audio at 5 MiB.
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_AUDIO_BYTES = 5 * 1024 * 1024

#: Longest side kept when re-encoding an oversized image.
MAX_IMAGE_SIDE = 2048

#: Sample rate the sound pipeline decodes; anything else is rejected there.
TARGET_SAMPLE_RATE = 32000

#: Longest stretch of a recording kept, so the upload stays inside
#: MAX_AUDIO_BYTES: 32000 Hz * 2 bytes * 78 s is just under 5 MiB.
MAX_AUDIO_SECONDS = 78.0

#: Prefixes that identify a container by its first bytes.
_SIGNATURES = (
    (b"\xff\xd8\xff", "image"),
    (b"\x89PNG\r\n\x1a\n", "image"),
    (b"GIF87a", "image"),
    (b"GIF89a", "image"),
    (b"BM", "image"),
    (b"RIFF", None),
    (b"II*\x00", "image"),
    (b"MM\x00*", "image"),
    (b"#!AMR", "audio"),
    (b"#!SILK", "audio"),
    (b"OggS", "audio"),
    (b"ID3", "audio"),
    (b"fLaC", "audio"),
    (b"\x1aE\xdf\xa3", "audio"),
)

_AUDIO_EXTENSIONS = frozenset(
    {
        ".wav",
        ".mp3",
        ".ogg",
        ".oga",
        ".opus",
        ".m4a",
        ".aac",
        ".flac",
        ".amr",
        ".silk",
        ".aiff",
        ".wma",
    }
)
_IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
        ".heic",
        ".heif",
    }
)


@dataclass(frozen=True)
class MediaItem:
    """One media payload extracted from a message.

    Attributes:
        kind: Either "image" or "audio".
        path: Local file holding the payload.
        origin: Where the payload came from: "message" or "quote".
    """

    kind: str
    path: Path
    origin: str = "message"


def probe_kind(path: Path) -> Optional[str]:
    """Classify a file as an image, an audio recording or neither.

    The container is read from the first bytes so that a voice message saved
    without an extension, or an image delivered as a bare file, still lands in
    the right pipeline.

    Args:
        path: File to inspect.

    Returns:
        "image", "audio", or None when the content is neither.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(64)
    except OSError:
        return None
    if not head:
        return None
    if head[:4] == b"RIFF" and len(head) >= 12:
        form = head[8:12]
        if form == b"WEBP":
            return "image"
        if form == b"WAVE":
            return "audio"
    for prefix, kind in _SIGNATURES:
        if kind is not None and head.startswith(prefix):
            return kind
    try:
        with Image.open(io.BytesIO(head)):
            return "image"
    except Exception:  # noqa: BLE001 - any decoder failure means "not an image"
        pass
    suffix = path.suffix.lower()
    if suffix in _AUDIO_EXTENSIONS:
        return "audio"
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    return None


async def _component_to_path(component, temp_dir: Path) -> Optional[Path]:
    """Materialize one message component into a local file.

    Args:
        component: AstrBot message component.
        temp_dir: Directory that owns the returned file.

    Returns:
        A local path, or None when the component carries no retrievable media.
    """
    convert = getattr(component, "convert_to_file_path", None)
    if convert is None:
        return None
    try:
        resolved = await convert()
    except Exception as exc:  # noqa: BLE001 - a bad attachment is not fatal
        _LOGGER.debug("could not resolve attachment: %s", exc)
        return None
    if not resolved:
        return None
    path = Path(str(resolved))
    if not path.exists():
        return None
    if path.parent != temp_dir:
        # Copy media that lives outside our temporary directory so that later
        # cleanup cannot delete platform owned files.
        target = temp_dir / ("media_" + uuid.uuid4().hex[:10] + (path.suffix or ".bin"))
        await asyncio.to_thread(shutil.copyfile, path, target)
        return target
    return path


def _walk_components(event) -> Iterable:
    """Yield every message component, including those inside a quote.

    Args:
        event: Message event.

    Returns:
        An iterator of (component, origin) pairs.
    """
    try:
        components = event.get_messages()
    except Exception:  # noqa: BLE001
        components = []
    for component in components or []:
        if type(component).__name__ == "Reply":
            for nested in getattr(component, "chain", None) or []:
                yield nested, "quote"
            continue
        yield component, "message"


async def collect_media(
    event,
    temp_dir: Path,
    *,
    max_items: int = 4,
    include_quote: bool = True,
) -> List[MediaItem]:
    """Extract every usable image and recording from a message.

    Args:
        event: Message event to read.
        temp_dir: Directory that owns any copied files.
        max_items: Upper bound on the number of returned items.
        include_quote: Whether media in the quoted message is considered.

    Returns:
        Media items in message order, at most max_items entries.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    items: List[MediaItem] = []
    seen: set = set()
    for component, origin in _walk_components(event):
        if len(items) >= max_items:
            break
        if origin == "quote" and not include_quote:
            continue
        name = type(component).__name__
        if name not in {"Image", "Record", "File", "Video"}:
            continue
        path = await _component_to_path(component, temp_dir)
        if path is None:
            continue
        kind = probe_kind(path)
        if kind is None:
            _LOGGER.debug("skipping unsupported attachment: %s", path.name)
            continue
        key = (kind, path.stat().st_size, path.name)
        if key in seen:
            continue
        seen.add(key)
        items.append(MediaItem(kind=kind, path=path, origin=origin))
    return items


def normalize_image(source: Path, target: Path, *, quality: int = 90) -> Path:
    """Re-encode an image into an upright, service friendly JPEG.

    The orientation tag is applied here and the caller must use the returned
    file for every later step, because detector boxes are expressed in the
    pixels of whatever frame was uploaded.

    Args:
        source: Image file as delivered by the platform.
        target: Destination path for the normalized JPEG.
        quality: JPEG quality passed to the encoder.

    Returns:
        The path that was written.

    Raises:
        ValueError: When the file cannot be decoded as an image.
    """
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            image = image.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - surfaced as a business error
        raise ValueError("unsupported image: %s" % exc) from exc
    if max(image.size) > MAX_IMAGE_SIDE:
        ratio = MAX_IMAGE_SIDE / float(max(image.size))
        size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
        image = image.resize(size, Image.LANCZOS)
    for attempt_quality in (quality, 80, 70, 60):
        image.save(target, format="JPEG", quality=attempt_quality, optimize=True)
        if target.stat().st_size <= MAX_IMAGE_BYTES:
            return target
    raise ValueError("image stays above the upload limit after re-encoding")


def _copy_as_target(source: Path, target: Path) -> Path:
    """Copy a recording that already has the format the service decodes.

    Args:
        source: Recording as delivered by the platform.
        target: Destination WAV path.

    Returns:
        The path that was written.
    """
    shutil.copyfile(source, target)
    return target


def _ffmpeg_binary(configured: str = "") -> str:
    """Resolve the ffmpeg executable.

    Args:
        configured: Optional command name or path from the plugin config.

    Returns:
        The command to run, or an empty string when ffmpeg is unavailable.
    """
    if configured:
        return configured
    return shutil.which("ffmpeg") or ""


def _looks_like_pcm_wav(path: Path) -> bool:
    """Check whether a file already is a 16 bit PCM WAV at the target rate.

    Args:
        path: Candidate file.

    Returns:
        True when the sound pipeline can read it without transcoding.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(64)
    except OSError:
        return False
    if len(head) < 36 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return False
    try:
        import wave

        with wave.open(str(path), "rb") as wav:
            return (
                wav.getnchannels() == 1
                and wav.getsampwidth() == 2
                and wav.getframerate() == TARGET_SAMPLE_RATE
            )
    except Exception:  # noqa: BLE001 - a malformed header is simply "no"
        return False


def _copy_as_target(source: Path, target: Path) -> Path:
    """Copy a recording that already has the format the service decodes.

    Args:
        source: Recording as delivered by the platform.
        target: Destination WAV path.

    Returns:
        The path that was written.
    """
    shutil.copyfile(source, target)
    return target


async def transcode_audio(
    source: Path,
    target: Path,
    *,
    ffmpeg: str = "",
    timeout_s: float = 60.0,
    max_seconds: float = MAX_AUDIO_SECONDS,
) -> Path:
    """Convert any recording into the mono 32 kHz 16 bit WAV the service decodes.

    A recording that already has the target format is copied instead of
    decoded again, and a long recording is cut after max_seconds so the
    upload stays inside the service size limit.

    Args:
        source: Recording as delivered by the platform.
        target: Destination WAV path.
        ffmpeg: Optional ffmpeg command from the plugin config.
        timeout_s: Wall clock budget for the conversion.
        max_seconds: Longest stretch kept, in seconds; 0 keeps everything.

    Returns:
        The path that was written.

    Raises:
        ValueError: When ffmpeg is missing or the conversion produced nothing.
    """
    if _looks_like_pcm_wav(source):
        return _copy_as_target(source, target)
    binary = _ffmpeg_binary(ffmpeg)
    if not binary:
        raise ValueError("ffmpeg is not available for audio decoding")
    command = [
        binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
    ]
    if max_seconds and max_seconds > 0:
        command.extend(["-t", "%.1f" % max_seconds])
    command.append(str(target))
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise ValueError("audio decoding timed out") from None
    detail = (stderr or b"").decode("utf-8", "ignore").strip()[:200]
    if process.returncode != 0:
        raise ValueError("audio decoding failed: %s" % (detail or "ffmpeg error"))
    if not target.exists() or target.stat().st_size <= 44:
        raise ValueError("audio decoding produced no samples: %s" % (detail or "empty"))
    return target


def trim_to_limit(path: Path, limit: int) -> Optional[Path]:
    """Check a prepared upload against the service size limit.

    Args:
        path: Prepared file.
        limit: Maximum accepted size in bytes.

    Returns:
        The path when it fits, otherwise None.
    """
    try:
        return path if path.stat().st_size <= limit else None
    except OSError:
        return None


def audio_duration_seconds(path: Path) -> Optional[float]:
    """Read the duration of a WAV file.

    Args:
        path: File to inspect.

    Returns:
        Duration in seconds, or None when the header cannot be read.
    """
    try:
        import wave

        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
    except Exception:  # noqa: BLE001
        return None
    if not rate:
        return None
    return frames / float(rate)


def read_pcm_samples(path: Path) -> Optional[tuple]:
    """Read a mono 16 bit WAV into a numpy array for spectrogram rendering.

    Args:
        path: WAV file produced by transcode_audio.

    Returns:
        A (samples, sample_rate) tuple, or None when unreadable.
    """
    try:
        import wave

        import numpy as np

        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            raw = wav.readframes(wav.getnframes())
    except Exception:  # noqa: BLE001
        return None
    if width != 2:
        return None
    samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate
