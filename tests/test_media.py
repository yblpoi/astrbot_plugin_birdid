"""Tests for the media helpers in bird.media.

Every image and recording is synthesized inside pytest's tmp_path, so the
suite runs without the identification service, without network access and
without ffmpeg.
"""

from __future__ import annotations

import math
import sys
import wave
from array import array
from pathlib import Path

import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bird import media  # noqa: E402  (path bootstrap must run first)

PNG_HEAD = b"\x89PNG\r\n\x1a\n"


def _write_bytes(path: Path, payload: bytes) -> Path:
    """Write raw bytes and return the path.

    Args:
        path: Destination path.
        payload: Content to write.

    Returns:
        The path that was written.
    """
    path.write_bytes(payload)
    return path


def _tone_payload(frames: int, *, channels: int = 1, amplitude: float = 0.5) -> bytes:
    """Build interleaved 16 bit samples for one sine tone.

    Args:
        frames: Number of sample frames.
        channels: Channels per frame.
        amplitude: Peak amplitude in the unit range.

    Returns:
        Signed 16 bit PCM bytes.
    """
    values = array("h")
    for index in range(frames):
        sample = int(
            32767 * amplitude * math.sin(2.0 * math.pi * 440.0 * index / 32000.0)
        )
        for _ in range(channels):
            values.append(sample)
    return values.tobytes()


def _write_wav(
    path: Path,
    payload: bytes,
    *,
    rate: int = 32000,
    channels: int = 1,
    sampwidth: int = 2,
) -> Path:
    """Write a WAV container around already encoded PCM bytes.

    Args:
        path: Destination path.
        payload: PCM bytes.
        rate: Sample rate stored in the header.
        channels: Channel count stored in the header.
        sampwidth: Bytes per sample stored in the header.

    Returns:
        The path that was written.
    """
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sampwidth)
        handle.setframerate(rate)
        handle.writeframes(payload)
    return path


def _write_solid_png(path: Path, size=(320, 160)) -> Path:
    """Write a one colour PNG.

    Args:
        path: Destination path.
        size: Image size as (width, height).

    Returns:
        The path that was written.
    """
    Image.new("RGB", size, (12, 34, 56)).save(path)
    return path


class _Event:
    """Minimal stand-in for a message event."""

    def __init__(self, messages) -> None:
        self._messages = messages

    def get_messages(self):
        """Return the components carried by this message."""
        return self._messages


def _component(class_name: str, path: Path | None):
    """Build a duck-typed component whose class name drives the dispatch.

    Args:
        class_name: Class name media.collect_media dispatches on.
        path: File the component resolves to, or None for an empty attachment.

    Returns:
        An object carrying an async convert_to_file_path method.
    """

    async def convert_to_file_path() -> str:
        return str(path) if path is not None else ""

    return type(
        class_name, (), {"convert_to_file_path": staticmethod(convert_to_file_path)}
    )()


def _reply(chain) -> object:
    """Build a quoted message component holding other components.

    Args:
        chain: Components carried by the quoted message.

    Returns:
        An object named Reply with a chain attribute.
    """
    return type("Reply", (), {"chain": list(chain)})()


def test_probe_kind_detects_png(tmp_path: Path) -> None:
    """A PNG payload is an image even without a telling suffix."""
    path = _write_bytes(tmp_path / "payload.bin", PNG_HEAD + b"\x00" * 16)
    assert media.probe_kind(path) == "image"


def test_probe_kind_detects_jpeg(tmp_path: Path) -> None:
    """The JPEG start of image marker is enough to classify a file."""
    path = _write_bytes(tmp_path / "payload", b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")
    assert media.probe_kind(path) == "image"


def test_probe_kind_detects_riff_wave(tmp_path: Path) -> None:
    """A RIFF container whose form is WAVE is a recording."""
    path = _write_bytes(tmp_path / "clip.bin", b"RIFF\x24\x00\x00\x00WAVEfmt ")
    assert media.probe_kind(path) == "audio"


def test_probe_kind_detects_riff_webp(tmp_path: Path) -> None:
    """A RIFF container whose form is WEBP stays an image."""
    path = _write_bytes(tmp_path / "still.bin", b"RIFF\x20\x00\x00\x00WEBPVP8 ")
    assert media.probe_kind(path) == "image"


def test_probe_kind_detects_id3_tag(tmp_path: Path) -> None:
    """An MP3 file that starts with an ID3 tag is a recording."""
    path = _write_bytes(tmp_path / "voice.dat", b"ID3\x04\x00\x00\x00\x00\x00\x00")
    assert media.probe_kind(path) == "audio"


def test_probe_kind_detects_ogg(tmp_path: Path) -> None:
    """The OggS page header is enough to classify a recording."""
    path = _write_bytes(tmp_path / "voice.dat", b"OggS\x00\x02\x00\x00")
    assert media.probe_kind(path) == "audio"


def test_probe_kind_falls_back_to_audio_suffix(tmp_path: Path) -> None:
    """Unknown bytes with an audio suffix are treated as a recording."""
    path = _write_bytes(tmp_path / "voice.mp3", b"\x00" * 64)
    assert media.probe_kind(path) == "audio"


def test_probe_kind_falls_back_to_image_suffix(tmp_path: Path) -> None:
    """Unknown bytes with an image suffix are treated as a picture."""
    path = _write_bytes(tmp_path / "photo.jpeg", b"\x00" * 64)
    assert media.probe_kind(path) == "image"


def test_probe_kind_returns_none_for_unknown_bytes(tmp_path: Path) -> None:
    """Content and suffix that both fail leave the file unclassified."""
    path = _write_bytes(tmp_path / "blob.dat", b"\x00" * 64)
    assert media.probe_kind(path) is None


def test_probe_kind_returns_none_for_empty_file(tmp_path: Path) -> None:
    """An empty attachment is not guessed from its suffix."""
    path = _write_bytes(tmp_path / "empty.png", b"")
    assert media.probe_kind(path) is None


def test_probe_kind_returns_none_for_missing_file(tmp_path: Path) -> None:
    """A file that vanished is reported as unclassified instead of raising."""
    assert media.probe_kind(tmp_path / "absent.png") is None


def test_probe_kind_ignores_truncated_riff(tmp_path: Path) -> None:
    """A RIFF header too short to carry a form is not guessed."""
    path = _write_bytes(tmp_path / "clip.bin", b"RIFF\x00\x00\x00\x00")
    assert media.probe_kind(path) is None


def test_normalize_image_writes_jpeg_within_limit(tmp_path: Path) -> None:
    """The normalized upload is a JPEG under the service size limit."""
    source = _write_solid_png(tmp_path / "source.png")
    target = media.normalize_image(source, tmp_path / "out.jpg")

    assert target == tmp_path / "out.jpg"
    assert target.stat().st_size <= media.MAX_IMAGE_BYTES
    with Image.open(target) as decoded:
        assert decoded.format == "JPEG"


def test_normalize_image_shrinks_long_side(tmp_path: Path) -> None:
    """An oversized picture keeps its aspect ratio below the long side cap."""
    source = _write_solid_png(tmp_path / "wide.png", size=(4000, 1000))
    target = media.normalize_image(source, tmp_path / "out.jpg")

    with Image.open(target) as decoded:
        assert decoded.size == (media.MAX_IMAGE_SIDE, 512)


def test_normalize_image_applies_exif_orientation(tmp_path: Path) -> None:
    """The orientation tag is applied, so the stored pixels are upright."""
    source = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (40, 20), (200, 10, 10)).save(source, format="JPEG", exif=exif)

    target = media.normalize_image(source, tmp_path / "out.jpg")

    with Image.open(target) as decoded:
        assert decoded.size == (20, 40)


def test_normalize_image_rejects_undecodable_input(tmp_path: Path) -> None:
    """A payload Pillow cannot decode is surfaced as a ValueError."""
    source = _write_bytes(tmp_path / "broken.jpg", b"not an image at all")
    with pytest.raises(ValueError, match="unsupported image"):
        media.normalize_image(source, tmp_path / "out.jpg")


def test_trim_to_limit_accepts_file_at_limit(tmp_path: Path) -> None:
    """A file of exactly the limit size still passes the check."""
    path = _write_bytes(tmp_path / "upload.jpg", b"x" * 10)
    assert media.trim_to_limit(path, 10) == path


def test_trim_to_limit_rejects_oversized_file(tmp_path: Path) -> None:
    """An over limit file is rejected, not truncated in place."""
    path = _write_bytes(tmp_path / "upload.jpg", b"x" * 11)

    assert media.trim_to_limit(path, 10) is None
    assert path.stat().st_size == 11


def test_trim_to_limit_returns_none_for_missing_file(tmp_path: Path) -> None:
    """A file that cannot be measured is reported as unusable."""
    assert media.trim_to_limit(tmp_path / "absent.bin", 10) is None


def test_audio_duration_seconds_reads_wav_length(tmp_path: Path) -> None:
    """The duration comes from the frame count and the sample rate."""
    path = _write_wav(tmp_path / "clip.wav", _tone_payload(48000))
    assert media.audio_duration_seconds(path) == pytest.approx(1.5)


def test_audio_duration_seconds_returns_none_for_non_wav(tmp_path: Path) -> None:
    """A file that is not a WAV container has no readable duration."""
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    assert media.audio_duration_seconds(path) is None


def test_read_pcm_samples_returns_float32_units(tmp_path: Path) -> None:
    """Decoded samples are float32, mono and tagged with their rate."""
    frames = 1600
    path = _write_wav(tmp_path / "clip.wav", _tone_payload(frames))

    samples, rate = media.read_pcm_samples(path)

    assert rate == 32000
    assert samples.shape == (frames,)
    assert str(samples.dtype) == "float32"


def test_read_pcm_samples_stays_in_unit_range(tmp_path: Path) -> None:
    """Full scale input maps into [-1, 1] without clipping or wrapping."""
    path = _write_wav(tmp_path / "clip.wav", _tone_payload(1600, amplitude=1.0))

    samples, _ = media.read_pcm_samples(path)

    assert float(samples.max()) <= 1.0
    assert float(samples.min()) >= -1.0
    assert float(samples.max()) > 0.9


def test_read_pcm_samples_downmixes_stereo(tmp_path: Path) -> None:
    """A stereo frame collapses to the mean of its channels."""
    frames = 100
    values = array("h")
    for _ in range(frames):
        values.append(16384)
        values.append(0)
    path = _write_wav(tmp_path / "stereo.wav", values.tobytes(), channels=2)

    samples, _ = media.read_pcm_samples(path)

    assert samples.shape == (frames,)
    assert float(samples[0]) == pytest.approx(0.25, abs=1e-4)


def test_read_pcm_samples_returns_none_for_non_wav(tmp_path: Path) -> None:
    """A file without a WAV header is reported as unreadable."""
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"ID3\x04\x00\x00")
    assert media.read_pcm_samples(path) is None


def test_read_pcm_samples_returns_none_for_wide_samples(tmp_path: Path) -> None:
    """A container whose samples are not 16 bit is refused."""
    path = _write_wav(tmp_path / "eight_bit.wav", bytes(64), sampwidth=1)
    assert media.read_pcm_samples(path) is None


async def test_transcode_audio_requires_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    """Without an ffmpeg binary the audio path fails with a business error."""
    monkeypatch.setattr(media, "_ffmpeg_binary", lambda configured="": "")

    with pytest.raises(ValueError, match="ffmpeg"):
        await media.transcode_audio(tmp_path / "in.wav", tmp_path / "out.wav")


async def test_collect_media_returns_direct_image(tmp_path: Path) -> None:
    """An image sent as the message itself is returned as a message item."""
    source = tmp_path / "src" / "bird.png"
    source.parent.mkdir()
    Image.new("RGB", (16, 16)).save(source)

    items = await media.collect_media(
        _Event([_component("Image", source)]), tmp_path / "work"
    )

    assert len(items) == 1
    assert items[0].kind == "image"
    assert items[0].origin == "message"
    assert items[0].path.is_file()


async def test_collect_media_reads_quoted_image(tmp_path: Path) -> None:
    """An image inside the quoted message is returned with a quote origin."""
    source = tmp_path / "src" / "quoted.png"
    source.parent.mkdir()
    Image.new("RGB", (16, 16)).save(source)

    event = _Event([_reply([_component("Image", source)])])
    items = await media.collect_media(event, tmp_path / "work")

    assert len(items) == 1
    assert items[0].origin == "quote"


async def test_collect_media_skips_quoted_media_when_disabled(tmp_path: Path) -> None:
    """The caller can ignore the quoted message on purpose."""
    source = tmp_path / "src" / "quoted.png"
    source.parent.mkdir()
    Image.new("RGB", (16, 16)).save(source)

    event = _Event([_reply([_component("Image", source)])])
    items = await media.collect_media(event, tmp_path / "work", include_quote=False)

    assert items == []


async def test_collect_media_returns_empty_for_text_message(tmp_path: Path) -> None:
    """A message without a supported component carries no media."""
    event = _Event([_component("Plain", None)])
    assert await media.collect_media(event, tmp_path / "work") == []


async def test_collect_media_respects_max_items(tmp_path: Path) -> None:
    """No more than max_items entries are returned."""
    directory = tmp_path / "src"
    directory.mkdir()
    components = []
    for index in range(3):
        source = directory / ("bird%d.png" % index)
        Image.new("RGB", (16, 16)).save(source)
        components.append(_component("Image", source))

    items = await media.collect_media(
        _Event(components), tmp_path / "work", max_items=2
    )

    assert len(items) == 2


async def test_collect_media_deduplicates_same_file(tmp_path: Path) -> None:
    """The same workdir file delivered twice is reported once."""
    source = tmp_path / "bird.png"
    Image.new("RGB", (16, 16)).save(source)
    event = _Event([_component("Image", source), _component("Image", source)])

    items = await media.collect_media(event, tmp_path)

    assert len(items) == 1


async def test_collect_media_ignores_unresolvable_component(tmp_path: Path) -> None:
    """A component whose resolver fails is skipped, not raised."""

    async def _explode() -> str:
        raise OSError("attachment is gone")

    broken = type("File", (), {"convert_to_file_path": _explode})()
    items = await media.collect_media(_Event([broken]), tmp_path / "work")

    assert items == []


async def test_collect_media_survives_get_messages_failure(tmp_path: Path) -> None:
    """An event that cannot list its components yields no media."""

    class _BrokenEvent:
        def get_messages(self):
            raise RuntimeError("no message payload")

    assert await media.collect_media(_BrokenEvent(), tmp_path / "work") == []
