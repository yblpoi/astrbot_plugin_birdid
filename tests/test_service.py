"""Tests for the request orchestrator in bird.service.

The backend, the archive and ffmpeg are replaced with in-memory doubles, so
the suite runs without the identification service, without network access and
without an audio decoder.
"""

from __future__ import annotations

import asyncio
import gzip
import math
import shutil
import sys
import wave
from array import array
from pathlib import Path

import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bird import service as service_module  # noqa: E402  (path bootstrap)
from bird.client import BirdResult, Detection, Species  # noqa: E402
from bird.media import MediaItem  # noqa: E402
from bird.names import NameResolver  # noqa: E402
from bird.render import crop_image, image_to_jpeg_bytes  # noqa: E402
from bird.service import BirdService, ServiceBusy  # noqa: E402

JPEG_SIGNATURE = b"\xff\xd8\xff"


class FakeClient:
    """Backend double that returns canned detections without any IO."""

    def __init__(self, detections=None, *, gate=None) -> None:
        self.detections = list(detections or [])
        self.images: list = []
        self.sounds: list = []
        self.health_calls = 0
        self._gate = gate

    async def health(self) -> dict:
        """Pretend to answer the health endpoint."""
        self.health_calls += 1
        return {"status": "ok"}

    async def identify_image(
        self,
        image_bytes: bytes,
        *,
        filename: str = "image.jpg",
        area: str = "",
        latitude=None,
        longitude=None,
        deadline_s=None,
    ) -> BirdResult:
        """Record the upload and return the canned detections."""
        self.images.append((image_bytes, filename, area, deadline_s))
        if self._gate is not None:
            await self._gate.wait()
        return BirdResult(detections=self.detections, kind="image", elapsed_ms=1.0)

    async def identify_sound(
        self,
        sound_bytes: bytes,
        *,
        filename: str = "sound.wav",
        area: str = "",
        latitude=None,
        longitude=None,
        deadline_s=None,
    ) -> BirdResult:
        """Record the upload and return the canned detections."""
        self.sounds.append((sound_bytes, filename, area, deadline_s))
        if self._gate is not None:
            await self._gate.wait()
        return BirdResult(detections=self.detections, kind="audio", elapsed_ms=1.0)


class FakeArchive:
    """Archive double that records the calls it receives."""

    def __init__(self, error: Exception | None = None) -> None:
        self.records: list = []
        self._error = error

    def record(self, **kwargs) -> dict:
        """Record one observation or fail on purpose."""
        self.records.append(kwargs)
        if self._error is not None:
            raise self._error
        return {"stored": True}


def _write_tone(path: Path, seconds: float = 1.0, rate: int = 32000) -> Path:
    """Write a mono 32 kHz sine WAV the audio pipeline accepts as-is.

    Args:
        path: Destination path.
        seconds: Clip length in seconds.
        rate: Sample rate.

    Returns:
        The path that was written.
    """
    values = array("h")
    for index in range(int(seconds * rate)):
        values.append(int(12000 * math.sin(2.0 * math.pi * 880.0 * index / rate)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(values.tobytes())
    return path


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    """Yield control until predicate returns True.

    Args:
        predicate: Callable checked on every iteration.
        timeout: Wall clock budget in seconds.

    Returns:
        None
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition was not reached in time")
        await asyncio.sleep(0.005)


@pytest.fixture
def archive() -> FakeArchive:
    """Return an archive double shared by one test."""
    return FakeArchive()


@pytest.fixture
def names(tmp_path: Path) -> NameResolver:
    """Return a resolver backed by a one row name table."""
    path = tmp_path / "names.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("egretta garzetta\t白鹭\tLittle Egret\n")
    return NameResolver(path)


@pytest.fixture
def detections() -> list:
    """Return one bird detection with a single candidate."""
    return [
        Detection(
            box=(20.0, 20.0, 120.0, 120.0),
            species=[
                Species(
                    confidence=92.5,
                    scientific="Egretta garzetta",
                    english="Little Egret",
                    animal_class="B",
                    common="Egretta garzetta",
                )
            ],
        )
    ]


@pytest.fixture
def image_item(tmp_path: Path) -> MediaItem:
    """Write a photo and wrap it as an image media item."""
    path = tmp_path / "bird.png"
    Image.new("RGB", (240, 180), (40, 90, 40)).save(path)
    return MediaItem(kind="image", path=path)


@pytest.fixture
def audio_item(tmp_path: Path) -> MediaItem:
    """Write a 32 kHz mono WAV and wrap it as an audio media item."""
    return MediaItem(kind="audio", path=_write_tone(tmp_path / "clip.wav"))


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Create the per request directory the service writes into."""
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture
def patch_transcode(monkeypatch) -> None:
    """Replace ffmpeg with a copy so the audio path stays offline."""

    async def _passthrough(source, target, *, ffmpeg="", timeout_s=60.0):
        shutil.copyfile(source, target)
        return target

    monkeypatch.setattr(service_module.media_tools, "transcode_audio", _passthrough)


async def test_identify_rejects_empty_items(workdir: Path, names, archive) -> None:
    """A message without media is a ValueError, not an empty reply."""
    service = BirdService(FakeClient(), names, archive)
    with pytest.raises(ValueError, match="no media"):
        await service.identify([], workdir)


async def test_identify_releases_worker_slot(
    image_item, workdir: Path, names, archive, detections
) -> None:
    """The slot taken for one request is released before it returns."""
    service = BirdService(FakeClient(detections), names, archive)

    await service.identify([image_item], workdir)

    assert service.running == 0
    assert service.waiting == 0


async def test_identify_image_marks_reply_as_image(
    image_item, workdir: Path, names, archive, detections
) -> None:
    """A picture request produces an image reply holding observations."""
    service = BirdService(FakeClient(detections), names, archive)

    reply = await service.identify([image_item], workdir)

    assert reply.kind == "image"
    assert len(reply.observations) == 1


async def test_identify_image_resolves_chinese_name(
    image_item, workdir: Path, names, archive, detections
) -> None:
    """The scientific name is translated through the name resolver."""
    service = BirdService(FakeClient(detections), names, archive)

    reply = await service.identify([image_item], workdir)

    observation = reply.observations[0]
    assert (observation.zh, observation.en) == ("白鹭", "Little Egret")
    # The image classifier puts a common name in the scientific slot, so a
    # picture reply never repeats it as though it were a Latin name.
    assert observation.scientific == ""
    assert observation.gate_box == (20.0, 20.0, 120.0, 120.0)


async def test_identify_image_returns_crop_jpeg(
    image_item, workdir: Path, names, archive, detections
) -> None:
    """Each rendered detection arrives as a JPEG attachment."""
    service = BirdService(FakeClient(detections), names, archive)

    reply = await service.identify([image_item], workdir)

    assert len(reply.images) == 1
    assert reply.images[0].startswith(JPEG_SIGNATURE)


async def test_identify_image_crops_only_the_top_detection(
    image_item, workdir: Path, names, archive, detections
) -> None:
    """A photo with several targets answers with a single picture."""
    extra = Detection(
        box=(140.0, 20.0, 220.0, 110.0),
        species=[
            Species(
                confidence=71.0,
                scientific="Ardea cinerea",
                english="Grey Heron",
                animal_class="B",
                common="Ardea cinerea",
            )
        ],
    )
    service = BirdService(FakeClient(detections + [extra]), names, archive)

    reply = await service.identify([image_item], workdir)

    assert len(reply.observations) == 2
    assert len(reply.images) == 1
    assert reply.images[0].startswith(JPEG_SIGNATURE)


async def test_identify_image_crops_the_bird_over_a_louder_object(
    image_item, workdir: Path, names, archive
) -> None:
    """A faint bird box wins the crop over a confident non-bird box."""
    person = Detection(
        box=(0.0, 0.0, 120.0, 180.0),
        species=[
            Species(
                confidence=91.0,
                scientific="person",
                english="person",
                animal_class="M",
                common="person",
            )
        ],
    )
    bird = Detection(
        box=(120.0, 0.0, 240.0, 180.0),
        species=[
            Species(
                confidence=30.0,
                scientific="Apus apus",
                english="Common Swift",
                animal_class="B",
                common="Apus apus",
            )
        ],
    )
    service = BirdService(FakeClient([person, bird]), names, archive)

    reply = await service.identify([image_item], workdir)

    expected = image_to_jpeg_bytes(
        crop_image(image_item.path, bird.box, margin_ratio=0.15)
    )
    assert reply.images == [expected]


async def test_identify_image_uploads_jpeg(
    image_item, workdir: Path, names, archive, detections
) -> None:
    """The client receives the normalized JPEG, not the source file."""
    client = FakeClient(detections)
    service = BirdService(client, names, archive)

    await service.identify([image_item], workdir)

    payload, filename, area, deadline_s = client.images[0]
    assert filename == "upload.jpg"
    assert payload.startswith(b"\xff\xd8\xff")
    assert deadline_s == service.request_timeout_s


async def test_identify_image_forwards_region_code(
    image_item, workdir: Path, names, archive, detections
) -> None:
    """A known location is turned into a region code for the backend."""
    client = FakeClient(detections)
    service = BirdService(client, names, archive)

    reply = await service.identify([image_item], workdir, location="云南")

    assert client.images[0][2] == "CN-YN"
    assert reply.location == "云南"


async def test_identify_prefers_image_when_both_present(
    tmp_path: Path, image_item, workdir: Path, names, archive, detections
) -> None:
    """A message carrying both kinds is treated as an image request."""
    client = FakeClient(detections)
    service = BirdService(client, names, archive)
    audio = MediaItem(kind="audio", path=tmp_path / "unused.wav")

    reply = await service.identify([audio, image_item], workdir)

    assert reply.kind == "image"
    assert client.sounds == []


async def test_identify_audio_marks_reply_as_audio(
    audio_item, workdir: Path, names, archive, detections, patch_transcode
) -> None:
    """A recording request produces an audio reply."""
    service = BirdService(FakeClient(detections), names, archive)

    reply = await service.identify([audio_item], workdir)

    assert reply.kind == "audio"
    assert len(reply.observations) == 1


async def test_identify_audio_keeps_the_scientific_name(
    audio_item, workdir: Path, names, archive, detections, patch_transcode
) -> None:
    """Recordings do report a scientific name, so the reply keeps it."""
    service = BirdService(FakeClient(detections), names, archive)

    reply = await service.identify([audio_item], workdir)

    assert reply.observations[0].scientific == "Egretta garzetta"


async def test_identify_audio_uploads_wav(
    audio_item, workdir: Path, names, archive, detections, patch_transcode
) -> None:
    """The client receives the transcoded 32 kHz WAV payload."""
    client = FakeClient(detections)
    service = BirdService(client, names, archive)

    await service.identify([audio_item], workdir)

    payload, filename, _, _ = client.sounds[0]
    assert filename == "upload.wav"
    assert payload.startswith(b"RIFF")


async def test_identify_audio_renders_spectrogram(
    audio_item, workdir: Path, names, archive, detections, patch_transcode
) -> None:
    """A decodable recording is drawn as a JPEG spectrogram."""
    service = BirdService(FakeClient(detections), names, archive)

    reply = await service.identify([audio_item], workdir)

    assert len(reply.images) == 1
    assert reply.images[0].startswith(JPEG_SIGNATURE)


async def test_identify_raises_service_busy_when_saturated(
    tmp_path: Path, image_item, names, archive, detections
) -> None:
    """A third request is refused at once while two are in flight."""
    gate = asyncio.Event()
    service = BirdService(
        FakeClient(detections, gate=gate), names, archive, max_in_flight=2
    )
    workdirs = []
    for index in range(2):
        path = tmp_path / ("work%d" % index)
        path.mkdir()
        workdirs.append(path)

    tasks = [
        asyncio.create_task(service.identify([image_item], workdir))
        for workdir in workdirs
    ]
    try:
        await _wait_until(lambda: service.running == 2)
        with pytest.raises(ServiceBusy):
            await service.identify([image_item], workdirs[0])
    finally:
        gate.set()
    replies = await asyncio.gather(*tasks)

    assert len(replies) == 2


async def test_identify_survives_archive_failure(
    image_item, workdir: Path, names, detections
) -> None:
    """A failing archive write is logged, not propagated to the caller."""
    broken = FakeArchive(error=RuntimeError("disk full"))
    service = BirdService(FakeClient(detections), names, broken)

    reply = await service.identify([image_item], workdir)

    assert reply.kind == "image"
    assert broken.records


async def test_archive_receives_resolved_species(
    image_item, workdir: Path, names, archive, detections
) -> None:
    """The archive stores the top species with its Chinese name."""
    service = BirdService(FakeClient(detections), names, archive)

    await service.identify([image_item], workdir, request_id="req-1")

    record = archive.records[0]
    assert record["source"] == "image"
    assert record["species"] == {"zh": "白鹭", "en": "Little Egret", "sci": ""}
    assert record["request"] == "req-1"


async def test_archive_keeps_the_scientific_name_of_recordings(
    audio_item, workdir: Path, names, archive, detections, patch_transcode
) -> None:
    """A recording archives the scientific name the backend reported."""
    service = BirdService(FakeClient(detections), names, archive)

    await service.identify([audio_item], workdir)

    assert archive.records[0]["species"]["sci"] == "Egretta garzetta"


async def test_observations_use_common_name_for_non_birds(
    image_item, workdir: Path, names, archive
) -> None:
    """A non bird detection keeps the label the backend already sent."""
    other = [
        Detection(
            box=(20.0, 20.0, 80.0, 80.0),
            species=[
                Species(
                    confidence=70.0,
                    scientific="Felis catus",
                    english="Cat",
                    animal_class="M",
                    common="猫",
                )
            ],
        )
    ]
    service = BirdService(FakeClient(other), names, archive)

    reply = await service.identify([image_item], workdir)

    assert (reply.observations[0].zh, reply.observations[0].en) == ("猫", "Cat")


async def test_identify_does_not_probe_backend_health(
    image_item, workdir: Path, names, archive, detections
) -> None:
    """Identification never waits on the health endpoint."""
    client = FakeClient(detections)
    service = BirdService(client, names, archive)

    await service.identify([image_item], workdir)

    assert client.health_calls == 0


async def test_constructor_clamps_limits(names, archive) -> None:
    """A non positive budget is raised instead of disabling the guard."""
    service = BirdService(
        FakeClient(), names, archive, max_in_flight=0, request_timeout_s=0.1
    )

    assert service.max_in_flight == 1
    assert service.request_timeout_s == 1.0
