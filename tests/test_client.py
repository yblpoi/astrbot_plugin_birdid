"""Tests for the backend client protocol handling."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest

from bird.client import (
    CODE_NO_ANIMALS,
    CODE_OK,
    CODE_PROCESSING,
    BirdClient,
    BusyError,
    ClientError,
    _decode_payload,
    _error_or_result,
    _parse_species,
)


def test_parse_species_splits_the_pipe_triple():
    item = [97.5, "麻雀|Eurasian Tree Sparrow|Passer montanus", 12, "B"]
    species = _parse_species(item)
    assert species.confidence == 97.5
    assert species.common == "麻雀"
    assert species.english == "Eurasian Tree Sparrow"
    assert species.scientific == "Passer montanus"
    assert species.animal_class == "B"


def test_parse_species_defaults_the_class_to_bird():
    assert _parse_species([1.0, "a|b|c"]).animal_class == "B"


def test_parse_species_falls_back_to_the_first_segment():
    species = _parse_species([50.0, "Common Kingfisher"])
    assert species.common == "Common Kingfisher"
    assert species.english == "Common Kingfisher"
    assert species.scientific == "Common Kingfisher"


def test_parse_species_rejects_malformed_entries():
    assert _parse_species("nope") is None
    assert _parse_species([1.0]) is None
    assert _parse_species(["x", "a|b|c"]) is None


def test_decode_payload_keeps_boxes_and_ranks():
    payload = [
        {"box": [1, 2, 3, 4], "list": [[90.0, "a|b|c", 1, "B"]]},
        {"box": [5, 6, 7, 8], "list": [[10.0, "d|e|f", 2, "B"]]},
    ]
    result = _decode_payload(payload, "image", 12.0)
    assert len(result.detections) == 2
    assert result.detections[0].box == (1.0, 2.0, 3.0, 4.0)
    assert result.top.species[0].scientific == "c"
    assert result.elapsed_ms == 12.0


def test_decode_payload_accepts_recordings_without_a_box():
    """Recordings answer with an empty box; the candidates still count."""
    payload = [
        {
            "box": [],
            "list": [
                [
                    88.0,
                    "Grey-cowled Wood Rail|Grey-cowled Wood Rail|Aramides cajaneus",
                    875,
                    "B",
                ]
            ],
        }
    ]
    result = _decode_payload(payload, "audio", 5.0)
    assert len(result.detections) == 1
    assert result.detections[0].box == ()
    assert result.detections[0].species[0].scientific == "Aramides cajaneus"


def test_decode_payload_drops_entries_without_candidates():
    payload = [{"box": [1, 2, 3, 4], "list": []}, {"box": [], "list": []}]
    assert _decode_payload(payload, "image", 1.0).detections == []


def test_decode_payload_ignores_malformed_boxes_and_entries():
    payload = [
        "nope",
        {"box": [1, 2, 3], "list": [[1.0, "a|b|c"]]},
        {"box": ["a", "b", "c", "d"], "list": [[1.0, "a|b|c"]]},
        {"box": [1, 2, 3, 4], "list": "nope"},
    ]
    assert _decode_payload(payload, "image", 1.0).detections == []


def test_error_or_result_returns_the_parsed_result():
    result = _error_or_result(
        CODE_OK, [{"box": [1, 2, 3, 4], "list": [[1.0, "a|b|c"]]}], "image", 3.0
    )
    assert len(result.detections) == 1


def test_error_or_result_maps_no_animals_to_an_empty_result():
    assert _error_or_result(CODE_NO_ANIMALS, None, "image", 3.0).detections == []


def test_error_or_result_raises_for_the_busy_message():
    with pytest.raises(BusyError):
        _error_or_result(1007, "server busy", "image", 1.0)


def test_error_or_result_raises_for_other_failures():
    with pytest.raises(ClientError) as info:
        _error_or_result(1002, "bad sound", "audio", 1.0)
    assert info.value.code == 1002


def test_processing_reply_is_reported_as_a_client_error():
    """A bare poll answer still pending must not look like a result."""
    with pytest.raises(ClientError) as info:
        _error_or_result(CODE_PROCESSING, "rid", "image", 1.0)
    assert info.value.code == CODE_PROCESSING


async def test_submit_reports_a_refused_connection_as_a_client_error():
    """A dead backend must surface as ClientError, not a raw transport error."""
    client = BirdClient("http://127.0.0.1:9", timeout_s=1.0)
    form = aiohttp.FormData()
    form.add_field("image", b"\xff\xd8\xff", filename="image.jpg")
    try:
        with pytest.raises(ClientError) as info:
            await client._submit(form, "image", 1.0)
    finally:
        await client.close()
    assert info.value.code == 0
    assert "127.0.0.1" not in str(info.value)


async def test_submit_keeps_a_deadline_as_a_timeout(monkeypatch):
    """A deadline must not be reported as a dead connection."""
    client = BirdClient("http://127.0.0.1:9", timeout_s=1.0)

    async def _boom(*args, **kwargs):
        raise asyncio.TimeoutError("identification did not finish in time")

    monkeypatch.setattr(client, "_submit_loop", _boom)
    form = aiohttp.FormData()
    form.add_field("image", b"\xff\xd8\xff", filename="image.jpg")
    try:
        with pytest.raises(asyncio.TimeoutError):
            await client._submit(form, "image", 1.0)
    finally:
        await client.close()
