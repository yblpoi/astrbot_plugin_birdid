"""Tests for the chat reply rendering."""

from __future__ import annotations

from bird.format import Observation, format_reply, format_status


def _observation(**overrides):
    """Build an observation with sensible defaults."""
    values = {
        "zh": "麻雀",
        "en": "Eurasian Tree Sparrow",
        "scientific": "Passer montanus",
        "confidence": 97.5,
    }
    values.update(overrides)
    return Observation(**values)


def test_label_shows_chinese_english_and_latin():
    assert _observation().label() == "麻雀 / Eurasian Tree Sparrow / Passer montanus"


def test_label_omits_a_repeated_english_name():
    observation = _observation(
        zh="普通翠鸟", en="Common Kingfisher", scientific="Common Kingfisher"
    )
    assert observation.label() == "普通翠鸟 / Common Kingfisher"


def test_label_omits_a_latin_slot_that_only_repeats_the_common_name():
    """The image classifier fills the scientific slot with a common name."""
    observation = _observation(
        zh="斑头大翠鸟", en="Blyth's Kingfisher", scientific="Blyths Kingfisher"
    )
    assert observation.label() == "斑头大翠鸟 / Blyth's Kingfisher"


def test_label_falls_back_when_every_name_is_missing():
    observation = _observation(zh="", en="", scientific="")
    assert observation.label() == "未知"


def test_reply_opens_with_the_gate_verdict():
    text = format_reply(
        [_observation(gate_box=(10, 20, 110, 220), gate_confidence=30.5)],
        kind="image",
        elapsed_ms=205.6,
    )
    lines = text.splitlines()
    assert lines[0] == "识别结果：鸟类（30.5%）"
    assert "1. 麻雀 / Eurasian Tree Sparrow / Passer montanus  97.5%" in text
    assert text.rstrip().endswith("耗时 206 ms")


def test_reply_hides_the_gate_box_coordinates():
    text = format_reply(
        [_observation(gate_box=(10, 20, 110, 220), gate_confidence=88.5)],
        kind="image",
        elapsed_ms=5.0,
    )
    assert "10,20-110,220" not in text
    assert "检测框" not in text


def test_reply_groups_candidates_by_their_gate_box():
    first = _observation(gate_box=(0, 0, 10, 10), gate_confidence=41.0)
    second = _observation(
        zh="白喉针尾雨燕",
        en="White-throated Needletail",
        scientific="Hirundapus caudacutus",
        confidence=12.0,
        gate_box=(20, 20, 90, 90),
        gate_confidence=7.5,
    )
    text = format_reply([first, second], kind="image", elapsed_ms=1.0)
    lines = text.splitlines()
    assert lines.count("识别结果：鸟类（41.0%）") == 1

    assert lines.count("识别结果：鸟类（7.5%）") == 1
    assert (
        "2. 白喉针尾雨燕 / White-throated Needletail / Hirundapus caudacutus  12.0%"
        in text
    )


def test_reply_headings_a_non_bird_with_its_own_name():
    text = format_reply(
        [
            _observation(
                zh="人",
                en="person",
                scientific="person",
                confidence=30.5,
                animal_class="M",
                gate_box=(58, 404, 1931, 1478),
                gate_confidence=30.5,
            )
        ],
        kind="image",
        elapsed_ms=1.0,
    )
    assert text.splitlines()[0] == "识别结果：人（30.5%）"


def test_reply_omits_the_score_when_the_gate_has_none():
    text = format_reply([_observation()], kind="audio", elapsed_ms=42.0)
    assert text.splitlines()[0] == "识别结果：鸟类"
    assert "耗时 42 ms" in text


def test_reply_explains_an_empty_result():
    text = format_reply([], kind="image", elapsed_ms=12.0)
    assert text.splitlines()[0] == "识别结果：未检测到鸟类。"
    assert "耗时 12 ms" in text


def test_reply_mentions_the_location_when_one_was_applied():
    text = format_reply([_observation()], kind="image", elapsed_ms=1.0, location="云南")
    assert "地点先验：云南" in text


def test_reply_reports_truncated_candidates():
    text = format_reply([_observation()], kind="image", elapsed_ms=1.0, truncated=2)
    assert "另有 2 个候选未列出。" in text


def test_reply_never_names_a_model_or_a_path():
    text = format_reply(
        [_observation(gate_box=(1, 2, 3, 4), gate_confidence=99.0)],
        kind="image",
        elapsed_ms=5.0,
    )
    for banned in ("yolo", "onnx", "fastvit", "birdnet", "npu", "/models", "ckpt"):
        assert banned not in text.lower()


def test_status_reports_the_backend_numbers():
    text = format_status(
        online=True,
        health={"status": "ok", "version": "0.1.0", "pending": 2, "max_queue": 32},
        queue=1,
        running=2,
        archive={"pending_mb": 1.5, "archive_mb": 20.0, "batches": 3},
        names={"count": 11587, "loaded": True},
    )
    assert "状态：在线" in text
    assert "版本：0.1.0" in text
    assert "后端队列：2/32" in text
    assert "本插件：排队 1，处理中 2" in text
    assert "名称库：11587 条" in text


def test_status_marks_an_offline_backend():
    text = format_status(
        online=False,
        health={},
        queue=0,
        running=0,
        archive={},
        names={"count": 0, "loaded": False},
    )
    assert "状态：离线" in text
    assert "（未加载）" in text


def test_status_keeps_a_small_archive_readable():
    text = format_status(
        online=True,
        health={"status": "ok", "pending": 0, "max_queue": 32},
        queue=0,
        running=0,
        archive={"pending_mb": 49.3 / 1024, "archive_mb": 0.0, "batches": 1},
        names={"count": 11587, "loaded": True},
    )
    assert "待打包 49.3 KB" in text
    assert "已归档 0 KB" in text
    assert "批次 1" in text


def test_status_uses_megabytes_once_the_archive_grows():
    text = format_status(
        online=True,
        health={},
        queue=0,
        running=0,
        archive={"pending_mb": 3.4, "archive_mb": 48.7, "batches": 6},
        names={"count": 10, "loaded": True},
    )
    assert "待打包 3.4 MB" in text
    assert "已归档 48.7 MB" in text
