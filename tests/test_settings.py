"""Tests for the runtime settings and the access policy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bird.settings import (  # noqa: E402  (path bootstrap)
    MAX_ENTRIES,
    Settings,
    SettingsError,
    normalize_class_code,
    normalize_patch,
    parse_entries,
)


def test_missing_keys_fall_back_to_the_defaults():
    settings = Settings.from_config({})
    assert settings.auto_reply is True
    assert settings.archive_enabled is True
    assert settings.group_whitelist_enabled is False
    assert settings.group_whitelist == ()
    assert settings.private_whitelist_enabled is False
    assert settings.private_whitelist == ()
    assert settings.class_code == "B"
    assert settings.api_area == ""
    assert settings.did == ""


def test_a_plain_object_without_get_keeps_the_defaults():
    settings = Settings.from_config(None)
    assert settings.auto_reply is True
    assert settings.class_code == "B"


def test_blank_switch_values_keep_the_default():
    settings = Settings.from_config({"auto_reply": "", "archive_enabled": None})
    assert settings.auto_reply is True
    assert settings.archive_enabled is True


def test_chinese_switch_words_are_understood():
    settings = Settings.from_config({"auto_reply": "关", "archive_enabled": "开"})
    assert settings.auto_reply is False
    assert settings.archive_enabled is True


def test_whitelists_accept_a_string_a_list_and_duplicates():
    settings = Settings.from_config(
        {
            "group_whitelist": "111、222,111",
            "private_whitelist": [" 42 ", "42", ""],
        }
    )
    assert settings.group_whitelist == ("111", "222")
    assert settings.private_whitelist == ("42",)


def test_parse_entries_is_bounded():
    raw = ["id%d" % index for index in range(MAX_ENTRIES + 50)]
    assert len(parse_entries(raw)) == MAX_ENTRIES


def test_class_code_keeps_only_known_letters():
    assert normalize_class_code("bmz") == "BM"
    assert normalize_class_code("B,M") == "BM"
    assert normalize_class_code(None) == ""
    assert normalize_class_code("") == ""


def test_allows_every_chat_when_no_whitelist_is_enabled():
    settings = Settings.from_config({})
    assert settings.allows(is_group=True, group_id="1") == (True, "ok")
    assert settings.allows(is_group=False, session_id="9") == (True, "ok")


def test_an_enabled_but_empty_whitelist_blocks_everyone():
    settings = Settings.from_config({"group_whitelist_enabled": True})
    allowed, reason = settings.allows(is_group=True, group_id="1")
    assert allowed is False
    assert reason == "group_not_whitelisted"


def test_a_group_whitelist_matches_the_group_id():
    settings = Settings.from_config(
        {"group_whitelist_enabled": True, "group_whitelist": ["7788"]}
    )
    assert settings.allows(is_group=True, group_id="7788")[0] is True
    assert settings.allows(is_group=True, group_id="1")[0] is False


def test_a_private_whitelist_matches_the_session_id():
    settings = Settings.from_config(
        {"private_whitelist_enabled": True, "private_whitelist": ["42"]}
    )
    assert settings.allows(is_group=False, session_id="42")[0] is True
    allowed, reason = settings.allows(is_group=False, session_id="7")
    assert allowed is False
    assert reason == "private_not_whitelisted"


def test_a_whitelisted_group_does_not_open_private_chats():
    settings = Settings.from_config(
        {
            "group_whitelist_enabled": True,
            "group_whitelist": ["1"],
            "private_whitelist_enabled": True,
            "private_whitelist": [],
        }
    )
    assert settings.allows(is_group=True, group_id="1")[0] is True
    assert settings.allows(is_group=False, session_id="1")[0] is False


def test_to_public_dict_round_trips_through_the_form():
    settings = Settings.from_config(
        {"group_whitelist": ["1"], "class_code": "bm", "did": "abc123"}
    )
    public = settings.to_public_dict()
    assert public["group_whitelist"] == ["1"]
    assert public["class_code"] == "BM"
    assert public["did"] == "abc123"
    assert isinstance(public["group_whitelist"], list)


def test_normalize_patch_keeps_only_known_keys():
    cleaned = normalize_patch({"auto_reply": False, "unexpected": "x"})
    assert cleaned == {"auto_reply": False}


def test_normalize_patch_normalises_every_field_type():
    cleaned = normalize_patch(
        {
            "auto_reply": "yes",
            "group_whitelist": "1,2",
            "class_code": "m",
            "did": "  abc  ",
        }
    )
    assert cleaned == {
        "auto_reply": True,
        "group_whitelist": ["1", "2"],
        "class_code": "M",
        "did": "abc",
    }


def test_normalize_patch_rejects_a_non_mapping():
    with pytest.raises(SettingsError):
        normalize_patch(["auto_reply"])


def test_normalize_patch_rejects_an_ambiguous_switch():
    with pytest.raises(SettingsError):
        normalize_patch({"auto_reply": "maybe"})


def test_normalize_patch_rejects_a_boolean_used_as_text():
    cleaned = normalize_patch({"did": True})
    assert cleaned == {"did": ""}


def test_image_mode_defaults_to_sending_every_picture():
    settings = Settings.from_config({})
    assert settings.image_mode == "all"


def test_image_mode_accepts_every_documented_value():
    for value in ("all", "crop", "spectrogram", "off"):
        assert Settings.from_config({"image_mode": value}).image_mode == value


def test_image_mode_reads_the_chinese_labels():
    assert Settings.from_config({"image_mode": "关闭"}).image_mode == "off"
    assert Settings.from_config({"image_mode": "仅照片"}).image_mode == "crop"
    assert Settings.from_config({"image_mode": "仅频谱图"}).image_mode == "spectrogram"


def test_image_mode_falls_back_instead_of_failing_an_old_config():
    assert Settings.from_config({"image_mode": "nonsense"}).image_mode == "all"
    assert Settings.from_config({"image_mode": None}).image_mode == "all"


def test_normalize_patch_carries_the_image_mode():
    assert normalize_patch({"image_mode": "关闭"}) == {"image_mode": "off"}


def test_a_group_is_matched_through_its_alias():
    group = "wechat_native:GroupMessage:main|group|room"
    settings = Settings.from_config(
        {"group_whitelist_enabled": True, "group_whitelist": [group]}
    )
    allowed, reason = settings.allows(
        is_group=True,
        chat_key=group + "|wxid_speaker",
        group_id="room",
        chat_alias=group,
    )
    assert allowed is True
    assert reason == "ok"
