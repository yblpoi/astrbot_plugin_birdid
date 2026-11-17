"""Tests for the location to region code mapping."""

from __future__ import annotations

from bird.regions import normalize_location, resolve_location


def test_normalize_strips_trailing_punctuation():
    assert normalize_location("  广东省。 ") == "广东省"


def test_normalize_lowercases_english_names():
    assert normalize_location("Hong Kong") == "hong kong"


def test_normalize_returns_empty_for_blank_text():
    assert normalize_location("   ") == ""


def test_province_name_resolves_to_its_code():
    assert resolve_location("广东省") == ("CN-GD", "广东")


def test_short_province_name_resolves_too():
    assert resolve_location("云南") == ("CN-YN", "云南")


def test_english_country_name_resolves_to_national_code():
    assert resolve_location("China") == ("CN", "中国")


def test_country_prefix_is_skipped_when_looking_up():
    assert resolve_location("中国 云南") == ("CN-YN", "云南")


def test_province_without_a_code_yields_no_prior():
    assert resolve_location("山东省") == (None, "山东")


def test_unknown_place_keeps_the_original_text():
    assert resolve_location("加拿大") == (None, "加拿大")


def test_empty_text_yields_no_prior():
    assert resolve_location("") == (None, "")
