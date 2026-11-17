"""Tests for the bilingual bird name table."""

from __future__ import annotations

import gzip

from bird.names import NameResolver, normalize_english, normalize_scientific


def _write_table(path, rows):
    """Write a gzipped name table with deterministic content."""
    body = "".join("%s\t%s\t%s\n" % row for row in rows)
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as handle:
        handle.write(body.encode("utf-8"))


def test_normalize_lowercases_and_collapses_separators():
    assert normalize_scientific("  Passer_Montanus  ") == "passer montanus"


def test_normalize_keeps_single_spaces_between_words():
    assert normalize_scientific("Parus   major") == "parus major"


def test_normalize_returns_empty_for_blank_input():
    assert normalize_scientific("   ") == ""


def test_normalize_returns_empty_for_empty_input():
    assert normalize_scientific("") == ""


def test_resolve_reads_chinese_and_english_names(tmp_path):
    table = tmp_path / "names.tsv.gz"
    _write_table(table, [("Passer montanus", "麻雀", "Eurasian Tree Sparrow")])
    resolver = NameResolver(table)
    assert resolver.resolve("Passer_montanus") == ("麻雀", "Eurasian Tree Sparrow")


def test_resolve_matches_regardless_of_case_and_spacing(tmp_path):
    table = tmp_path / "names.tsv.gz"
    _write_table(table, [("Parus major", "大山雀", "Great Tit")])
    resolver = NameResolver(table)
    assert resolver.resolve("  parus   MAJOR ") == ("大山雀", "Great Tit")


def test_resolve_uses_fallbacks_when_the_name_is_missing(tmp_path):
    table = tmp_path / "names.tsv.gz"
    _write_table(table, [("Parus major", "大山雀", "Great Tit")])
    resolver = NameResolver(table)
    assert resolver.resolve(
        "Unknownea birdia", fallback_zh="未知", fallback_en="Unknown"
    ) == (
        "未知",
        "Unknown",
    )


def test_resolve_returns_empty_fallbacks_by_default(tmp_path):
    table = tmp_path / "names.tsv.gz"
    _write_table(table, [])
    resolver = NameResolver(table)
    assert resolver.resolve("Unknownea birdia") == ("", "")


def test_resolve_skips_malformed_rows(tmp_path):
    table = tmp_path / "names.tsv.gz"
    with gzip.GzipFile(filename=str(table), mode="wb", mtime=0) as handle:
        handle.write(
            b"only-one-column\nPasser montanus\t\xe9\xba\xbb\xe9\x9b\x80\tSparrow\n"
        )
    resolver = NameResolver(table)
    assert resolver.resolve("Passer montanus") == ("麻雀", "Sparrow")


def test_missing_table_leaves_the_resolver_empty(tmp_path):
    resolver = NameResolver(tmp_path / "absent.tsv.gz")
    assert resolver.resolve("Passer montanus", fallback_en="Sparrow") == ("", "Sparrow")


def test_len_counts_rows_and_marks_the_table_loaded(tmp_path):
    table = tmp_path / "names.tsv.gz"
    _write_table(table, [("Parus major", "大山雀", "Great Tit")])
    resolver = NameResolver(table)
    assert len(resolver) == 1 and resolver.loaded


def test_normalize_english_folds_apostrophes_and_hyphens():
    assert normalize_english("Abbott's Babbler") == "abbotts babbler"
    assert normalize_english("Abbotts_Babbler") == "abbotts babbler"
    assert normalize_english("Black-browed Bushtit") == "black browed bushtit"
    assert normalize_english("  ") == ""


def test_resolve_falls_back_to_the_english_index(tmp_path):
    """The image classifier reports common names in the scientific slot."""
    table = tmp_path / "names.tsv.gz"
    _write_table(table, [("Passer montanus", "麻雀", "Eurasian Tree Sparrow")])
    resolver = NameResolver(table)
    assert resolver.resolve("Eurasian_Tree_Sparrow") == (
        "麻雀",
        "Eurasian Tree Sparrow",
    )


def test_resolve_prefers_the_scientific_index(tmp_path):
    table = tmp_path / "names.tsv.gz"
    _write_table(
        table,
        [
            ("Parus major", "大山雀", "Great Tit"),
            ("Fake birdia", "假鸟", "Parus major"),
        ],
    )
    resolver = NameResolver(table)
    assert resolver.resolve("Parus major") == ("大山雀", "Great Tit")


def test_bundled_table_resolves_a_common_name():
    resolver = NameResolver()
    zh, _ = resolver.resolve("Common Kingfisher")
    assert zh == "普通翠鸟"


def test_bundled_table_holds_the_expected_species():
    """The shipped table must resolve a well known Chinese species."""
    resolver = NameResolver()
    zh, en = resolver.resolve("Passer montanus")
    assert zh == "麻雀" and "Sparrow" in en
