"""Tests for the in-memory dashboard counters."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bird.stats import (  # noqa: E402  (path bootstrap)
    KIND_AUDIO,
    KIND_IMAGE,
    OUTCOME_EMPTY,
    OUTCOME_OK,
    StatsCollector,
)


def test_an_empty_collector_reports_zeros():
    snapshot = StatsCollector().snapshot()
    assert snapshot["total"] == 0
    assert snapshot["success_rate"] == 0.0
    assert snapshot["latency"] == {"avg": 0.0, "max": 0.0, "p95": 0.0, "count": 0}
    assert snapshot["top_species"] == []
    assert snapshot["recent"] == []


def test_one_success_is_counted_everywhere():
    collector = StatsCollector()
    collector.record(
        kind=KIND_IMAGE,
        outcome=OUTCOME_OK,
        elapsed_ms=120.0,
        trigger="auto",
        species="麻雀",
    )
    snapshot = collector.snapshot()
    assert snapshot["total"] == 1
    assert snapshot["by_outcome"] == {OUTCOME_OK: 1}
    assert snapshot["by_kind"][KIND_IMAGE] == 1
    assert snapshot["by_trigger"]["auto"] == 1
    assert snapshot["success_rate"] == 1.0
    assert snapshot["top_species"] == [{"name": "麻雀", "count": 1}]
    assert snapshot["daily"][-1]["ok"] == 1


def test_an_unknown_kind_is_counted_as_an_image():
    collector = StatsCollector()
    collector.record(kind="video", outcome=OUTCOME_OK)
    assert collector.snapshot()["by_kind"][KIND_IMAGE] == 1


def test_an_unknown_trigger_is_not_invented():
    collector = StatsCollector()
    collector.record(kind=KIND_AUDIO, outcome=OUTCOME_OK, trigger="scheduled")
    assert collector.snapshot()["by_trigger"] == {
        "auto": 0,
        "command": 0,
        "tool": 0,
    }


def test_a_non_positive_latency_is_ignored():
    collector = StatsCollector()
    collector.record(kind=KIND_IMAGE, outcome=OUTCOME_OK, elapsed_ms=0.0)
    assert collector.snapshot()["latency"]["count"] == 0


def test_an_empty_result_still_counts_as_a_success():
    collector = StatsCollector()
    collector.record(kind=KIND_IMAGE, outcome=OUTCOME_EMPTY)
    snapshot = collector.snapshot()
    assert snapshot["success_rate"] == 1.0
    assert snapshot["daily"][-1]["empty"] == 1


def test_a_species_is_only_ranked_after_a_real_result():
    collector = StatsCollector()
    collector.record(kind=KIND_IMAGE, outcome=OUTCOME_OK, species="麻雀")
    collector.record(kind=KIND_IMAGE, outcome="error", species="乌鸦")
    assert collector.snapshot()["top_species"] == [{"name": "麻雀", "count": 1}]


def test_a_blank_species_is_dropped():
    collector = StatsCollector()
    collector.record(kind=KIND_IMAGE, outcome=OUTCOME_OK, species="   ")
    assert collector.snapshot()["top_species"] == []


def test_rejected_messages_are_split_by_reason():
    collector = StatsCollector()
    collector.record_rejected("whitelist")
    collector.record_rejected("whitelist")
    collector.record_rejected("disabled")
    assert collector.snapshot()["rejected"] == {"whitelist": 2, "disabled": 1}


def test_an_unknown_rejection_reason_is_counted_as_disabled():
    collector = StatsCollector()
    collector.record_rejected("spam")
    assert collector.snapshot()["rejected"] == {"whitelist": 0, "disabled": 1}


def test_the_recent_list_is_bounded():
    collector = StatsCollector()
    for index in range(60):
        # A zero latency is ignored by design, so every sample starts at 1.
        collector.record(
            kind=KIND_IMAGE, outcome=OUTCOME_OK, elapsed_ms=float(index + 1)
        )
    snapshot = collector.snapshot()
    assert len(snapshot["recent"]) == 24
    assert snapshot["latency"]["count"] == 60


def test_latency_summary_uses_the_collected_samples():
    collector = StatsCollector()
    for value in (10.0, 20.0, 30.0, 40.0):
        collector.record(kind=KIND_IMAGE, outcome=OUTCOME_OK, elapsed_ms=value)
    latency = collector.snapshot()["latency"]
    assert latency["max"] == 40.0
    assert latency["avg"] == 25.0
    assert latency["count"] == 4


def test_top_species_is_bounded_and_ordered():
    collector = StatsCollector()
    for name, times in (("麻雀", 3), ("乌鸦", 1), ("喜鹊", 2)):
        for _ in range(times):
            collector.record(kind=KIND_IMAGE, outcome=OUTCOME_OK, species=name)
    assert collector.top_species(limit=2) == [
        {"name": "麻雀", "count": 3},
        {"name": "喜鹊", "count": 2},
    ]
    assert collector.top_species(limit=0) == []
