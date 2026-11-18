"""Build the chat reply for an identification.

The reply is deliberately terse. The gate verdict collapses into one heading,
``识别结果：鸟类（30.5%）``, followed by the species the classifier proposes and
their confidence, the cropped images and the spectrogram, plus how long the
request took. Raw gate boxes never reach the chat: they are training data, not
something a reader needs. Nothing here mentions model names, file paths, device
names or any other detail about the machine behind the service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = ["Observation", "format_reply", "format_status"]


def _comparable(value: str) -> str:
    """Normalize a name so repeats can be detected.

    Args:
        value: Any name fragment.

    Returns:
        A case folded, single spaced form with the punctuation the
        classifiers drop already folded away.
    """
    cleaned = value.replace("_", " ")
    for mark in ("'", "\u2019", "`", "\u00b4"):
        cleaned = cleaned.replace(mark, "")
    for mark in ("-", "\u2010"):
        cleaned = cleaned.replace(mark, " ")
    return " ".join(cleaned.split()).casefold()


@dataclass(frozen=True)
class Observation:
    """One species guess ready for display.

    Attributes:
        zh: Chinese name, may be empty when the table has no row.
        en: English name.
        scientific: Scientific name.
        confidence: Confidence in percent.
        gate_box: Gate box for this observation, None for audio.
        gate_confidence: Detector confidence for the gate box.
        animal_class: One letter class code; B means bird.
    """

    zh: str
    en: str
    scientific: str
    confidence: float
    gate_box: Optional[Sequence[float]] = None
    gate_confidence: float = 0.0
    animal_class: str = "B"

    def label(self) -> str:
        """Return the bilingual name used in the reply.

        Returns:
            A string such as "麻雀 / Eurasian Tree Sparrow / Passer montanus".
        """
        parts: List[str] = []
        if self.zh:
            parts.append(self.zh)
        if self.en and self.en != self.zh:
            parts.append(self.en)
        latin = self.scientific
        seen = {_comparable(part) for part in parts}
        if latin and _comparable(latin) not in seen:
            parts.append(latin)
        return " / ".join(parts) if parts else "未知"


def _group_by_gate(observations: Sequence[Observation]) -> List[List[Observation]]:
    """Group candidates that came from the same gate box.

    The service emits one gate box per detection and lists its candidates in
    order, so candidates sharing a box are always adjacent.

    Args:
        observations: Species guesses in display order.

    Returns:
        Groups of adjacent observations, in the original order.
    """
    groups: List[List[Observation]] = []
    keys: List[tuple] = []
    for item in observations:
        key = tuple(item.gate_box) if item.gate_box else ()
        if groups and keys[-1] == key:
            groups[-1].append(item)
        else:
            keys.append(key)
            groups.append([item])
    return groups


def _gate_heading(members: Sequence[Observation]) -> str:
    """Render the one heading that replaces the raw gate box.

    Args:
        members: Observations produced by one gate box.

    Returns:
        A line such as "识别结果：鸟类（30.5%）", or "识别结果：鸟类" when the
        gate reported no score (recordings).
    """
    first = members[0]
    if first.animal_class == "B":
        label = "鸟类"
    else:
        label = first.zh or first.en or "未知目标"
    score = first.gate_confidence
    if score > 0:
        return "识别结果：%s（%.1f%%）" % (label, score)
    return "识别结果：%s" % label


def format_reply(
    observations: Sequence[Observation],
    *,
    kind: str,
    elapsed_ms: float,
    location: str = "",
    truncated: int = 0,
) -> str:
    """Render the text half of the reply.

    Each gate box contributes one heading, and the candidates it produced are
    listed under it, so a reader sees "识别结果：鸟类（30.5%）" instead of the
    coordinates the gate worked on.

    Args:
        observations: Species guesses in display order.
        kind: "image" or "audio".
        elapsed_ms: Total processing time in milliseconds.
        location: Optional normalized location that was applied.
        truncated: Number of further guesses that were left out.

    Returns:
        The reply text, without trailing whitespace.
    """
    lines: List[str] = []
    if not observations:
        lines.append("识别结果：未检测到鸟类。")
        if kind == "image":
            lines.append("可以试试换成主体更清楚、鸟占画面更大的照片。")
        else:
            lines.append("可以试试更长一些、背景噪声更小的录音。")
        lines.append("")
        lines.append("耗时 %d ms" % int(round(elapsed_ms)))
        return "\n".join(lines)

    index = 0
    for group in _group_by_gate(observations):
        lines.append(_gate_heading(group))
        for item in group:
            index += 1
            lines.append("%d. %s  %.1f%%" % (index, item.label(), item.confidence))
    if truncated > 0:
        lines.append("另有 %d 个候选未列出。" % truncated)
    if location:
        lines.append("地点先验：%s" % location)
    lines.append("")
    lines.append("耗时 %d ms" % int(round(elapsed_ms)))
    return "\n".join(lines)


def _format_size(megabytes: float) -> str:
    """Render a size so small values stay readable.

    Args:
        megabytes: Size in mebibytes.

    Returns:
        The size with a KB or MB unit; zero renders as ``0 KB``.
    """
    try:
        value = float(megabytes)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0:
        return "0 KB"
    if value < 1.0:
        return "%.1f KB" % (value * 1024.0)
    return "%.1f MB" % value


def format_status(
    *,
    online: bool,
    health: dict,
    queue: int,
    running: int,
    archive: dict,
    names: dict,
    location_hint: str = "",
) -> str:
    """Render the administrator status reply.

    Args:
        online: Whether the backend answered its health endpoint.
        health: Raw health payload from the backend.
        queue: Requests waiting for a worker slot in this plugin.
        running: Requests currently being processed by this plugin.
        archive: Archive counters.
        names: Name table counters.
        location_hint: Extra note about the location mapping.

    Returns:
        The status text.
    """
    lines = ["鸟类识别后端"]
    lines.append("状态：%s" % ("在线" if online else "离线"))
    if health:
        pending = health.get("pending")
        max_queue = health.get("max_queue")
        version = health.get("version")
        if version:
            lines.append("版本：%s" % version)
        if pending is not None and max_queue is not None:
            lines.append("后端队列：%s/%s" % (pending, max_queue))
        state = health.get("status")
        if state:
            lines.append("后端负载：%s" % state)
    lines.append("本插件：排队 %d，处理中 %d" % (queue, running))
    lines.append(
        "归档：待打包 %s，已归档 %s，批次 %d"
        % (
            _format_size(archive.get("pending_mb", 0.0)),
            _format_size(archive.get("archive_mb", 0.0)),
            archive.get("batches", 0),
        )
    )
    lines.append(
        "名称库：%s 条%s"
        % (
            names.get("count", 0),
            "" if names.get("loaded") else "（未加载）",
        )
    )
    if location_hint:
        lines.append(location_hint)
    return "\n".join(lines)
