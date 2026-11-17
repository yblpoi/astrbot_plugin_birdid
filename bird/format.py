"""Build the chat reply for an identification.

The reply is deliberately terse. It shows the species the models propose, the
gate boxes the detector produced, the cropped images and the spectrogram, plus
how long the request took. It never mentions model names, file paths, device
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


def _format_box(box: Optional[Sequence[float]]) -> str:
    """Format a gate box for display.

    Args:
        box: Four coordinates, or None.

    Returns:
        A compact "x1,y1-x2,y2" string, or an empty string.
    """
    if not box or len(box) != 4:
        return ""
    try:
        values = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return ""
    return "%d,%d-%d,%d" % (values[0], values[1], values[2], values[3])


def format_reply(
    observations: Sequence[Observation],
    *,
    kind: str,
    elapsed_ms: float,
    location: str = "",
    truncated: int = 0,
) -> str:
    """Render the text half of the reply.

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
        lines.append("没有识别到鸟类。")
        if kind == "image":
            lines.append("可以试试换成主体更清楚、鸟占画面更大的照片。")
        else:
            lines.append("可以试试更长一些、背景噪声更小的录音。")
        lines.append("")
        lines.append("耗时 %d ms" % int(round(elapsed_ms)))
        return "\n".join(lines)

    birds = [item for item in observations if item.animal_class == "B"]
    if birds:
        lines.append("识别结果")
    else:
        lines.append("画面里没有鸟类，检测到的是")
    for index, item in enumerate(observations, start=1):
        head = "%d. %s  %.1f%%" % (index, item.label(), item.confidence)
        lines.append(head)
        box = _format_box(item.gate_box)
        if box:
            lines.append("   检测框 %s  门禁 %.1f%%" % (box, item.gate_confidence))
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
