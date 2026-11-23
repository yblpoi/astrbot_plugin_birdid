"""Runtime view of the plugin settings and its access policy.

The configuration that ships in ``_conf_schema.json`` is a flat mapping, but a
few of its entries are switches that have to be understood together: the
whitelist toggles only mean something next to the lists they guard, and the
element selector has to survive the round trip through the dashboard unchanged.
Everything here therefore normalises once, at load time, and hands the rest of
the plugin a frozen snapshot instead of raw JSON values.

Nothing in this module imports AstrBot, so the policy can be unit tested
without the framework installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

__all__ = [
    "DEFAULT_CLASS_CODE",
    "DEFAULT_IMAGE_MODE",
    "IMAGE_MODES",
    "Settings",
    "SettingsError",
    "normalize_class_code",
    "normalize_image_mode",
    "normalize_patch",
    "parse_entries",
]

DEFAULT_CLASS_CODE = "B"

#: Element letters accepted by the Dongniao ``class`` field.
CLASS_LETTERS = ("B", "M", "A", "R", "F")

#: Pictures the bot may send after a successful identification.
#:
#: ``all`` keeps every generated picture, ``crop`` only keeps the cropped
#: photo of a picture request, ``spectrogram`` only keeps the rendered
#: spectrogram of a recording, and ``off`` sends text alone.
IMAGE_MODES = ("all", "crop", "spectrogram", "off")
DEFAULT_IMAGE_MODE = "all"

#: Guard rails so a malformed dashboard payload cannot grow the config.
MAX_ENTRIES = 200
MAX_ENTRY_LENGTH = 128
MAX_TEXT_LENGTH = 200

BOOL_FIELDS = (
    "auto_reply",
    "archive_enabled",
    "group_whitelist_enabled",
    "private_whitelist_enabled",
)
LIST_FIELDS = ("group_whitelist", "private_whitelist")
TEXT_FIELDS = ("class_code", "api_area", "did", "model_names", "image_mode")


_TRUE_WORDS = frozenset({"1", "true", "yes", "on", "enabled", "开", "是"})
_FALSE_WORDS = frozenset({"0", "false", "no", "off", "disabled", "关", "否"})


class SettingsError(ValueError):
    """Raised when a settings patch cannot be normalised."""


def normalize_class_code(value: Any) -> str:
    """Reduce a raw selector to the letters the protocol understands.

    The protocol accepts any combination of B, M, A, R and F. Unknown letters
    are dropped instead of rejected so a typo narrows the search less than a
    hard failure would.

    Args:
        value: Raw selector from the configuration or the dashboard.

    Returns:
        The sanitised selector, possibly empty when nothing valid remains.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    seen = []
    for char in value.upper():
        if char in CLASS_LETTERS and char not in seen:
            seen.append(char)
    return "".join(seen)


def normalize_image_mode(value: Any) -> str:
    """Reduce a raw picture mode to one of :data:`IMAGE_MODES`.

    Unknown values fall back to the default instead of raising, so an older
    configuration file keeps working after a downgrade.

    Args:
        value: Raw mode from the configuration or the dashboard.

    Returns:
        One of the accepted mode names.
    """
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in IMAGE_MODES:
            return text
        aliases = {
            "1": "all",
            "true": "all",
            "on": "all",
            "开": "all",
            "开启": "all",
            "0": "off",
            "false": "off",
            "关": "off",
            "关闭": "off",
            "photo": "crop",
            "crop_only": "crop",
            "仅照片": "crop",
            "spectrogram_only": "spectrogram",
            "仅频谱图": "spectrogram",
        }
        if text in aliases:
            return aliases[text]
    return DEFAULT_IMAGE_MODE


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce one configuration value to a boolean.

    Args:
        value: Raw value; None and blank strings mean "use the default".
        default: Value used when nothing usable is present.

    Returns:
        The resolved boolean.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if not text:
            return default
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
    return default


def _split_raw(value: Any) -> Iterable[str]:
    """Yield the individual entries of a raw list-like value.

    Args:
        value: A sequence, a set or a single string with separators.

    Returns:
        An iterable of raw entry strings.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        for chunk in value.replace("、", ",").replace("\n", ",").split(","):
            yield chunk
        return
    if isinstance(value, Mapping):
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            if isinstance(item, str):
                yield item
            elif item is not None and not isinstance(item, Mapping):
                yield str(item)


def parse_entries(value: Any) -> Tuple[str, ...]:
    """Normalise a whitelist into a deduplicated tuple of identifiers.

    Blank entries are dropped, surrounding whitespace is removed and the
    result is capped so a runaway config cannot exhaust memory.

    Args:
        value: Raw list, comma separated string or single identifier.

    Returns:
        The cleaned identifiers in their original order.
    """
    seen = []
    lookup = set()
    for raw in _split_raw(value):
        text = " ".join(str(raw).split())
        if not text:
            continue
        text = text[:MAX_ENTRY_LENGTH]
        if text in lookup:
            continue
        lookup.add(text)
        seen.append(text)
        if len(seen) >= MAX_ENTRIES:
            break
    return tuple(seen)


def _as_text(value: Any, default: str = "") -> str:
    """Coerce one configuration value to a trimmed single-line string.

    Args:
        value: Raw value.
        default: Value used when nothing usable is present.

    Returns:
        The cleaned text.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    text = " ".join(str(value).split())
    if not text:
        return default
    return text[:MAX_TEXT_LENGTH]


def _matches(candidates: Sequence[str], *values: str) -> bool:
    """Return whether any identifier matches one of the candidate entries.

    A whitelist entry may name a whole chat (``platform:type:id``), a bare
    session id or a group id, so every identifier the event can offer is
    compared against every entry.

    Args:
        candidates: Normalised whitelist entries.
        values: Identifiers offered by the event; blanks are ignored.
    Returns:
        True when the chat is listed.
    """
    if not candidates:
        return False
    wanted = {value for value in values if value}
    if not wanted:
        return False
    lowered = {value.casefold() for value in wanted}
    for entry in candidates:
        if entry in wanted:
            return True
        if entry.casefold() in lowered:
            return True
    return False


@dataclass(frozen=True)
class Settings:
    """Frozen snapshot of the plugin settings.

    Attributes:
        auto_reply: Whether media is recognised without an explicit request.
        archive_enabled: Whether observations are written to disk at all.
        group_whitelist_enabled: Whether group chats are filtered.
        group_whitelist: Group chats allowed to reach the recogniser.
        private_whitelist_enabled: Whether direct chats are filtered.
        private_whitelist: Direct chats allowed to reach the recogniser.
        class_code: Element selector forwarded to the service.
        api_area: Optional Dongniao region code used as a prior.
        did: Optional device identifier forwarded to the service.
        model_names: Optional model names shown in the status reply.
        image_mode: Which pictures a successful reply may carry.
    """

    auto_reply: bool = True
    archive_enabled: bool = True
    group_whitelist_enabled: bool = False
    group_whitelist: Tuple[str, ...] = ()
    private_whitelist_enabled: bool = False
    private_whitelist: Tuple[str, ...] = ()
    class_code: str = DEFAULT_CLASS_CODE
    api_area: str = ""
    did: str = ""
    model_names: str = ""
    image_mode: str = DEFAULT_IMAGE_MODE

    @classmethod
    def from_config(cls, config: Any) -> "Settings":
        """Build a snapshot from the raw plugin configuration.

        Args:
            config: Mapping holding the plugin settings; missing keys fall
                back to their defaults and unusable values are ignored.

        Returns:
            The normalised snapshot.
        """
        if hasattr(config, "get"):
            get = config.get
        else:

            def get(key: str, default: Any = None) -> Any:
                """Fall back to the default for a plain configuration object."""
                return default

        return cls(
            auto_reply=_as_bool(get("auto_reply", True), True),
            archive_enabled=_as_bool(get("archive_enabled", True), True),
            group_whitelist_enabled=_as_bool(
                get("group_whitelist_enabled", False), False
            ),
            group_whitelist=parse_entries(get("group_whitelist", ())),
            private_whitelist_enabled=_as_bool(
                get("private_whitelist_enabled", False), False
            ),
            private_whitelist=parse_entries(get("private_whitelist", ())),
            class_code=normalize_class_code(get("class_code", DEFAULT_CLASS_CODE)),
            api_area=_as_text(get("api_area", "")),
            did=_as_text(get("did", "")),
            model_names=_as_text(get("model_names", "")),
            image_mode=normalize_image_mode(get("image_mode", DEFAULT_IMAGE_MODE)),
        )

    def to_public_dict(self) -> Dict[str, Any]:
        """Return the snapshot in the shape the dashboard consumes.

        Returns:
            A JSON friendly mapping with lists instead of tuples.
        """
        return {
            "auto_reply": self.auto_reply,
            "archive_enabled": self.archive_enabled,
            "group_whitelist_enabled": self.group_whitelist_enabled,
            "group_whitelist": list(self.group_whitelist),
            "private_whitelist_enabled": self.private_whitelist_enabled,
            "private_whitelist": list(self.private_whitelist),
            "class_code": self.class_code,
            "api_area": self.api_area,
            "did": self.did,
            "model_names": self.model_names,
            "image_mode": self.image_mode,
        }

    def allows(
        self,
        *,
        is_group: bool,
        chat_key: str = "",
        session_id: str = "",
        group_id: str = "",
        chat_alias: str = "",
    ) -> Tuple[bool, str]:
        """Decide whether one chat may reach the recogniser.

        A whitelist that is enabled but empty blocks every chat in that
        category: an empty list reads as "nobody", not as "everybody".

        Args:
            is_group: Whether the message came from a group chat.
            chat_key: Unified message origin of the chat.
            session_id: Platform session id of the chat.
            group_id: Group identifier; blank outside group chats.
            chat_alias: Group chat without its trailing sender segment. A
                platform that keeps one session per speaker appends the
                sender to the session id, so an entry picked from the group
                itself would otherwise never match the live event.

        Returns:
            A (allowed, reason) tuple where reason names the blocking rule.
        """
        if is_group:
            if not self.group_whitelist_enabled:
                return True, "ok"
            if _matches(
                self.group_whitelist, chat_key, session_id, group_id, chat_alias
            ):
                return True, "ok"
            return False, "group_not_whitelisted"
        if not self.private_whitelist_enabled:
            return True, "ok"
        if _matches(self.private_whitelist, chat_key, session_id):
            return True, "ok"
        return False, "private_not_whitelisted"


def _strict_bool(key: str, value: Any) -> bool:
    """Normalise one boolean patch value, rejecting ambiguous input.

    Args:
        key: Field name, used in the error message.
        value: Raw value from the dashboard.

    Returns:
        The boolean value.

    Raises:
        SettingsError: The value is a string that is neither truthy nor falsy.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
    raise SettingsError("%s 需要是开关值" % key)


def normalize_patch(patch: Any) -> Dict[str, Any]:
    """Normalise a settings patch coming from the dashboard.

    Only known fields are returned, so an unexpected key can never end up in
    the configuration file.

    Args:
        patch: Mapping submitted by the page.

    Returns:
        A mapping of normalised values, ready to be merged into the config.

    Raises:
        SettingsError: The patch is not a mapping or a field is malformed.
    """
    if not isinstance(patch, Mapping):
        raise SettingsError("设置内容格式不正确")
    cleaned: Dict[str, Any] = {}
    for key in BOOL_FIELDS:
        if key in patch:
            cleaned[key] = _strict_bool(key, patch[key])
    for key in LIST_FIELDS:
        if key in patch:
            cleaned[key] = list(parse_entries(patch[key]))
    for key in TEXT_FIELDS:
        if key not in patch:
            continue
        raw = patch[key]
        if key == "class_code":
            cleaned[key] = normalize_class_code(raw)
            continue
        if key == "image_mode":
            cleaned[key] = normalize_image_mode(raw)
            continue
        cleaned[key] = _as_text(raw)
    return cleaned
