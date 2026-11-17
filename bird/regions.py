"""Map free form location text onto Dongniao region codes.

The identification service accepts an area code and turns it into a
representative coordinate for its geographic prior. Only the codes listed in
its regions table are known: unknown codes are silently ignored there, so this
module never invents coordinates. Provinces without a code fall through to no
prior at all, which is safer than snapping them to the national centroid.

Location text arrives from chat, so the matcher accepts the usual Chinese
short forms ("广东", "广东省") as well as English country names.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

__all__ = ["REGION_CODES", "normalize_location", "resolve_location"]

#: Region codes known to the service, mapped to their display name.
REGION_CODES: Dict[str, str] = {
    "CN": "中国",
    "CN-BJ": "北京",
    "CN-SH": "上海",
    "CN-GD": "广东",
    "CN-ZJ": "浙江",
    "CN-JS": "江苏",
    "CN-SC": "四川",
    "CN-YN": "云南",
    "CN-HE": "河北",
    "CN-LN": "辽宁",
    "CN-AM": "澳门",
    "CN-HK": "香港",
    "CN-TW": "台湾",
}

#: Suffixes stripped before a lookup, longest first.
_SUFFIXES = (
    "维吾尔自治区",
    "壮族自治区",
    "回族自治区",
    "特别行政区",
    "自治区",
    "省",
    "市",
    "地区",
)

_ALIASES: Dict[str, str] = {
    # Provincial level units that do have a code.
    "北京": "CN-BJ",
    "北京市": "CN-BJ",
    "上海": "CN-SH",
    "上海市": "CN-SH",
    "广东": "CN-GD",
    "广东省": "CN-GD",
    "浙江": "CN-ZJ",
    "浙江省": "CN-ZJ",
    "江苏": "CN-JS",
    "江苏省": "CN-JS",
    "四川": "CN-SC",
    "四川省": "CN-SC",
    "云南": "CN-YN",
    "云南省": "CN-YN",
    "河北": "CN-HE",
    "河北省": "CN-HE",
    "辽宁": "CN-LN",
    "辽宁省": "CN-LN",
    "澳门": "CN-AM",
    "澳门特别行政区": "CN-AM",
    "香港": "CN-HK",
    "香港特别行政区": "CN-HK",
    "台湾": "CN-TW",
    "台湾省": "CN-TW",
    "中国": "CN",
    "中国大陆": "CN",
    # English names for the same units.
    "beijing": "CN-BJ",
    "shanghai": "CN-SH",
    "guangdong": "CN-GD",
    "zhejiang": "CN-ZJ",
    "jiangsu": "CN-JS",
    "sichuan": "CN-SC",
    "yunnan": "CN-YN",
    "hebei": "CN-HE",
    "liaoning": "CN-LN",
    "macau": "CN-AM",
    "macao": "CN-AM",
    "hong kong": "CN-HK",
    "hongkong": "CN-HK",
    "taiwan": "CN-TW",
    "china": "CN",
    "prc": "CN",
}

#: Provinces and regions the service has no coordinates for. They are listed so
#: the reply can stay quiet instead of guessing a national centroid.
_UNSUPPORTED = (
    "天津",
    "重庆",
    "山东",
    "河南",
    "湖北",
    "湖南",
    "福建",
    "江西",
    "安徽",
    "山西",
    "陕西",
    "甘肃",
    "贵州",
    "广西",
    "海南",
    "新疆",
    "西藏",
    "青海",
    "宁夏",
    "内蒙古",
    "吉林",
    "黑龙江",
)

_SPACE_RE = re.compile(r"[\s,，、/]+")


def normalize_location(text: str) -> str:
    """Normalize free form location text for lookup.

    Args:
        text: Location fragment typed by the user.

    Returns:
        Lowercase text with surrounding punctuation and spacing removed. An
        empty string is returned when nothing usable is left.
    """
    if not text:
        return ""
    cleaned = text.strip().strip("。.,，、;；:：!！?？()（）[]【】")
    cleaned = _SPACE_RE.sub(" ", cleaned)
    return cleaned.strip().lower()


def resolve_location(text: str) -> Tuple[Optional[str], str]:
    """Resolve a location fragment into a service region code.

    The lookup tries the raw text, then the text with an administrative suffix
    removed, then the same candidates without a leading country qualifier such
    as "中国".

    Args:
        text: Location fragment typed by the user, for example "云南省".

    Returns:
        A two-tuple of the region code and a display name. The code is None
        when the fragment names a place the service cannot locate, in which
        case no geographic prior is sent at all.
    """
    for candidate in _candidates(normalize_location(text)):
        code = _ALIASES.get(candidate)
        if code is None:
            code = _ALIASES.get(_strip_suffix(candidate))
        if code is not None:
            return code, REGION_CODES.get(code, candidate)
        short = _strip_suffix(candidate)
        if short in _UNSUPPORTED:
            return None, short
    return None, text.strip()


def _candidates(normalized: str):
    """Yield lookup candidates from most to least specific.

    Args:
        normalized: Text produced by normalize_location.

    Returns:
        An iterator of candidate keys.
    """
    if not normalized:
        return
    yield normalized
    parts = normalized.split(" ")
    if len(parts) > 1:
        yield "".join(parts)
        yield parts[-1]
    if len(normalized) > 2 and normalized[0] in "中国":
        yield normalized[1:]


def _strip_suffix(value: str) -> str:
    """Remove one administrative suffix from a location candidate.

    Args:
        value: Normalized location text.

    Returns:
        The text without its suffix, or the input when no suffix matched.
    """
    for suffix in _SUFFIXES:
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)]
    return value
