"""Bilingual bird name resolution for identification results.

The identification backends only return scientific (Latin) names for birds:
the image pipeline reports Genus species in every label segment and the
audio pipeline reports the BirdNET English common name plus the scientific
name.  This module turns those labels into a Chinese name and an English name
using a compact table shipped with the plugin.

The table is a gzipped TSV with three tab separated columns:

    <normalized scientific name> TAB <chinese name> TAB <english name>

Names are normalized before lookup so that the different spellings used by the
classifiers (Genus_species, trailing spaces, mixed case) all resolve to the
same row.

The image classifier labels its candidates with an English common name instead
of the scientific name, so the table is also indexed by its English column and
a lookup that misses the scientific names falls back to that index.

A second table, ``bird_aliases.tsv.gz``, maps the classifier class names that
are missing from the first table onto the same three columns. Those classes
use taxonomy revisions the main table spells differently, so they are
resolved once at build time instead of being guessed at run time.
"""

from __future__ import annotations

import gzip
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

__all__ = [
    "DEFAULT_ALIAS_PATH",
    "DEFAULT_DATA_PATH",
    "NameResolver",
    "normalize_english",
    "normalize_scientific",
]

_LOGGER = logging.getLogger(__name__)

#: Table bundled with the plugin, relative to this module.
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "bird_names.tsv.gz"

#: Extra rows for classifier class names the main table does not carry.
DEFAULT_ALIAS_PATH = Path(__file__).resolve().parent / "data" / "bird_aliases.tsv.gz"


def normalize_english(value: str) -> str:
    """Normalize an English common name for reverse lookup.

    The classifiers drop apostrophes (``Abbotts_Babbler`` for
    ``Abbott's Babbler``) and are inconsistent about hyphens, so those
    characters are folded away before the table is indexed.

    Args:
        value: Raw English name reported by a classifier or stored in the
            table.

    Returns:
        The lowercase, single spaced key used for the reverse index.
    """
    if not value:
        return ""
    cleaned = value.replace("_", " ")
    for mark in ("'", "\u2019", "`", "\u00b4"):
        cleaned = cleaned.replace(mark, "")
    cleaned = cleaned.replace("-", " ").replace("\u2010", " ")
    return " ".join(cleaned.strip().lower().split())


def normalize_scientific(value: str) -> str:
    """Normalize a scientific name for table lookup.

    Args:
        value: Raw scientific name, possibly using underscores instead of
            spaces and carrying stray whitespace or mixed case.

    Returns:
        The lowercase, single-spaced form used as the table key. An empty
        string is returned for values without any usable characters.
    """
    if not value:
        return ""
    return " ".join(value.replace("_", " ").strip().lower().split())


class NameResolver:
    """Look up Chinese and English names for scientific names.

    The table is loaded lazily on first use and then kept in memory. Loading
    is guarded by a lock so concurrent first requests cannot read the file
    twice.

    Attributes:
        path: Table file that backs this resolver.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        alias_path: Optional[Path] = None,
    ) -> None:
        """Initialize the resolver without touching the filesystem.

        Args:
            path: Optional table path; defaults to the bundled table.
            alias_path: Optional alias table; defaults to the bundled file
                that sits beside the main table.
        """
        self.path = Path(path) if path is not None else DEFAULT_DATA_PATH
        if alias_path is not None:
            self.alias_path: Optional[Path] = Path(alias_path)
        elif path is None:
            self.alias_path = DEFAULT_ALIAS_PATH
        else:
            self.alias_path = None
        self._names: Dict[str, Tuple[str, str]] = {}
        self._by_english: Dict[str, Tuple[str, str]] = {}
        self._lock = threading.Lock()
        self._loaded = False

    def __len__(self) -> int:
        """Return the number of loaded rows, loading the table when needed."""
        self.load()
        return len(self._names)

    @property
    def loaded(self) -> bool:
        """Whether the table has already been read from disk."""
        return self._loaded

    def load(self) -> None:
        """Read the name table into memory.

        A missing or unreadable table is not fatal: identification still
        works, it only falls back to whatever the backend reported. The
        failure is logged once and the resolver keeps an empty table.

        Returns:
            None
        """
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            names: Dict[str, Tuple[str, str]] = {}
            by_english: Dict[str, Tuple[str, str]] = {}
            try:
                with gzip.open(self.path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) != 3:
                            continue
                        key = normalize_scientific(parts[0])
                        if not key:
                            continue
                        row = (parts[1].strip(), parts[2].strip())
                        names[key] = row
                        alias = normalize_english(row[1])
                        if alias and alias not in by_english:
                            by_english[alias] = row
                self._read_aliases(by_english)
            except FileNotFoundError:
                _LOGGER.warning("bird name table not found: %s", self.path)
            except (OSError, EOFError, UnicodeDecodeError) as exc:
                _LOGGER.warning("bird name table unreadable (%s): %s", self.path, exc)
            self._names = names
            self._by_english = by_english
            self._loaded = True

    def resolve(
        self,
        scientific: str,
        *,
        fallback_zh: str = "",
        fallback_en: str = "",
    ) -> Tuple[str, str]:
        """Resolve a species label to a (Chinese, English) pair.

        The label is looked up as a scientific name first and as an English
        common name second, because the image classifier reports common names
        in the scientific name slot.

        Args:
            scientific: Scientific or English name reported by the backend.
            fallback_zh: Chinese name used when the table has no row.
            fallback_en: English name used when the table has no row.

        Returns:
            A two-tuple of Chinese and English names. Either element can be
            an empty string, which tells the caller to omit that half.
        """
        self.load()
        for key, table in (
            (normalize_scientific(scientific), self._names),
            (normalize_english(scientific), self._by_english),
        ):
            if not key:
                continue
            row = table.get(key)
            if row is not None:
                zh, en = row
                return zh or fallback_zh, en or fallback_en
        return fallback_zh, fallback_en

    def _read_aliases(self, by_english: Dict[str, Tuple[str, str]]) -> None:
        """Merge the alias table into the English index.

        The alias table only covers classifier class names that the main
        table spells differently, so a missing or unreadable file is not an
        error: the resolver simply keeps the rows it already has.

        Args:
            by_english: English index built from the main table, updated in
                place.

        Returns:
            None
        """
        if self.alias_path is None:
            return
        try:
            with gzip.open(self.alias_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 3:
                        continue
                    key = normalize_english(parts[0])
                    if not key or key in by_english:
                        continue
                    by_english[key] = (parts[1].strip(), parts[2].strip())
        except FileNotFoundError:
            _LOGGER.warning("bird alias table not found: %s", self.alias_path)
        except (OSError, EOFError, UnicodeDecodeError) as exc:
            _LOGGER.warning(
                "bird alias table unreadable (%s): %s", self.alias_path, exc
            )
