"""
Text normalization helpers.

We normalize:
- phonetic strings (Latin-ish, IPA-ish, dataset-specific)
- Arabic text (optional, for mapping/inspection)
"""
from __future__ import annotations

import re


_WS_RE = re.compile(r"\s+")
# Keep letters, digits, basic phonetic punctuation used in transliterations.
_PHONETIC_ALLOWED_RE = re.compile(r"[^a-zA-Z0-9ʔʕḥḍṣṭẓāīūáíúâîûäëïöü'`\- ]+")

_ARABIC_DIACRITICS_RE = re.compile(
    "["  # Arabic diacritics / harakat
    "\u0610-\u061A"
    "\u064B-\u065F"
    "\u0670"
    "\u06D6-\u06ED"
    "]+"
)


def normalize_phonetic(text: str) -> str:
    """
    Normalize Quran-MD phonetic transliteration strings and model outputs.

    Steps:
    - trim
    - lowercase
    - remove weird punctuation
    - collapse whitespace
    """
    if text is None:
        return ""
    t = text.strip().lower()
    t = _PHONETIC_ALLOWED_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def strip_arabic_diacritics(text: str) -> str:
    if text is None:
        return ""
    return _ARABIC_DIACRITICS_RE.sub("", text)


def normalize_arabic(text: str) -> str:
    """
    Lightweight Arabic normalization (for mapping/debug):
    - strip diacritics
    - unify alef variants
    - normalize ta marbuta / ya
    - remove tatweel
    - collapse whitespace
    """
    if text is None:
        return ""
    t = strip_arabic_diacritics(text)
    # tatweel
    t = t.replace("\u0640", "")
    # alef variants
    t = re.sub(r"[إأآا]", "ا", t)
    # ya variants
    t = t.replace("ى", "ي")
    # ta marbuta
    t = t.replace("ة", "ه")
    # hamza on waw/ya
    t = t.replace("ؤ", "و").replace("ئ", "ي")
    # whitespace
    t = _WS_RE.sub(" ", t).strip()
    return t
