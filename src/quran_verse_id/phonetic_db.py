"""
Phonetic Quran DB.

This DB maps each (surah, ayah) -> canonical phonetic string.

We build it from `Buraaq/quran-md-words` where each word has:
- surah_id
- ayah_id
- word_idx (position in ayah)
- word_tr (phonetic transliteration)

Then we concatenate word_tr in order to get an ayah-level phonetic target.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import json

from .normalizer import normalize_phonetic


@dataclass(frozen=True)
class VerseKey:
    surah: int
    ayah: int

    def as_str(self) -> str:
        return f"{self.surah}:{self.ayah}"


@dataclass
class VerseEntry:
    surah: int
    ayah: int
    phonetic: str

    @property
    def key(self) -> VerseKey:
        return VerseKey(self.surah, self.ayah)


class PhoneticQuranDB:
    def __init__(self, entries: List[VerseEntry]):
        if not entries:
            raise ValueError("PhoneticQuranDB requires non-empty entries")
        self.entries = entries
        self._by_key: Dict[VerseKey, VerseEntry] = {e.key: e for e in entries}

    def get(self, surah: int, ayah: int) -> VerseEntry | None:
        return self._by_key.get(VerseKey(surah, ayah))

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load_json(cls, path: str | Path) -> "PhoneticQuranDB":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Phonetic DB not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        entries = []
        for row in data:
            entries.append(
                VerseEntry(
                    surah=int(row["surah"]),
                    ayah=int(row["ayah"]),
                    phonetic=normalize_phonetic(row["phonetic"]),
                )
            )
        return cls(entries)

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [{"surah": e.surah, "ayah": e.ayah, "phonetic": e.phonetic} for e in self.entries]
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
