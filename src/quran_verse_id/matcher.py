"""
Candidate retrieval for verse recognition.

We:
- take a phonetic transcript (from the CTC model)
- fuzzy match against a phonetic Quran DB
- return top-K candidates for forced-alignment scoring
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set

from rapidfuzz import fuzz, process

from .normalizer import normalize_phonetic
from .phonetic_db import PhoneticQuranDB, VerseEntry


def _trigrams(s: str) -> Set[str]:
    s = s.replace(" ", "_")
    if len(s) < 3:
        return {s} if s else set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


class TrigramIndex:
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.inv: Dict[str, List[int]] = {}
        for i, t in enumerate(texts):
            for tri in _trigrams(t):
                self.inv.setdefault(tri, []).append(i)

    def candidate_indices(self, query: str, *, min_pool: int = 1500, max_pool: int = 6000) -> List[int]:
        """
        Returns a pool of candidate indices based on trigram overlap.
        """
        q_tris = _trigrams(query)
        if not q_tris:
            return list(range(len(self.texts)))

        # Count hits per entry
        counts: Dict[int, int] = {}
        for tri in q_tris:
            for idx in self.inv.get(tri, []):
                counts[idx] = counts.get(idx, 0) + 1

        if not counts:
            return list(range(len(self.texts)))

        # Sort by overlap desc
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        pool_size = max(min_pool, min(max_pool, len(ranked)))
        return [idx for idx, _ in ranked[:pool_size]]


@dataclass
class Candidate:
    surah: int
    ayah: int
    ayah_end: int | None
    text_clean: str          # phonetic candidate string for scoring
    match_score: float       # fuzzy match score (0-100)


class CandidateRetriever:
    def __init__(self, db: PhoneticQuranDB):
        # Ensure stable ordering by (surah, ayah)
        self.db_entries: List[VerseEntry] = sorted(db.entries, key=lambda e: (e.surah, e.ayah))
        self.texts: List[str] = [e.phonetic for e in self.db_entries]
        self.index = TrigramIndex(self.texts)
        # Build quick lookup by (surah, ayah) -> position
        self.pos: Dict[Tuple[int, int], int] = {(e.surah, e.ayah): i for i, e in enumerate(self.db_entries)}

    def get_phonetic(self, surah: int, ayah: int) -> str | None:
        i = self.pos.get((surah, ayah))
        if i is None:
            return None
        return self.texts[i]

    def next_verse(self, surah: int, ayah: int) -> Tuple[int, int] | None:
        i = self.pos.get((surah, ayah))
        if i is None:
            return None
        if i + 1 >= len(self.db_entries):
            return None
        nxt = self.db_entries[i + 1]
        return (nxt.surah, nxt.ayah)

    def search(self, transcript: str, *, top_k: int = 50) -> List[Candidate]:
        q = normalize_phonetic(transcript)
        if not q:
            return []

        pool = self.index.candidate_indices(q)
        # Use rapidfuzz over the pooled candidates for speed.
        pooled_texts = [self.texts[i] for i in pool]
        matches = process.extract(
            q,
            pooled_texts,
            scorer=fuzz.QRatio,
            limit=top_k,
        )

        candidates: List[Candidate] = []
        for _matched_text, score, pool_idx in matches:
            global_idx = pool[pool_idx]
            entry = self.db_entries[global_idx]
            candidates.append(
                Candidate(
                    surah=entry.surah,
                    ayah=entry.ayah,
                    ayah_end=None,
                    text_clean=entry.phonetic,
                    match_score=float(score) / 100.0,
                )
            )
        return candidates

    def expand_spans(self, starts: List[Candidate], *, max_span_ayahs: int = 3) -> List[Candidate]:
        """
        For each start candidate, create span candidates by concatenating the next N ayat
        (within the DB ordering). This helps with multi-ayah clips.
        """
        out: List[Candidate] = []
        seen = set()

        for c in starts:
            cur_surah, cur_ayah = c.surah, c.ayah
            phon_parts = []
            for span_len in range(1, max_span_ayahs + 1):
                ph = self.get_phonetic(cur_surah, cur_ayah)
                if ph is None:
                    break
                phon_parts.append(ph)
                text = " ".join(phon_parts)
                ayah_end = cur_ayah
                key = (c.surah, c.ayah, ayah_end)
                if key not in seen:
                    out.append(
                        Candidate(
                            surah=c.surah,
                            ayah=c.ayah,
                            ayah_end=ayah_end if ayah_end != c.ayah else None,
                            text_clean=text,
                            match_score=c.match_score,
                        )
                    )
                    seen.add(key)

                nxt = self.next_verse(cur_surah, cur_ayah)
                if nxt is None:
                    break
                # Only allow spans that stay in the same surah (optional but safer).
                if nxt[0] != c.surah:
                    break
                cur_surah, cur_ayah = nxt

        # Sort: prefer higher fuzzy match_score initially (CTC scoring will override)
        out.sort(key=lambda x: x.match_score, reverse=True)
        return out
