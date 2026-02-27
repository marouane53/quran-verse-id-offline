"""
Build a phonetic Quran DB from `Buraaq/quran-md-words`.

Outputs JSON like:
[
  {"surah": 1, "ayah": 1, "phonetic": "..."},
  ...
]

This is used for:
- training targets (ayah-level)
- candidate retrieval by fuzzy matching
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm

from quran_verse_id.normalizer import normalize_phonetic


def _pick(row: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSON path, e.g. data/quran_phonetic_db.json")
    ap.add_argument("--split", default="train", help="Dataset split")
    ap.add_argument("--dataset", default="Buraaq/quran-md-words", help="HF dataset name")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)

    # Collect word_tr by verse
    words_by_verse: Dict[Tuple[int, int], List[Tuple[int, str]]] = defaultdict(list)

    for row in tqdm(ds, desc="Reading quran-md-words"):
        surah = _pick(row, ["surah_id", "surah", "surah_number", "Surah_ID", "surahId"])
        ayah = _pick(row, ["ayah_id", "ayah", "ayah_number", "Aya_ID", "ayahId"])
        word_idx = _pick(row, ["word_idx", "word_index", "word_id", "wordNumber", "word_pos"])
        word_tr = _pick(row, ["word_tr", "transliteration", "word_transliteration", "phonetic", "word_phonetic"])

        if surah is None or ayah is None or word_idx is None or word_tr is None:
            # Fail loudly with diagnostic
            raise KeyError(
                f"Row is missing required keys. Available keys: {sorted(row.keys())}. "
                f"Found surah={surah}, ayah={ayah}, word_idx={word_idx}, word_tr={word_tr}"
            )

        surah_i = int(surah)
        ayah_i = int(ayah)
        word_i = int(word_idx)
        tr = normalize_phonetic(str(word_tr))
        if not tr:
            continue
        words_by_verse[(surah_i, ayah_i)].append((word_i, tr))

    entries = []
    for (surah, ayah), items in sorted(words_by_verse.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        items_sorted = sorted(items, key=lambda x: x[0])
        phonetic = " ".join([tr for _, tr in items_sorted])
        entries.append({"surah": surah, "ayah": ayah, "phonetic": phonetic})

    out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} verses to {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
