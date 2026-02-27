"""
Build a small benchmark corpus from public sources.

This is NOT meant to replicate the exact offline-tarteel corpus.
If you want the exact same 54-sample corpus, use:
  scripts/import_offline_tarteel_benchmark.sh /path/to/offline-tarteel

This script downloads a configurable set of EveryAyah MP3 files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.request import urlretrieve


def everyayah_url(surah: int, ayah: int, reciter: str = "Alafasy_128kbps") -> str:
    # EveryAyah uses 3-digit surah + 3-digit ayah
    return f"https://everyayah.com/data/{reciter}/{surah:03d}{ayah:03d}.mp3"


DEFAULT_VERSES: List[Tuple[int, int]] = [
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (2, 255),
    (112, 1),
    (112, 2),
    (112, 3),
    (112, 4),
    (113, 1),
    (113, 2),
    (113, 3),
    (113, 4),
    (113, 5),
    (114, 1),
    (114, 2),
    (114, 3),
    (114, 4),
    (114, 5),
    (114, 6),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/test_corpus", help="Output corpus directory")
    ap.add_argument("--everyayah", type=int, default=20, help="How many EveryAyah clips to download")
    ap.add_argument("--reciter", default="Alafasy_128kbps", help="EveryAyah reciter folder")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    verses = DEFAULT_VERSES[: max(1, min(args.everyayah, len(DEFAULT_VERSES)))]

    samples: List[Dict] = []
    for surah, ayah in verses:
        fid = f"everyayah_{surah:03d}{ayah:03d}"
        fname = f"{fid}.mp3"
        url = everyayah_url(surah, ayah, reciter=args.reciter)
        dst = out_dir / fname
        if not dst.exists():
            print(f"Downloading {url} -> {dst}")
            urlretrieve(url, dst)
        samples.append(
            {
                "id": fid,
                "file": fname,
                "expected": {"surah": surah, "ayah": ayah, "ayah_end": None},
                "source": "everyayah",
                "category": "public",
            }
        )

    manifest = {"samples": samples}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest.json with {len(samples)} samples to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
