#!/usr/bin/env python3
"""
Rebuild missing long/multi EveryAyah benchmark audio files from manifest entries.

Usage:
  python scripts/fetch_missing_everyayah_samples.py \
    --corpus benchmark/test_corpus \
    --reciter Alafasy_128kbps
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def everyayah_url(surah: int, ayah: int, reciter: str) -> str:
    return f"https://everyayah.com/data/{reciter}/{surah:03d}{ayah:03d}.mp3"


def download_to_wav(url: str, out_wav: Path) -> None:
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            url,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
    )


def build_multi_with_gap(surah: int, ayah_start: int, ayah_end: int, out_wav: Path, reciter: str) -> None:
    with tempfile.TemporaryDirectory(prefix="everyayah_multi_") as td:
        tmp = Path(td)
        segs: list[Path] = []

        gap = tmp / "gap.wav"
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "0.5",
                "-c:a",
                "pcm_s16le",
                str(gap),
            ]
        )

        for i, ayah in enumerate(range(ayah_start, ayah_end + 1)):
            part = tmp / f"seg_{i:03d}.wav"
            url = everyayah_url(surah, ayah, reciter)
            download_to_wav(url, part)
            segs.append(part)

        concat_list = tmp / "concat.txt"
        with concat_list.open("w", encoding="utf-8") as f:
            for i, seg in enumerate(segs):
                f.write(f"file '{seg.as_posix()}'\n")
                if i != len(segs) - 1:
                    f.write(f"file '{gap.as_posix()}'\n")

        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(out_wav),
            ]
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="benchmark/test_corpus", help="Corpus directory containing manifest.json")
    ap.add_argument("--reciter", default="Alafasy_128kbps", help="EveryAyah reciter folder")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    manifest_path = corpus / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])

    missing = []
    for s in samples:
        fname = s.get("file", "")
        if not (fname.startswith("long_") or fname.startswith("multi_")):
            continue
        out = corpus / fname
        if not out.exists():
            missing.append((s, out))

    if not missing:
        print("No missing long/multi files.")
        return 0

    print(f"Missing long/multi files: {len(missing)}")
    generated = 0
    failed = 0

    for s, out in missing:
        sid = s.get("id", out.name)
        surah = int(s["surah"])
        ayah = int(s["ayah"])
        ayah_end = s.get("ayah_end")
        print(f"Generating {sid} -> {out.name}")

        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            if ayah_end is None:
                url = everyayah_url(surah, ayah, args.reciter)
                download_to_wav(url, out)
            else:
                build_multi_with_gap(surah, ayah, int(ayah_end), out, args.reciter)
            generated += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED {sid}: {e}")

    print(f"Done. generated={generated} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
