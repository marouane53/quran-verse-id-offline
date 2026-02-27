"""
Build a small benchmark corpus from public sources.

This is NOT meant to replicate the exact offline-tarteel corpus.
If you want the exact same 54-sample corpus, use:
  scripts/import_offline_tarteel_benchmark.sh /path/to/offline-tarteel

This script can build a mixed corpus:
- EveryAyah public MP3 references
- RetaSy quranic_audio_dataset samples (via Hugging Face datasets)
"""
from __future__ import annotations

import argparse
import json
import shutil
import wave
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.request import urlretrieve

import numpy as np


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


def _pick(row: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _write_wav_from_array(audio_array: Any, sampling_rate: int, dst: Path) -> None:
    wav = np.asarray(audio_array, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav[:, 0]
    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767.0).astype(np.int16)

    with wave.open(str(dst), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)  # int16
        f.setframerate(int(sampling_rate))
        f.writeframes(pcm16.tobytes())


def _save_audio(audio_obj: Any, dst: Path) -> bool:
    if isinstance(audio_obj, dict):
        src_path = audio_obj.get("path")
        if isinstance(src_path, str) and src_path:
            p = Path(src_path)
            if p.exists():
                shutil.copy2(p, dst)
                return True

        arr = audio_obj.get("array")
        if arr is not None:
            sr = int(audio_obj.get("sampling_rate") or 16000)
            _write_wav_from_array(arr, sr, dst)
            return True

        blob = audio_obj.get("bytes")
        if isinstance(blob, (bytes, bytearray)) and len(blob) > 0:
            dst.write_bytes(bytes(blob))
            return True
        return False

    if isinstance(audio_obj, str):
        if audio_obj.startswith("http://") or audio_obj.startswith("https://"):
            urlretrieve(audio_obj, dst)
            return True
        p = Path(audio_obj)
        if p.exists():
            shutil.copy2(p, dst)
            return True

    return False


def build_everyayah_samples(out_dir: Path, n: int, reciter: str) -> List[Dict[str, Any]]:
    verses = DEFAULT_VERSES[: max(0, min(n, len(DEFAULT_VERSES)))]
    samples: List[Dict[str, Any]] = []
    for surah, ayah in verses:
        fid = f"everyayah_{surah:03d}{ayah:03d}"
        fname = f"{fid}.mp3"
        url = everyayah_url(surah, ayah, reciter=reciter)
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
    return samples


def build_retasy_samples(out_dir: Path, n: int, split: str, seed: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []

    from datasets import Audio, load_dataset

    print(f"Loading RetaSy dataset split={split} (streaming)")
    ds = load_dataset("RetaSy/quranic_audio_dataset", split=split, streaming=True)
    try:
        # Keep decode disabled so we can copy path/bytes without requiring torchcodec.
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        # Keep best-effort behavior when schema differs.
        pass
    try:
        ds = ds.shuffle(seed=seed, buffer_size=max(1000, n * 50))
    except Exception:
        pass

    samples: List[Dict[str, Any]] = []
    for row in ds:
        surah = _pick(row, ["surah_id", "surah", "surah_number", "Surah_ID", "surahId"])
        ayah = _pick(row, ["ayah_id", "ayah", "ayah_number", "Aya_ID", "ayahId"])
        audio = _pick(row, ["audio", "Audio", "audio_file", "audio_path", "file", "path", "url"])

        if surah is None or ayah is None or audio is None:
            continue

        try:
            surah_i = int(surah)
            ayah_i = int(ayah)
        except Exception:
            continue

        fid = f"retasy_{len(samples):03d}_{surah_i:03d}{ayah_i:03d}"
        fname = f"{fid}.wav"
        dst = out_dir / fname

        if not dst.exists():
            ok = _save_audio(audio, dst)
            if not ok:
                continue

        samples.append(
            {
                "id": fid,
                "file": fname,
                "expected": {"surah": surah_i, "ayah": ayah_i, "ayah_end": None},
                "source": "retasy",
                "category": "public",
            }
        )

        if len(samples) >= n:
            break

    if len(samples) < n:
        print(f"Warning: requested {n} RetaSy samples but collected {len(samples)}.")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/test_corpus", help="Output corpus directory")
    ap.add_argument("--everyayah", type=int, default=20, help="How many EveryAyah clips to download")
    ap.add_argument("--retasy", type=int, default=0, help="How many RetaSy clips to download")
    ap.add_argument("--retasy-split", default="train", help="RetaSy split name")
    ap.add_argument("--reciter", default="Alafasy_128kbps", help="EveryAyah reciter folder")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed for RetaSy sampling")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples: List[Dict] = []
    samples.extend(build_everyayah_samples(out_dir, args.everyayah, args.reciter))
    samples.extend(build_retasy_samples(out_dir, args.retasy, args.retasy_split, args.seed))

    manifest = {"samples": samples}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote manifest.json with {len(samples)} samples to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
