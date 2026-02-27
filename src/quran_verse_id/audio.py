"""
Audio utilities.

We decode arbitrary audio formats (mp3, m4a, wav, etc.) to mono 16kHz float32 using ffmpeg.

Why ffmpeg?
- Robust format support
- Deterministic output
- Avoids Python codec edge cases
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np


class AudioDecodeError(RuntimeError):
    pass


def load_audio(
    path: str | Path,
    *,
    sample_rate: int = 16000,
    mono: bool = True,
) -> np.ndarray:
    """
    Decode an audio file to a numpy float32 waveform.

    Args:
        path: input audio file
        sample_rate: target sample rate
        mono: convert to mono

    Returns:
        waveform: shape [num_samples], dtype float32, range roughly [-1, 1]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    ac = "1" if mono else "2"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(p),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        ac,
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise AudioDecodeError(
            f"ffmpeg failed decoding {p} (exit={e.returncode}). Output:\n{e.output.decode('utf-8', errors='replace')}"
        ) from e

    if not out:
        raise AudioDecodeError(f"ffmpeg produced empty output for {p}")

    audio = np.frombuffer(out, dtype=np.float32)
    if audio.size == 0:
        raise AudioDecodeError(f"Decoded audio is empty for {p}")

    # If stereo requested, return shape [num_samples, channels]
    if not mono:
        audio = audio.reshape(-1, 2)

    return audio


def audio_duration_s(num_samples: int, sample_rate: int = 16000) -> float:
    return float(num_samples) / float(sample_rate)
