"""
Experiment: phonetic_ctc_align

CTC forced alignment using a Quran phonetic CTC model.

Default model:
- TBOGamer22/wav2vec2-quran-phonetics

You can override with a local finetuned checkpoint by setting:
- QURAN_CTC_CHECKPOINT=/path/to/checkpoint_dir
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from quran_verse_id.audio import load_audio
from quran_verse_id.ctc_scorer import ctc_forward_logp
from quran_verse_id.matcher import CandidateRetriever
from quran_verse_id.normalizer import normalize_phonetic
from quran_verse_id.phonetic_db import PhoneticQuranDB

DEFAULT_MODEL_ID = "TBOGamer22/wav2vec2-quran-phonetics"

_MODEL = None
_PROCESSOR = None
_DEVICE = None

_DB = None
_RETRIEVER = None


def _get_device() -> torch.device:
    # Prefer CUDA, then Apple Silicon MPS, then CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: Optional[str] = None) -> tuple[Wav2Vec2ForCTC, Wav2Vec2Processor, torch.device]:
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None and _PROCESSOR is not None and _DEVICE is not None:
        return _MODEL, _PROCESSOR, _DEVICE

    _DEVICE = _get_device()

    model_id_or_path = checkpoint or os.environ.get("QURAN_CTC_CHECKPOINT") or DEFAULT_MODEL_ID
    _PROCESSOR = Wav2Vec2Processor.from_pretrained(model_id_or_path)
    _MODEL = Wav2Vec2ForCTC.from_pretrained(model_id_or_path)
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    return _MODEL, _PROCESSOR, _DEVICE


def load_phonetic_db(db_path: Optional[str] = None) -> tuple[PhoneticQuranDB, CandidateRetriever]:
    global _DB, _RETRIEVER
    if _DB is not None and _RETRIEVER is not None:
        return _DB, _RETRIEVER

    p = db_path or os.environ.get("QURAN_PHONETIC_DB")
    if not p:
        raise RuntimeError(
            "Phonetic DB path not provided. Pass --phonetic-db to benchmark.runner or set QURAN_PHONETIC_DB."
        )
    _DB = PhoneticQuranDB.load_json(p)
    _RETRIEVER = CandidateRetriever(_DB)
    return _DB, _RETRIEVER


@torch.inference_mode()
def predict(
    audio_path: str,
    *,
    phonetic_db_path: Optional[str] = None,
    top_k: int = 50,
    max_span_ayahs: int = 3,
) -> Dict:
    """
    Predict (surah, ayah) for an audio file.
    Returns dict compatible with offline-tarteel benchmark schema.
    """
    model, processor, device = load_model()
    _db, retriever = load_phonetic_db(phonetic_db_path)

    wav = load_audio(audio_path, sample_rate=16000, mono=True)
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    logits = model(input_values).logits[0]  # [T, V]
    log_probs = F.log_softmax(logits, dim=-1)

    pred_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(pred_ids.unsqueeze(0))[0]
    transcript_clean = normalize_phonetic(transcript)

    # Candidate retrieval
    starts = retriever.search(transcript_clean, top_k=top_k)
    if not starts:
        return {"surah": None, "ayah": None, "ayah_end": None, "confidence": 0.0, "transcript": transcript_clean}

    candidates = retriever.expand_spans(starts, max_span_ayahs=max_span_ayahs)

    blank_id = processor.tokenizer.pad_token_id
    if blank_id is None:
        # common fallback; wav2vec2 uses pad as blank but some tokenizers use 0
        blank_id = 0

    best = None
    best_score = -1e18

    for cand in candidates:
        # Tokenize the phonetic target
        # NOTE: Wav2Vec2 CTC tokenizers typically operate char-level.
        target_ids = processor.tokenizer(cand.text_clean).input_ids
        # Some tokenizers may add special tokens; strip anything >= vocab size as a guard.
        vocab_size = log_probs.shape[-1]
        target_ids = [tid for tid in target_ids if 0 <= tid < vocab_size and tid != blank_id]
        if not target_ids:
            continue

        res = ctc_forward_logp(log_probs, target_ids, blank_id=blank_id)

        # Length-normalize (prevents bias toward short targets)
        norm = max(1, res.target_len)
        score = res.logp / float(norm)

        if score > best_score:
            best_score = score
            best = cand

    if best is None:
        # fallback to best fuzzy start
        s = starts[0]
        return {"surah": s.surah, "ayah": s.ayah, "ayah_end": None, "confidence": float(s.match_score), "transcript": transcript_clean}

    # Confidence: softmax over top few (cheap heuristic)
    # Here we just squash normalized score to (0,1) using sigmoid-ish mapping.
    confidence = float(1.0 / (1.0 + torch.exp(torch.tensor(-best_score)).item()))

    return {
        "surah": int(best.surah),
        "ayah": int(best.ayah),
        "ayah_end": int(best.ayah_end) if best.ayah_end is not None else None,
        "confidence": confidence,
        "transcript": transcript_clean,
    }


def model_size_bytes() -> int:
    """
    Approximate in-memory model size in bytes (parameters only).
    For true on-device size, export to ONNX + quantize.
    """
    model, _processor, _device = load_model()
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return int(total)
