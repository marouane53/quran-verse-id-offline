# Quran Offline Verse ID (Phonetic CTC)

Offline Quran verse recognition (identify the recited **ayah**) with **no internet**.

This repo is a **new, clean** implementation that targets your constraints:
- **95%+ recall goal**
- **<200MB on-device**
- **<1s latency** (after ONNX + INT8 quantization)

## Why this repo should succeed (vs the failed shrink attempts)

Your best result so far was **CTC forced-alignment** (high accuracy) but using a **1.2GB wav2vec2-large** model.
Shrinking failed because the student backbone was English-only.

This repo fixes that by switching to an **Arabic/Quran-aware CTC backbone**:

- **Default backbone:** `TBOGamer22/wav2vec2-quran-phonetics` (Hugging Face)
  - wav2vec2-base sized (**~94M params**) trained on Quranic phonetic targets.
  - outputs a **phonetic transcript**, which is much closer to the signal than Arabic orthography.
- **Training data:** `Buraaq/quran-md-ayahs` (**~187K ayah samples, 30 reciters**) for verse-level finetuning.
- **Text targets:** built from `Buraaq/quran-md-words` (**word-level phonetic transliteration**) to construct a **phonetic Quran DB** (one canonical phonetic string per ayah).

Then we do the same winning approach:
1. Run CTC model once → frame logits
2. Retrieve top-K candidate ayat by fuzzy match on the **phonetic transcript**
3. Re-score candidates via **CTC forced alignment** (forward algorithm / `ctc_loss`)
4. Return `(surah, ayah)` (and optionally `ayah_end` for multi-ayah spans)

Finally, we export to **ONNX** and run **INT8 quantized** inference for the phone.

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

You need `ffmpeg` installed (already present on most dev machines). Check:

```bash
ffmpeg -version
```

### 2) Get benchmark corpus

**Option A (recommended): import the exact corpus from your original repo**:

```bash
bash scripts/import_offline_tarteel_benchmark.sh /path/to/offline-tarteel
```

This copies:
- `benchmark/test_corpus/*.wav|*.mp3|*.m4a`
- `benchmark/test_corpus/manifest.json`

**Option B: build a fresh corpus from public sources** (EveryAyah + RetaSy):

```bash
python scripts/build_benchmark_from_sources.py --everyayah 20 --retasy 30
```

### 3) Build the phonetic Quran DB (required)

This creates `data/quran_phonetic_db.json` (one entry per ayah).

```bash
python scripts/build_phonetic_db.py --out data/quran_phonetic_db.json
```

### 4) Run the benchmark

```bash
python -m benchmark.runner   --experiment phonetic_ctc_align   --corpus benchmark/test_corpus   --phonetic-db data/quran_phonetic_db.json
```

---

## Finetune on Quran-MD (recommended)

This adapts the phonetic CTC model to **verse-length** audio and multi-reciter variability.

```bash
python scripts/train_finetune_phonetic_ctc.py   --output-dir models/phonetic_ctc_finetuned   --phonetic-db data/quran_phonetic_db.json   --max-steps 20000   --per-device-batch 8   --grad-accum 4
```

---

## Export to ONNX + INT8 quantize (for mobile)

```bash
python scripts/export_onnx.py   --checkpoint models/phonetic_ctc_finetuned   --out onnx/phonetic_ctc.onnx

python scripts/quantize_onnx.py   --in onnx/phonetic_ctc.onnx   --out onnx/phonetic_ctc.int8.onnx
```

---

## Repo layout

- `src/quran_verse_id/` core library (audio IO, phonetic DB, candidate retrieval, CTC scoring)
- `experiments/phonetic_ctc_align/` the inference experiment used by the benchmark
- `benchmark/runner.py` evaluates an experiment against a test corpus
- `scripts/` data + model utility scripts
- `tests/` unit tests

---

## Notes on targets

This repo uses **phonetic targets** (from Quran-MD word transliteration) because:
- Arabic orthography loses vowels (and Quran recitation is vowel-rich).
- Phonetic CTC alignment is materially easier and more stable.

If you want to run an orthographic Arabic pipeline too, you can add a second experiment later.
