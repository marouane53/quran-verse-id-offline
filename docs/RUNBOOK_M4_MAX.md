# Quran Verse ID Offline: M4 Max Runbook

This runbook prepares the environment and gives deterministic commands for:
- baseline benchmark
- short MPS finetune
- post-finetune benchmark

It avoids long unattended execution by default.

## 0) Prerequisites

Run from repo root:

```bash
cd /Users/marouane/Documents/code/quran-verse-id-offline
```

Check tools:

```bash
python3.12 --version
ffmpeg -version | head -n 1
git --version
```

## 1) Create `.venv` (Python 3.12) and install deps

```bash
cd /Users/marouane/Documents/code/quran-verse-id-offline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Sanity import check:

```bash
python - <<'PY'
import torch, torchaudio, transformers, datasets, rapidfuzz, onnxruntime, pytest
print("deps-ok")
print("torch", torch.__version__)
print("mps_available", getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())
PY
```

## 2) Get original benchmark corpus (54 samples)

Clone original repo once:

```bash
mkdir -p /Users/marouane/Documents/code/external
cd /Users/marouane/Documents/code/external
if [ ! -d offline-tarteel ]; then
  git clone https://github.com/yazinsai/offline-tarteel
fi
```

Import corpus into this repo:

```bash
cd /Users/marouane/Documents/code/quran-verse-id-offline
bash scripts/import_offline_tarteel_benchmark.sh /Users/marouane/Documents/code/external/offline-tarteel
```

Validate manifest and file presence:

```bash
python - <<'PY'
import json
from pathlib import Path

corpus = Path("benchmark/test_corpus")
manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
samples = manifest.get("samples", [])
missing = [s["file"] for s in samples if not (corpus / s["file"]).exists()]
print("sample_count:", len(samples))
print("missing_files:", len(missing))
if missing:
    print("first_missing:", missing[:10])
PY
```

Fallback only if exact corpus import is incomplete:

```bash
python scripts/build_benchmark_from_sources.py --everyayah 23 --retasy 31 --out benchmark/test_corpus
```

## 3) Build phonetic Quran DB

```bash
cd /Users/marouane/Documents/code/quran-verse-id-offline
source .venv/bin/activate
python scripts/build_phonetic_db.py --out data/quran_phonetic_db.json
```

Quick check:

```bash
python - <<'PY'
import json
data = json.load(open("data/quran_phonetic_db.json"))
print("phonetic_db_entries:", len(data))
print("first_entry:", data[0])
PY
```

## 4) Run unit tests and CLI sanity checks

```bash
cd /Users/marouane/Documents/code/quran-verse-id-offline
source .venv/bin/activate
pytest -q
python -m benchmark.runner --help
python scripts/train_finetune_phonetic_ctc.py --help
python scripts/build_benchmark_from_sources.py --help
```

## 5) Baseline benchmark

```bash
cd /Users/marouane/Documents/code/quran-verse-id-offline
source .venv/bin/activate
python -m benchmark.runner \
  --experiment phonetic_ctc_align \
  --corpus benchmark/test_corpus \
  --phonetic-db data/quran_phonetic_db.json \
  --save
```

## 6) Short finetune on Apple GPU (MPS)

Fast smoke profile:

```bash
cd /Users/marouane/Documents/code/quran-verse-id-offline
source .venv/bin/activate
python scripts/train_finetune_phonetic_ctc.py \
  --output-dir models/phonetic_ctc_finetuned \
  --phonetic-db data/quran_phonetic_db.json \
  --device mps \
  --max-train-samples 12000 \
  --max-val-samples 1000 \
  --max-steps 1000 \
  --per-device-batch 2 \
  --grad-accum 8 \
  --eval-steps 200 \
  --save-steps 200
```

Notes:
- `--fp16` is CUDA-only in this script; do not use it on MPS.
- Reduce `--per-device-batch` if you hit memory pressure.

## 7) Post-finetune benchmark

```bash
cd /Users/marouane/Documents/code/quran-verse-id-offline
source .venv/bin/activate
export QURAN_CTC_CHECKPOINT="$(cat models/phonetic_ctc_finetuned/latest)"
python -m benchmark.runner \
  --experiment phonetic_ctc_align \
  --corpus benchmark/test_corpus \
  --phonetic-db data/quran_phonetic_db.json \
  --save
```

## 8) Optional long run (20k steps)

Use only when you want the full training recipe:

```bash
cd /Users/marouane/Documents/code/quran-verse-id-offline
source .venv/bin/activate
python scripts/train_finetune_phonetic_ctc.py \
  --output-dir models/phonetic_ctc_finetuned_full \
  --phonetic-db data/quran_phonetic_db.json \
  --device mps \
  --max-steps 20000 \
  --per-device-batch 2 \
  --grad-accum 8 \
  --eval-steps 1000 \
  --save-steps 1000
```

## 9) Aggregate results (baseline vs finetuned)

This prints a compact table from the latest two benchmark result JSON files:

```bash
cd /Users/marouane/Documents/code/quran-verse-id-offline
source .venv/bin/activate
python - <<'PY'
import json
from pathlib import Path

res_dir = Path("benchmark/results")
files = sorted(res_dir.glob("phonetic_ctc_align_*.json"))
if len(files) < 2:
    print("Need at least 2 result files in benchmark/results")
    raise SystemExit(1)

baseline, finetuned = files[-2], files[-1]

def load(p):
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "file": p.name,
        "recall": d["recall"] * 100.0,
        "precision": d["precision"] * 100.0,
        "seqacc": d["sequence_accuracy"] * 100.0,
        "latency": d["avg_latency_s"],
        "samples": d["total"],
    }

b = load(baseline)
f = load(finetuned)

print("baseline :", b["file"])
print("finetuned:", f["file"])
print("")
print(f"{'Metric':<12} {'Baseline':>12} {'Finetuned':>12} {'Delta':>12}")
print("-" * 52)
print(f"{'Recall %':<12} {b['recall']:>11.2f} {f['recall']:>11.2f} {f['recall']-b['recall']:>11.2f}")
print(f"{'Precision %':<12} {b['precision']:>11.2f} {f['precision']:>11.2f} {f['precision']-b['precision']:>11.2f}")
print(f"{'SeqAcc %':<12} {b['seqacc']:>11.2f} {f['seqacc']:>11.2f} {f['seqacc']-b['seqacc']:>11.2f}")
print(f"{'Latency s':<12} {b['latency']:>11.3f} {f['latency']:>11.3f} {f['latency']-b['latency']:>11.3f}")
print(f"{'Samples':<12} {b['samples']:>12} {f['samples']:>12} {'-':>12}")
PY
```

## 10) Optional ONNX export + quantization

```bash
cd /Users/marouane/Documents/code/quran-verse-id-offline
source .venv/bin/activate
export QURAN_CTC_CHECKPOINT="$(cat models/phonetic_ctc_finetuned/latest)"
python scripts/export_onnx.py --checkpoint "$QURAN_CTC_CHECKPOINT" --out onnx/phonetic_ctc.onnx
python scripts/quantize_onnx.py --in onnx/phonetic_ctc.onnx --out onnx/phonetic_ctc.int8.onnx
ls -lh onnx/phonetic_ctc.onnx onnx/phonetic_ctc.int8.onnx
```
