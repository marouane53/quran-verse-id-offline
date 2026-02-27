#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
REQ_FILE="$ROOT_DIR/requirements.txt"
DEFAULT_PYTHON_BIN="${PYTHON_BIN:-python3.12}"
DEFAULT_OLD_REPO_PATH="${OLD_REPO_PATH:-/Users/marouane/Documents/code/external/offline-tarteel}"
DEFAULT_CORPUS_DIR="$ROOT_DIR/benchmark/test_corpus"
DEFAULT_PHONETIC_DB="$ROOT_DIR/data/quran_phonetic_db.json"
DEFAULT_RESULTS_DIR="$ROOT_DIR/benchmark/results"

# Historical numbers from offline-tarteel report table (CTC alignment).
OLD_RECALL_PCT="${OLD_RECALL_PCT:-83.0}"
OLD_PRECISION_PCT="${OLD_PRECISION_PCT:-83.0}"
OLD_SEQACC_PCT="${OLD_SEQACC_PCT:-81.0}"
OLD_LATENCY_S="${OLD_LATENCY_S:-5.0}"
OLD_SIZE_BYTES="${OLD_SIZE_BYTES:-1288490188}"

banner() {
  cat <<'EOF'
==============================================================
 Quran Verse ID Offline - Interactive Benchmark CLI
==============================================================
EOF
}

pause() {
  echo
  read -r -p "Press Enter to continue..."
}

ensure_venv_exists() {
  if [[ ! -d "$VENV_DIR" ]]; then
    echo ".venv not found at: $VENV_DIR"
    read -r -p "Create + install environment now? [Y/n]: " ans
    ans="${ans:-Y}"
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      setup_env
    else
      echo "Cannot proceed without a virtual environment."
      return 1
    fi
  fi
}

activate_venv() {
  ensure_venv_exists
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
}

setup_env() {
  echo "Setting up environment in $VENV_DIR"
  cd "$ROOT_DIR"
  if [[ ! -d "$VENV_DIR" ]]; then
    "$DEFAULT_PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  python -m pip install -U pip
  pip install -r "$REQ_FILE"
  pip install -e "$ROOT_DIR"
  python --version
  pip --version
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("mps_available", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY
}

import_exact_corpus() {
  local src
  read -r -p "Path to offline-tarteel repo [$DEFAULT_OLD_REPO_PATH]: " src
  src="${src:-$DEFAULT_OLD_REPO_PATH}"
  activate_venv
  cd "$ROOT_DIR"
  bash "$ROOT_DIR/scripts/import_offline_tarteel_benchmark.sh" "$src"
  validate_corpus
}

build_public_corpus() {
  local everyayah retasy reciter split out seed
  read -r -p "EveryAyah sample count [23]: " everyayah
  everyayah="${everyayah:-23}"
  read -r -p "RetaSy sample count [31]: " retasy
  retasy="${retasy:-31}"
  read -r -p "RetaSy split [train]: " split
  split="${split:-train}"
  read -r -p "EveryAyah reciter [Alafasy_128kbps]: " reciter
  reciter="${reciter:-Alafasy_128kbps}"
  read -r -p "Seed [42]: " seed
  seed="${seed:-42}"
  read -r -p "Output corpus dir [$DEFAULT_CORPUS_DIR]: " out
  out="${out:-$DEFAULT_CORPUS_DIR}"

  activate_venv
  cd "$ROOT_DIR"
  python scripts/build_benchmark_from_sources.py \
    --everyayah "$everyayah" \
    --retasy "$retasy" \
    --retasy-split "$split" \
    --reciter "$reciter" \
    --seed "$seed" \
    --out "$out"
  validate_corpus "$out"
}

validate_corpus() {
  local corpus
  corpus="${1:-$DEFAULT_CORPUS_DIR}"
  activate_venv
  cd "$ROOT_DIR"
  python - "$corpus" <<'PY'
import json
import sys
from pathlib import Path

corpus = Path(sys.argv[1])
manifest = corpus / "manifest.json"
if not manifest.exists():
    print(f"ERROR: missing manifest: {manifest}")
    raise SystemExit(1)

obj = json.loads(manifest.read_text(encoding="utf-8"))
samples = obj.get("samples", [])
missing = [s.get("file") for s in samples if not (corpus / s.get("file", "")).exists()]

source_counts = {}
for s in samples:
    src = s.get("source", "?")
    source_counts[src] = source_counts.get(src, 0) + 1

print(f"corpus_dir: {corpus}")
print(f"sample_count: {len(samples)}")
print(f"source_counts: {source_counts}")
print(f"missing_files: {len(missing)}")
if missing:
    print("first_missing:", missing[:20])
PY
}

build_phonetic_db() {
  local out
  read -r -p "Phonetic DB output path [$DEFAULT_PHONETIC_DB]: " out
  out="${out:-$DEFAULT_PHONETIC_DB}"
  activate_venv
  cd "$ROOT_DIR"
  python scripts/build_phonetic_db.py --out "$out"
  python - "$out" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print("ERROR: phonetic DB file not found")
    raise SystemExit(1)
data = json.loads(p.read_text(encoding="utf-8"))
print("phonetic_db_entries:", len(data))
print("first_entry:", data[0] if data else None)
PY
}

run_tests() {
  activate_venv
  cd "$ROOT_DIR"
  pytest -q
}

list_experiments() {
  local exp
  local i=1
  local exp_dir="$ROOT_DIR/experiments"
  local -a exps=()
  while IFS= read -r -d '' exp; do
    exps+=("$(basename "$exp")")
  done < <(find "$exp_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

  if [[ "${#exps[@]}" -eq 0 ]]; then
    echo "No experiment directories found under $exp_dir"
    return 1
  fi

  echo "Available experiments:"
  for exp in "${exps[@]}"; do
    if [[ -f "$exp_dir/$exp/run.py" ]]; then
      echo "  $i) $exp"
      i=$((i + 1))
    fi
  done
}

choose_experiment() {
  local exp_dir="$ROOT_DIR/experiments"
  local -a runnable=()
  local exp
  while IFS= read -r -d '' exp; do
    if [[ -f "$exp/run.py" ]]; then
      runnable+=("$(basename "$exp")")
    fi
  done < <(find "$exp_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

  if [[ "${#runnable[@]}" -eq 0 ]]; then
    echo "ERROR: no runnable experiments found."
    return 1
  fi

  echo "Choose experiment:" >&2
  local i
  for i in "${!runnable[@]}"; do
    printf "  %d) %s\n" "$((i + 1))" "${runnable[$i]}" >&2
  done
  read -r -p "Selection [1]: " sel >&2
  sel="${sel:-1}"
  if ! [[ "$sel" =~ ^[0-9]+$ ]]; then
    echo "Invalid selection." >&2
    return 1
  fi
  if (( sel < 1 || sel > ${#runnable[@]} )); then
    echo "Selection out of range." >&2
    return 1
  fi
  echo "${runnable[$((sel - 1))]}"
}

run_benchmark() {
  local exp corpus db ckpt fast use_fast top_k max_span fast_flag
  exp="$(choose_experiment)" || return 1
  read -r -p "Corpus dir [$DEFAULT_CORPUS_DIR]: " corpus
  corpus="${corpus:-$DEFAULT_CORPUS_DIR}"
  read -r -p "Phonetic DB path [$DEFAULT_PHONETIC_DB]: " db
  db="${db:-$DEFAULT_PHONETIC_DB}"
  read -r -p "Checkpoint override (empty = default): " ckpt
  ckpt="${ckpt:-}"
  read -r -p "Fast mode? [Y/n]: " fast
  fast="${fast:-Y}"
  use_fast=0
  if [[ "$fast" =~ ^[Yy]$ ]]; then
    use_fast=1
  fi
  if [[ "$use_fast" -eq 1 ]]; then
    read -r -p "top_k [20]: " top_k
    top_k="${top_k:-20}"
    read -r -p "max_span_ayahs [2]: " max_span
    max_span="${max_span:-2}"
    fast_flag="--fast"
  else
    read -r -p "top_k [50]: " top_k
    top_k="${top_k:-50}"
    read -r -p "max_span_ayahs [3]: " max_span
    max_span="${max_span:-3}"
    fast_flag=""
  fi

  activate_venv
  cd "$ROOT_DIR"
  if [[ -n "$ckpt" ]]; then
    QURAN_CTC_CHECKPOINT="$ckpt" python -m benchmark.runner \
      --experiment "$exp" \
      --corpus "$corpus" \
      --phonetic-db "$db" \
      $fast_flag \
      --top-k "$top_k" \
      --max-span-ayahs "$max_span" \
      --save
  else
    python -m benchmark.runner \
      --experiment "$exp" \
      --corpus "$corpus" \
      --phonetic-db "$db" \
      $fast_flag \
      --top-k "$top_k" \
      --max-span-ayahs "$max_span" \
      --save
  fi
}

run_short_finetune() {
  local out db device
  read -r -p "Output dir [$ROOT_DIR/models/phonetic_ctc_finetuned]: " out
  out="${out:-$ROOT_DIR/models/phonetic_ctc_finetuned}"
  read -r -p "Phonetic DB path [$DEFAULT_PHONETIC_DB]: " db
  db="${db:-$DEFAULT_PHONETIC_DB}"
  read -r -p "Device [mps]: " device
  device="${device:-mps}"

  activate_venv
  cd "$ROOT_DIR"
  python scripts/train_finetune_phonetic_ctc.py \
    --output-dir "$out" \
    --phonetic-db "$db" \
    --device "$device" \
    --max-train-samples 12000 \
    --max-val-samples 1000 \
    --max-steps 1000 \
    --per-device-batch 2 \
    --grad-accum 8 \
    --eval-steps 200 \
    --save-steps 200
}

run_full_finetune_20k() {
  local out db device confirm
  read -r -p "Output dir [$ROOT_DIR/models/phonetic_ctc_finetuned_full]: " out
  out="${out:-$ROOT_DIR/models/phonetic_ctc_finetuned_full}"
  read -r -p "Phonetic DB path [$DEFAULT_PHONETIC_DB]: " db
  db="${db:-$DEFAULT_PHONETIC_DB}"
  read -r -p "Device [mps]: " device
  device="${device:-mps}"
  read -r -p "This is a long run. Continue? [y/N]: " confirm
  confirm="${confirm:-N}"
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    return 0
  fi

  activate_venv
  cd "$ROOT_DIR"
  python scripts/train_finetune_phonetic_ctc.py \
    --output-dir "$out" \
    --phonetic-db "$db" \
    --device "$device" \
    --max-steps 20000 \
    --per-device-batch 2 \
    --grad-accum 8 \
    --eval-steps 1000 \
    --save-steps 1000
}

benchmark_with_latest_checkpoint() {
  local exp corpus db latest_file ckpt_dir fast use_fast top_k max_span fast_flag
  exp="$(choose_experiment)" || return 1
  read -r -p "Corpus dir [$DEFAULT_CORPUS_DIR]: " corpus
  corpus="${corpus:-$DEFAULT_CORPUS_DIR}"
  read -r -p "Phonetic DB path [$DEFAULT_PHONETIC_DB]: " db
  db="${db:-$DEFAULT_PHONETIC_DB}"
  read -r -p "Path to latest checkpoint pointer file [$ROOT_DIR/models/phonetic_ctc_finetuned/latest]: " latest_file
  latest_file="${latest_file:-$ROOT_DIR/models/phonetic_ctc_finetuned/latest}"

  if [[ ! -f "$latest_file" ]]; then
    echo "Missing latest pointer file: $latest_file"
    return 1
  fi
  ckpt_dir="$(<"$latest_file")"
  if [[ ! -d "$ckpt_dir" ]]; then
    echo "Checkpoint directory from latest file does not exist: $ckpt_dir"
    return 1
  fi
  read -r -p "Fast mode? [Y/n]: " fast
  fast="${fast:-Y}"
  use_fast=0
  if [[ "$fast" =~ ^[Yy]$ ]]; then
    use_fast=1
  fi
  if [[ "$use_fast" -eq 1 ]]; then
    read -r -p "top_k [20]: " top_k
    top_k="${top_k:-20}"
    read -r -p "max_span_ayahs [2]: " max_span
    max_span="${max_span:-2}"
    fast_flag="--fast"
  else
    read -r -p "top_k [50]: " top_k
    top_k="${top_k:-50}"
    read -r -p "max_span_ayahs [3]: " max_span
    max_span="${max_span:-3}"
    fast_flag=""
  fi

  activate_venv
  cd "$ROOT_DIR"
  QURAN_CTC_CHECKPOINT="$ckpt_dir" python -m benchmark.runner \
    --experiment "$exp" \
    --corpus "$corpus" \
    --phonetic-db "$db" \
    $fast_flag \
    --top-k "$top_k" \
    --max-span-ayahs "$max_span" \
    --save
}

show_latest_result() {
  activate_venv
  cd "$ROOT_DIR"
  python - "$DEFAULT_RESULTS_DIR" <<'PY'
import json
import sys
from pathlib import Path

res_dir = Path(sys.argv[1])
files = sorted(res_dir.glob("*.json"))
if not files:
    print(f"No result files found in: {res_dir}")
    raise SystemExit(1)

latest = files[-1]
d = json.loads(latest.read_text(encoding="utf-8"))
print("latest_file:", latest.name)
print("experiment:", d.get("experiment"))
print("samples:", d.get("total"))
print("recall_pct:", round(float(d.get("recall", 0.0)) * 100.0, 3))
print("precision_pct:", round(float(d.get("precision", 0.0)) * 100.0, 3))
print("seqacc_pct:", round(float(d.get("sequence_accuracy", 0.0)) * 100.0, 3))
print("avg_latency_s:", round(float(d.get("avg_latency_s", 0.0)), 4))
print("model_size_bytes:", d.get("model_size_bytes"))
cfg = d.get("runner_config")
if cfg:
    print("runner_config:", cfg)
PY
}

compare_latest_to_old() {
  activate_venv
  cd "$ROOT_DIR"
  python - \
    "$DEFAULT_RESULTS_DIR" \
    "$OLD_RECALL_PCT" \
    "$OLD_PRECISION_PCT" \
    "$OLD_SEQACC_PCT" \
    "$OLD_LATENCY_S" \
    "$OLD_SIZE_BYTES" <<'PY'
import json
import sys
from pathlib import Path

res_dir = Path(sys.argv[1])
old_recall = float(sys.argv[2])
old_precision = float(sys.argv[3])
old_seqacc = float(sys.argv[4])
old_latency = float(sys.argv[5])
old_size = float(sys.argv[6])

files = sorted(res_dir.glob("*.json"))
if not files:
    print(f"No result files found in: {res_dir}")
    raise SystemExit(1)
latest = files[-1]
d = json.loads(latest.read_text(encoding="utf-8"))

cur = {
    "recall_pct": float(d.get("recall", 0.0)) * 100.0,
    "precision_pct": float(d.get("precision", 0.0)) * 100.0,
    "seqacc_pct": float(d.get("sequence_accuracy", 0.0)) * 100.0,
    "latency_s": float(d.get("avg_latency_s", 0.0)),
    "size_bytes": float(d.get("model_size_bytes") or 0.0),
}
old = {
    "recall_pct": old_recall,
    "precision_pct": old_precision,
    "seqacc_pct": old_seqacc,
    "latency_s": old_latency,
    "size_bytes": old_size,
}

print("latest_file:", latest.name)
print("comparison_reference: offline-tarteel ctc-alignment table numbers")
print("")
print(f"{'Metric':<14} {'Current':>12} {'Old':>12} {'Delta':>12}")
print("-" * 54)
for k, label in [
    ("recall_pct", "Recall (%)"),
    ("precision_pct", "Precision (%)"),
    ("seqacc_pct", "SeqAcc (%)"),
    ("latency_s", "Latency (s)"),
    ("size_bytes", "Size (bytes)"),
]:
    c = cur[k]
    o = old[k]
    delta = c - o
    print(f"{label:<14} {c:>12.3f} {o:>12.3f} {delta:>12.3f}")
PY
}

compare_two_results() {
  local first second
  read -r -p "First result json path: " first
  read -r -p "Second result json path: " second
  if [[ ! -f "$first" || ! -f "$second" ]]; then
    echo "Both result files must exist."
    return 1
  fi
  activate_venv
  cd "$ROOT_DIR"
  python - "$first" "$second" <<'PY'
import json
import sys
from pathlib import Path

f1 = Path(sys.argv[1])
f2 = Path(sys.argv[2])
d1 = json.loads(f1.read_text(encoding="utf-8"))
d2 = json.loads(f2.read_text(encoding="utf-8"))

def row(d):
    return {
        "recall_pct": float(d.get("recall", 0.0)) * 100.0,
        "precision_pct": float(d.get("precision", 0.0)) * 100.0,
        "seqacc_pct": float(d.get("sequence_accuracy", 0.0)) * 100.0,
        "latency_s": float(d.get("avg_latency_s", 0.0)),
        "size_bytes": float(d.get("model_size_bytes") or 0.0),
    }

a = row(d1)
b = row(d2)
print("first_file:", f1)
print("second_file:", f2)
print("")
print(f"{'Metric':<14} {'First':>12} {'Second':>12} {'Delta':>12}")
print("-" * 54)
for k, label in [
    ("recall_pct", "Recall (%)"),
    ("precision_pct", "Precision (%)"),
    ("seqacc_pct", "SeqAcc (%)"),
    ("latency_s", "Latency (s)"),
    ("size_bytes", "Size (bytes)"),
]:
    x = a[k]
    y = b[k]
    print(f"{label:<14} {x:>12.3f} {y:>12.3f} {y-x:>12.3f}")
PY
}

export_onnx() {
  local ckpt out
  read -r -p "Checkpoint dir or HF model id: " ckpt
  if [[ -z "$ckpt" ]]; then
    echo "Checkpoint is required."
    return 1
  fi
  read -r -p "Output ONNX path [$ROOT_DIR/onnx/phonetic_ctc.onnx]: " out
  out="${out:-$ROOT_DIR/onnx/phonetic_ctc.onnx}"
  activate_venv
  cd "$ROOT_DIR"
  python scripts/export_onnx.py --checkpoint "$ckpt" --out "$out"
}

quantize_onnx() {
  local inp out
  read -r -p "Input ONNX path [$ROOT_DIR/onnx/phonetic_ctc.onnx]: " inp
  inp="${inp:-$ROOT_DIR/onnx/phonetic_ctc.onnx}"
  read -r -p "Output INT8 ONNX path [$ROOT_DIR/onnx/phonetic_ctc.int8.onnx]: " out
  out="${out:-$ROOT_DIR/onnx/phonetic_ctc.int8.onnx}"
  activate_venv
  cd "$ROOT_DIR"
  python scripts/quantize_onnx.py --in "$inp" --out "$out"
  ls -lh "$inp" "$out"
}

print_menu() {
  echo
  echo "Choose an action:"
  echo "  1) Setup/repair .venv + install deps"
  echo "  2) Import exact corpus from offline-tarteel"
  echo "  3) Build public corpus (EveryAyah + RetaSy)"
  echo "  4) Validate corpus files"
  echo "  5) Build phonetic DB"
  echo "  6) Run tests"
  echo "  7) Run benchmark (choose experiment)"
  echo "  8) Run short finetune (1k steps)"
  echo "  9) Run full finetune (20k steps)"
  echo " 10) Benchmark with latest finetuned checkpoint"
  echo " 11) Show latest benchmark result"
  echo " 12) Compare latest result vs old offline-tarteel numbers"
  echo " 13) Compare any two result JSON files"
  echo " 14) Export ONNX"
  echo " 15) Quantize ONNX (INT8)"
  echo "  0) Exit"
}

run_action() {
  local label="$1"
  shift
  ( "$@" )
  local code=$?
  if [[ "$code" -eq 0 ]]; then
    echo "Action completed: $label"
    return 0
  fi
  echo "Action failed [$code]: $label"
  return 0
}

main() {
  banner
  echo "Project root: $ROOT_DIR"
  echo "Default old repo path: $DEFAULT_OLD_REPO_PATH"
  echo "Default corpus dir: $DEFAULT_CORPUS_DIR"
  echo "Default phonetic DB: $DEFAULT_PHONETIC_DB"
  echo "Default results dir: $DEFAULT_RESULTS_DIR"

  while true; do
    print_menu
    read -r -p "Selection: " choice
    case "${choice:-}" in
      1) run_action "setup_env" setup_env ;;
      2) run_action "import_exact_corpus" import_exact_corpus ;;
      3) run_action "build_public_corpus" build_public_corpus ;;
      4)
        cdir=""
        read -r -p "Corpus dir [$DEFAULT_CORPUS_DIR]: " cdir
        cdir="${cdir:-$DEFAULT_CORPUS_DIR}"
        run_action "validate_corpus" validate_corpus "$cdir"
        ;;
      5) run_action "build_phonetic_db" build_phonetic_db ;;
      6) run_action "run_tests" run_tests ;;
      7) run_action "run_benchmark" run_benchmark ;;
      8) run_action "run_short_finetune" run_short_finetune ;;
      9) run_action "run_full_finetune_20k" run_full_finetune_20k ;;
      10) run_action "benchmark_with_latest_checkpoint" benchmark_with_latest_checkpoint ;;
      11) run_action "show_latest_result" show_latest_result ;;
      12) run_action "compare_latest_to_old" compare_latest_to_old ;;
      13) run_action "compare_two_results" compare_two_results ;;
      14) run_action "export_onnx" export_onnx ;;
      15) run_action "quantize_onnx" quantize_onnx ;;
      0)
        echo "Bye."
        exit 0
        ;;
      *)
        echo "Invalid selection."
        ;;
    esac
    pause
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
