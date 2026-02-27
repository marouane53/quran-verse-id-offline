"""
Benchmark runner.

Runs an experiment against a corpus described by `manifest.json`.

Manifest format (compatible with offline-tarteel):

{
  "samples": [
    {
      "id": "retasy_000",
      "file": "retasy_000.wav",
      "expected": {"surah": 1, "ayah": 1, "ayah_end": null},
      "category": "short",
      "source": "retasy"
    },
    ...
  ]
}
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def score_sequence(expected: List[Dict], predicted: List[Dict]) -> Dict[str, float]:
    """
    Score predicted verse sequence vs expected.

    - recall: |intersection| / |expected|
    - precision: |intersection| / |predicted|
    - sequence_accuracy: 1 if exact ordered match, else 0
    """
    exp = [(v["surah"], v["ayah"]) for v in expected]
    pred = [(v["surah"], v["ayah"]) for v in predicted]

    exp_set = set(exp)
    pred_set = set(pred)
    correct = len(exp_set & pred_set)

    recall = correct / len(exp) if exp else 0.0
    precision = correct / len(pred) if pred else 0.0
    seq_acc = 1.0 if exp == pred and exp else 0.0
    return {"recall": recall, "precision": precision, "sequence_accuracy": seq_acc}


def _expand_span(surah: int, ayah: int, ayah_end: Optional[int]) -> List[Dict]:
    if surah is None or ayah is None:
        return []
    if ayah_end is None or ayah_end == ayah:
        return [{"surah": int(surah), "ayah": int(ayah)}]
    if ayah_end < ayah:
        ayah, ayah_end = ayah_end, ayah
    return [{"surah": int(surah), "ayah": int(a)} for a in range(int(ayah), int(ayah_end) + 1)]


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.1f} GB"
    return f"{size_bytes / (1024**2):.0f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, help="Experiment directory name under ./experiments/")
    ap.add_argument("--corpus", required=True, help="Path to corpus directory (contains manifest.json)")
    ap.add_argument("--phonetic-db", default=None, help="Path to phonetic Quran DB json (required for phonetic experiments)")
    ap.add_argument("--save", action="store_true", help="Save results JSON to benchmark/results/")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    corpus_dir = Path(args.corpus)
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])

    exp_mod = importlib.import_module(f"experiments.{args.experiment}.run")

    # model size (best-effort)
    size = None
    if hasattr(exp_mod, "model_size_bytes"):
        try:
            size = int(exp_mod.model_size_bytes())
        except Exception:
            size = None

    total_recall = 0.0
    total_precision = 0.0
    total_seqacc = 0.0
    latencies: List[float] = []
    per_sample: List[Dict] = []

    for s in samples:
        audio_path = str(corpus_dir / s["file"])

        if "expected_verses" in s:
            expected_seq = s["expected_verses"]
        elif "expected" in s:
            e = s["expected"]
            expected_seq = _expand_span(e.get("surah"), e.get("ayah"), e.get("ayah_end"))
        else:
            expected_seq = _expand_span(s.get("surah"), s.get("ayah"), s.get("ayah_end"))

        start = time.perf_counter()
        try:
            pred = exp_mod.predict(audio_path, phonetic_db_path=args.phonetic_db)
        except TypeError:
            # experiment signature doesn't accept phonetic_db_path
            pred = exp_mod.predict(audio_path)
        elapsed = time.perf_counter() - start

        predicted_seq = _expand_span(pred.get("surah"), pred.get("ayah"), pred.get("ayah_end"))

        scores = score_sequence(expected_seq, predicted_seq)
        total_recall += scores["recall"]
        total_precision += scores["precision"]
        total_seqacc += scores["sequence_accuracy"]
        latencies.append(elapsed)

        per_sample.append(
            {
                "id": s.get("id"),
                "file": s.get("file"),
                "expected": expected_seq,
                "predicted": predicted_seq,
                "raw_pred": pred,
                "recall": scores["recall"],
                "precision": scores["precision"],
                "sequence_accuracy": scores["sequence_accuracy"],
                "latency_s": elapsed,
            }
        )

    n = len(samples)
    avg_latency = sum(latencies) / n if n else 0.0
    results = {
        "experiment": args.experiment,
        "recall": total_recall / n if n else 0.0,
        "precision": total_precision / n if n else 0.0,
        "sequence_accuracy": total_seqacc / n if n else 0.0,
        "total": n,
        "avg_latency_s": avg_latency,
        "model_size_bytes": size,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "per_sample": per_sample,
    }

    print()
    print(f"Experiment: {args.experiment}")
    print(f"Corpus:     {corpus_dir}")
    if size is not None:
        print(f"Model size: {_format_size(size)} (params only, rough)")
    print()
    print(f"{'Metric':<18} {'Value':>12}")
    print("-" * 32)
    print(f"{'Recall':<18} {results['recall']*100:>11.1f}%")
    print(f"{'Precision':<18} {results['precision']*100:>11.1f}%")
    print(f"{'SeqAcc':<18} {results['sequence_accuracy']*100:>11.1f}%")
    print(f"{'Avg latency':<18} {results['avg_latency_s']:>11.3f}s")
    print(f"{'Samples':<18} {results['total']:>12d}")
    print()

    if args.save:
        out_dir = root / "benchmark" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_path = out_dir / f"{args.experiment}_{ts}.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
