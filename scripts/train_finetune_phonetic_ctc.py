"""
Finetune the phonetic CTC model on ayah-level Quran recitation.

Default:
- base model: TBOGamer22/wav2vec2-quran-phonetics
- dataset:   Buraaq/quran-md-ayahs  (ayah-level audio across ~30 reciters)

Targets:
- built from Buraaq/quran-md-words (word_tr) using scripts/build_phonetic_db.py
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import Audio, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, get_linear_schedule_with_warmup

from quran_verse_id.normalizer import normalize_phonetic


def load_phonetic_map(path: str) -> Dict[Tuple[int, int], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[Tuple[int, int], str] = {}
    for row in data:
        out[(int(row["surah"]), int(row["ayah"]))] = normalize_phonetic(row["phonetic"])
    if len(out) < 6000:
        raise ValueError(f"Phonetic DB seems too small ({len(out)}). Did build_phonetic_db succeed?")
    return out


class AyahDataset(Dataset):
    def __init__(self, hf_ds, phonetic_map: Dict[Tuple[int, int], str]):
        self.ds = hf_ds
        self.ph_map = phonetic_map

        sample = self.ds[0]
        self.surah_key = next((k for k in ["surah_id", "surah", "surah_number", "Surah_ID"] if k in sample), None)
        self.ayah_key = next((k for k in ["ayah_id", "ayah", "ayah_number", "Aya_ID"] if k in sample), None)
        if self.surah_key is None or self.ayah_key is None:
            raise KeyError(f"Could not find surah/ayah keys in dataset columns: {sorted(sample.keys())}")
        if "audio" not in sample:
            raise KeyError(f"Dataset does not contain 'audio' column. Columns: {sorted(sample.keys())}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        surah = int(row[self.surah_key])
        ayah = int(row[self.ayah_key])
        target = self.ph_map.get((surah, ayah), "")
        audio = row["audio"]
        wav = audio["array"]
        sr = int(audio["sampling_rate"])
        return {"wav": wav, "sr": sr, "target": target, "surah": surah, "ayah": ayah}


@dataclass
class Batch:
    input_values: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class DataCollatorCTC:
    def __init__(self, processor: Wav2Vec2Processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Batch:
        wavs = [f["wav"] for f in features]
        srs = [int(f["sr"]) for f in features]
        if len(set(srs)) != 1:
            raise ValueError(f"Mismatched sampling rates in batch: {set(srs)}")
        sr = srs[0]

        inputs = self.processor(wavs, sampling_rate=sr, return_tensors="pt", padding=True)

        with self.processor.as_target_processor():
            labels = self.processor([f["target"] for f in features], return_tensors="pt", padding=True).input_ids

        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        return Batch(
            input_values=inputs.input_values,
            attention_mask=inputs.attention_mask,
            labels=labels,
        )


def save_checkpoint(out_dir: Path, model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, step: int):
    ckpt_dir = out_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    (out_dir / "latest").write_text(str(ckpt_dir), encoding="utf-8")


@torch.inference_mode()
def evaluate(model: Wav2Vec2ForCTC, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    for batch in loader:
        input_values = batch.input_values.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)
        out = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
        loss = float(out.loss.item())
        total_loss += loss * input_values.size(0)
        total += input_values.size(0)
    model.train()
    return total_loss / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--phonetic-db", required=True, help="Path to data/quran_phonetic_db.json")
    ap.add_argument("--model", default="TBOGamer22/wav2vec2-quran-phonetics")
    ap.add_argument("--dataset", default="Buraaq/quran-md-ayahs")
    ap.add_argument("--split", default="train")
    ap.add_argument("--val-split", default=None, help="Optional split name for validation (else we create a split)")
    ap.add_argument("--val-size", type=float, default=0.01, help="If val-split not provided, fraction for validation")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max-steps", type=int, default=20000, help="Number of optimizer updates (not microbatches)")
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    phon_map = load_phonetic_map(args.phonetic_db)

    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model)
    model.to(device)
    model.train()

    print(f"Loading dataset: {args.dataset} ({args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    if args.val_split:
        val_ds = load_dataset(args.dataset, split=args.val_split).cast_column("audio", Audio(sampling_rate=16000))
        train_ds = ds
    else:
        ds = ds.shuffle(seed=args.seed)
        val_n = max(1, int(len(ds) * args.val_size))
        val_ds = ds.select(range(val_n))
        train_ds = ds.select(range(val_n, len(ds)))

    train = AyahDataset(train_ds, phon_map)
    val = AyahDataset(val_ds, phon_map)

    collator = DataCollatorCTC(processor)
    train_loader = DataLoader(
        train,
        batch_size=args.per_device_batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val,
        batch_size=args.per_device_batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and torch.cuda.is_available()))

    update_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=args.max_steps, desc="updates")
    data_iter = iter(train_loader)

    while update_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        micro_step += 1

        input_values = batch.input_values.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)

        with torch.cuda.amp.autocast(enabled=(args.fp16 and torch.cuda.is_available())):
            out = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
            loss = out.loss / float(args.grad_accum)

        scaler.scale(loss).backward()

        if micro_step % args.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_step += 1

            pbar.update(1)
            pbar.set_postfix({"loss": float(out.loss.item()), "lr": scheduler.get_last_lr()[0]})

            if update_step % args.eval_steps == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"\nstep={update_step} val_loss={val_loss:.4f}")

            if update_step % args.save_steps == 0:
                save_checkpoint(out_dir, model, processor, update_step)

    save_checkpoint(out_dir, model, processor, update_step)
    print(f"Training done. Latest checkpoint: {(out_dir / 'latest').read_text().strip()}")


if __name__ == "__main__":
    raise SystemExit(main())
