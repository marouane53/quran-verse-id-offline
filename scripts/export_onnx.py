"""
Export a (finetuned) Wav2Vec2ForCTC checkpoint to ONNX.

Example:
  python scripts/export_onnx.py --checkpoint models/phonetic_ctc_finetuned/checkpoint-20000 --out onnx/model.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Local checkpoint dir or HF model id")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
    model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint)
    model.eval()

    # Dummy audio: 2 seconds
    dummy = torch.randn(1, 32000, dtype=torch.float32)
    attn = torch.ones_like(dummy, dtype=torch.long)

    dynamic_axes = {
        "input_values": {0: "batch", 1: "samples"},
        "attention_mask": {0: "batch", 1: "samples"},
        "logits": {0: "batch", 1: "frames"},
    }

    torch.onnx.export(
        model,
        (dummy, attn),
        f=str(out_path),
        input_names=["input_values", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
    )

    print(f"Wrote ONNX to {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
