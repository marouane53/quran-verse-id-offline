"""
Quantize an ONNX model (dynamic INT8 weights).

Example:
  python scripts/quantize_onnx.py --in onnx/model.onnx --out onnx/model.int8.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input ONNX")
    ap.add_argument("--out", required=True, help="Output quantized ONNX")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(inp)

    quantize_dynamic(
        model_input=str(inp),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )

    print(f"Wrote quantized ONNX to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
