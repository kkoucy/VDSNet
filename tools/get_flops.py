#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch
from thop import profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import build_model, load_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Profile VDSNet parameters and computation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    model = build_model(config, training=False).to(args.device).eval()
    sample = torch.randn(1, 3, args.height, args.width, device=args.device)
    macs, executed_parameters = profile(model, inputs=(sample,), verbose=False)
    all_parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f"Model: {config['model']}")
    print(f"Input: 1 x 3 x {args.height} x {args.width}")
    print(f"Parameters (THOP executed graph): {executed_parameters / 1e6:.4f} M")
    print(f"Parameters (all registered):      {all_parameters / 1e6:.4f} M")
    print(f"MACs (THOP):                      {macs / 1e9:.4f} G")
    print(f"FLOPs (1 MAC = 2 FLOPs):         {2 * macs / 1e9:.4f} G")
    print()
    print("Note: the manuscript follows the common THOP convention and reports its")
    print("MAC count in the GFLOPs column. Sorting/indexing and custom selective-scan")
    print("kernels are not fully represented by generic THOP operator hooks.")


if __name__ == "__main__":
    main()

