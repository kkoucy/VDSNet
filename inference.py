#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from data import IMAGE_EXTENSIONS
from utils import build_model, load_config, load_model_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Enhance underwater images with VDSNet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True, help="Image or image directory")
    parser.add_argument("--output", required=True, help="Output image or directory")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def image_paths(path):
    path = Path(path)
    if path.is_file():
        return [path], path.parent
    if not path.is_dir():
        raise FileNotFoundError(path)
    paths = sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No images found in {path}")
    return paths, path


def load_image(path, device):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)


def save_image(tensor, path):
    array = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def main():
    args = parse_args()
    device = torch.device(args.device)
    config = load_config(args.config)
    model = build_model(config, training=False).to(device)
    _, incompatible = load_model_weights(model, args.weights, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    model.eval()

    paths, input_root = image_paths(args.input)
    output = Path(args.output)
    output_is_file = len(paths) == 1 and output.suffix.lower() in IMAGE_EXTENSIONS
    divisor = 16 if config["model"] == "VDSNetS" else 8
    with torch.inference_mode():
        for path in tqdm(paths, unit="image"):
            tensor = load_image(path, device)
            height, width = tensor.shape[-2:]
            pad_h = (divisor - height % divisor) % divisor
            pad_w = (divisor - width % divisor) % divisor
            if pad_h or pad_w:
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
            prediction = model(tensor)[..., :height, :width]
            destination = output if output_is_file else output / path.relative_to(input_root)
            save_image(prediction, destination)


if __name__ == "__main__":
    main()

