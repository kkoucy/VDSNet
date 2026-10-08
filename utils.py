from pathlib import Path
import random

import numpy as np
import torch
import yaml

from models import VDSNet, VDSNetS


MODEL_CLASSES = {"VDSNet": VDSNet, "VDSNetS": VDSNetS}


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_model(config, training=False):
    network = dict(config["network"])
    if not training:
        network["dino_model_path"] = None
    model_name = config["model"]
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model {model_name}; choose from {sorted(MODEL_CLASSES)}")
    return MODEL_CLASSES[model_name](**network)


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("params_ema", "params", "model", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def load_model_weights(model, path, strict=True):
    checkpoint = torch.load(path, map_location="cpu")
    state = _unwrap_state_dict(checkpoint)
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("dino_model."):
            continue
        cleaned[key] = value
    incompatible = model.load_state_dict(cleaned, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("dino_model.")]
    if strict and (missing or incompatible.unexpected_keys):
        raise RuntimeError(
            f"Checkpoint mismatch; missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return checkpoint, incompatible


def restoration_state_dict(model):
    if hasattr(model, "module"):
        model = model.module
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("dino_model.")
    }


def set_random_seed(seed, rank=0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path, root):
    if path is None:
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(root) / path
