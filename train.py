#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from data import PairedImageDataset
from losses import EdgeLoss, restoration_losses
from utils import (
    build_model,
    load_config,
    load_model_weights,
    resolve_path,
    restoration_state_dict,
    set_random_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VDSNet or VDSNet-S")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    return parser.parse_args()


def distributed_context(local_rank):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
    else:
        rank = 0
        local_rank = 0
    if not torch.cuda.is_available():
        raise RuntimeError("Training VDSNet requires a CUDA GPU")
    return distributed, rank, world_size, torch.device("cuda", local_rank)


def learning_rate(iteration, initial_lr, scheduler):
    periods = [int(value) for value in scheduler["periods"]]
    weights = [float(value) for value in scheduler["restart_weights"]]
    eta_mins = [float(value) for value in scheduler["eta_mins"]]
    start = 0
    for period, weight, eta_min in zip(periods, weights, eta_mins):
        end = start + period
        if iteration <= end:
            progress = max(0, iteration - start) / period
            peak = initial_lr * weight
            return eta_min + 0.5 * (peak - eta_min) * (1 + math.cos(math.pi * progress))
        start = end
    return eta_mins[-1]


def temperature(iteration, config):
    start = float(config["start"])
    end = float(config["end"])
    duration = int(config["decay_iterations"])
    progress = min(1.0, max(0.0, iteration / duration))
    return start + progress * (end - start)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_sum = 0.0
    count = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        divisor = 16 if model.__class__.__name__ == "VDSNetS" else 8
        height, width = inputs.shape[-2:]
        pad_h = (divisor - height % divisor) % divisor
        pad_w = (divisor - width % divisor) % divisor
        inputs = torch.nn.functional.pad(inputs, (0, pad_w, 0, pad_h), mode="reflect")
        predictions = model(inputs).clamp(0, 1)[..., :height, :width]
        mse = (predictions - targets).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
        psnr_sum += (-10.0 * torch.log10(mse)).sum().item()
        count += targets.shape[0]
    model.train()
    return psnr_sum / max(1, count)


def save_checkpoint(path, model, optimizer, iteration, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "params": restoration_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "config": config,
        },
        path,
    )


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    config = load_config(args.config)
    distributed, rank, world_size, device = distributed_context(args.local_rank)
    set_random_seed(config.get("seed", 100), rank)

    for split in ("train", "val"):
        for key in ("input_dir", "target_dir"):
            config["datasets"][split][key] = str(resolve_path(config["datasets"][split][key], repo_root))
    dino_path = config["network"].get("dino_model_path")
    if dino_path is not None:
        config["network"]["dino_model_path"] = str(resolve_path(dino_path, repo_root))

    train_cfg = config["datasets"]["train"]
    train_set = PairedImageDataset(
        train_cfg["input_dir"],
        train_cfg["target_dir"],
        crop_size=train_cfg.get("crop_size"),
        augment=train_cfg.get("geometric_augment", True),
        repeat=train_cfg.get("repeat", 1),
    )
    sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(config.get("num_workers", 0)) > 0,
    )

    val_loader = None
    if rank == 0:
        val_cfg = config["datasets"]["val"]
        val_set = PairedImageDataset(val_cfg["input_dir"], val_cfg["target_dir"])
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(config, training=True).to(device)
    optimizer_cfg = dict(config["train"]["optimizer"])
    optimizer_type = optimizer_cfg.pop("type")
    if optimizer_type != "AdamW":
        raise ValueError("The release trainer currently supports AdamW only")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        **optimizer_cfg,
    )

    start_iteration = 0
    pretrained = config["train"].get("pretrained")
    if pretrained:
        load_model_weights(model, resolve_path(pretrained, repo_root), strict=True)
    resume = config["train"].get("resume")
    if resume:
        checkpoint, _ = load_model_weights(model, resolve_path(resume, repo_root), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint["iteration"])

    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    raw_model = model.module if hasattr(model, "module") else model
    edge_loss = EdgeLoss().to(device)
    training = config["train"]
    loss_weights = training["loss_weights"]
    output_dir = resolve_path(training["output_dir"], repo_root)
    initial_lr = float(optimizer_cfg["lr"])
    total_iterations = int(training["total_iterations"])
    iteration = start_iteration
    epoch = 0
    start_time = time.time()

    if rank == 0:
        trainable = sum(parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad)
        print(
            f"Training {config['model']} on {world_size} GPU(s); "
            f"batch/GPU={train_cfg['batch_size']}, global batch={int(train_cfg['batch_size']) * world_size}, "
            f"trainable parameters={trainable / 1e6:.3f}M"
        )

    while iteration < total_iterations:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in train_loader:
            if iteration >= total_iterations:
                break
            iteration += 1
            current_lr = learning_rate(iteration, initial_lr, training["scheduler"])
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            current_temperature = temperature(iteration, training["temperature"])
            raw_model.set_mvgl_temperature(current_temperature)

            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            total_loss, loss_values = restoration_losses(
                raw_model, predictions, targets, loss_weights, edge_loss
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                float(training.get("gradient_clip_norm", 0.01)),
            )
            optimizer.step()

            if rank == 0 and iteration % int(training["print_every"]) == 0:
                elapsed = time.time() - start_time
                details = " ".join(f"{name}={value.item():.4f}" for name, value in loss_values.items())
                print(
                    f"iter={iteration}/{total_iterations} lr={current_lr:.3e} "
                    f"T={current_temperature:.3f} total={total_loss.item():.4f} {details} "
                    f"elapsed={elapsed / 3600:.2f}h",
                    flush=True,
                )

            run_validation = iteration % int(training["validate_every"]) == 0
            if run_validation and rank == 0:
                score = validate(raw_model, val_loader, device)
                print(f"validation iter={iteration}: PSNR={score:.4f} dB", flush=True)
            if run_validation and distributed:
                dist.barrier()

            if rank == 0 and iteration % int(training["save_every"]) == 0:
                save_checkpoint(output_dir / f"net_g_{iteration}.pth", model, optimizer, iteration, config)
        epoch += 1

    if rank == 0:
        save_checkpoint(output_dir / "net_g_latest.pth", model, optimizer, iteration, config)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

