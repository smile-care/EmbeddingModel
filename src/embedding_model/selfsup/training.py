"""Shared runtime utilities for the independent self-supervised trainer."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("DDP 自监督训练当前要求 CUDA/NCCL")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size


def seed_everything(seed: int, rank: int = 0) -> None:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_optimizer(
    student: nn.Module,
    *,
    learning_rate: float,
    backbone_lr_ratio: float,
    weight_decay: float,
) -> Optimizer:
    backbone_parameters: list[nn.Parameter] = []
    other_parameters: list[nn.Parameter] = []
    for name, parameter in student.named_parameters():
        if ".backbone." in f".{name}.":
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    if not backbone_parameters or not other_parameters:
        raise ValueError("无法为 selfsup student 划分 backbone/head 参数组")
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": learning_rate * backbone_lr_ratio},
            {"params": other_parameters, "lr": learning_rate},
        ],
        weight_decay=weight_decay,
    )


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps 必须小于 total_steps")

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


def cosine_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    progress = min(max(step / max(1, total_steps), 0.0), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def linear_warmup(start: float, end: float, step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return end
    return start + (end - start) * step / warmup_steps


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(data: dict[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(destination)


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def set_backbone_trainable(student: nn.Module, trainable: bool) -> None:
    raw_student = unwrap_module(student)
    raw_student.encoder.set_backbone_trainable(trainable)


@torch.no_grad()
def representation_diagnostics(view1: torch.Tensor, view2: torch.Tensor) -> dict[str, float]:
    normalized1 = torch.nn.functional.normalize(view1.float(), dim=1)
    normalized2 = torch.nn.functional.normalize(view2.float(), dim=1)
    consistency = (normalized1 * normalized2).sum(dim=1).mean()
    centered = torch.cat([normalized1, normalized2], dim=0)
    centered = centered - centered.mean(dim=0)
    singular_values = torch.linalg.svdvals(centered)
    probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return {
        "augmentation_consistency": float(consistency.item()),
        "representation_effective_rank": float(effective_rank.item()),
    }


def checkpoint_payload(
    *,
    method: str,
    global_step: int,
    student: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    teacher: nn.Module | None = None,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "method": method,
        "global_step": global_step,
        "student_state_dict": unwrap_module(student).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": {"selfsup": config},
        "rng_state": capture_rng_state(),
    }
    if teacher is not None:
        payload["teacher_state_dict"] = teacher.state_dict()
    if criterion is not None:
        payload["criterion_state_dict"] = criterion.state_dict()
    return payload


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    student: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    teacher: nn.Module | None = None,
    criterion: nn.Module | None = None,
    expected_method: str | None = None,
    map_location: torch.device | str = "cpu",
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if expected_method is not None and checkpoint.get("method") != expected_method:
        raise ValueError(
            f"checkpoint method={checkpoint.get('method')!r}，当前配置 method={expected_method!r}，禁止跨方法恢复"
        )
    unwrap_module(student).load_state_dict(checkpoint["student_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
    if teacher is not None:
        teacher.load_state_dict(checkpoint["teacher_state_dict"], strict=True)
    if criterion is not None and "criterion_state_dict" in checkpoint:
        criterion.load_state_dict(checkpoint["criterion_state_dict"], strict=True)
    restore_rng_state(checkpoint.get("rng_state"))
    return int(checkpoint["global_step"]) + 1


def build_inference_payload(
    *,
    method: str,
    encoder: nn.Module,
    config: dict[str, Any],
    global_step: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": method,
        "global_step": global_step,
        "encoder_state_dict": encoder.state_dict(),
        "config": {"selfsup": config},
        "embedding_source": "representations",
    }
