"""Independent mask-aware self-supervised trainer (VICReg or DINO-style).

This entry intentionally does not import SupCon models, losses, queues, samplers,
or training helpers. Only the shared DINOv3 backbone implementation is reused by
``embedding_model.selfsup``.

Examples:
  python scripts_training/train_selfsup.py --config configs/selfsup_config.yaml
  torchrun --nproc_per_node=4 scripts_training/train_selfsup.py --config configs/selfsup_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedding_model.selfsup.config import load_selfsup_config
from src.embedding_model.selfsup.datasets import (
    SelfSupervisedDataset,
    UniformStepBatchSampler,
    selfsup_collate,
)
from src.embedding_model.selfsup.losses import DINOLoss, VICRegLoss
from src.embedding_model.selfsup.models import (
    DINOHead,
    EMATeacher,
    VICRegProjector,
    build_mask_aware_encoder,
)
from src.embedding_model.selfsup.models.heads import SelfSupervisedNetwork
from src.embedding_model.selfsup.training import (
    atomic_torch_save,
    build_inference_payload,
    build_optimizer,
    build_warmup_cosine_scheduler,
    checkpoint_payload,
    cosine_schedule,
    init_distributed,
    linear_warmup,
    load_checkpoint,
    representation_diagnostics,
    seed_everything,
    seed_worker,
    set_backbone_trainable,
    unwrap_module,
)

try:
    import wandb
except ImportError:
    wandb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask-aware VICReg / DINO 自监督训练")
    parser.add_argument("--config", default="configs/selfsup_config.yaml", help="自监督 YAML 配置")
    parser.add_argument("--resume", default=None, help="恢复训练 checkpoint")
    return parser.parse_args()


def load_data_sources(config_path: str | Path) -> list[dict[str, Any]]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"数据配置不存在: {path}")
    with path.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)
    try:
        sources = data_config["scenes"]["train"]
    except (KeyError, TypeError) as error:
        raise ValueError("数据配置必须包含 scenes.train 列表") from error
    if not isinstance(sources, list) or not sources:
        raise ValueError("scenes.train 必须是非空列表")
    return sources


def build_student(config: dict[str, Any], method: str) -> SelfSupervisedNetwork:
    model_config = config["model"]
    encoder = build_mask_aware_encoder(model_config, project_root=ROOT, load_pretrained=True)
    representation_dim = int(model_config["representation_dim"])
    projection_dim = int(model_config["projection_dim"])
    hidden_dim = int(model_config.get("projection_hidden_dim", 2048))
    if method == "vicreg":
        head: nn.Module = VICRegProjector(representation_dim, hidden_dim, projection_dim)
    else:
        dino_config = config["dino"]
        head = DINOHead(
            input_dim=representation_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=int(dino_config.get("bottleneck_dim", 256)),
            num_prototypes=int(dino_config["num_prototypes"]),
        )
    return SelfSupervisedNetwork(encoder, head)


def build_criterion(config: dict[str, Any], method: str, device: torch.device) -> nn.Module:
    if method == "vicreg":
        vicreg = config["vicreg"]
        return VICRegLoss(
            invariance_weight=float(vicreg["invariance_weight"]),
            variance_weight=float(vicreg["variance_weight"]),
            covariance_weight=float(vicreg["covariance_weight"]),
            variance_target=float(vicreg["variance_target"]),
            eps=float(vicreg.get("eps", 1e-4)),
        ).to(device)
    dino = config["dino"]
    return DINOLoss(
        num_prototypes=int(dino["num_prototypes"]),
        center_momentum=float(dino["center_momentum"]),
    ).to(device)


def reduce_metrics(metrics: dict[str, float], device: torch.device, world_size: int) -> dict[str, float]:
    if world_size == 1:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values)
    values /= world_size
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


def train_step(
    *,
    method: str,
    student: nn.Module,
    teacher: EMATeacher | None,
    criterion: nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    gradient_clip: float,
    global_step: int,
    total_steps: int,
    method_config: dict[str, Any],
    compute_diagnostics: bool,
) -> tuple[dict[str, float], bool]:
    student.train()
    images1 = batch["view1_image"].to(device, non_blocking=True)
    masks1 = batch["view1_mask"].to(device, non_blocking=True)
    images2 = batch["view2_image"].to(device, non_blocking=True)
    masks2 = batch["view2_mask"].to(device, non_blocking=True)
    mask_sums = torch.cat(
        [masks1.flatten(1).sum(dim=1), masks2.flatten(1).sum(dim=1)],
        dim=0,
    )
    mask_metrics = {
        "mask_fraction": float(
            (0.5 * (masks1.float().mean() + masks2.float().mean())).item()
        ),
        "empty_mask_rate": float((mask_sums <= 0).float().mean().item()),
    }

    optimizer.zero_grad(set_to_none=True)
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=amp_enabled):
        student1 = student(images1, masks1)
        student2 = student(images2, masks2)
        if method == "vicreg":
            loss_outputs = criterion(student1["outputs"], student2["outputs"])
            teacher_momentum = None
            teacher_temperature = None
        else:
            if teacher is None:
                raise RuntimeError("DINO 训练缺少 EMA teacher")
            with torch.no_grad():
                teacher1 = teacher(images1, masks1)
                teacher2 = teacher(images2, masks2)
            teacher_temperature = linear_warmup(
                float(method_config["teacher_temperature_start"]),
                float(method_config["teacher_temperature_end"]),
                global_step,
                int(method_config["teacher_temperature_warmup_steps"]),
            )
            teacher_momentum = cosine_schedule(
                float(method_config["teacher_momentum_start"]),
                float(method_config["teacher_momentum_end"]),
                global_step,
                total_steps,
            )
            loss_outputs = criterion(
                student1["outputs"],
                student2["outputs"],
                teacher1["outputs"],
                teacher2["outputs"],
                student_temperature=float(method_config["student_temperature"]),
                teacher_temperature=teacher_temperature,
            )

        loss = loss_outputs["loss"]

    finite_flag = torch.tensor(
        1 if torch.isfinite(loss) else 0,
        device=device,
        dtype=torch.int32,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
    if finite_flag.item() == 0:
        optimizer.zero_grad(set_to_none=True)
        failed_metrics = {
            key: 0.0
            for key in loss_outputs
        }
        failed_metrics.update(
            {
                "loss": 0.0,
                "grad_norm": 0.0,
                "nonfinite_steps": 1.0,
                "augmentation_consistency": 0.0,
            }
        )
        failed_metrics.update(mask_metrics)
        if compute_diagnostics:
            failed_metrics["representation_effective_rank"] = 0.0
        if teacher_momentum is not None and teacher_temperature is not None:
            failed_metrics["teacher_momentum"] = teacher_momentum
            failed_metrics["teacher_temperature"] = teacher_temperature
        return failed_metrics, False

    previous_scale = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), gradient_clip)
    scaler.step(optimizer)
    scaler.update()
    update_succeeded = not scaler.is_enabled() or scaler.get_scale() >= previous_scale

    if method == "dino" and update_succeeded:
        assert teacher is not None and teacher_momentum is not None
        teacher.update(unwrap_module(student), teacher_momentum)

    metrics = {
        key: float(value.detach().float().item())
        for key, value in loss_outputs.items()
    }
    metrics["grad_norm"] = float(torch.as_tensor(grad_norm).detach().float().item())
    metrics["nonfinite_steps"] = 0.0
    metrics.update(mask_metrics)
    normalized1 = torch.nn.functional.normalize(student1["representations"].detach().float(), dim=1)
    normalized2 = torch.nn.functional.normalize(student2["representations"].detach().float(), dim=1)
    metrics["augmentation_consistency"] = float((normalized1 * normalized2).sum(dim=1).mean().item())
    if compute_diagnostics:
        metrics.update(representation_diagnostics(student1["representations"], student2["representations"]))
    if teacher_momentum is not None and teacher_temperature is not None:
        metrics["teacher_momentum"] = teacher_momentum
        metrics["teacher_temperature"] = teacher_temperature
    return metrics, update_succeeded


def main() -> None:
    args = parse_args()
    local_rank, rank, world_size = init_distributed()
    is_main = rank == 0
    config = load_selfsup_config(args.config)
    method = str(config["method"]).lower()
    training_config = config["training"]
    output_config = config["output"]
    seed = int(training_config.get("seed", 0))
    seed_everything(seed, rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log = lambda message: print(message, flush=True) if is_main else None
    log(f"Self-supervised method={method}, device={device}, world_size={world_size}")

    source_configs = load_data_sources(config["data"]["data_config_path"])
    dataset = SelfSupervisedDataset(
        source_configs,
        project_root=ROOT,
        image_size=int(config["data"]["image_size"]),
        augmentation_config=config["data"].get("augmentation", {}),
        training=True,
    )
    log(
        f"统一样本池: samples={len(dataset)}, configured_sources={len(source_configs)}, "
        f"deduplicated={dataset.duplicate_count}, missing_masks={dataset.missing_mask_count}"
    )

    total_steps = int(training_config["total_steps"])
    batch_size = int(config["data"]["batch_size"])
    sampler = UniformStepBatchSampler(
        len(dataset),
        batch_size,
        total_steps,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    num_workers = int(config["data"].get("num_workers", 4))
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=selfsup_collate,
        num_workers=num_workers,
        pin_memory=bool(config["data"].get("pin_memory", True)),
        persistent_workers=num_workers > 0,
        prefetch_factor=int(config["data"].get("prefetch_factor", 2)) if num_workers > 0 else None,
        worker_init_fn=seed_worker,
    )
    log(f"uniform sampling: batch/GPU={batch_size}, effective_batch={batch_size * world_size}")

    student = build_student(config, method).to(device)
    freeze_steps = int(training_config.get("backbone_freeze_steps", 0))
    set_backbone_trainable(student, freeze_steps <= 0)
    teacher = EMATeacher(student).to(device) if method == "dino" else None
    criterion = build_criterion(config, method, device)

    if world_size > 1:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    optimizer = build_optimizer(
        student,
        learning_rate=float(training_config["learning_rate"]),
        backbone_lr_ratio=float(training_config.get("backbone_lr_ratio", 0.05)),
        weight_decay=float(training_config.get("weight_decay", 0.05)),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(training_config.get("warmup_steps", 5000)),
        min_lr_ratio=float(training_config.get("min_lr_ratio", 0.01)),
    )
    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_step = 1
    if args.resume:
        start_step = load_checkpoint(
            args.resume,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            teacher=teacher,
            criterion=criterion,
            expected_method=method,
            map_location=device,
        )
        sampler.set_start_step(start_step)
        set_backbone_trainable(student, start_step > freeze_steps)
        log(f"已恢复 {args.resume}，从 step {start_step} 继续")

    output_tokens = {"method": method, "backbone": str(config["model"]["backbone"])}
    checkpoint_dir = Path(str(output_config["checkpoint_dir"]).format(**output_tokens))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(output_config.get("use_wandb", False))
    if use_wandb and wandb is None:
        raise RuntimeError("use_wandb=true，但 hjh 环境未安装 wandb")
    if is_main and use_wandb:
        wandb.init(
            project=str(output_config.get("wandb_project", "industrial-selfsup")),
            name=(
                str(output_config["wandb_run_name"]).format(**output_tokens)
                if output_config.get("wandb_run_name")
                else None
            ),
            config=config,
        )

    def save_checkpoint(global_step: int, archive: bool = False) -> None:
        if not is_main:
            return
        payload = checkpoint_payload(
            method=method,
            global_step=global_step,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            teacher=teacher,
            criterion=criterion,
        )
        atomic_torch_save(payload, checkpoint_dir / "current_checkpoint.pth")
        if archive:
            atomic_torch_save(payload, checkpoint_dir / f"checkpoint_step_{global_step}.pth")

        inference_encoder = (
            teacher.model.encoder if teacher is not None else unwrap_module(student).encoder
        )
        inference_payload = build_inference_payload(
            method=method,
            encoder=inference_encoder,
            config=config,
            global_step=global_step,
        )
        atomic_torch_save(inference_payload, checkpoint_dir / "inference_model.pth")

    log_interval = int(training_config.get("log_interval", 50))
    current_interval = int(training_config.get("current_model_interval_steps", 1000))
    archive_interval = int(training_config.get("save_interval_steps", 20000))
    gradient_clip = float(training_config.get("gradient_clip", 2.0))
    recent_losses: deque[float] = deque(maxlen=max(1, log_interval))
    train_iterator = iter(dataloader)
    progress = tqdm(range(start_step, total_steps + 1), disable=not is_main, desc=f"SelfSup/{method}")

    for global_step in progress:
        if freeze_steps > 0 and global_step == freeze_steps + 1:
            set_backbone_trainable(student, True)
            log(f"Step {global_step}: 已解冻 backbone")

        batch = next(train_iterator)
        metrics, _ = train_step(
            method=method,
            student=student,
            teacher=teacher,
            criterion=criterion,
            batch=batch,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            gradient_clip=gradient_clip,
            global_step=global_step,
            total_steps=total_steps,
            method_config=config[method],
            compute_diagnostics=(global_step == start_step or global_step % log_interval == 0),
        )
        scheduler.step()
        metrics = reduce_metrics(metrics, device, world_size)
        recent_losses.append(metrics["loss"])
        metrics["window_loss"] = sum(recent_losses) / len(recent_losses)
        metrics["backbone_lr"] = optimizer.param_groups[0]["lr"]
        metrics["head_lr"] = optimizer.param_groups[1]["lr"]

        if is_main:
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                rank=f"{metrics.get('representation_effective_rank', 0.0):.1f}",
                lr=f"{metrics['backbone_lr']:.2e}",
            )
        if global_step == start_step or global_step % log_interval == 0:
            log(
                f"Step {global_step}: loss={metrics['loss']:.4f}, "
                f"consistency={metrics.get('augmentation_consistency', 0.0):.4f}, "
                f"effective_rank={metrics.get('representation_effective_rank', 0.0):.2f}"
            )
            if is_main and use_wandb:
                wandb.log(metrics, step=global_step)

        save_current = current_interval > 0 and global_step % current_interval == 0
        save_archive = archive_interval > 0 and global_step % archive_interval == 0
        if save_current or save_archive:
            if world_size > 1:
                dist.barrier()
            save_checkpoint(global_step, archive=save_archive)
            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.barrier()
    save_checkpoint(total_steps, archive=True)
    if world_size > 1:
        dist.barrier()
    if is_main and use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()
    log(f"训练完成，产物目录: {checkpoint_dir}")


if __name__ == "__main__":
    main()
