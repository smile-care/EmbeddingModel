"""CPU-friendly forward/backward smoke tests for both selfsup methods and backbones."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedding_model.selfsup.losses import DINOLoss, VICRegLoss
from src.embedding_model.selfsup.models.encoder import MaskAwareEncoder
from src.embedding_model.selfsup.models.heads import DINOHead, SelfSupervisedNetwork, VICRegProjector
from src.embedding_model.selfsup.models.pooling import ConvNeXtMaskAggregator, ViTMaskAggregator
from src.embedding_model.selfsup.models.teacher_student import EMATeacher
from src.embedding_model.selfsup.training import build_optimizer
from src.embedding_model.supcon.models.backbone.dinov3_convnext import (
    DINOv3ConvNext,
    DINOv3ConvNextConfig,
)
from src.embedding_model.supcon.models.backbone.dinov3_vit import DINOv3ViT, DINOv3ViTConfig
from scripts_training.train_selfsup import train_step


def tiny_encoder(backbone_type: str) -> MaskAwareEncoder:
    if backbone_type == "convnext":
        config = DINOv3ConvNextConfig(
            hidden_sizes=[8, 16, 32, 64],
            depths=[1, 1, 1, 1],
            image_size=32,
        )
        backbone = DINOv3ConvNext(config)
        aggregator = ConvNeXtMaskAggregator(
            config.hidden_sizes,
            representation_dim=32,
            use_layers=[1, 2, 3],
            pooling_mode="fg_context_delta",
        )
        return MaskAwareEncoder(backbone, aggregator, backbone_type="convnext")

    config = DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=8,
        # The shared ViT expects pretrained weights to initialize register tokens.
        # This random-weight smoke model therefore disables them explicitly.
        num_register_tokens=0,
        pos_embed_rescale=None,
    )
    backbone = DINOv3ViT(config)
    aggregator = ViTMaskAggregator(32, representation_dim=32, cls_weight=0.0)
    return MaskAwareEncoder(backbone, aggregator, backbone_type="vit")


def run_vicreg(backbone_type: str, images: torch.Tensor, masks: torch.Tensor) -> None:
    student = SelfSupervisedNetwork(tiny_encoder(backbone_type), VICRegProjector(32, 64, 16))
    optimizer = build_optimizer(student, learning_rate=1e-3, backbone_lr_ratio=0.1, weight_decay=0.0)
    metrics, succeeded = train_step(
        method="vicreg",
        student=student,
        teacher=None,
        criterion=VICRegLoss(),
        batch=build_batch(images, masks),
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        device=torch.device("cpu"),
        amp_enabled=False,
        gradient_clip=2.0,
        global_step=1,
        total_steps=2,
        method_config={},
        compute_diagnostics=True,
    )
    assert succeeded and torch.isfinite(torch.tensor(metrics["loss"]))


def run_dino(backbone_type: str, images: torch.Tensor, masks: torch.Tensor) -> None:
    student = SelfSupervisedNetwork(tiny_encoder(backbone_type), DINOHead(32, 64, 16, 32))
    teacher = EMATeacher(student)
    optimizer = build_optimizer(student, learning_rate=1e-3, backbone_lr_ratio=0.1, weight_decay=0.0)
    metrics, succeeded = train_step(
        method="dino",
        student=student,
        teacher=teacher,
        criterion=DINOLoss(num_prototypes=32),
        batch=build_batch(images, masks),
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        device=torch.device("cpu"),
        amp_enabled=False,
        gradient_clip=2.0,
        global_step=1,
        total_steps=2,
        method_config={
            "teacher_temperature_start": 0.04,
            "teacher_temperature_end": 0.07,
            "teacher_temperature_warmup_steps": 1,
            "teacher_momentum_start": 0.996,
            "teacher_momentum_end": 1.0,
            "student_temperature": 0.1,
        },
        compute_diagnostics=True,
    )
    assert succeeded and torch.isfinite(torch.tensor(metrics["loss"]))


def build_batch(images: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "view1_image": images,
        "view1_mask": masks,
        "view2_image": images.flip(-1),
        "view2_mask": masks.flip(-1),
    }


def main() -> None:
    torch.manual_seed(7)
    images = torch.randn(4, 3, 32, 32)
    masks = torch.zeros(4, 1, 32, 32)
    masks[:, :, 8:24, 8:24] = 1.0
    for backbone_type in ("convnext", "vit"):
        run_vicreg(backbone_type, images, masks)
        run_dino(backbone_type, images, masks)
        print(f"{backbone_type}: VICReg and DINO smoke tests passed")


if __name__ == "__main__":
    main()
