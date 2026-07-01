"""
Smoke test for ConvNeXt SupCon model setup.

Loads configs/supcon_config.yaml, builds a ConvNeXtModel, creates one mock
image/mask batch, runs a single forward pass, and reports module params/FLOPs.
"""
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from thop import profile as thop_profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedding_model.supcon.models.backbone.dinov3_convnext import DINOv3ConvNextConfig
from src.embedding_model.supcon.models.convnext_model import ConvNeXtModel
from src.embedding_model.utils.config_loader import load_config


THOP_NOTE = "THOP reports MACs; FLOPs below use the common convention FLOPs = 2 * MACs."


def resolve_image_size(image_size_config) -> int:
    if isinstance(image_size_config, int):
        return image_size_config
    if isinstance(image_size_config, (list, tuple)) and image_size_config:
        return int(max(image_size_config))
    raise ValueError(f"Invalid image_size config: {image_size_config!r}")


def count_params(module: nn.Module) -> Dict[str, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return {"total": total, "trainable": trainable}


def format_count(value: float) -> str:
    units = ["", "K", "M", "G", "T"]
    value = float(value)
    for unit in units:
        if abs(value) < 1000.0:
            return f"{value:.2f}{unit}"
        value /= 1000.0
    return f"{value:.2f}P"


class BackboneProfileWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor):
        return self.backbone(images, output_hidden_states=True)


class ModelProfileWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, masks: torch.Tensor):
        return self.model(images, masks, return_features=True)


def profile_macs(module: nn.Module, inputs: tuple) -> int:
    macs, _ = thop_profile(module, inputs=inputs, verbose=False)
    return int(macs)


def print_module_stats(name: str, module: nn.Module, macs: int) -> None:
    params = count_params(module)
    flops = 2 * macs
    print(
        f"{name:<18} "
        f"params={format_count(params['total']):>8} "
        f"trainable={format_count(params['trainable']):>8} "
        f"MACs={format_count(macs):>9} "
        f"FLOPs={format_count(flops):>9}"
    )


def main() -> None:
    config_path = ROOT / "configs" / "supcon_config.yaml"
    config = load_config(str(config_path))["supcon"]
    model_config = config["model"]

    backbone_name = model_config.get("backbone")
    if not backbone_name or not backbone_name.startswith("convnext_"):
        raise ValueError(f"This smoke test expects a ConvNeXt backbone, got: {backbone_name!r}")

    backbone_cfg_path = ROOT / "configs" / "backbone" / f"{backbone_name}.yaml"
    backbone_ckpt_path = ROOT / "pretrain_ckpts" / f"{backbone_name}.pth"
    image_size = resolve_image_size(config["data"].get("image_size", 224))

    backbone_cfg = DINOv3ConvNextConfig.from_yaml(str(backbone_cfg_path))
    convnext_config = model_config.get("convnext", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtModel(
        backbone_cfg=backbone_cfg,
        ckpt_path=str(backbone_ckpt_path),
        embedding_dim=int(model_config["embedding_dim"]),
        image_size=image_size,
        freeze_backbone=False,
        use_layers=list(convnext_config.get("use_layers", [1, 2, 3])),
    ).to(device)
    model.eval()

    batch_size = 2
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    masks = torch.ones(batch_size, 1, image_size, image_size, device=device)

    with torch.no_grad():
        outputs = model(images, masks, return_features=True)

    with torch.no_grad():
        backbone_features = model.backbone(images, output_hidden_states=True)
        fused_features = model.feature_fusion(backbone_features, masks)

    backbone_profile = BackboneProfileWrapper(model.backbone).to(device).eval()
    model_profile = ModelProfileWrapper(model).to(device).eval()

    module_macs = {
        "backbone": profile_macs(
            backbone_profile,
            (images,),
        ),
        "feature_fusion": profile_macs(
            model.feature_fusion,
            (backbone_features, masks),
        ),
        "projection_head": profile_macs(
            model.projection_head,
            (fused_features,),
        ),
        "total_model": profile_macs(
            model_profile,
            (images, masks),
        ),
    }

    print(f"device: {device}")
    print(f"backbone: {backbone_name}")
    print(f"image_size: {image_size}")
    print(f"batch_size: {batch_size}")
    print(f"use_layers: {convnext_config.get('use_layers', [1, 2, 3])}")
    print(f"features: {tuple(outputs['features'].shape)}")
    print(f"embeddings: {tuple(outputs['embeddings'].shape)}")
    print(f"embedding_norm: {outputs['embeddings'].norm(dim=1).tolist()}")
    print()
    print(THOP_NOTE)
    print_module_stats("backbone", model.backbone, module_macs["backbone"])
    print_module_stats("feature_fusion", model.feature_fusion, module_macs["feature_fusion"])
    print_module_stats("projection_head", model.projection_head, module_macs["projection_head"])
    print_module_stats("total_model", model, module_macs["total_model"])


if __name__ == "__main__":
    main()
