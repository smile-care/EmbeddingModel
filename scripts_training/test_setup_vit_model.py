"""
Smoke test for ViT SupCon model setup.

Builds a ViT model through the unified SupCon factory, runs one mock forward,
and reports module params/MACs/FLOPs with THOP.
"""
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from thop import profile as thop_profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from embedding_model.supcon.build import build_supcon_model
from embedding_model.utils.config_loader import load_config


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
    print(
        f"{name:<18} "
        f"params={format_count(params['total']):>8} "
        f"trainable={format_count(params['trainable']):>8} "
        f"MACs={format_count(macs):>9} "
        f"FLOPs={format_count(2 * macs):>9}"
    )


def main() -> None:
    config = load_config(str(ROOT / "configs" / "supcon_config.yaml"))["supcon"]
    model_config = dict(config["model"])
    backbone_name = model_config.get("backbone")
    if not str(backbone_name).startswith("vit"):
        backbone_name = "vits16"
        model_config["backbone"] = backbone_name

    model_config["backbone_config_path"] = str(ROOT / "configs" / "backbone" / f"{backbone_name}.yaml")
    model_config["pretrained_path"] = str(ROOT / "pretrain_ckpts" / f"{backbone_name}.pth")

    image_size = resolve_image_size(config["data"].get("image_size", 224))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_supcon_model(
        model_config,
        image_size=image_size,
        use_moco=False,
        freeze_backbone=False,
        device=device,
    )
    model.eval()

    batch_size = 2
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    masks = torch.ones(batch_size, 1, image_size, image_size, device=device)

    with torch.no_grad():
        outputs = model(images, masks, return_features=True)
        cls_token, patch_tokens = model.backbone(images, output_hidden_states=True)
        fused_features = model.mask_pooling(cls_token, patch_tokens, masks)

    module_macs = {
        "backbone": profile_macs(BackboneProfileWrapper(model.backbone).to(device).eval(), (images,)),
        "mask_pooling": profile_macs(model.mask_pooling, (cls_token, patch_tokens, masks)),
        "projection_head": profile_macs(model.projection_head, (fused_features,)),
        "total_model": profile_macs(ModelProfileWrapper(model).to(device).eval(), (images, masks)),
    }

    print(f"device: {device}")
    print(f"backbone: {backbone_name}")
    print(f"image_size: {image_size}")
    print(f"batch_size: {batch_size}")
    print(f"cls_weight: {model_config.get('vit', {}).get('cls_weight', 0.3)}")
    print(f"representations: {tuple(outputs['representations'].shape)}")
    print(f"projections: {tuple(outputs['projections'].shape)}")
    print(f"projection_norm: {outputs['projections'].norm(dim=1).tolist()}")
    print()
    print(THOP_NOTE)
    print_module_stats("backbone", model.backbone, module_macs["backbone"])
    print_module_stats("mask_pooling", model.mask_pooling, module_macs["mask_pooling"])
    print_module_stats("projection_head", model.projection_head, module_macs["projection_head"])
    print_module_stats("total_model", model, module_macs["total_model"])


if __name__ == "__main__":
    main()
