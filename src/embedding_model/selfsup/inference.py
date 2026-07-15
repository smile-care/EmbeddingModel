"""Load and run self-supervised inference encoders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import MaskAwareEncoder, build_mask_aware_encoder


def load_inference_encoder(
    checkpoint_path: str | Path,
    *,
    project_root: str | Path,
    device: torch.device,
) -> tuple[MaskAwareEncoder, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 1 or "encoder_state_dict" not in checkpoint:
        raise ValueError("仅支持 selfsup format_version=1 的 inference_model.pth")
    try:
        config = checkpoint["config"]["selfsup"]
    except (KeyError, TypeError) as error:
        raise ValueError("selfsup inference checkpoint 缺少完整配置") from error
    encoder = build_mask_aware_encoder(
        config["model"],
        project_root=project_root,
        load_pretrained=False,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    encoder.eval()
    return encoder, config


@torch.no_grad()
def extract_representations(
    encoder: MaskAwareEncoder,
    images: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.normalize(encoder(images, masks), dim=1, p=2, eps=1e-8)
