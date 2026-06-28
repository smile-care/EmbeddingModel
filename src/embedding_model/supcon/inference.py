"""SupCon/MoCo inference utilities: load a trained model and compute embeddings.

Pure DL code (no platform/DB/config-file knowledge).  The caller passes a fully
resolved ``model_config`` dict plus ``image_size`` / ``use_moco``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from embedding_model.supcon.build import build_supcon_model
from embedding_model.supcon.datasets.supcon_dataset import MaskSoftDilation


def load_supcon_model(
    checkpoint_path: str | Path,
    model_config: dict[str, Any],
    image_size: int,
    use_moco: bool,
    device: torch.device,
) -> torch.nn.Module:
    """Build the model and load a trained checkpoint (strict=False)."""
    model, _queue = build_supcon_model(
        model_config,
        image_size=image_size,
        use_moco=use_moco,
        moco_config=model_config.get("moco", {}),
        freeze_backbone=False,
        device=device,
    )
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    image_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return image_tf, mask_tf


@torch.no_grad()
def compute_backbone_embeddings(
    model_config: dict[str, Any],
    image_paths: list[Path],
    image_size: int,
    mask_paths: list[Path | None] | None = None,
    batch_size: int = 16,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Compute embeddings from the *pretrained backbone only* (no trained head).

    Builds the model so the backbone loads its pretrained weights, then uses
    mask-weighted global average pooling over the last backbone feature map as the
    embedding.  This lets the platform offer a usable "default encoder" before any
    experiment has been trained, with deterministic output (only pretrained
    weights are involved — the randomly-initialized FPN / projection head are not).
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _queue = build_supcon_model(
        model_config,
        image_size=image_size,
        use_moco=False,
        moco_config=model_config.get("moco", {}),
        freeze_backbone=True,
        device=dev,
    )
    model.eval()
    image_tf, mask_tf = _build_transforms(image_size)
    mask_dilation = MaskSoftDilation({"enabled": True})

    if mask_paths is None:
        mask_paths = [None] * len(image_paths)

    out: list[np.ndarray] = []
    for i in range(0, len(image_paths), batch_size):
        batch_imgs: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor] = []
        for img_p, mask_p in zip(image_paths[i : i + batch_size], mask_paths[i : i + batch_size]):
            try:
                image = Image.open(img_p).convert("RGB")
                image_tensor = image_tf(image)
            except OSError:
                image_tensor = torch.zeros(3, image_size, image_size)

            if mask_p is not None and Path(mask_p).is_file():
                try:
                    mask_img = Image.open(mask_p).convert("L")
                    mask_tensor = (mask_tf(mask_img) > 0.5).float()
                except OSError:
                    mask_tensor = torch.ones(1, image_size, image_size)
            else:
                mask_tensor = torch.ones(1, image_size, image_size)
            mask_tensor = mask_dilation(mask_tensor)

            batch_imgs.append(image_tensor)
            batch_masks.append(mask_tensor)

        x = torch.stack(batch_imgs).to(dev)
        m = torch.stack(batch_masks).to(dev)

        feats = model.backbone(x, output_hidden_states=True)
        last = feats[-1]  # (B, C, h, w)
        m_ds = F.interpolate(m, size=last.shape[-2:], mode="area")
        num = (last * m_ds).sum(dim=(2, 3))
        den = m_ds.sum(dim=(2, 3)).clamp_min(1e-6)
        pooled = num / den
        emb = F.normalize(pooled, dim=1, p=2, eps=1e-8)
        out.append(emb.cpu().numpy())

    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(out).astype(np.float32, copy=False)


@torch.no_grad()
def compute_supcon_embeddings(
    checkpoint_path: str | Path,
    image_paths: list[Path],
    model_config: dict[str, Any],
    image_size: int,
    use_moco: bool = False,
    mask_paths: list[Path | None] | None = None,
    batch_size: int = 16,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Compute L2-normalized embeddings for a list of (image, optional mask) pairs."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_supcon_model(checkpoint_path, model_config, image_size, use_moco, dev)
    image_tf, mask_tf = _build_transforms(image_size)
    mask_dilation = MaskSoftDilation({"enabled": True})

    if mask_paths is None:
        mask_paths = [None] * len(image_paths)

    out: list[np.ndarray] = []
    for i in range(0, len(image_paths), batch_size):
        batch_imgs: list[torch.Tensor] = []
        batch_masks: list[torch.Tensor] = []
        for img_p, mask_p in zip(image_paths[i : i + batch_size], mask_paths[i : i + batch_size]):
            try:
                image = Image.open(img_p).convert("RGB")
                image_tensor = image_tf(image)
            except OSError:
                image_tensor = torch.zeros(3, image_size, image_size)

            if mask_p is not None and Path(mask_p).is_file():
                try:
                    mask_img = Image.open(mask_p).convert("L")
                    mask_tensor = (mask_tf(mask_img) > 0.5).float()
                except OSError:
                    mask_tensor = torch.ones(1, image_size, image_size)
            else:
                mask_tensor = torch.ones(1, image_size, image_size)
            mask_tensor = mask_dilation(mask_tensor)

            batch_imgs.append(image_tensor)
            batch_masks.append(mask_tensor)

        x = torch.stack(batch_imgs).to(dev)
        m = torch.stack(batch_masks).to(dev)

        if use_moco:
            pred = model(x, m, mode="query", return_features=False, return_segmentation=False)
        else:
            pred = model(x, m, return_features=False, return_segmentation=False)
        emb = F.normalize(pred["embeddings"], dim=1, p=2, eps=1e-8)
        out.append(emb.cpu().numpy())

    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(out).astype(np.float32, copy=False)
