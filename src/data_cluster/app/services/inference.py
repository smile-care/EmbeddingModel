"""Platform inference wrapper.

Builds an inference model config from ``configs/data_cluster.yaml`` defaults plus
the experiment's stored config overrides, then delegates the actual embedding
computation to :mod:`embedding_model.supcon.inference`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from embedding_model.supcon.inference import compute_backbone_embeddings as _compute_backbone_embeddings
from embedding_model.supcon.inference import compute_supcon_embeddings as _compute_embeddings
from embedding_model.utils.config_loader import load_config

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DC_CONFIG_PATH = _REPO_ROOT / "configs" / "data_cluster.yaml"

# Stable id used by the API/UI to mean "no trained model — use the pretrained backbone".
DEFAULT_MODEL_ID = "default"
DEFAULT_MODEL_NAME = "默认预训练模型"


def _coerce_bool(cfg: dict[str, Any], *keys: str, default: bool) -> bool:
    for key in keys:
        if key not in cfg:
            continue
        value = cfg[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _coerce_image_size(cfg: dict[str, Any]) -> int | list[int] | None:
    value = cfg.get("imageSize", cfg.get("image_size"))
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, list):
        out = [int(x) for x in value if isinstance(x, (int, float)) and int(x) > 0]
        return out or None
    return None


def _load_dc_config() -> dict[str, Any]:
    try:
        return load_config(str(_DC_CONFIG_PATH))
    except Exception:
        return {}


def build_infer_config(exp_config: dict[str, Any] | None) -> tuple[dict[str, Any], int, bool]:
    """Merge data_cluster.yaml defaults with exp_config overrides.

    Returns ``(model_config, image_size, use_moco)`` ready for embedding_model.
    """
    dc_cfg = _load_dc_config()
    dc = dc_cfg.get("data_cluster", {}) if dc_cfg else {}
    exp_cfg = exp_config if isinstance(exp_config, dict) else {}

    data_cfg = dict(dc.get("data", {}))
    model_cfg = dict(dc.get("model", {}))
    moco_cfg = dict(dc.get("moco", {}))

    image_size = _coerce_image_size(exp_cfg)
    if image_size is not None:
        data_cfg["image_size"] = image_size
    elif data_cfg.get("image_size") is None:
        data_cfg["image_size"] = 224

    backbone = exp_cfg.get("backbone") or model_cfg.get("backbone") or "convnext_small"
    backbone_name = exp_cfg.get("modelName") or exp_cfg.get("model_name") or backbone
    if isinstance(backbone_name, str) and backbone_name.strip():
        model_cfg["backbone"] = backbone_name.strip()

    embedding_dim = exp_cfg.get("embeddingDim") or exp_cfg.get("embedding_dim")
    if isinstance(embedding_dim, int) and embedding_dim > 0:
        model_cfg["embedding_dim"] = embedding_dim
    elif "embedding_dim" not in model_cfg:
        model_cfg["embedding_dim"] = 128

    if "projection_hidden_dims" not in model_cfg and "projection_head" not in model_cfg:
        model_cfg["projection_hidden_dims"] = [256, 128]

    use_moco = _coerce_bool(exp_cfg, "useMoCo", "use_moco", default=moco_cfg.get("enabled", False))
    moco_cfg["enabled"] = use_moco
    model_cfg["moco"] = moco_cfg

    image_size_cfg = data_cfg["image_size"]
    if isinstance(image_size_cfg, list):
        model_image_size = int(max(image_size_cfg))
    else:
        model_image_size = int(image_size_cfg)

    return model_cfg, model_image_size, use_moco


def compute_supcon_embeddings(
    checkpoint_path: str | Path,
    image_paths: list[Path],
    mask_paths: list[Path | None] | None = None,
    exp_config: dict[str, Any] | None = None,
    batch_size: int = 16,
    device: str | None = None,
) -> np.ndarray:
    """Compute embeddings for dataset crops using the experiment's trained model."""
    model_cfg, image_size, use_moco = build_infer_config(exp_config)
    return _compute_embeddings(
        checkpoint_path=checkpoint_path,
        image_paths=image_paths,
        model_config=model_cfg,
        image_size=image_size,
        use_moco=use_moco,
        mask_paths=mask_paths,
        batch_size=batch_size,
        device=device,
    )


def _build_default_model_config() -> tuple[dict[str, Any], int]:
    """Resolve the default pretrained-backbone config from data_cluster.yaml.

    Returns ``(model_config, image_size)``.  ``pretrained_path`` is only set when
    the weights file actually exists, so a missing checkpoint degrades to a
    randomly-initialized backbone instead of crashing.
    """
    dc = _load_dc_config().get("data_cluster", {})
    training = dc.get("training", {}) if isinstance(dc, dict) else {}
    backbones = dc.get("backbones", {}) if isinstance(dc, dict) else {}
    model_cfg = dict(dc.get("model", {})) if isinstance(dc, dict) else {}

    default_backbone = (
        training.get("default_backbone")
        or model_cfg.get("backbone")
        or "convnext_small"
    )
    model_cfg["backbone"] = default_backbone

    info = backbones.get(default_backbone, {}) if isinstance(backbones, dict) else {}
    if isinstance(info, dict):
        bb_cfg = info.get("backbone_config")
        if bb_cfg:
            bb_path = Path(bb_cfg)
            if not bb_path.is_absolute():
                bb_path = (_REPO_ROOT / bb_path).resolve()
            model_cfg["backbone_config_path"] = str(bb_path)
        pretrained = info.get("pretrained_path")
        if pretrained:
            p = Path(pretrained)
            if not p.is_absolute():
                p = (_REPO_ROOT / p).resolve()
            if p.is_file():
                model_cfg["pretrained_path"] = str(p)

    data_cfg = dc.get("data", {}) if isinstance(dc, dict) else {}
    image_size = data_cfg.get("image_size", 224)
    if isinstance(image_size, list):
        image_size = int(max(image_size)) if image_size else 224
    else:
        image_size = int(image_size)
    return model_cfg, image_size


def compute_default_embeddings(
    image_paths: list[Path],
    mask_paths: list[Path | None] | None = None,
    batch_size: int = 16,
    device: str | None = None,
) -> np.ndarray:
    """Compute embeddings using the default pretrained backbone (no trained head)."""
    model_cfg, image_size = _build_default_model_config()
    return _compute_backbone_embeddings(
        model_config=model_cfg,
        image_paths=image_paths,
        image_size=image_size,
        mask_paths=mask_paths,
        batch_size=batch_size,
        device=device,
    )
