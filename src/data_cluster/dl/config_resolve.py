"""Resolve training/inference config: SupCon architecture from supcon_config.yaml,
platform paths and web UI defaults from data_cluster.yaml."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from embedding_model.utils.config_loader import load_config

_REPO_ROOT = Path(__file__).resolve().parents[3]
DC_CONFIG_PATH = _REPO_ROOT / "configs" / "data_cluster.yaml"
DEFAULT_SUPCON_CONFIG_PATH = _REPO_ROOT / "configs" / "supcon_config.yaml"


def load_dc_config() -> dict[str, Any]:
    try:
        return load_config(str(DC_CONFIG_PATH))
    except Exception:
        return {}


def supcon_config_path() -> Path:
    dc = load_dc_config().get("data_cluster", {})
    raw = dc.get("supcon_config_path", "configs/supcon_config.yaml")
    p = Path(str(raw))
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def load_supcon_section() -> dict[str, Any]:
    """``supcon`` block from supcon_config.yaml (model / moco / loss / …)."""
    cfg = load_config(str(supcon_config_path()))
    section = cfg.get("supcon", {})
    return copy.deepcopy(section) if isinstance(section, dict) else {}


def load_dc_section() -> dict[str, Any]:
    section = load_dc_config().get("data_cluster", {})
    return copy.deepcopy(section) if isinstance(section, dict) else {}


def resolve_backbone_pretrained_path(
    backbone_name: str,
    exp_config: dict[str, Any] | None = None,
) -> str | None:
    """Platform checkpoint path from data_cluster.yaml ``backbones``."""
    dc = load_dc_section()
    backbones = dc.get("backbones", {})

    if isinstance(exp_config, dict):
        raw = exp_config.get("pretrainedPath") or exp_config.get("pretrained_path")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    info = backbones.get(backbone_name, {})
    if isinstance(info, dict) and info.get("pretrained_path"):
        return str(info["pretrained_path"])
    return None


def resolve_backbone_config_path(backbone_name: str) -> str | None:
    dc = load_dc_section()
    info = dc.get("backbones", {}).get(backbone_name, {})
    if not isinstance(info, dict):
        return None
    raw = info.get("backbone_config")
    if not raw:
        return None
    p = Path(str(raw))
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return str(p)


def build_platform_supcon_base() -> dict[str, Any]:
    """Merge SupCon defaults with DataCluster platform overrides (not exp-specific).

    Architecture (model / moco / loss) comes entirely from supcon_config.yaml.
    data_cluster.yaml supplies backbone checkpoint paths, web training defaults,
    and DataLoader-oriented data settings.
    """
    sup = load_supcon_section()
    dc = load_dc_section()

    merged: dict[str, Any] = {
        "data": copy.deepcopy(sup.get("data", {})),
        "training": copy.deepcopy(sup.get("training", {})),
        "model": copy.deepcopy(sup.get("model", {})),
        "moco": copy.deepcopy(sup.get("moco", {})),
        "loss": copy.deepcopy(sup.get("loss", {})),
        "training_strategy": copy.deepcopy(sup.get("training_strategy", {})),
    }

    dc_data = dc.get("data", {})
    if isinstance(dc_data, dict):
        for key in (
            "image_size",
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
            "repeat_factor",
            "mask_dilation",
        ):
            if key in dc_data:
                merged["data"][key] = copy.deepcopy(dc_data[key])

    dc_training = dc.get("training", {})
    if isinstance(dc_training, dict):
        for key in (
            "epochs",
            "learning_rate",
            "weight_decay",
            "backbone_lr_ratio",
            "save_interval",
            "use_amp",
            "eval_interval",
            "lr_scheduler",
        ):
            if key in dc_training:
                merged["training"][key] = dc_training[key]

    dc_strategy = dc.get("training_strategy", {})
    if isinstance(dc_strategy, dict) and "freeze_backbone_epochs" in dc_strategy:
        merged["training_strategy"]["freeze_backbone_epochs"] = dc_strategy[
            "freeze_backbone_epochs"
        ]

    default_backbone = (dc_training or {}).get("default_backbone")
    if isinstance(default_backbone, str) and default_backbone.strip():
        merged["model"]["backbone"] = default_backbone.strip()

    return merged
