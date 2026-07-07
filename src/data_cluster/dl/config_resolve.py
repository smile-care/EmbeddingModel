"""Resolve training/inference config: SupCon architecture from supcon_config.yaml,
platform paths and web UI defaults from data_cluster.yaml."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from embedding_model.utils.config_loader import load_config
from embedding_model.supcon.datasets.augmentations import deep_merge_dict

_REPO_ROOT = Path(__file__).resolve().parents[3]
DC_CONFIG_PATH = _REPO_ROOT / "configs" / "data_cluster.yaml"
DEFAULT_SUPCON_CONFIG_PATH = _REPO_ROOT / "configs" / "supcon_config.yaml"
DEFAULT_EMBEDDING_SOURCE = "representations"

_EMBEDDING_SOURCE_ALIASES = {
    "representations": "representations",
    "representation": "representations",
    "features": "representations",
    "feature": "representations",
    "h": "representations",
    "projections": "projections",
    "projection": "projections",
    "embeddings": "projections",
    "embedding": "projections",
    "z": "projections",
}


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


def normalize_embedding_source(value: Any, default: str = DEFAULT_EMBEDDING_SOURCE) -> str:
    """Normalize configured embedding source to a model output key."""
    if value is None:
        value = default
    key = str(value).strip().lower()
    try:
        return _EMBEDDING_SOURCE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted({"representations", "projections"}))
        raise ValueError(f"embedding_source 必须是 {valid}，当前为 {value!r}") from exc


def resolve_embedding_source(
    base_config: dict[str, Any] | None = None,
    exp_config: dict[str, Any] | None = None,
) -> str:
    """Resolve app embedding source from base config, with experiment override."""
    base_config = base_config if isinstance(base_config, dict) else {}
    inference_cfg = base_config.get("inference", {})
    default = DEFAULT_EMBEDDING_SOURCE
    if isinstance(inference_cfg, dict):
        default = normalize_embedding_source(
            inference_cfg.get("embedding_source", inference_cfg.get("embeddingSource")),
            default=DEFAULT_EMBEDDING_SOURCE,
        )

    if isinstance(exp_config, dict):
        override = exp_config.get("embeddingSource", exp_config.get("embedding_source"))
        if override is not None:
            return normalize_embedding_source(override, default=default)

    return default


def _get_pretrained_models_section(dc: dict[str, Any]) -> dict[str, Any]:
    """``pretrained_models`` block (legacy key ``backbones`` still accepted)."""
    section = dc.get("pretrained_models")
    if isinstance(section, dict) and section:
        return section
    legacy = dc.get("backbones", {})
    return legacy if isinstance(legacy, dict) else {}


def resolve_backbone_pretrained_path(
    backbone_name: str,
    exp_config: dict[str, Any] | None = None,
) -> str | None:
    """Resolve pretrained weights for an embedding model variant.

    Priority:
      1. explicit experiment override
      2. data_cluster.yaml ``pretrained_models.<name>.pretrained_path``
      3. repo default ``pretrain_ckpts/<name>.pth`` when present
    """
    dc = load_dc_section()
    models = _get_pretrained_models_section(dc)

    if isinstance(exp_config, dict):
        raw = exp_config.get("pretrainedPath") or exp_config.get("pretrained_path")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    info = models.get(backbone_name, {})
    if isinstance(info, dict) and info.get("pretrained_path"):
        return str(info["pretrained_path"])
    default_path = _REPO_ROOT / "pretrain_ckpts" / f"{backbone_name}.pth"
    if default_path.is_file():
        return str(default_path)
    return None


def resolve_backbone_config_path(backbone_name: str) -> str | None:
    dc = load_dc_section()
    info = _get_pretrained_models_section(dc).get(backbone_name, {})
    if isinstance(info, dict):
        raw = info.get("backbone_config")
        if raw:
            p = Path(str(raw))
            if not p.is_absolute():
                p = (_REPO_ROOT / p).resolve()
            return str(p)

    default_path = _REPO_ROOT / "configs" / "backbone" / f"{backbone_name}.yaml"
    if default_path.is_file():
        return str(default_path)
    return None


def list_default_models() -> list[dict[str, str]]:
    """Configured DINOv3 RAW models for inference (``default_models`` block)."""
    dc = load_dc_section()
    raw = dc.get("default_models", {})
    if not isinstance(raw, dict):
        return []
    out: list[dict[str, str]] = []
    for model_id, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or model_id
        out.append({"id": str(model_id), "name": str(name)})
    return out


def default_model_ids() -> frozenset[str]:
    return frozenset(m["id"] for m in list_default_models())


def resolve_default_model(model_id: str) -> tuple[str, str | None, str | None]:
    """Resolve one DINOv3 RAW model from ``default_models.<model_id>``.

    Returns ``(backbone_name, backbone_config_path, pretrained_path)``.
    """
    dc = load_dc_section()
    models = dc.get("default_models", {})
    if not isinstance(models, dict):
        raise ValueError(f"Unknown default model: {model_id}")
    entry = models.get(model_id)
    if not isinstance(entry, dict):
        raise ValueError(f"Unknown default model: {model_id}")

    backbone = entry.get("backbone") or model_id
    backbone_config_path = resolve_backbone_config_path(backbone)

    raw = entry.get("pretrained_path")
    if isinstance(raw, str) and raw.strip():
        p = Path(raw.strip())
        pretrained_path = str(p if p.is_absolute() else (_REPO_ROOT / p).resolve())
    else:
        default_path = _REPO_ROOT / "pretrain_ckpts" / f"{backbone}.pth"
        pretrained_path = str(default_path) if default_path.is_file() else None

    return backbone, backbone_config_path, pretrained_path


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
        "inference": copy.deepcopy(sup.get("inference", {})),
    }

    dc_data = dc.get("data", {})
    if isinstance(dc_data, dict):
        for key in (
            "image_size",
            "batch_size",
        ):
            if key in dc_data:
                merged["data"][key] = copy.deepcopy(dc_data[key])
        if isinstance(dc_data.get("train_augmentation"), dict):
            base_aug = merged["data"].get("train_augmentation")
            if isinstance(base_aug, dict):
                merged["data"]["train_augmentation"] = deep_merge_dict(
                    base_aug,
                    dc_data["train_augmentation"],
                )
            else:
                merged["data"]["train_augmentation"] = copy.deepcopy(dc_data["train_augmentation"])

    dc_training = dc.get("training", {})
    if isinstance(dc_training, dict):
        for key in (
            "learning_rate",
            "use_amp",
            "early_stop_patience",
            "early_stop_min_delta",
            "lr_scheduler",
            "freeze_backbone",
            "use_eval",
            "device",
        ):
            if key in dc_training:
                merged["training"][key] = dc_training[key]

    default_backbone = (dc_training or {}).get("default_backbone")
    if isinstance(default_backbone, str) and default_backbone.strip():
        merged["model"]["backbone"] = default_backbone.strip()

    dc_inference = dc.get("inference", {})
    if isinstance(dc_inference, dict):
        for key in ("embedding_source", "embeddingSource"):
            if key in dc_inference:
                merged["inference"]["embedding_source"] = normalize_embedding_source(dc_inference[key])

    merged["inference"]["embedding_source"] = resolve_embedding_source(merged)

    return merged
