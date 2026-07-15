"""Configuration loading and validation for self-supervised training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SUPPORTED_METHODS = {"vicreg", "dino"}


def load_selfsup_config(path: str | Path) -> dict[str, Any]:
    """Load a self-supervised YAML config and fail early on invalid values."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"自监督配置不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict) or not isinstance(raw.get("selfsup"), dict):
        raise ValueError("配置根节点必须包含 selfsup 字典")

    config = raw["selfsup"]
    method = str(config.get("method", "")).lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"selfsup.method 必须是 {sorted(SUPPORTED_METHODS)}，当前为 {method!r}")

    for section in ("model", "data", "training", "output", method):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"selfsup.{section} 必须是字典")

    _require_positive(config["data"], "batch_size", "selfsup.data")
    _require_positive(config["training"], "total_steps", "selfsup.training")
    _require_positive(config["training"], "learning_rate", "selfsup.training")
    _require_positive(config["model"], "representation_dim", "selfsup.model")
    _require_positive(config["model"], "projection_dim", "selfsup.model")

    total_steps = int(config["training"]["total_steps"])
    warmup_steps = int(config["training"].get("warmup_steps", 0))
    freeze_steps = int(config["training"].get("backbone_freeze_steps", 0))
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps 必须满足 0 <= warmup_steps < total_steps")
    if not 0 <= freeze_steps <= total_steps:
        raise ValueError("backbone_freeze_steps 必须在 [0, total_steps] 内")
    backbone_lr_ratio = float(config["training"].get("backbone_lr_ratio", 0.05))
    if not 0.0 <= backbone_lr_ratio <= 1.0:
        raise ValueError("backbone_lr_ratio 必须在 [0, 1] 内")

    image_size = config["data"].get("image_size", 224)
    if not isinstance(image_size, int) or image_size <= 0:
        raise ValueError(f"selfsup.data.image_size 必须是正整数，当前为 {image_size!r}")
    sampling = str(config["data"].get("sampling", "uniform"))
    if sampling != "uniform":
        raise ValueError(f"selfsup.data.sampling 当前只支持 uniform，当前为 {sampling!r}")

    if method == "vicreg":
        for key in ("invariance_weight", "variance_weight", "covariance_weight", "variance_target"):
            _require_positive(config["vicreg"], key, "selfsup.vicreg", allow_zero=key != "variance_target")
    else:
        _require_positive(config["dino"], "num_prototypes", "selfsup.dino")
        _require_positive(config["dino"], "student_temperature", "selfsup.dino")
        _require_positive(config["dino"], "teacher_temperature_start", "selfsup.dino")
        _require_positive(config["dino"], "teacher_temperature_end", "selfsup.dino")
        _require_probability(config["dino"], "teacher_momentum_start", "selfsup.dino")
        _require_probability(config["dino"], "teacher_momentum_end", "selfsup.dino")
        _require_probability(config["dino"], "center_momentum", "selfsup.dino")
        if config["dino"]["teacher_momentum_start"] > config["dino"]["teacher_momentum_end"]:
            raise ValueError("teacher_momentum_start 不能大于 teacher_momentum_end")

    return config


def _require_positive(
    section: dict[str, Any],
    key: str,
    prefix: str,
    *,
    allow_zero: bool = False,
) -> None:
    value = section.get(key)
    valid = isinstance(value, (int, float)) and (value >= 0 if allow_zero else value > 0)
    if not valid:
        operator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{prefix}.{key} 必须 {operator}，当前为 {value!r}")


def _require_probability(section: dict[str, Any], key: str, prefix: str) -> None:
    value = section.get(key)
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{prefix}.{key} 必须在 [0, 1]，当前为 {value!r}")
