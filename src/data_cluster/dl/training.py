from __future__ import annotations

import copy
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from sqlalchemy.orm import Session

from data_cluster.app.config import Settings, get_settings
from data_cluster.app.database import SessionLocal
from data_cluster.app.models.db import CropImage, Experiment, ExperimentSample
from data_cluster.app.services.storage import experiment_checkpoint_run_dir
from data_cluster.dl.config_resolve import (
    build_platform_supcon_base,
    resolve_backbone_pretrained_path,
    resolve_embedding_source,
)
from data_cluster.dl.step_budget import recommend_total_steps
from data_cluster.dl.trainer import SupconTrainer
from embedding_model.utils.logging import setup_logger

TRAINING_ALGORITHM = "supcon_db_indexed_training"
WEIGHTS_FILENAME = "current_model.pth"
TRAINING_INFO_FILENAME = "training_info.csv"
DATASET_CSV_FILENAME = "dataset.csv"
DEFAULT_VAL_RATIO = 0.2
MIN_CLASSES_FOR_TRAINING = 2
MIN_CROPS_PER_CLASS = 2
# Web 小样本 finetune 默认 weight decay：常为冻结 backbone、只训 fusion/head，
# 且 total_steps 较短；supcon 全量预训练用的 0.05 对此场景偏强，0.01 更稳妥。
WEB_DEFAULT_WEIGHT_DECAY = 0.01
WEB_DEFAULT_DEVICE = "gpu"

# Global stop-flag registry: experiment_id -> True means "please stop"
_stop_flags: dict[str, bool] = {}


def request_stop(experiment_id: str) -> None:
    """Signal a running training job to stop at the next checkpoint."""
    _stop_flags[experiment_id] = True


def _checkpoint_relative_to_project(settings: Settings, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(settings.project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _url_to_fs_path(settings: Settings, url: str | None) -> Path | None:
    if not url:
        return None
    relative = str(url).removeprefix("/static/")
    return (settings.upload_dir / relative).resolve()


def _coerce_id_list(config: dict[str, Any] | None, *keys: str) -> list[str]:
    if not config:
        return []
    for key in keys:
        raw = config.get(key)
        if isinstance(raw, list):
            return [str(item) for item in raw if str(item).strip()]
    return []


def _coerce_float(config: dict[str, Any] | None, *keys: str, default: float) -> float:
    if not config:
        return default
    for key in keys:
        raw = config.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return default


def _coerce_int(config: dict[str, Any] | None, *keys: str, default: int) -> int:
    if not config:
        return default
    for key in keys:
        raw = config.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return default


def _coerce_bool(config: dict[str, Any] | None, *keys: str, default: bool) -> bool:
    if not config:
        return default
    for key in keys:
        raw = config.get(key)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            lowered = raw.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
    return default


def _coerce_device_choice(config: dict[str, Any] | None, *keys: str, default: str = WEB_DEFAULT_DEVICE) -> str:
    """解析训练设备选项：``cpu`` 或 ``gpu``。"""
    if config:
        for key in keys:
            raw = config.get(key)
            if isinstance(raw, str) and raw.strip():
                choice = raw.strip().lower()
                if choice in {"cpu", "gpu"}:
                    return choice
                raise ValueError(f"不支持的 device: {raw!r}，可选 cpu / gpu")
    normalized = str(default).strip().lower()
    return normalized if normalized in {"cpu", "gpu"} else WEB_DEFAULT_DEVICE


def resolve_training_device(choice: str) -> torch.device:
    """把 ``cpu`` / ``gpu`` 配置项解析为 ``torch.device``。"""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "训练配置 device=gpu，但当前环境 CUDA 不可用。"
                "请检查 GPU 驱动、CUDA 安装，或将 device 改为 cpu。"
            )
        return torch.device("cuda")
    raise ValueError(f"不支持的 device: {choice!r}，可选 cpu / gpu")


def _coerce_image_size(config: dict[str, Any] | None) -> int | list[int] | None:
    if not config:
        return None
    for key in ("imageSize", "image_size"):
        raw = config.get(key)
        if isinstance(raw, int):
            return raw
        if isinstance(raw, list) and raw and all(isinstance(v, int) for v in raw):
            return raw
    return None


def _alias_similarity_metric_block(metric_block: dict[str, Any] | None) -> None:
    if not isinstance(metric_block, dict):
        return
    if metric_block.get("PosSim") is None and metric_block.get("margin_pos_sim") is not None:
        metric_block["PosSim"] = metric_block["margin_pos_sim"]
    if metric_block.get("NegSim") is None and metric_block.get("margin_neg_sim") is not None:
        metric_block["NegSim"] = metric_block["margin_neg_sim"]
    if metric_block.get("Margin") is None and metric_block.get("margin") is not None:
        metric_block["Margin"] = metric_block["margin"]


def _alias_similarity_series(series: dict[str, Any] | None) -> None:
    if not isinstance(series, dict):
        return
    if series.get("PosSims") is None and series.get("posSims") is not None:
        series["PosSims"] = series["posSims"]
    if series.get("NegSims") is None and series.get("negSims") is not None:
        series["NegSims"] = series["negSims"]


def normalize_experiment_metrics(metrics: dict[str, Any] | None, *, _inplace: bool = False) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return metrics

    # deepcopy only when the caller needs the original preserved (default).
    # Internal callers that already own the dict pass _inplace=True to skip the copy.
    normalized = metrics if _inplace else copy.deepcopy(metrics)
    _alias_similarity_metric_block(normalized)
    _alias_similarity_metric_block(normalized.get("val"))
    _alias_similarity_series(normalized.get("liveSeries"))

    summary = normalized.get("summary")
    if isinstance(summary, dict):
        _alias_similarity_metric_block(summary)
        _alias_similarity_metric_block(summary.get("last_val_metrics"))
        _alias_similarity_series(summary.get("liveSeries"))

    return normalized


def _stable_crop_path(settings: Settings, crop: CropImage) -> Path:
    if crop.file_path:
        return Path(crop.file_path).resolve()
    guessed = _url_to_fs_path(settings, crop.url)
    if guessed is None:
        raise ValueError(f"Crop {crop.id} 缺失 file_path/url")
    return guessed


def _stable_mask_path(settings: Settings, crop: CropImage) -> Path | None:
    if crop.mask_path:
        return Path(crop.mask_path).resolve()
    return _url_to_fs_path(settings, crop.mask_url)


def _resolve_experiment_crops(db: Session, exp: Experiment, settings: Settings) -> list[CropImage]:
    if not exp.dataset_id:
        raise ValueError("训练任务缺少 dataset_id，无法构建 crop 训练集")

    config = exp.config if isinstance(exp.config, dict) else {}
    crop_ids = _coerce_id_list(
        config,
        "trainingCropIds",
        "training_crop_ids",
        "selectedCropIds",
        "cropIds",
    )

    query = (
        db.query(CropImage)
        .filter(CropImage.dataset_id == exp.dataset_id)
    )

    if crop_ids:
        rows = query.filter(CropImage.id.in_(crop_ids)).all()
        order = {crop_id: idx for idx, crop_id in enumerate(crop_ids)}
        rows.sort(key=lambda row: order.get(row.id, len(order)))
        return rows

    rows = query.all()
    rows.sort(key=lambda row: (row.source_image_id, row.instance_index, row.id))
    return rows


def _build_experiment_samples(
    db: Session,
    exp: Experiment,
    settings: Settings,
) -> list[ExperimentSample]:
    crops = _resolve_experiment_crops(db, exp, settings)
    if not crops:
        raise ValueError("当前实验没有可训练的 crop 样本，请先生成裁剪图并完成选择")

    crop_ids = [crop.id for crop in crops]
    category_rows = (
        db.query(
            CropImage.id,
            CropImage.source_image_id,
            CropImage.class_id,
        )
        .filter(CropImage.id.in_(crop_ids))
        .all()
    )
    category_name_rows = {
        crop.id: crop.defect_class.name
        for crop in crops
        if crop.defect_class is not None
    }
    category_by_crop = {
        row.id: {
            "source_image_id": row.source_image_id,
            "category_name": category_name_rows.get(row.id, ""),
        }
        for row in category_rows
    }

    grouped: dict[str, list[CropImage]] = defaultdict(list)
    for crop in crops:
        meta = category_by_crop.get(crop.id)
        if meta is None or not meta["category_name"]:
            continue
        grouped[str(meta["category_name"])].append(crop)

    if len(grouped) < MIN_CLASSES_FOR_TRAINING:
        raise ValueError(f"训练至少需要 {MIN_CLASSES_FOR_TRAINING} 个类别的 crop 样本")

    too_few = {
        name: len(crops_in_cat)
        for name, crops_in_cat in grouped.items()
        if len(crops_in_cat) < MIN_CROPS_PER_CLASS
    }
    if too_few:
        detail = "，".join(f"{name}: {n} 张" for name, n in sorted(too_few.items()))
        raise ValueError(
            f"每个类别至少需要 {MIN_CROPS_PER_CLASS} 张训练样本，以下类别不足（{detail}）"
        )

    samples: list[ExperimentSample] = []
    sort_order = 0
    for category_name in sorted(grouped):
        category_crops = sorted(
            grouped[category_name],
            key=lambda crop: (crop.source_image_id, crop.instance_index, crop.id),
        )

        for idx, crop in enumerate(category_crops):
            crop_path = _stable_crop_path(settings, crop)
            mask_path = _stable_mask_path(settings, crop)
            if not crop_path.exists():
                raise ValueError(f"训练 crop 文件不存在: {crop_path}")
            if mask_path is None or not mask_path.exists():
                raise ValueError(f"训练 mask 文件不存在: {mask_path}")

            meta = category_by_crop[crop.id]
            sample = ExperimentSample(
                experiment_id=exp.id,
                crop_image_id=crop.id,
                # 当前平台阶段：训练与验证使用同一份样本，先全部记为 train
                split="train",
                sort_order=sort_order,
                category_name=category_name,
                source_image_id=meta["source_image_id"],
                crop_url=crop.url,
                crop_path=str(crop_path),
                mask_url=crop.mask_url,
                mask_path=str(mask_path),
            )
            samples.append(sample)
            sort_order += 1

    return samples


def prepare_experiment_samples(db: Session, exp: Experiment, settings: Settings) -> tuple[int, int]:
    db.query(ExperimentSample).filter(ExperimentSample.experiment_id == exp.id).delete()
    samples = _build_experiment_samples(db, exp, settings)
    for sample in samples:
        db.add(sample)
    db.flush()

    train_count = sum(1 for sample in samples if sample.split == "train")
    val_count = sum(1 for sample in samples if sample.split == "val")

    config = dict(exp.config or {})
    config["resolvedTrainingCropIds"] = [sample.crop_image_id for sample in samples if sample.crop_image_id]
    config["resolvedCategoryNames"] = sorted({sample.category_name for sample in samples})
    config["resolvedTrainCount"] = train_count
    config["resolvedValCount"] = val_count
    exp.config = config
    return train_count, val_count


def _train_category_counts(exp: Experiment) -> list[int]:
    """训练集每个类别的样本数（用于 ``recommend_total_steps`` 的覆盖度估算）。"""
    counter = Counter(sample.category_name for sample in exp.samples if sample.split == "train")
    return list(counter.values())


def get_experiment_sample_counts(exp: Experiment) -> tuple[int, int, int]:
    total = len(exp.samples)
    train_count = sum(1 for sample in exp.samples if sample.split == "train")
    val_count = sum(1 for sample in exp.samples if sample.split == "val")
    return total, train_count, val_count


def _write_training_info_csv(
    path: Path,
    exp: Experiment,
    weights_path: Path,
    settings: Settings,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_samples, train_samples, val_samples = get_experiment_sample_counts(exp)
    rows: list[tuple[str, str]] = [
        ("experiment_id", exp.id),
        ("experiment_name", exp.name),
        ("model_display_name", exp.model),
        ("dataset_id", exp.dataset_id or ""),
        ("dataset_name", exp.dataset),
        ("algorithm", TRAINING_ALGORITHM),
        ("status", exp.status),
        ("duration", exp.duration or ""),
        ("accuracy", exp.accuracy or ""),
        ("progress", str(exp.progress)),
        ("sample_count", str(total_samples)),
        ("train_sample_count", str(train_samples)),
        ("val_sample_count", str(val_samples)),
        ("checkpoint_absolute_path", str(weights_path.resolve())),
        ("checkpoint_relative_path", _checkpoint_relative_to_project(settings, weights_path)),
        ("weights_filename", WEIGHTS_FILENAME),
    ]
    if exp.config:
        rows.append(("hyperparameters_json", json.dumps(exp.config, ensure_ascii=False)))
    if exp.metrics:
        rows.append(("metrics_json", json.dumps(exp.metrics, ensure_ascii=False)))

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(rows)


def _write_dataset_csv(path: Path, exp: Experiment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "category_name",
        "crop_image_id",
        "source_image_id",
        "image_path",
        "mask_path",
        "crop_url",
        "mask_url",
    ]
    sorted_samples = sorted(exp.samples, key=lambda item: item.sort_order)
    has_val_split = any(sample.split == "val" for sample in sorted_samples)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sample in sorted_samples:
            w.writerow(
                {
                    "split": sample.split,
                    "category_name": sample.category_name,
                    "crop_image_id": sample.crop_image_id or "",
                    "source_image_id": sample.source_image_id or "",
                    "image_path": sample.crop_path,
                    "mask_path": sample.mask_path or "",
                    "crop_url": sample.crop_url or "",
                    "mask_url": sample.mask_url or "",
                }
            )
            # 如果未单独划分 val，则写一份同样的样本给 val，确保“训练=验证同一份”。
            if not has_val_split and sample.split == "train":
                w.writerow(
                    {
                        "split": "val",
                        "category_name": sample.category_name,
                        "crop_image_id": sample.crop_image_id or "",
                        "source_image_id": sample.source_image_id or "",
                        "image_path": sample.crop_path,
                        "mask_path": sample.mask_path or "",
                        "crop_url": sample.crop_url or "",
                        "mask_url": sample.mask_url or "",
                    }
                )


def _build_data_config(exp: Experiment, manifest_path: Path, supcon_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = exp.config if isinstance(exp.config, dict) else {}
    sup_data = ((supcon_config or {}).get("supcon") or {}).get("data") or {}
    default_mask_dilation = sup_data.get("mask_dilation") if isinstance(sup_data.get("mask_dilation"), dict) else {}
    mask_enabled = config.get("maskDilationEnabled")
    if mask_enabled is None:
        mask_enabled = config.get("mask_dilation_enabled")

    mask_dilation = {
        "enabled": bool(default_mask_dilation.get("enabled", False)),
    }
    if isinstance(mask_enabled, bool):
        mask_dilation["enabled"] = mask_enabled
    elif isinstance(mask_enabled, str):
        lowered = mask_enabled.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            mask_dilation["enabled"] = True
        elif lowered in {"0", "false", "no", "off"}:
            mask_dilation["enabled"] = False

    return {
        "name": exp.name,
        "dataset_type": "data_cluster_triplets",
        "manifest_path": str(manifest_path.resolve()),
        "mask_dilation": mask_dilation,
    }


def _load_dc_config() -> dict[str, Any]:
    from data_cluster.dl.config_resolve import load_dc_config

    return load_dc_config()


def _resolve_backbone_pretrained_path(backbone_name: str, exp_config: dict[str, Any] | None = None) -> str | None:
    return resolve_backbone_pretrained_path(backbone_name, exp_config)


def _build_supcon_config(exp: Experiment, run_dir: Path) -> dict[str, Any]:
    """从 supcon_config.yaml + DataCluster 平台配置 + 实验参数构建 SupconTrainer 配置。"""
    config = exp.config if isinstance(exp.config, dict) else {}
    sup = build_platform_supcon_base()

    sup["output"] = {
        "checkpoint_dir": str(run_dir.resolve()),
        "log_dir": str((run_dir / "logs").resolve()),
    }

    # --- 数据参数覆盖 ---
    image_size = _coerce_image_size(config)
    if image_size is not None:
        sup["data"]["image_size"] = image_size
    sup["data"]["batch_size"] = _coerce_int(config, "batchSize", "batch_size", default=sup["data"].get("batch_size", 16))

    # --- 训练参数覆盖 ---
    # 默认训练总量按「类别对共现覆盖」理论估算（见 step_budget.py），而不是
    # 写死的固定值：数据集越大/类别越不均衡，所需训练量不同；配合下方的
    # early stopping，这里给得宽松一些也无妨——真正训练不到这个上限就会
    # 因为验证 margin 不再提升而提前停止。用户仍可显式传 totalSteps 覆盖。
    try:
        smart_default_total_steps = recommend_total_steps(
            _train_category_counts(exp), sup["data"]["batch_size"]
        )
    except Exception:
        smart_default_total_steps = int(sup["training"].get("total_steps", 1000))
    sup["training"]["total_steps"] = max(
        1, _coerce_int(config, "totalSteps", "total_steps", default=smart_default_total_steps)
    )
    sup["training"]["learning_rate"] = _coerce_float(
        config, "learningRate", "learning_rate", default=float(sup["training"].get("learning_rate", 1e-4))
    )
    sup["training"]["weight_decay"] = _coerce_float(
        config, "weightDecay", "weight_decay", default=WEB_DEFAULT_WEIGHT_DECAY
    )
    sup["training"]["use_amp"] = _coerce_bool(
        config, "useAmp", "use_amp", default=bool(sup["training"].get("use_amp", False))
    )
    # 早停：连续 N 次验证 margin 未提升（提升幅度 < min_delta）即提前结束训练，
    # 避免小数据集把 total_steps 上限的宽松预算真的全部跑完。0 = 关闭早停。
    sup["training"]["early_stop_patience"] = max(
        0,
        _coerce_int(
            config, "earlyStopPatience", "early_stop_patience",
            default=int(sup["training"].get("early_stop_patience", 5)),
        ),
    )
    sup["training"]["early_stop_min_delta"] = _coerce_float(
        config, "earlyStopMinDelta", "early_stop_min_delta",
        default=float(sup["training"].get("early_stop_min_delta", 0.005)),
    )
    # SupconTrainer 需要 lr_scheduler 键（'cosine' | 'step'），data_cluster.yaml 默认 cosine
    lr_scheduler = config.get("lrScheduler") or config.get("lr_scheduler")
    if isinstance(lr_scheduler, str) and lr_scheduler.strip():
        sup["training"]["lr_scheduler"] = lr_scheduler.strip().lower()
    else:
        sup["training"].setdefault("lr_scheduler", "cosine")

    # 是否冻结 backbone（整个训练过程）：为 False 时 backbone 与其余参数
    # 使用同一个全局 learning_rate，不再单独设置 LR ratio。
    sup["training"]["freeze_backbone"] = _coerce_bool(
        config, "freezeBackbone", "freeze_backbone",
        default=bool(sup["training"].get("freeze_backbone", False)),
    )
    # true：每个检查点都做验证（含早停）；false：训练过程不验证，结束后 eval 一次。
    sup["training"]["use_eval"] = _coerce_bool(
        config, "useEval", "use_eval",
        default=bool(sup["training"].get("use_eval", True)),
    )
    sup["training"]["device"] = _coerce_device_choice(
        config, "device", "trainingDevice",
        default=str(sup["training"].get("device", WEB_DEFAULT_DEVICE)),
    )
    sup.setdefault("inference", {})
    sup["inference"]["embedding_source"] = resolve_embedding_source(sup, config)

    # --- 模型参数覆盖（架构默认来自 supcon_config.yaml）---
    backbone = config.get("backbone") or config.get("modelName") or config.get("model_name")
    if isinstance(backbone, str) and backbone.strip():
        sup["model"]["backbone"] = backbone.strip()
    elif not sup["model"].get("backbone"):
        sup["model"]["backbone"] = "convnext_tiny"

    embedding_dim = config.get("embeddingDim") or config.get("embedding_dim")
    if isinstance(embedding_dim, int) and embedding_dim > 0:
        sup["model"]["embedding_dim"] = embedding_dim

    # 预训练权重路径：从 data_cluster.yaml 的 pretrained_models.<backbone>.pretrained_path 加载
    backbone_name = sup["model"].get("backbone", "convnext_tiny")
    pretrained_path = _resolve_backbone_pretrained_path(backbone_name, config)
    if pretrained_path:
        sup["model"]["pretrained_path"] = pretrained_path
        print(f"[training] backbone='{backbone_name}' 默认加载预训练权重: {pretrained_path}")
    else:
        print(f"[training] backbone='{backbone_name}' 未找到预训练权重路径，将从随机/HF 初始化")

    # --- MoCo 参数覆盖（默认来自 supcon_config.yaml）---
    use_moco = _coerce_bool(config, "useMoCo", "use_moco", default=sup["moco"].get("enabled", False))
    sup["moco"]["enabled"] = use_moco
    queue_size = config.get("queueSize") or config.get("queue_size")
    if queue_size is not None:
        sup["moco"]["queue_size"] = queue_size

    return {"supcon": sup}


def _format_quality(summary: dict[str, Any] | None) -> str:
    """实验质量摘要（用于列表展示）：以验证集 Margin 为主，缺失时退回训练损失。"""
    if not summary:
        return "---"
    val_metrics = summary.get("last_val_metrics")
    if isinstance(val_metrics, dict) and val_metrics.get("margin") is not None:
        return f"Margin {float(val_metrics['margin']):.4f}"
    train_metrics = summary.get("last_train_metrics")
    if isinstance(train_metrics, dict) and train_metrics.get("loss") is not None:
        return f"loss={float(train_metrics['loss']):.4f}"
    return "---"


def run_training_job(experiment_id: str) -> None:
    settings = get_settings()
    settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    db = SessionLocal()
    started_at = time.time()
    try:
        exp = (
            db.query(Experiment)
            .filter(Experiment.id == experiment_id)
            .first()
        )
        if not exp:
            return

        # 只更新「本次运行」字段；结果字段（status/metrics/checkpoint_path）保持上一次成功的值，
        # 重训进行中/中止/失败都不会清掉上一次成功的 dashboard 与推理可用模型。
        exp.run_status = "Running"
        exp.run_progress = 0.0
        exp.run_metrics = {"stage": "preparing_dataset", "startedAt": started_at}
        db.commit()

        if not exp.samples:
            prepare_experiment_samples(db, exp, settings)
            db.commit()
            db.refresh(exp)

        run_dir = experiment_checkpoint_run_dir(settings, exp.model, exp.id)
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset_csv = run_dir / DATASET_CSV_FILENAME
        _write_dataset_csv(dataset_csv, exp)

        # 验证集与训练集使用同一份样本；是否在每个检查点 eval 由
        # training.use_eval 控制（false 时仅在训练结束后 eval 一次）。
        use_eval = True
        supcon_config = _build_supcon_config(exp, run_dir)
        logger = setup_logger(
            f"experiment_{exp.id}",
            log_dir=supcon_config["supcon"]["output"]["log_dir"],
        )
        device_choice = str(supcon_config["supcon"]["training"].get("device", WEB_DEFAULT_DEVICE))
        device = resolve_training_device(device_choice)
        logger.info(f"训练设备: {device} (config device={device_choice})")
        data_config = _build_data_config(exp, dataset_csv, supcon_config=supcon_config)
        pretrained_path = None
        if isinstance(exp.config, dict):
            raw_pretrained = exp.config.get("pretrainedPath") or exp.config.get("pretrained_path")
            if isinstance(raw_pretrained, str) and raw_pretrained.strip():
                pretrained_path = raw_pretrained.strip()

        # Accumulate per-checkpoint series for live chart rendering. Training
        # loss and validation now share the same checkpoint cadence (see
        # trainer.py), so all series below are always the same length and
        # already aligned to `live_steps` — no separate epoch bookkeeping needed.
        live_steps: list[int] = []
        live_train_losses: list[float] = []
        live_val_losses: list[float] = []
        live_margins: list[float] = []
        live_pos_sims: list[float] = []
        live_neg_sims: list[float] = []
        training_started_at: float | None = None

        def on_progress(payload: dict[str, Any]) -> bool:
            """Called at each training checkpoint (~10 per run). Returns True
            to signal the trainer to stop. Checkpoints are infrequent enough
            that every callback invocation writes straight to the DB."""
            nonlocal training_started_at
            if training_started_at is None:
                training_started_at = time.time()

            tm = payload.get("train_metrics") or {}
            vm = normalize_experiment_metrics({"val": payload.get("val_metrics") or {}})["val"]
            step = int(payload.get("step", 0) or 0)

            if tm.get("loss") is not None:
                live_train_losses.append(float(tm["loss"]))
                live_steps.append(step)
            if vm.get("loss") is not None:
                live_val_losses.append(float(vm["loss"]))
                if vm.get("margin") is not None:
                    live_margins.append(float(vm["margin"]))
                if vm.get("PosSim") is not None:
                    live_pos_sims.append(float(vm["PosSim"]))
                if vm.get("NegSim") is not None:
                    live_neg_sims.append(float(vm["NegSim"]))

            # 先检查停止标志（不需要 DB IO，极低开销）
            if _stop_flags.get(experiment_id):
                return True

            total_steps = int(payload.get("total_steps", 1) or 1)
            current = db.get(Experiment, experiment_id)
            if not current:
                return False
            current.run_progress = float(payload.get("progress", 0.0))
            current.run_metrics = normalize_experiment_metrics(
                {
                    "stage": "training",
                    "step": step,
                    "totalSteps": total_steps,
                    "startedAt": started_at,
                    "trainingStartedAt": training_started_at,
                    "train": tm,
                    "val": vm,
                    "liveSeries": {
                        "steps": list(live_steps),
                        "trainLosses": list(live_train_losses),
                        "valLosses": list(live_val_losses),
                        "margins": list(live_margins),
                        "posSims": list(live_pos_sims),
                        "negSims": list(live_neg_sims),
                    },
                },
                _inplace=True,
            )
            db.commit()
            return False

        trainer = SupconTrainer(
            supcon_config=supcon_config,
            logger=logger,
            data_config_paths=[data_config],
            use_eval=use_eval,
            pretrained_path=pretrained_path,
            device=device,
            progress_callback=on_progress,
        )
        summary = trainer.run()

        exp = db.get(Experiment, experiment_id)
        if not exp:
            return

        was_stopped = _stop_flags.pop(experiment_id, False)
        checkpoint_path = Path(summary["checkpoint_path"])

        # Embed the accumulated live series into the final summary so the
        # completed dashboard can render per-checkpoint loss / margin curves.
        summary["liveSeries"] = {
            "steps": live_steps,
            "trainLosses": live_train_losses,
            "valLosses": live_val_losses,
            "margins": live_margins,
            "posSims": live_pos_sims,
            "negSims": live_neg_sims,
        }
        summary = normalize_experiment_metrics({"summary": summary}, _inplace=True)["summary"]

        if was_stopped:
            # 中止视为「未完成」：不提升为结果，仅记录本次运行状态；保留上一次成功的
            # 结果字段与 checkpoint，使 dashboard 继续显示上一次成功页 + 中止横幅。
            exp.run_status = "Stopped"
            exp.run_metrics = normalize_experiment_metrics(
                {
                    "stage": "stopped",
                    "summary": summary,
                    "stepsRun": summary.get("steps_run", live_steps[-1] if live_steps else 0),
                },
                _inplace=True,
            )
            db.commit()
        else:
            # 成功完成：提升为 last-good 结果，并清空本次运行的实时指标。
            exp.run_status = "Completed"
            exp.run_progress = 100.0
            exp.run_metrics = {"stage": "completed"}
            exp.status = "Completed"
            exp.progress = 100.0
            exp.duration = f"{int(time.time() - started_at)}s"
            exp.accuracy = _format_quality(summary)
            if checkpoint_path.exists():
                exp.checkpoint_path = str(checkpoint_path.resolve())
            exp.metrics = normalize_experiment_metrics(
                {
                    "stage": "completed",
                    "summary": summary,
                    "trainSampleCount": sum(1 for sample in exp.samples if sample.split == "train"),
                    "valSampleCount": sum(1 for sample in exp.samples if sample.split == "val"),
                },
                _inplace=True,
            )
            db.commit()

            _write_training_info_csv(run_dir / TRAINING_INFO_FILENAME, exp, checkpoint_path, settings)
    except Exception as exc:
        _stop_flags.pop(experiment_id, None)
        exp = db.get(Experiment, experiment_id)
        if exp:
            # 失败视为「未完成」：只更新本次运行状态，保留上一次成功的结果与 checkpoint。
            exp.run_status = "Failed"
            exp.run_metrics = {
                "stage": "failed",
                "error": str(exc),
            }
            db.commit()
    finally:
        db.close()
