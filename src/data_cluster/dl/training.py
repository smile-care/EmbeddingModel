from __future__ import annotations

import copy
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from data_cluster.app.config import Settings, get_settings
from data_cluster.app.database import SessionLocal
from data_cluster.app.models.db import CropImage, Experiment, ExperimentSample
from data_cluster.app.services.storage import experiment_checkpoint_run_dir
from data_cluster.dl.trainer import SupconTrainer
from embedding_model.utils.config_loader import load_config
from embedding_model.utils.logging import setup_logger

TRAINING_ALGORITHM = "supcon_db_indexed_training"
WEIGHTS_FILENAME = "current_model.pth"
TRAINING_INFO_FILENAME = "training_info.csv"
DATASET_CSV_FILENAME = "dataset.csv"
DEFAULT_VAL_RATIO = 0.2
MIN_CLASSES_FOR_TRAINING = 2
MIN_CROPS_PER_CLASS = 2
# DataCluster 自有配置文件
DC_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "data_cluster.yaml"
)

# Global stop-flag registry: experiment_id -> True means "please stop"
_stop_flags: dict[str, bool] = {}


def request_stop(experiment_id: str) -> None:
    """Signal a running training job to stop after the current epoch."""
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
    mask_method = config.get("maskDilationMethod") or config.get("mask_dilation_method")
    mask_enabled = config.get("maskDilationEnabled")
    if mask_enabled is None:
        mask_enabled = config.get("mask_dilation_enabled")
    mask_fast_gamma = config.get("maskDilationFastGamma")
    if mask_fast_gamma is None:
        mask_fast_gamma = config.get("mask_dilation_fast_gamma")

    mask_dilation = {
        "enabled": bool(default_mask_dilation.get("enabled", True)),
        "method": str(default_mask_dilation.get("method", "fast_pool")).strip().lower(),
        "fast_gamma": float(default_mask_dilation.get("fast_gamma", 0.8)),
    }
    if isinstance(mask_enabled, bool):
        mask_dilation["enabled"] = mask_enabled
    elif isinstance(mask_enabled, str):
        lowered = mask_enabled.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            mask_dilation["enabled"] = True
        elif lowered in {"0", "false", "no", "off"}:
            mask_dilation["enabled"] = False
    if isinstance(mask_method, str) and mask_method.strip():
        mask_dilation["method"] = mask_method.strip().lower()
    if mask_fast_gamma is not None:
        try:
            mask_dilation["fast_gamma"] = float(mask_fast_gamma)
        except (TypeError, ValueError):
            pass

    return {
        "name": exp.name,
        "dataset_type": "data_cluster_triplets",
        "manifest_path": str(manifest_path.resolve()),
        "mask_dilation": mask_dilation,
    }


def _load_dc_config() -> dict[str, Any]:
    """加载 DataCluster 平台配置。"""
    try:
        return load_config(str(DC_CONFIG_PATH))
    except Exception:
        return {}


def _resolve_backbone_pretrained_path(backbone_name: str, exp_config: dict[str, Any] | None = None) -> str | None:
    """从 DataCluster 配置中解析 backbone 对应的预训练权重路径。"""
    dc_cfg = _load_dc_config()
    backbones = dc_cfg.get("data_cluster", {}).get("backbones", {})

    # exp_config 中可显式指定 pretrainedPath
    if isinstance(exp_config, dict):
        raw = exp_config.get("pretrainedPath") or exp_config.get("pretrained_path")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

    # 从 data_cluster.yaml 的 backbones 段查找
    info = backbones.get(backbone_name, {})
    if isinstance(info, dict) and info.get("pretrained_path"):
        return str(info["pretrained_path"])

    return None


def _build_supcon_config(exp: Experiment, run_dir: Path) -> dict[str, Any]:
    """从 DataCluster 配置文件 + 实验参数构建 SupconTrainer 所需配置。"""
    config = exp.config if isinstance(exp.config, dict) else {}
    dc_cfg = _load_dc_config()
    dc = dc_cfg.get("data_cluster", {}) if dc_cfg else {}

    # 以 data_cluster.yaml 中的 training / model / moco / loss / data 段为默认值
    sup = {
        "data": copy.deepcopy(dc.get("data", {})),
        "training": copy.deepcopy(dc.get("training", {})),
        "model": copy.deepcopy(dc.get("model", {})),
        "moco": copy.deepcopy(dc.get("moco", {})),
        "loss": copy.deepcopy(dc.get("loss", {})),
        "training_strategy": copy.deepcopy(dc.get("training_strategy", {})),
        "output": {
            "checkpoint_dir": str(run_dir.resolve()),
            "log_dir": str((run_dir / "logs").resolve()),
        },
    }

    # --- 数据参数覆盖 ---
    image_size = _coerce_image_size(config)
    if image_size is not None:
        sup["data"]["image_size"] = image_size
    sup["data"]["batch_size"] = _coerce_int(config, "batchSize", "batch_size", default=sup["data"].get("batch_size", 16))
    sup["data"]["repeat_factor"] = max(
        1,
        _coerce_int(config, "repeatFactor", "repeat_factor", default=int(sup["data"].get("repeat_factor", 1))),
    )
    sup["data"]["num_workers"] = _coerce_int(
        config, "numWorkers", "num_workers", default=sup["data"].get("num_workers", 4)
    )
    sup["data"]["pin_memory"] = _coerce_bool(
        config, "pinMemory", "pin_memory", default=sup["data"].get("pin_memory", True)
    )
    sup["data"]["persistent_workers"] = _coerce_bool(
        config, "persistentWorkers", "persistent_workers",
        default=bool(sup["data"].get("persistent_workers", True))
    )
    sup["data"]["prefetch_factor"] = _coerce_int(
        config, "prefetchFactor", "prefetch_factor", default=int(sup["data"].get("prefetch_factor", 2))
    )

    # --- 训练参数覆盖 ---
    sup["training"]["epochs"] = _coerce_int(config, "epochs", default=sup["training"].get("epochs", 100))
    sup["training"]["learning_rate"] = _coerce_float(
        config, "learningRate", "learning_rate", default=float(sup["training"].get("learning_rate", 1e-4))
    )
    sup["training"]["weight_decay"] = _coerce_float(
        config, "weightDecay", "weight_decay", default=float(sup["training"].get("weight_decay", 0.05))
    )
    sup["training"]["backbone_lr_ratio"] = _coerce_float(
        config, "backboneLrRatio", "backbone_lr_ratio",
        default=float(sup["training"].get("backbone_lr_ratio", 0.1)),
    )
    sup["training"]["save_interval"] = _coerce_int(
        config, "saveInterval", "save_interval", default=sup["training"].get("save_interval", 5)
    )
    sup["training"]["use_amp"] = _coerce_bool(
        config, "useAmp", "use_amp", default=bool(sup["training"].get("use_amp", False))
    )
    sup["training"]["eval_interval"] = max(
        1,
        _coerce_int(config, "evalInterval", "eval_interval", default=int(sup["training"].get("eval_interval", 1))),
    )
    # SupconTrainer 需要 lr_scheduler 键（'cosine' | 'step'），data_cluster.yaml 默认 cosine
    lr_scheduler = config.get("lrScheduler") or config.get("lr_scheduler")
    if isinstance(lr_scheduler, str) and lr_scheduler.strip():
        sup["training"]["lr_scheduler"] = lr_scheduler.strip().lower()
    else:
        sup["training"].setdefault("lr_scheduler", "cosine")

    # --- 训练策略覆盖 ---
    sup["training_strategy"]["freeze_backbone_epochs"] = _coerce_int(
        config, "freezeBackboneEpochs", "freeze_backbone_epochs",
        default=sup["training_strategy"].get("freeze_backbone_epochs", 0),
    )

    # --- 模型参数覆盖 ---
    # backbone 选择优先级：实验显式指定 > data_cluster.yaml training.default_backbone
    #                       > model.backbone > convnext_small
    backbone = config.get("backbone") or config.get("modelName") or config.get("model_name")
    if isinstance(backbone, str) and backbone.strip():
        sup["model"]["backbone"] = backbone.strip()
    elif not sup["model"].get("backbone"):
        default_backbone = dc.get("training", {}).get("default_backbone")
        sup["model"]["backbone"] = (default_backbone or "convnext_small")

    embedding_dim = config.get("embeddingDim") or config.get("embedding_dim")
    if isinstance(embedding_dim, int) and embedding_dim > 0:
        sup["model"]["embedding_dim"] = embedding_dim

    # 归一化 projection head 配置键：SupconTrainer/工厂同时支持
    # model.projection_hidden_dims（扁平）与 model.projection_head.hidden_dims（嵌套）。
    if "projection_head" not in sup["model"] and sup["model"].get("projection_hidden_dims"):
        sup["model"]["projection_head"] = {"hidden_dims": list(sup["model"]["projection_hidden_dims"])}

    # 预训练权重路径：默认从 data_cluster.yaml 的 backbones.<backbone>.pretrained_path 加载
    backbone_name = sup["model"].get("backbone", "convnext_small")
    pretrained_path = _resolve_backbone_pretrained_path(backbone_name, config)
    if pretrained_path:
        sup["model"]["pretrained_path"] = pretrained_path
        print(f"[training] backbone='{backbone_name}' 默认加载预训练权重: {pretrained_path}")
    else:
        print(f"[training] backbone='{backbone_name}' 未找到预训练权重路径，将从随机/HF 初始化")

    # --- MoCo 参数覆盖 ---
    use_moco = _coerce_bool(config, "useMoCo", "use_moco", default=sup["moco"].get("enabled", False))
    sup["moco"]["enabled"] = use_moco
    sup["moco"]["queue_size"] = _coerce_int(
        config, "queueSize", "queue_size", default=sup["moco"].get("queue_size", 16384)
    )

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
        exp.run_metrics = {"stage": "preparing_dataset"}
        db.commit()

        if not exp.samples:
            prepare_experiment_samples(db, exp, settings)
            db.commit()
            db.refresh(exp)

        run_dir = experiment_checkpoint_run_dir(settings, exp.model, exp.id)
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset_csv = run_dir / DATASET_CSV_FILENAME
        _write_dataset_csv(dataset_csv, exp)

        # 训练阶段需求：验证集与训练集使用同一份样本，始终开启 eval。
        use_eval = True
        supcon_config = _build_supcon_config(exp, run_dir)
        logger = setup_logger(
            f"experiment_{exp.id}",
            log_dir=supcon_config["supcon"]["output"]["log_dir"],
        )
        data_config = _build_data_config(exp, dataset_csv, supcon_config=supcon_config)
        pretrained_path = None
        if isinstance(exp.config, dict):
            raw_pretrained = exp.config.get("pretrainedPath") or exp.config.get("pretrained_path")
            if isinstance(raw_pretrained, str) and raw_pretrained.strip():
                pretrained_path = raw_pretrained.strip()

        # Accumulate per-epoch series for live chart rendering
        live_train_losses: list[float] = []
        live_val_losses: list[float] = []
        live_margins: list[float] = []
        live_pos_sims: list[float] = []
        live_neg_sims: list[float] = []

        # 降频写 DB：每 DB_WRITE_INTERVAL 个 epoch 写一次，减少训练主线程阻塞
        # 前端轮询间隔已改为 2s，写入间隔 3 epoch 完全够用
        _DB_WRITE_INTERVAL = 3
        _last_db_write_epoch: list[int] = [0]  # 用 list 做闭包可变引用

        def on_progress(payload: dict[str, Any]) -> bool:
            """Called after each epoch. Returns True to signal trainer to stop."""
            tm = payload.get("train_metrics") or {}
            vm = normalize_experiment_metrics({"val": payload.get("val_metrics") or {}})["val"]

            if tm.get("loss") is not None:
                live_train_losses.append(float(tm["loss"]))
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

            epoch = payload.get("epoch", 0)
            total_epochs = payload.get("total_epochs", 1)
            is_last_epoch = (epoch >= total_epochs)
            epochs_since_write = epoch - _last_db_write_epoch[0]

            # 未到写入间隔且不是最后一 epoch，跳过 DB 写入
            if epochs_since_write < _DB_WRITE_INTERVAL and not is_last_epoch:
                return False

            _last_db_write_epoch[0] = epoch
            current = db.get(Experiment, experiment_id)
            if not current:
                return False
            current.run_progress = float(payload.get("progress", 0.0))
            current.run_metrics = normalize_experiment_metrics(
                {
                    "stage": "training",
                    "epoch": epoch,
                    "totalEpochs": total_epochs,
                    "train": tm,
                    "val": vm,
                    "liveSeries": {
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
            progress_callback=on_progress,
        )
        summary = trainer.run()

        exp = db.get(Experiment, experiment_id)
        if not exp:
            return

        was_stopped = _stop_flags.pop(experiment_id, False)
        checkpoint_path = Path(summary["checkpoint_path"])

        # Embed the accumulated live series into the final summary so the
        # completed dashboard can render per-epoch loss / margin curves.
        summary["liveSeries"] = {
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
                    "epochsRun": len(live_train_losses),
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
