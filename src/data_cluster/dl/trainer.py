"""Web 平台训练业务层：简单的监督对比 (SupCon) finetune。

只做必要的事：
  * 加载数据集（训练 / 验证共用同一份样本，由 manifest 提供）
  * 构建模型并加载预训练 backbone（委托 ``build_supcon_model``）
  * 标准监督对比损失训练，可选先冻结 backbone 若干 epoch 再解冻
  * 每个 epoch 通过 ``progress_callback`` 上报进度与指标
  * 训练结束（或用户中止）时保存唯一 checkpoint、绘制 loss 曲线

刻意不包含 MoCo、多尺度训练、重复采样、混合精度 (AMP) 等额外功能；
如需这些能力请使用 ``scripts/train_supcon.py`` 独立训练脚本。
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from embedding_model.supcon.build import (
    build_supcon_model,
    extract_model_state_dict,
    is_full_supcon_checkpoint,
    is_full_supcon_state_dict,
    normalize_state_dict_for_supcon_model,
)
from embedding_model.supcon.datasets.manifest_triplet_dataset import DataClusterTripletDataset
from embedding_model.supcon.datasets.supcon_dataset import SupConDataset
from embedding_model.supcon.models.losses import SupervisedContrastiveLoss
from embedding_model.utils.config_loader import load_config
from embedding_model.utils.metrics import similarity_distribution_stats
from embedding_model.utils.visualization import plot_loss_curve

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Web 平台常见场景是几十张图的小样本 finetune：一个 epoch 可能只有个位数
# batch。此时多进程 DataLoader 每个 epoch 重新调度 worker 的 IPC 开销，
# 比几个 batch 本身的计算时间还长，表现为「每个 epoch 训练完就卡一下」。
# 低于该阈值时自动退化为主进程同步加载（num_workers=0），数据集较大时
# 仍使用配置的多进程 worker。
MIN_BATCHES_PER_EPOCH_FOR_WORKERS = 8


class SupconTrainer:
    """监督对比 finetune pipeline。

    ``__init__`` 完成输出目录、数据/DataLoader、模型与预训练加载、损失、优化器与调度器；
    ``run()`` 执行 epoch 循环、验证、进度回调、日志与保存。
    """

    def __init__(
        self,
        supcon_config: dict,
        logger,
        data_config_paths: list,
        *,
        use_eval: bool = True,
        pretrained_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.supcon_config = supcon_config
        self.logger = logger
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_eval = use_eval
        self.pretrained_path = pretrained_path
        self.progress_callback = progress_callback

        sup = supcon_config["supcon"]
        self.data_config = sup["data"]
        self.training_config = sup["training"]
        self.model_config = sup["model"]
        self.loss_config = sup["loss"]
        self.training_strategy = sup.get("training_strategy", {})

        self.checkpoint_dir = Path(sup["output"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 单一训练尺度（不支持多尺度）：list 配置取最大边长。
        image_size_cfg = self.data_config.get("image_size", 224)
        if isinstance(image_size_cfg, (list, tuple)):
            self.image_size = int(max(image_size_cfg)) if image_size_cfg else 224
        else:
            self.image_size = int(image_size_cfg)

        self.eval_interval = max(1, int(self.training_config.get("eval_interval", 1)))
        self.non_blocking = self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self._setup_data(data_config_paths)
        self._setup_model()
        self.criterion = SupervisedContrastiveLoss(
            temperature=self.loss_config["supcon"]["temperature"],
        ).to(self.device)
        self._setup_optimizer_scheduler()

        self.freeze_epochs = int(self.training_strategy.get("freeze_backbone_epochs", 0))
        self.start_epoch = 1

    # ------------------------------------------------------------------ data
    def _resolve_dataset_cls(self, data_cfg: dict):
        dataset_type = str(data_cfg.get("dataset_type", "")).strip().lower()
        if dataset_type in {"data_cluster_triplets", "triplets", "manifest_triplets"}:
            return DataClusterTripletDataset
        return SupConDataset

    def _build_loader_kwargs(self, dataset_len: int, batch_size: int) -> dict:
        """按数据集大小自适应选择 num_workers（见 ``MIN_BATCHES_PER_EPOCH_FOR_WORKERS``）。"""
        num_workers = int(self.data_config.get("num_workers", 4))
        batches_per_epoch = math.ceil(dataset_len / batch_size) if batch_size > 0 else 0
        if num_workers > 0 and batches_per_epoch < MIN_BATCHES_PER_EPOCH_FOR_WORKERS:
            self.logger.info(
                f"每 epoch 仅 {batches_per_epoch} 个 batch（阈值 {MIN_BATCHES_PER_EPOCH_FOR_WORKERS}），"
                f"关闭多进程 DataLoader (num_workers {num_workers}->0) 避免小数据集下的 worker 调度开销。"
            )
            num_workers = 0

        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": bool(self.data_config.get("pin_memory", True)),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(self.data_config.get("persistent_workers", True))
            loader_kwargs["prefetch_factor"] = max(2, int(self.data_config.get("prefetch_factor", 2)))
        return loader_kwargs

    def _setup_data(self, data_config_paths: list) -> None:
        """构建数据集与 DataLoader。训练集与验证集共用同一份样本。"""
        logger = self.logger
        data_config_path = data_config_paths[0] if data_config_paths else None
        if isinstance(data_config_path, dict):
            data_config = data_config_path
        elif data_config_path:
            data_config = load_config(data_config_path)
        else:
            raise ValueError("缺少数据配置 (data_config_paths)")
        if "scenes" in data_config:
            raise ValueError("当前训练器仅支持单 dataset 配置。")

        dataset_cls = self._resolve_dataset_cls(data_config)
        train_dataset = dataset_cls(
            data_config=data_config,
            split="train",
            image_size=self.image_size,
        )
        categories = sorted(list(train_dataset.categories))
        logger.info(f"类别数: {len(categories)}, 缺陷类别: {categories}")
        logger.info(f"训练样本: {len(train_dataset)}, 图像尺度: {self.image_size}")

        batch_size = min(self.data_config["batch_size"], max(1, len(train_dataset)))
        train_loader_kwargs = self._build_loader_kwargs(len(train_dataset), batch_size)

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **train_loader_kwargs
        )

        self.val_dataloader = None
        if self.use_eval:
            # 验证集与训练集共用同一份样本，仅关闭数据增强（split='val'）。
            val_dataset = dataset_cls(
                data_config=data_config,
                split="val",
                image_size=self.image_size,
            )
            logger.info(f"验证样本: {len(val_dataset)} (与训练集共用)")
            val_loader_kwargs = self._build_loader_kwargs(len(val_dataset), batch_size)
            self.val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, **val_loader_kwargs
            )

    # ----------------------------------------------------------------- model
    def _resolve_pretrained_path(self) -> Optional[Path]:
        """实验 ``pretrainedPath`` 优先，否则使用 ``model_config['pretrained_path']``。"""
        if self.pretrained_path:
            return Path(self.pretrained_path)
        raw = self.model_config.get("pretrained_path")
        return Path(raw) if raw else None

    def _setup_model(self) -> None:
        freeze_backbone = int(self.training_strategy.get("freeze_backbone_epochs", 0)) > 0
        self.logger.info("创建模型...")

        model_config = dict(self.model_config)
        pretrained_path = self._resolve_pretrained_path()
        # 完整 SupCon 权重（含 fusion/head）在构建后一次性加载，避免 backbone 构造器只读 stages.*
        if pretrained_path and pretrained_path.is_file() and is_full_supcon_checkpoint(pretrained_path):
            model_config.pop("pretrained_path", None)

        model, _queue = build_supcon_model(
            model_config,
            image_size=self.image_size,
            use_moco=False,
            freeze_backbone=freeze_backbone,
            device=self.device,
            logger=self.logger,
        )
        self.model = model
        self._load_pretrained_weights_if_any()

    def _load_pretrained_weights_if_any(self) -> None:
        """加载预训练权重到模型 (strict=True)。

        * 完整 SupCon checkpoint（含 feature_fusion / projection_head）：
          加载 backbone + fusion + head；MoCo 格式会自动剥离 query_encoder 前缀。
          键不匹配时立即报错，避免静默部分加载。
        * 仅 backbone 权重（如 DINOv3 原始 ``pretrain_ckpts/*.pth``）：
          已在 ``build_supcon_model`` 构建时由 backbone 加载，此处仅记录日志。
        """
        path = self._resolve_pretrained_path()
        if not path:
            return
        if not path.is_file():
            self.logger.warning(f"预训练权重文件不存在，跳过加载: {path}")
            return

        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        if not isinstance(ckpt, dict):
            self.logger.warning("预训练文件不是字典，跳过加载")
            return

        state = extract_model_state_dict(ckpt)
        if is_full_supcon_state_dict(state):
            state = normalize_state_dict_for_supcon_model(state)
            self.logger.info(f"加载完整 SupCon 预训练权重 (strict=True): {path}")
            self.model.load_state_dict(state, strict=True)
            self.logger.info(f"预训练权重已加载，共 {len(state)} 个参数张量")
        else:
            self.logger.info(f"backbone 预训练权重已在模型构建时加载: {path}")

    def _setup_optimizer_scheduler(self) -> None:
        training_config = self.training_config
        backbone_lr_ratio = float(training_config.get("backbone_lr_ratio", 0.1))
        base_lr = float(training_config["learning_rate"])

        backbone_params, other_params = [], []
        for name, param in self.model.named_parameters():
            (backbone_params if "backbone" in name else other_params).append(param)

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": base_lr * backbone_lr_ratio},
                {"params": other_params, "lr": base_lr},
            ],
            weight_decay=float(training_config["weight_decay"]),
        )

        epochs = int(training_config["epochs"])
        if str(training_config.get("lr_scheduler", "cosine")).lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, epochs // 3), gamma=0.1
            )

    # ----------------------------------------------------------------- train
    def train_epoch(self, epoch: int) -> dict:
        """训练一个 epoch：双视图监督对比损失。"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            v1_img = batch["view1_image"].to(self.device, non_blocking=self.non_blocking)
            v1_mask = batch["view1_mask"].to(self.device, non_blocking=self.non_blocking)
            v2_img = batch["view2_image"].to(self.device, non_blocking=self.non_blocking)
            v2_mask = batch["view2_mask"].to(self.device, non_blocking=self.non_blocking)
            labels = batch["label"].to(self.device, non_blocking=self.non_blocking)

            out1 = self.model(v1_img, v1_mask, return_features=False)
            out2 = self.model(v2_img, v2_mask, return_features=False)

            # 同一样本的 view1/view2 共享 label，构成 positive pair
            embeddings = torch.cat([out1["embeddings"], out2["embeddings"]], dim=0)
            labels_dup = torch.cat([labels, labels], dim=0)
            loss = self.criterion(embeddings, labels_dup)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}

    @torch.no_grad()
    def validate(self) -> dict:
        """在共用数据上计算嵌入相似度分布 (PosSim / NegSim / Margin)。"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_embeddings, all_labels = [], []

        for batch in tqdm(self.val_dataloader, desc="Validating"):
            v1_img = batch["view1_image"].to(self.device, non_blocking=self.non_blocking)
            v1_mask = batch["view1_mask"].to(self.device, non_blocking=self.non_blocking)
            labels = batch["label"].to(self.device, non_blocking=self.non_blocking)

            out = self.model(v1_img, v1_mask, return_features=False)
            emb = F.normalize(out["embeddings"], dim=1, p=2, eps=1e-8)

            loss = self.criterion(emb, labels)
            total_loss += loss.item()
            num_batches += 1
            all_embeddings.append(emb.cpu())
            all_labels.append(labels.cpu())

        if all_embeddings:
            emb_all = torch.cat(all_embeddings, dim=0).to(self.device)
            lbl_all = torch.cat(all_labels, dim=0).to(self.device)
            sim = similarity_distribution_stats(emb_all, lbl_all)
            margin_pos_sim = sim["pos_sim"]
            margin_neg_sim = sim["neg_sim"]
            margin = sim["margin"]
        else:
            margin_pos_sim = margin_neg_sim = margin = 0.0

        return {
            "loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "margin_pos_sim": margin_pos_sim,
            "margin_neg_sim": margin_neg_sim,
            "margin": margin,
        }

    # ------------------------------------------------------------------- run
    def _purge_legacy_periodic_checkpoints(self) -> None:
        """Web 训练只保留最终权重，清理历史 periodic checkpoint。"""
        for old in self.checkpoint_dir.glob("checkpoint_epoch_*.pth"):
            try:
                old.unlink()
            except OSError as exc:
                self.logger.warning(f"无法删除旧 checkpoint {old}: {exc}")

    def _save_checkpoint(self, epoch: int, train_metrics: dict) -> Path:
        """保存唯一最终权重 ``current_model.pth``（供推理使用）。"""
        self._purge_legacy_periodic_checkpoints()
        path = self.checkpoint_dir / "current_model.pth"
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_metrics["loss"],
            "config": self.supcon_config,
        }
        torch.save(payload, path)
        return path

    def run(self) -> dict:
        self.logger.info("开始训练...")
        self._purge_legacy_periodic_checkpoints()
        train_losses: list[float] = []
        val_losses: list[float] = []
        val_epochs: list[int] = []
        total_epochs = int(self.training_config["epochs"])
        last_train_metrics = None
        last_val_metrics = None
        last_epoch = 0

        for epoch in range(self.start_epoch, total_epochs + 1):
            if self.freeze_epochs > 0 and epoch == self.freeze_epochs + 1:
                self.logger.info("解冻 backbone 参数...")
                self.model.unfreeze_all()

            train_metrics = self.train_epoch(epoch)
            train_losses.append(train_metrics["loss"])
            last_train_metrics = train_metrics
            last_epoch = epoch

            val_metrics = None
            should_eval = (
                self.use_eval
                and self.val_dataloader is not None
                and (
                    epoch == self.start_epoch
                    or epoch % self.eval_interval == 0
                    or epoch == total_epochs
                )
            )
            if should_eval:
                val_metrics = self.validate()
                val_losses.append(val_metrics["loss"])
                val_epochs.append(epoch)
                last_val_metrics = val_metrics

            self.scheduler.step()

            if self.progress_callback is not None:
                should_stop = self.progress_callback(
                    {
                        "epoch": epoch,
                        "total_epochs": total_epochs,
                        "progress": (epoch / total_epochs) * 100.0,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                    }
                )
                if should_stop:
                    self.logger.info(f"训练在 epoch {epoch} 被用户中止。")
                    break

            log_msg = (
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                f"lr={self.scheduler.get_last_lr()[0]:.6f}"
            )
            if val_metrics is not None:
                log_msg += (
                    f"\n  [val] loss={val_metrics['loss']:.4f}, "
                    f"PosSim={val_metrics['margin_pos_sim']:.4f}, "
                    f"NegSim={val_metrics['margin_neg_sim']:.4f}, "
                    f"Margin={val_metrics['margin']:.4f}"
                )
            self.logger.info(log_msg)

        if last_train_metrics is not None:
            self._save_checkpoint(last_epoch, last_train_metrics)

        plot_loss_curve(
            train_losses,
            val_losses if self.use_eval else [],
            val_epochs=val_epochs if self.use_eval else None,
            save_path=str(self.checkpoint_dir / "loss_curve.png"),
            title="SupCon Training Loss",
        )

        self.logger.info("训练完成！")
        return {
            "epochs": total_epochs,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_epochs": val_epochs,
            "last_train_metrics": last_train_metrics,
            "last_val_metrics": last_val_metrics,
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_path": str((self.checkpoint_dir / "current_model.pth").resolve()),
            "loss_curve_path": str((self.checkpoint_dir / "loss_curve.png").resolve()),
        }
