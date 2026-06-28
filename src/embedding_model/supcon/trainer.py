"""
SupCon监督对比学习训练脚本
"""
import argparse
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from embedding_model.supcon.build import build_supcon_model
from embedding_model.supcon.datasets.manifest_triplet_dataset import DataClusterTripletDataset
from embedding_model.supcon.datasets.supcon_dataset import (IndexWithScale, MultiScaleBatchSampler, SupConDataset,
                                                            multi_scale_collate_fn)
from embedding_model.supcon.models.losses import (ComprehensiveSegmentationLoss,
                                                  SupervisedContrastiveLoss)
from embedding_model.supcon.models.moco_loss import MoCoLoss
from embedding_model.supcon.models.moco_queue import MoCoQueue
from embedding_model.utils.config_loader import load_config
from embedding_model.utils.logging import setup_logger
from embedding_model.utils.metrics import knn_evaluation, similarity_distribution_stats
from embedding_model.utils.visualization import plot_loss_curve

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class RepeatDataset(Dataset):
    """Wrap a dataset and repeat it virtually without duplicating files in memory."""

    def __init__(self, inner_dataset: Dataset, repeat_factor: int) -> None:
        self.inner_dataset = inner_dataset
        self.repeat_factor = max(1, int(repeat_factor))
        self.base_len = len(inner_dataset)

    def __len__(self) -> int:
        return self.base_len * self.repeat_factor

    def __getitem__(self, idx):
        if self.base_len <= 0:
            raise IndexError("empty dataset")
        if isinstance(idx, IndexWithScale):
            mapped = IndexWithScale(idx=int(idx.idx % self.base_len), image_size=idx.image_size)
            return self.inner_dataset[mapped]
        return self.inner_dataset[idx % self.base_len]



class SupconTrainer(object):
    """SupCon / MoCo 监督对比学习训练 pipeline。

    在 ``__init__`` 中完成：输出目录、数据与 DataLoader、相似度矩阵、模型与 MoCo 队列、
    可选 ``pretrained_path`` 加载预训练权重、损失函数、优化器与调度器。
    ``run()`` 仅执行 epoch 循环、验证、日志与保存。
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
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_eval = use_eval
        self.pretrained_path = pretrained_path
        self.progress_callback = progress_callback

        self.checkpoint_dir = Path(supcon_config['supcon']['output']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._setup_data_pipeline(data_config_paths)
        self._setup_model_and_moco_queue()
        self._setup_losses()
        self._setup_optimizer_scheduler()

        self.use_amp = bool(self.training_config.get('use_amp', False)) and self.device.type == 'cuda'
        self.eval_interval = max(1, int(self.training_config.get('eval_interval', 1)))
        self.non_blocking_transfer = self.device.type == 'cuda'
        self.amp_autocast_dtype = torch.float16 if self.device.type == 'cuda' else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        self.logger.info(f"训练加速配置: use_amp={self.use_amp}, eval_interval={self.eval_interval}")

        self.start_epoch = 1
        self.best_margin = float('inf')

    def _setup_data_pipeline(self, data_config_paths: list) -> None:
        """单数据集模式：构建数据集、DataLoader、相似度矩阵。"""
        logger = self.logger
        supcon_config = self.supcon_config

        logger.info("加载数据集...")
        data_config_path = data_config_paths[0] if data_config_paths else 'configs/data_config_zhenyu.yaml'
        logger.info(f"数据配置文件: {data_config_path}")
        if isinstance(data_config_path, dict):
            data_config = data_config_path
        else:
            data_config = load_config(data_config_path)
        if 'scenes' in data_config:
            raise ValueError("当前训练器仅支持单dataset配置，不再支持scenes多场景配置。")

        image_size_config = supcon_config['supcon']['data'].get('image_size', 224)
        if isinstance(image_size_config, int):
            image_sizes = [image_size_config]
            use_multiscale = False
        elif isinstance(image_size_config, list):
            image_sizes = image_size_config
            use_multiscale = len(image_sizes) > 1
        else:
            raise ValueError(f"image_size必须是int或List[int]，当前为{type(image_size_config)}")
        self.image_sizes = image_sizes
        self.use_multiscale = use_multiscale

        model_image_size = max(image_sizes)
        val_image_size = max(image_sizes)
        self.model_image_size = model_image_size
        self.val_image_size = val_image_size
        logger.info(f"图像尺度配置: {image_sizes}")
        if use_multiscale:
            logger.info(f"启用多尺度训练: 训练时每个batch随机选择 {image_sizes} 中的一个尺度")
            logger.info(f"验证集固定使用尺度: {val_image_size}")
        logger.info(f"模型初始化使用尺度: {model_image_size}")

        dataset_cls = self._resolve_dataset_cls(data_config)
        train_dataset = dataset_cls(
            data_config=data_config,
            split='train',
            image_size=image_sizes,
        )
        val_dataset = None
        if self.use_eval:
            val_dataset = dataset_cls(
                data_config=data_config,
                split='val',
                image_size=val_image_size,
            )

        categories = sorted(list(train_dataset.categories))
        num_classes = len(categories)

        repeat_factor = int(supcon_config['supcon']['data'].get('repeat_factor', 1))
        repeat_factor = max(1, repeat_factor)
        effective_train_dataset: Dataset = RepeatDataset(train_dataset, repeat_factor) if repeat_factor > 1 else train_dataset

        logger.info(f"类别数: {num_classes}, 缺陷类别: {categories}")
        logger.info(
            f"训练样本: raw={len(train_dataset)}, effective={len(effective_train_dataset)}, repeat_factor={repeat_factor}"
        )
        if val_dataset is not None:
            logger.info(f"验证样本: {len(val_dataset)}")

        batch_size = min(supcon_config['supcon']['data']['batch_size'], max(1, len(effective_train_dataset)))
        num_workers = supcon_config['supcon']['data']['num_workers']
        pin_memory = supcon_config['supcon']['data']['pin_memory']
        persistent_workers = bool(supcon_config['supcon']['data'].get('persistent_workers', True))
        prefetch_factor = int(supcon_config['supcon']['data'].get('prefetch_factor', 2))
        common_loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            common_loader_kwargs["persistent_workers"] = persistent_workers
            common_loader_kwargs["prefetch_factor"] = max(2, prefetch_factor)

        if use_multiscale:
            batch_sampler = MultiScaleBatchSampler(
                dataset=effective_train_dataset,
                batch_size=batch_size,
                image_sizes=image_sizes,
                shuffle=True,
            )
            self.train_dataloader = DataLoader(
                effective_train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=multi_scale_collate_fn,
                **common_loader_kwargs,
            )
        else:
            self.train_dataloader = DataLoader(
                effective_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                **common_loader_kwargs,
            )

        self.val_dataloader = None
        if self.use_eval and val_dataset is not None:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                **common_loader_kwargs,
            )

    def _resolve_dataset_cls(self, data_cfg: dict):
        dataset_type = str(data_cfg.get('dataset_type', '')).strip().lower()
        if dataset_type in {'data_cluster_triplets', 'triplets', 'manifest_triplets'}:
            return DataClusterTripletDataset
        return SupConDataset

    def _setup_model_and_moco_queue(self) -> None:
        """模型与单个 MoCo 队列。"""
        logger = self.logger
        supcon_config = self.supcon_config

        moco_config = supcon_config['supcon'].get('moco', {})
        use_moco = moco_config.get('enabled', False)
        self.moco_config = moco_config
        self.use_moco = use_moco

        logger.info("创建模型...")
        model_config = supcon_config['supcon']['model']
        self.model_config = model_config
        freeze_backbone = supcon_config['supcon']['training_strategy'].get('freeze_backbone_epochs', 0) > 0

        model, moco_queue = self.build_model(
            model_config=model_config,
            moco_config=moco_config,
            use_moco=use_moco,
            image_size=self.model_image_size,
            freeze_backbone=freeze_backbone,
        )
        self.model = model

        self.moco_queue = moco_queue if (use_moco and moco_queue is not None) else None

        self._load_pretrained_weights_if_any()

    def _load_pretrained_weights_if_any(self) -> None:
        """若提供 ``pretrained_path``，将权重加载到 ``self.model``（``strict=False``）。"""
        path = self.pretrained_path
        if not path:
            return
        p = Path(path)
        if not p.is_file():
            self.logger.warning(f"预训练权重文件不存在，跳过加载: {p}")
            return
        self.logger.info(f"加载预训练权重: {p}")
        ckpt = torch.load(str(p), map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                state = ckpt['state_dict']
            else:
                state = ckpt
        else:
            self.logger.warning("预训练文件不是字典，跳过加载")
            return
        incompatible = self.model.load_state_dict(state, strict=False)
        mk = getattr(incompatible, 'missing_keys', []) or []
        uk = getattr(incompatible, 'unexpected_keys', []) or []
        self.logger.info(
            f"预训练权重已合并到模型 (strict=False)，missing_keys={len(mk)}, unexpected_keys={len(uk)}",
        )

    def _setup_losses(self) -> None:
        """对比损失、分割损失及权重。"""
        supcon_config = self.supcon_config
        loss_config = supcon_config['supcon']['loss']
        self.loss_config = loss_config
        enable_segmentation = self.model_config.get('segmentation', {}).get('enabled', True)
        self.enable_segmentation = enable_segmentation

        criterion, seg_criterion = self.build_loss_func(
            loss_config=loss_config,
            moco_config=self.moco_config,
            use_moco=self.use_moco,
            enable_segmentation=enable_segmentation,
        )
        self.criterion = criterion
        self.seg_criterion = seg_criterion

        seg_loss_weight = loss_config.get('segmentation', {}).get('weight', 0.5)
        self.seg_loss_weight = seg_loss_weight
        if enable_segmentation:
            self.logger.info(f"分割损失权重: {seg_loss_weight}")

        # 预建验证用 criterion（MoCo 验证时不需要队列，用标准 SupCon loss）
        # 避免 validate() 内每个 batch 重复创建和搬运到 GPU
        if self.use_moco:
            from embedding_model.supcon.models.losses import SupervisedContrastiveLoss as _SCL
            self.val_criterion = _SCL(
                temperature=loss_config['supcon']['temperature'],
            ).to(self.device)
        else:
            self.val_criterion = None  # 非 MoCo 时直接复用 self.criterion

    def _setup_optimizer_scheduler(self) -> None:
        """优化器、学习率调度器、冻结 epoch 数。"""
        supcon_config = self.supcon_config
        model = self.model
        use_moco = self.use_moco

        training_config = supcon_config['supcon']['training']
        self.training_config = training_config
        backbone_lr_ratio = training_config.get('backbone_lr_ratio', 0.1)

        backbone_params = []
        other_params = []
        if use_moco:
            for name, param in model.query_encoder.named_parameters():
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        else:
            for name, param in model.named_parameters():
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': float(training_config['learning_rate']) * backbone_lr_ratio},
            {'params': other_params, 'lr': float(training_config['learning_rate'])},
        ], weight_decay=training_config['weight_decay'])

        if training_config['lr_scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['epochs'],
                eta_min=1e-6,
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config['epochs'] // 3,
                gamma=0.1,
            )

        training_strategy = supcon_config['supcon']['training_strategy']
        self.freeze_epochs = training_strategy.get('freeze_backbone_epochs', 0)

    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        use_moco: bool = False,
        moco_queue: Optional[MoCoQueue] = None,
        seg_criterion: Optional[nn.Module] = None,
        seg_loss_weight: float = 0.5,
    ) -> dict:
        """
        训练一个epoch
        
        Args:
            model: 模型（SupConModel或MoCoModel）
            dataloader: 数据加载器
            criterion: 对比损失函数
            optimizer: 优化器
            epoch: 当前epoch
            use_moco: 是否使用MoCo
            moco_queue: MoCo队列（如果使用MoCo）
            seg_criterion: 分割损失函数（可选）
            seg_loss_weight: 分割损失权重
        """
        model.train()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_seg_loss = 0.0
        total_pos_loss = 0.0
        total_neg_loss = 0.0
        num_batches = 0
        skipped_batches = 0
        nan_embedding_count = 0
        nan_loss_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # 获取数据（同时使用view1和view2）
            view1_images = batch['view1_image'].to(self.device, non_blocking=self.non_blocking_transfer)
            view1_masks = batch['view1_mask'].to(self.device, non_blocking=self.non_blocking_transfer)
            view2_images = batch['view2_image'].to(self.device, non_blocking=self.non_blocking_transfer)
            view2_masks = batch['view2_mask'].to(self.device, non_blocking=self.non_blocking_transfer)
            labels = batch['label'].to(self.device, non_blocking=self.non_blocking_transfer)
            
            if use_moco:
                if moco_queue is None:
                    raise RuntimeError("use_moco=True 但 moco_queue 未初始化")
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_autocast_dtype,
                ):
                    # MoCo训练流程
                    # 1. 使用query_encoder计算view1的query embeddings
                    query_outputs1 = model(view1_images, view1_masks, mode='query', return_features=False, return_segmentation=(seg_criterion is not None))
                    query_embeddings1 = query_outputs1['embeddings']

                    # 2. 使用momentum_encoder计算view2的key embeddings（用于positive pairs和更新队列）
                    with torch.no_grad():
                        key_outputs2 = model(view2_images, view2_masks, mode='key', return_features=False, return_segmentation=False)
                        key_embeddings2 = key_outputs2['embeddings']

                    # 3. 获取队列中的负样本
                    queue_embeddings, queue_labels = moco_queue.get_queue(device=self.device)

                    # 4. 检查embeddings是否包含NaN或Inf
                    if (torch.isnan(query_embeddings1).any() or torch.isinf(query_embeddings1).any() or
                        torch.isnan(key_embeddings2).any() or torch.isinf(key_embeddings2).any()):
                        nan_embedding_count += 1
                        skipped_batches += 1
                        if nan_embedding_count <= 5:
                            print(f"警告：Epoch {epoch}, Batch {num_batches}: embeddings包含NaN或Inf，跳过此batch")
                        continue

                    # 5. 计算MoCo loss（结合当前batch和队列中的负样本）
                    contrastive_loss = criterion(
                        query_embeddings=query_embeddings1,
                        key_embeddings=key_embeddings2,
                        query_labels=labels,
                        queue_embeddings=queue_embeddings,
                        queue_labels=queue_labels
                    )

                    # 计算分割损失（如果启用）
                    seg_loss = torch.tensor(0.0, device=self.device)
                    if seg_criterion is not None and 'segmentation' in query_outputs1:
                        seg_pred = query_outputs1['segmentation']  # (B, 1, H, W)
                        seg_loss = seg_criterion(seg_pred, view1_masks)

                    # 总损失
                    loss = contrastive_loss + seg_loss_weight * seg_loss
                
                # 6. 检查loss是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() != loss.item():
                    nan_loss_count += 1
                    skipped_batches += 1
                    if nan_loss_count <= 5:
                        print(f"警告：Epoch {epoch}, Batch {num_batches}: loss为NaN或Inf，跳过此batch")
                    continue
                
                # 7. 反向传播
                optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.query_encoder.parameters(), max_norm=2.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.query_encoder.parameters(), max_norm=2.0)
                    optimizer.step()
                
                # 8. 更新队列（FIFO）
                with torch.no_grad():
                    moco_queue.enqueue(key_embeddings2, labels)
                
                # 9. 动量更新momentum_encoder
                with torch.no_grad():
                    model.momentum_update()
            else:
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_autocast_dtype,
                ):
                    # 标准SupCon训练流程
                    # 前向传播：分别计算view1和view2的embedding
                    outputs1 = model(view1_images, view1_masks, return_features=False, return_segmentation=(seg_criterion is not None))
                    embeddings1 = outputs1['embeddings']

                    outputs2 = model(view2_images, view2_masks, return_features=False, return_segmentation=False)
                    embeddings2 = outputs2['embeddings']

                    # 拼接view1和view2的embedding，形成2B大小的batch
                    # embeddings: [view1_0, view1_1, ..., view1_B-1, view2_0, view2_1, ..., view2_B-1]
                    # labels_duplicated: [label_0, label_1, ..., label_B-1, label_0, label_1, ..., label_B-1]
                    # 这样同一个样本的view1和view2（索引i和i+B）有相同的label，会被视为positive pair
                    embeddings = torch.cat([embeddings1, embeddings2], dim=0)  # (2B, D)
                    labels_duplicated = torch.cat([labels, labels], dim=0)  # (2B,)

                    # 检查embeddings是否包含NaN或Inf
                    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                        nan_embedding_count += 1
                        skipped_batches += 1
                        if nan_embedding_count <= 5:  # 只打印前5次警告
                            print(f"警告：Epoch {epoch}, Batch {num_batches}: embeddings包含NaN或Inf，跳过此batch")
                        continue

                    # 计算SupCon loss
                    # 在loss计算中：
                    # - 同一个样本的view1和view2（相同label）会被视为positive pair，它们的embedding会被拉近
                    # - 不同样本之间根据相似度矩阵判断是否为positive pair
                    contrastive_loss = criterion(embeddings, labels_duplicated)

                    # 计算分割损失（如果启用）
                    seg_loss = torch.tensor(0.0, device=self.device)
                    if seg_criterion is not None and 'segmentation' in outputs1:
                        seg_pred = outputs1['segmentation']  # (B, 1, H, W)
                        seg_loss = seg_criterion(seg_pred, view1_masks)

                    # 总损失
                    loss = contrastive_loss + seg_loss_weight * seg_loss
                
                # 检查loss是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() != loss.item():
                    nan_loss_count += 1
                    skipped_batches += 1
                    if nan_loss_count <= 5:  # 只打印前5次警告
                        print(f"警告：Epoch {epoch}, Batch {num_batches}: loss为NaN或Inf，跳过此batch")
                    continue
                
                # 反向传播
                optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
            
            total_loss += loss.item()
            if use_moco:
                total_contrastive_loss += contrastive_loss.item()
            else:
                total_contrastive_loss += contrastive_loss.item()
            if seg_criterion is not None:
                total_seg_loss += seg_loss.item()
            
            # 收集pos_loss和neg_loss（仅MoCo）
            if use_moco and hasattr(criterion, 'last_pos_loss') and criterion.last_pos_loss is not None:
                total_pos_loss += criterion.last_pos_loss
            if use_moco and hasattr(criterion, 'last_neg_loss') and criterion.last_neg_loss is not None:
                total_neg_loss += criterion.last_neg_loss
            
            num_batches += 1
            
            postfix_dict = {'loss': loss.item()}
            if seg_criterion is not None:
                postfix_dict['seg_loss'] = seg_loss.item()
            pbar.set_postfix(postfix_dict)
        
        # 打印统计信息
        if skipped_batches > 0:
            print(f"Epoch {epoch}统计: 跳过{skipped_batches}个batch (NaN embedding: {nan_embedding_count}, NaN loss: {nan_loss_count})")
        
        result = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'contrastive_loss': total_contrastive_loss / num_batches if num_batches > 0 else 0.0,
            'skipped_batches': skipped_batches,
        }
        
        # 添加分割损失信息
        if seg_criterion is not None and num_batches > 0:
            result['seg_loss'] = total_seg_loss / num_batches
            if hasattr(seg_criterion, 'last_loss_details') and seg_criterion.last_loss_details:
                details = seg_criterion.last_loss_details
                result['seg_pred_ratio'] = details['pred_foreground_ratio']
                result['seg_gt_ratio'] = details['gt_foreground_ratio']
        
        # 添加pos_loss和neg_loss（仅MoCo）
        if use_moco:
            if num_batches > 0:
                result['pos_loss'] = total_pos_loss / num_batches
                result['neg_loss'] = total_neg_loss / num_batches if total_neg_loss > 0 else 0.0
            else:
                result['pos_loss'] = 0.0
                result['neg_loss'] = 0.0
        
        return result
    
    
    def validate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        use_moco: bool = False,
    ) -> dict:
        """
        验证函数，使用相似度分布分析作为评估指标
        
        评估指标（基于整个验证集计算，而非batch平均）：
        - PosSim: 正样本平均相似度（越大越好，通常趋近1）
        - NegSim: 负样本平均相似度（越小越好，接近0或负值）
        - Margin: 正负样本间隔（越大越好，表示判别性越强）
        
        Args:
            model: 模型（SupConModel或MoCoModel）
            dataloader: 数据加载器
            criterion: 损失函数
            use_moco: 是否使用MoCo（如果使用MoCo，验证时只使用query_encoder）
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        skipped_batches = 0
        nan_embedding_count = 0
        nan_loss_count = 0
        
        # 收集所有验证集的embeddings和labels（用于全局相似度分布分析）
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # 验证时只使用view1（不需要数据增强）
                view1_images = batch['view1_image'].to(self.device, non_blocking=self.non_blocking_transfer)
                view1_masks = batch['view1_mask'].to(self.device, non_blocking=self.non_blocking_transfer)
                labels = batch['label'].to(self.device, non_blocking=self.non_blocking_transfer)

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_autocast_dtype,
                ):
                    # 前向传播：计算embedding
                    if use_moco:
                        # MoCo验证：只使用query_encoder
                        outputs1 = model(view1_images, view1_masks, mode='query', return_features=False)
                    else:
                        # 标准SupCon验证
                        outputs1 = model(view1_images, view1_masks, return_features=False)
                    embeddings1 = outputs1['embeddings']
                    
                    # 归一化embeddings（L2归一化）
                    embeddings1 = F.normalize(embeddings1, dim=1, p=2, eps=1e-8)
                
                # 检查embeddings是否包含NaN或Inf
                if torch.isnan(embeddings1).any() or torch.isinf(embeddings1).any():
                    nan_embedding_count += 1
                    skipped_batches += 1
                    if nan_embedding_count <= 5:  # 只打印前5次警告
                        print(f"警告：验证时embeddings包含NaN或Inf，跳过此batch")
                    continue
                
                # 对于MoCo，验证时使用标准SupCon loss（因为不需要队列）
                # 对于标准SupCon，使用原有的loss
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_autocast_dtype,
                ):
                    if use_moco:
                        loss = self.val_criterion(embeddings1, labels)
                    else:
                        loss = criterion(embeddings1, labels)
                
                # 检查loss是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() != loss.item():
                    nan_loss_count += 1
                    skipped_batches += 1
                    if nan_loss_count <= 5:  # 只打印前5次警告
                        print(f"警告：验证时loss为NaN或Inf，跳过此batch")
                    continue
                
                # 收集embeddings和labels（用于全局相似度分布分析）
                all_embeddings.append(embeddings1.cpu())  # 移到CPU以节省GPU内存
                all_labels.append(labels.cpu())
                
                total_loss += loss.item()
                num_batches += 1
        
        # 打印统计信息
        if skipped_batches > 0:
            print(f"验证统计: 跳过{skipped_batches}个batch (NaN embedding: {nan_embedding_count}, NaN loss: {nan_loss_count})")
        
        # 基于整个验证集计算相似度分布统计（核心评估指标）
        if len(all_embeddings) > 0:
            # 拼接所有embeddings和labels
            all_embeddings_tensor = torch.cat(all_embeddings, dim=0)  # (N, D)
            all_labels_tensor = torch.cat(all_labels, dim=0)  # (N,)
            
            # 将tensor移回device进行计算
            all_embeddings_tensor = all_embeddings_tensor.to(self.device)
            all_labels_tensor = all_labels_tensor.to(self.device)
            
            # 1. 计算相似度分布统计（Margin指标）
            sim_stats = similarity_distribution_stats(all_embeddings_tensor, all_labels_tensor)
            margin_pos_sim = sim_stats['pos_sim']
            margin_neg_sim = sim_stats['neg_sim']
            margin = sim_stats['margin']
            
            # 2. 计算kNN评估指标
            knn_stats = knn_evaluation(all_embeddings_tensor, all_labels_tensor, k=10)
            knn_accuracy = knn_stats.get('knn_accuracy', 0.0)
        else:
            margin_pos_sim = 0.0
            margin_neg_sim = 0.0
            margin = 0.0
            knn_accuracy = 0.0
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            # 相似度分布统计（Margin指标）
            'margin_pos_sim': margin_pos_sim,
            'margin_neg_sim': margin_neg_sim,
            'margin': margin,
            # kNN评估指标（独立指标）
            'knn_accuracy': knn_accuracy,
            'skipped_batches': skipped_batches,
        }
    
    
    def build_model(
        self,
        model_config: dict,
        moco_config: dict,
        use_moco: bool,
        image_size: int,
        freeze_backbone: bool,
    ) -> tuple:
        """构建模型，委托给 embedding_model 统一建模工厂 ``build_supcon_model``。

        Returns:
            (model, moco_queue): 模型和 MoCo 队列（不使用 MoCo 时为 None）。
        """
        return build_supcon_model(
            model_config,
            image_size=image_size,
            use_moco=use_moco,
            moco_config=moco_config,
            freeze_backbone=freeze_backbone,
            device=self.device,
            logger=self.logger,
        )
    
    
    def build_loss_func(
        self,
        loss_config: dict,
        moco_config: dict,
        use_moco: bool,
        enable_segmentation: bool = True,
    ) -> tuple:
        """
        构建损失函数
        
        Args:
            loss_config: 损失函数配置
            moco_config: MoCo配置
            use_moco: 是否使用MoCo
            enable_segmentation: 是否启用分割分支
            
        Returns:
            (对比损失函数, 分割损失函数) 或 (对比损失函数, None)
        """
        if use_moco:
            # MoCo Loss
            moco_loss_type = moco_config.get('loss_type', 'supervised')  # 'supervised' 或 'standard'
            self.logger.info(f"使用MoCoLoss（类型: {moco_loss_type}）")
            
            if moco_loss_type == 'supervised':
                # 监督对比loss（仅同label为positive，不使用相似度矩阵）
                # 获取负样本惩罚参数（从loss_config中读取，如果不存在则使用默认值）
                neg_weight = loss_config.get('neg_weight', 0.5)
                margin = loss_config.get('margin', 0.0)
                self.logger.info(f"  负样本惩罚权重: {neg_weight}, margin: {margin}")
                criterion = MoCoLoss(
                    temperature=loss_config['supcon']['temperature'],
                    loss_type='supervised',
                    neg_weight=neg_weight,
                    margin=margin
                ).to(self.device)
            else:
                # 标准MoCo loss（InfoNCE）
                # 获取负样本惩罚参数（从loss_config中读取，如果不存在则使用默认值）
                neg_weight = loss_config.get('neg_weight', 0.5)
                margin = loss_config.get('margin', 0.0)
                self.logger.info(f"  负样本惩罚权重: {neg_weight}, margin: {margin}")
                criterion = MoCoLoss(
                    temperature=loss_config['supcon']['temperature'],
                    loss_type='standard',
                    neg_weight=neg_weight,
                    margin=margin
                ).to(self.device)
        else:
            # 标准SupCon Loss（仅同label为positive，不使用相似度矩阵）
            self.logger.info("使用SupervisedContrastiveLoss（标准SupCon模式，仅同label为positive）")
            criterion = SupervisedContrastiveLoss(
                temperature=loss_config['supcon']['temperature'],
            ).to(self.device)
        
        # 构建分割损失函数
        seg_criterion = None
        if enable_segmentation:
            seg_config = loss_config.get('segmentation', {})
            self.logger.info("构建分割损失函数（ComprehensiveSegmentationLoss）")
            self.logger.info(f"  BCE权重: {seg_config.get('bce_weight', 0.4)}, "
                       f"Dice权重: {seg_config.get('dice_weight', 0.4)}, "
                       f"比例约束权重: {seg_config.get('ratio_weight', 0.1)}")
            seg_criterion = ComprehensiveSegmentationLoss(
                bce_weight=seg_config.get('bce_weight', 0.4),
                dice_weight=seg_config.get('dice_weight', 0.4),
                ratio_weight=seg_config.get('ratio_weight', 0.1),
                target_foreground_ratio=seg_config.get('target_foreground_ratio', 0.1),
                max_foreground_ratio=seg_config.get('max_foreground_ratio', 0.1)
            ).to(self.device)
        
        return criterion, seg_criterion

    def run(self) -> dict:
        """训练循环（数据/模型/优化器/损失均在 `__init__` 中完成）。"""
        self.logger.info("开始训练...")
        train_losses = []
        val_losses = []
        total_epochs = self.training_config['epochs']
        last_train_metrics = None
        last_val_metrics = None

        for epoch in range(self.start_epoch, total_epochs + 1):
            # 解冻backbone（如果需要）
            if epoch == self.freeze_epochs + 1 and self.freeze_epochs > 0:
                self.logger.info("解冻backbone参数...")
                self.model.unfreeze_all()

            train_metrics = self.train_epoch(
                self.model,
                self.train_dataloader,
                self.criterion,
                self.optimizer,
                epoch,
                use_moco=self.use_moco,
                moco_queue=self.moco_queue,
                seg_criterion=self.seg_criterion,
                seg_loss_weight=self.seg_loss_weight,
            )
            train_losses.append(train_metrics['loss'])
            last_train_metrics = train_metrics

            val_metrics = None
            should_eval = (
                self.use_eval
                and self.val_dataloader is not None
                and (epoch % self.eval_interval == 0 or epoch == total_epochs)
            )
            if should_eval:
                val_metrics = self.validate(
                    self.model,
                    self.val_dataloader,
                    self.criterion,
                    use_moco=self.use_moco,
                )
                val_losses.append(val_metrics['loss'])
                last_val_metrics = val_metrics

            # 更新学习率
            self.scheduler.step()

            if self.progress_callback is not None:
                should_stop = self.progress_callback(
                    {
                        'epoch': epoch,
                        'total_epochs': total_epochs,
                        'progress': (epoch / total_epochs) * 100.0,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                    }
                )
                if should_stop:
                    self.logger.info(f"训练在 epoch {epoch} 被用户中止。")
                    break

            # 记录日志
            log_msg = (
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"lr={self.scheduler.get_last_lr()[0]:.6f}"
            )
            if self.use_moco:
                if 'pos_loss' in train_metrics:
                    log_msg += f", pos_loss={train_metrics['pos_loss']:.4f}"
                if 'neg_loss' in train_metrics:
                    log_msg += f", neg_loss={train_metrics['neg_loss']:.4f}"
            log_msg += f"\n  [train] default: loss={train_metrics['loss']:.4f}"
            if val_metrics is not None:
                log_msg += (
                    f"\n  [val] 汇总: loss={val_metrics['loss']:.4f}, "
                    f"PosSim={val_metrics.get('margin_pos_sim', 0):.4f}, "
                    f"NegSim={val_metrics.get('margin_neg_sim', 0):.4f}, "
                    f"Margin={val_metrics.get('margin', 0):.4f}, kNN={val_metrics.get('knn_accuracy', 0):.4f}"
                )
                if self.moco_queue is not None:
                    log_msg += f"\n  moco_queue full: {self.moco_queue.is_full()}"
            self.logger.info(log_msg)

            # 保存 checkpoint
            if epoch % self.training_config.get('save_interval', 5) == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'train_loss': train_metrics['loss'],
                    'best_margin': self.best_margin,
                    'config': self.supcon_config,
                }
                if val_metrics is not None:
                    checkpoint['val_loss'] = val_metrics['loss']
                torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': train_metrics['loss'],
                'best_margin': self.best_margin,
                'config': self.supcon_config,
            }
            if self.use_moco and self.moco_queue is not None:
                checkpoint_data['moco_queue_state'] = self.moco_queue.state_dict()
            torch.save(checkpoint_data, self.checkpoint_dir / "current_model.pth")

        # 绘制损失曲线
        plot_loss_curve(
            train_losses,
            val_losses if self.use_eval else [],
            save_path=str(self.checkpoint_dir / "loss_curve.png"),
            title="SupCon Training Loss",
        )

        self.logger.info("训练完成！")
        return {
            'epochs': total_epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'last_train_metrics': last_train_metrics,
            'last_val_metrics': last_val_metrics,
            'checkpoint_dir': str(self.checkpoint_dir),
            'checkpoint_path': str((self.checkpoint_dir / "current_model.pth").resolve()),
            'loss_curve_path': str((self.checkpoint_dir / "loss_curve.png").resolve()),
        }
