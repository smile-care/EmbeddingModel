"""
MoCo负样本队列管理器
维护FIFO队列存储embeddings、labels 和 scene_id，用于对比学习；
多场景时可按 scene_id 检索，仅用同场景负样本计算 loss。
"""
from typing import Optional, Union

import torch
import torch.nn as nn


class MoCoQueue(nn.Module):
    """
    MoCo负样本队列管理器

    功能：
    - 维护FIFO队列存储 embeddings、labels、scene_ids
    - 队列大小固定，新样本加入时自动移除最旧的样本
    - 队列存储在CPU以节省GPU显存，需要时移到GPU
    - scene_ids 用于多场景时按场景检索，计算对比 loss 时可只取同场景负样本
    """

    def __init__(self, queue_size: int = 16384, embedding_dim: int = 128):
        """
        初始化MoCo队列

        Args:
            queue_size: 队列大小（默认16384，适合显存受限）
            embedding_dim: embedding维度
        """
        super().__init__()
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim

        # 注册队列buffer（存储在CPU）
        # queue: (queue_size, embedding_dim)
        # labels: (queue_size,)
        # scene_ids: (queue_size,) 场景标识，-1 表示未使用/不按场景过滤
        self.register_buffer('queue', torch.randn(queue_size, embedding_dim))
        self.register_buffer('labels', torch.zeros(queue_size, dtype=torch.long))
        self.register_buffer('scene_ids', torch.full((queue_size,), -1, dtype=torch.long))

        # 队列指针：指向下一个要写入的位置
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))

        # 队列是否已经填满过（用于判断队列状态）
        self.register_buffer('is_filled', torch.zeros(1, dtype=torch.bool))

        # 初始化队列（归一化）
        self.queue = torch.nn.functional.normalize(self.queue, dim=1, p=2, eps=1e-8)

    @torch.no_grad()
    def enqueue(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        scene_id: Optional[Union[int, torch.Tensor]] = None,
    ):
        """
        将新的 embeddings、labels 加入队列（FIFO），可选写入场景标识。

        Args:
            embeddings: 新的 embeddings (B, D)，已归一化
            labels: 新的 labels (B,)
            scene_id: 场景标识。int 表示本 batch 同属该场景；(B,) 表示每个样本的场景 id；
                       None 表示不记录场景（保持 -1），单场景或不需要按场景过滤时可省略
        """
        batch_size = embeddings.shape[0]

        # 确保embeddings已归一化
        embeddings = torch.nn.functional.normalize(embeddings, dim=1, p=2, eps=1e-8)

        # 将embeddings和labels移到CPU（如果当前在GPU上）
        embeddings = embeddings.cpu()
        labels = labels.cpu()

        if scene_id is not None:
            if isinstance(scene_id, int):
                scene_ids = torch.full((batch_size,), scene_id, dtype=torch.long, device=embeddings.device)
            else:
                scene_ids = scene_id.cpu() if scene_id.is_cuda else scene_id
        else:
            scene_ids = None

        ptr = int(self.ptr)

        if batch_size >= self.queue_size:
            self.queue = embeddings[-self.queue_size:]
            self.labels = labels[-self.queue_size:]
            if scene_ids is not None:
                self.scene_ids = scene_ids[-self.queue_size:]
            self.ptr[0] = 0
            self.is_filled[0] = True
        else:
            end_ptr = ptr + batch_size

            if end_ptr <= self.queue_size:
                self.queue[ptr:end_ptr] = embeddings
                self.labels[ptr:end_ptr] = labels
                if scene_ids is not None:
                    self.scene_ids[ptr:end_ptr] = scene_ids
                if end_ptr == self.queue_size:
                    self.is_filled[0] = True
            else:
                remaining = self.queue_size - ptr
                self.queue[ptr:] = embeddings[:remaining]
                self.labels[ptr:] = labels[:remaining]
                if scene_ids is not None:
                    self.scene_ids[ptr:] = scene_ids[:remaining]
                self.queue[: batch_size - remaining] = embeddings[remaining:]
                self.labels[: batch_size - remaining] = labels[remaining:]
                if scene_ids is not None:
                    self.scene_ids[: batch_size - remaining] = scene_ids[remaining:]
                self.is_filled[0] = True

            self.ptr[0] = end_ptr % self.queue_size

    def get_queue(
        self,
        device: Optional[torch.device] = None,
        scene_id: Optional[int] = None,
    ) -> tuple:
        """
        获取队列中的 embeddings 和 labels，可选仅返回指定场景的条目（便于按场景计算对比 loss）。

        Args:
            device: 目标设备（若为 None，返回 CPU 上的 tensor）
            scene_id: 若给定，只返回 scene_ids == scene_id 的条目；None 表示返回全部

        Returns:
            (queue, labels): 队列中的 embeddings 和 labels，形状 (N, D) 与 (N,)
        """
        ptr = int(self.ptr)

        if self.is_filled[0]:
            queue = self.queue
            labels = self.labels
            scene_ids = self.scene_ids
        else:
            queue = self.queue[:ptr]
            labels = self.labels[:ptr]
            scene_ids = self.scene_ids[:ptr]

        if scene_id is not None:
            mask = scene_ids == scene_id
            queue = queue[mask]
            labels = labels[mask]

        if device is not None:
            queue = queue.to(device)
            labels = labels.to(device)

        return queue, labels
    
    def size(self) -> int:
        """
        返回当前队列的有效大小
        
        Returns:
            队列的有效大小
        """
        # 如果队列已填满过，永远返回queue_size
        if self.is_filled[0]:
            return self.queue_size
        else:
            # 队列未满，返回ptr（已填充的大小）
            return int(self.ptr)
    
    def is_full(self) -> bool:
        """
        检查队列是否已满
        
        Returns:
            如果队列已满返回True，否则返回False
        """
        return bool(self.is_filled[0])
    
    def reset(self):
        """重置队列（清空所有数据）"""
        self.queue = torch.randn(self.queue_size, self.embedding_dim)
        self.queue = torch.nn.functional.normalize(self.queue, dim=1, p=2, eps=1e-8)
        self.labels.zero_()
        self.scene_ids.fill_(-1)
        self.ptr[0] = 0
        self.is_filled[0] = False

