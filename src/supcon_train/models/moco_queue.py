"""
MoCo负样本队列管理器
维护FIFO队列存储embeddings和labels，用于对比学习
"""
import torch
import torch.nn as nn


class MoCoQueue(nn.Module):
    """
    MoCo负样本队列管理器
    
    功能：
    - 维护FIFO队列存储embeddings和labels
    - 队列大小固定，新样本加入时自动移除最旧的样本
    - 队列存储在CPU以节省GPU显存，需要时移到GPU
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
        self.register_buffer('queue', torch.randn(queue_size, embedding_dim))
        self.register_buffer('labels', torch.zeros(queue_size, dtype=torch.long))
        
        # 队列指针：指向下一个要写入的位置
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
        
        # 队列是否已经填满过（用于判断队列状态）
        self.register_buffer('is_filled', torch.zeros(1, dtype=torch.bool))
        
        # 初始化队列（归一化）
        self.queue = torch.nn.functional.normalize(self.queue, dim=1, p=2, eps=1e-8)
    
    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        将新的embeddings和labels加入队列（FIFO）
        
        Args:
            embeddings: 新的embeddings (B, D)，已归一化
            labels: 新的labels (B,)
        """
        batch_size = embeddings.shape[0]
        
        # 确保embeddings已归一化
        embeddings = torch.nn.functional.normalize(embeddings, dim=1, p=2, eps=1e-8)
        
        # 将embeddings和labels移到CPU（如果当前在GPU上）
        embeddings = embeddings.cpu()
        labels = labels.cpu()
        
        # 计算要替换的位置
        ptr = int(self.ptr)
        
        # 如果batch_size超过队列大小，只保留最后queue_size个
        if batch_size >= self.queue_size:
            # 直接替换整个队列
            self.queue = embeddings[-self.queue_size:]
            self.labels = labels[-self.queue_size:]
            self.ptr[0] = 0
            self.is_filled[0] = True
        else:
            # 替换从ptr开始的batch_size个位置
            # 如果队列未满，直接追加；如果队列已满，FIFO替换
            end_ptr = ptr + batch_size
            
            if end_ptr <= self.queue_size:
                # 未超出队列范围
                self.queue[ptr:end_ptr] = embeddings
                self.labels[ptr:end_ptr] = labels
                if end_ptr == self.queue_size:
                    # 队列已满
                    self.is_filled[0] = True
            else:
                # 超出队列范围，需要循环
                remaining = self.queue_size - ptr
                self.queue[ptr:] = embeddings[:remaining]
                self.labels[ptr:] = labels[:remaining]
                self.queue[:batch_size - remaining] = embeddings[remaining:]
                self.labels[:batch_size - remaining] = labels[remaining:]
                # 队列已经循环一次，标记为已填满
                self.is_filled[0] = True
            
            # 更新指针（循环）
            self.ptr[0] = (end_ptr % self.queue_size)
    
    def get_queue(self, device: torch.device = None) -> tuple:
        """
        获取队列中的所有embeddings和labels
        
        Args:
            device: 目标设备（如果为None，返回CPU上的tensor）
            
        Returns:
            (queue, labels): 队列中的embeddings和labels
        """
        ptr = int(self.ptr)
        
        # 如果队列已填满过，返回整个队列
        if self.is_filled[0]:
            queue = self.queue
            labels = self.labels
        else:
            # 队列未满，只返回已填充的部分 [0:ptr]
            queue = self.queue[:ptr]
            labels = self.labels[:ptr]
        
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
        self.ptr[0] = 0
        self.is_filled[0] = False

