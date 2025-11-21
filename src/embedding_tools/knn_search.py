"""
kNN搜索工具
"""
import numpy as np
import faiss
from typing import List, Tuple, Optional
import torch


class KNNSearcher:
    """kNN搜索器（使用Faiss）"""
    
    def __init__(
        self,
        embeddings: np.ndarray,
        use_gpu: bool = True,
        metric: str = 'L2'
    ):
        """
        初始化kNN搜索器
        
        Args:
            embeddings: 特征向量矩阵 (N, D)
            use_gpu: 是否使用GPU
            metric: 距离度量 ('L2' 或 'IP' for inner product)
        """
        self.embeddings = embeddings
        self.n_samples, self.dim = embeddings.shape
        
        # 创建Faiss索引
        if metric == 'L2':
            index = faiss.IndexFlatL2(self.dim)
        else:  # IP (Inner Product)
            index = faiss.IndexFlatIP(self.dim)
            # 归一化embeddings用于cosine相似度
            faiss.normalize_L2(embeddings)
        
        # 使用GPU（如果可用）
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            self.index = index
        
        # 添加向量到索引
        self.index.add(embeddings.astype('float32'))
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索k个最近邻
        
        Args:
            query: 查询向量 (D,) 或 (B, D)
            k: 返回的最近邻数量
            
        Returns:
            (distances, indices) 距离和索引
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype('float32')
        distances, indices = self.index.search(query, k)
        
        return distances, indices
    
    def search_by_index(
        self,
        query_idx: int,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        通过索引搜索（查询样本本身会被排除）
        
        Args:
            query_idx: 查询样本索引
            k: 返回的最近邻数量
            
        Returns:
            (distances, indices)
        """
        query = self.embeddings[query_idx:query_idx+1].astype('float32')
        distances, indices = self.index.search(query, k + 1)  # +1 因为包含自己
        
        # 排除自己
        mask = indices[0] != query_idx
        return distances[0][mask][:k], indices[0][mask][:k]
    
    def get_neighbor_labels(
        self,
        query_idx: int,
        labels: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        获取邻居的标签并计算一致性
        
        Args:
            query_idx: 查询样本索引
            labels: 所有样本的标签数组
            k: 邻居数量
            
        Returns:
            (neighbor_indices, neighbor_labels, consistency_ratio)
        """
        _, neighbor_indices = self.search_by_index(query_idx, k)
        neighbor_labels = labels[neighbor_indices]
        query_label = labels[query_idx]
        
        # 计算一致性（邻居中与查询样本同label的比例）
        consistency = np.mean(neighbor_labels == query_label)
        
        return neighbor_indices, neighbor_labels, consistency

