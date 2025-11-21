"""
离群点检测
"""
import numpy as np
from typing import Dict, List, Tuple
from ..embedding_tools.knn_search import KNNSearcher
from ..embedding_tools.statistics import compute_embedding_statistics


class OutlierDetector:
    """离群点检测器"""
    
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        instance_ids: np.ndarray
    ):
        """
        初始化检测器
        
        Args:
            embeddings: 特征向量 (N, D)
            labels: 标签 (N,)
            instance_ids: 实例ID (N,)
        """
        self.embeddings = embeddings
        self.labels = labels
        self.instance_ids = instance_ids
        self.stats = compute_embedding_statistics(embeddings, labels)
        self.knn_searcher = KNNSearcher(embeddings, use_gpu=True)
    
    def detect_outliers(
        self,
        top_percent: float = 5.0,
        k: int = 10,
        alpha: float = 0.6,
        beta: float = 0.4
    ) -> List[Dict]:
        """
        检测离群点
        
        Args:
            top_percent: 返回前百分之几的离群点
            k: kNN的k值
            alpha: 类内距离z-score权重
            beta: kNN一致性权重
            alpha + beta 应该等于 1
            
        Returns:
            离群点列表
        """
        n_samples = len(self.embeddings)
        scores = []
        
        for i in range(n_samples):
            label = self.labels[i]
            
            # 1. 计算到类中心的距离z-score
            label_mask = self.labels == label
            label_indices = np.where(label_mask)[0]
            label_idx_in_class = np.where(label_indices == i)[0][0]
            
            z_scores = self.stats['z_scores'][label]
            z_score = z_scores[label_idx_in_class]
            
            # 2. 计算kNN一致性
            _, neighbor_labels, consistency = self.knn_searcher.get_neighbor_labels(
                i, self.labels, k
            )
            inconsistency = 1.0 - consistency
            
            # 3. 综合评分
            score = alpha * abs(z_score) + beta * inconsistency
            
            scores.append({
                'instance_id': self.instance_ids[i],
                'label': label,
                'score': score,
                'z_score': z_score,
                'knn_consistency': consistency,
                'distance_to_center': self.stats['sample_to_center_distances'][label][label_idx_in_class],
                'index': i
            })
        
        # 排序
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回top_percent
        n_outliers = max(1, int(n_samples * top_percent / 100))
        return scores[:n_outliers]
    
    def get_class_outliers(
        self,
        label: str,
        top_n: int = 10
    ) -> List[Dict]:
        """
        获取特定类别的离群点
        
        Args:
            label: 类别标签
            top_n: 返回前N个离群点
            
        Returns:
            离群点列表
        """
        mask = self.labels == label
        label_indices = np.where(mask)[0]
        
        if len(label_indices) < 2:
            return []
        
        # 计算该类内所有样本的z-score
        z_scores = self.stats['z_scores'][label]
        
        # 计算kNN一致性
        consistencies = []
        for idx in label_indices:
            _, _, consistency = self.knn_searcher.get_neighbor_labels(
                idx, self.labels, k=10
            )
            consistencies.append(consistency)
        
        # 综合评分
        scores = []
        for i, idx in enumerate(label_indices):
            z_score = z_scores[i]
            consistency = consistencies[i]
            score = 0.6 * abs(z_score) + 0.4 * (1.0 - consistency)
            
            scores.append({
                'instance_id': self.instance_ids[idx],
                'label': label,
                'score': score,
                'z_score': z_score,
                'knn_consistency': consistency,
                'index': idx
            })
        
        # 排序
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        return scores[:top_n]

