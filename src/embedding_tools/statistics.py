"""
Embedding统计工具
"""
import numpy as np
from typing import Dict, Tuple
from ..utils.metrics import (
    compute_intra_class_distance,
    compute_inter_class_distance
)


def compute_embedding_statistics(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict:
    """
    计算embedding统计信息
    
    Args:
        embeddings: 特征向量 (N, D)
        labels: 标签 (N,)
        
    Returns:
        统计信息字典
    """
    # 类内距离
    intra_distances = compute_intra_class_distance(embeddings, labels)
    
    # 类间距离
    inter_distance_matrix, class_centers = compute_inter_class_distance(embeddings, labels)
    
    # 计算每个样本到类中心的距离
    unique_labels = np.unique(labels)
    sample_to_center_distances = {}
    z_scores = {}
    
    for label in unique_labels:
        mask = labels == label
        class_embeddings = embeddings[mask]
        center = class_centers[label]
        
        # 计算距离
        distances = np.linalg.norm(class_embeddings - center, axis=1)
        sample_to_center_distances[label] = distances
        
        # 计算z-score
        mean_dist = np.mean(distances)
        std_dist = np.std(distances) + 1e-8
        z_scores[label] = (distances - mean_dist) / std_dist
    
    return {
        'intra_class_distances': intra_distances,
        'inter_class_distance_matrix': inter_distance_matrix,
        'class_centers': class_centers,
        'sample_to_center_distances': sample_to_center_distances,
        'z_scores': z_scores,
        'unique_labels': unique_labels
    }

