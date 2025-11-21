"""
评估指标
"""
import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)
from typing import List, Dict, Tuple, Optional


def compute_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    pred_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算聚类评估指标
    
    Args:
        embeddings: 特征向量 (N, D)
        labels: 真实标签 (N,)
        pred_labels: 预测标签（如果已有聚类结果）
        
    Returns:
        指标字典
    """
    metrics = {}
    
    if pred_labels is not None:
        # NMI (Normalized Mutual Information)
        metrics['nmi'] = normalized_mutual_info_score(labels, pred_labels)
        
        # ARI (Adjusted Rand Index)
        metrics['ari'] = adjusted_rand_score(labels, pred_labels)
    
    # Silhouette Score
    if len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(embeddings, labels)
    
    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        指标字典（准确率、精确率、召回率、F1）
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


def compute_intra_class_distance(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    计算类内平均距离
    
    Args:
        embeddings: 特征向量 (N, D)
        labels: 标签 (N,)
        
    Returns:
        每个类别的类内平均距离
    """
    unique_labels = np.unique(labels)
    intra_distances = {}
    
    for label in unique_labels:
        mask = labels == label
        class_embeddings = embeddings[mask]
        
        if len(class_embeddings) < 2:
            intra_distances[label] = 0.0
            continue
        
        # 计算类中心
        center = np.mean(class_embeddings, axis=0)
        
        # 计算到中心的距离
        distances = np.linalg.norm(class_embeddings - center, axis=1)
        intra_distances[label] = np.mean(distances)
    
    return intra_distances


def compute_inter_class_distance(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    计算类间距离
    
    Args:
        embeddings: 特征向量 (N, D)
        labels: 标签 (N,)
        
    Returns:
        (距离矩阵, 类别中心字典)
    """
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    distance_matrix = np.zeros((n_classes, n_classes))
    class_centers = {}
    
    # 计算每个类别的中心
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_embeddings = embeddings[mask]
        center = np.mean(class_embeddings, axis=0)
        class_centers[label] = center
    
    # 计算类间距离
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                dist = np.linalg.norm(class_centers[label1] - class_centers[label2])
                distance_matrix[i, j] = dist
    
    return distance_matrix, class_centers

