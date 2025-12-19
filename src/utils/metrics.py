"""
评估指标
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)
from sklearn.neighbors import KNeighborsClassifier
from typing import List, Dict, Tuple, Optional



def similarity_distribution_stats(embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    计算相似度分布统计指标（论文级评估指标）
    
    指标定义：
    - PosSim: 正样本对的平均余弦相似度（同类别样本）
    - NegSim: 负样本对的平均余弦相似度（不同类别样本）
    - Margin: 正负样本相似度间隔（PosSim - NegSim）
    
    Args:
        embeddings: [N, D] 已归一化的embedding向量
        labels: [N] 样本标签
        
    Returns:
        dict: 包含 pos_sim, neg_sim, margin 的字典
    """
    # 计算余弦相似度矩阵
    sim = embeddings @ embeddings.T  # (N, N)
    
    # 构建标签mask
    labels_expanded = labels.unsqueeze(1)  # (N, 1)
    
    # 正样本mask：相同标签且排除对角线（自己和自己）
    pos_mask = (labels_expanded == labels_expanded.T) & (~torch.eye(len(labels), device=labels.device, dtype=bool))
    
    # 负样本mask：不同标签
    neg_mask = (labels_expanded != labels_expanded.T)
    
    # 计算正样本平均相似度
    if pos_mask.sum() > 0:
        pos_sim = sim[pos_mask].mean().item()
    else:
        pos_sim = 0.0
    
    # 计算负样本平均相似度
    if neg_mask.sum() > 0:
        neg_sim = sim[neg_mask].mean().item()
    else:
        neg_sim = 0.0
    
    # 计算正负样本间隔（核心指标）
    margin = pos_sim - neg_sim
    
    return {
        'pos_sim': pos_sim,
        'neg_sim': neg_sim,
        'margin': margin
    }



def cosine_similarity_stats(embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    计算余弦相似度统计指标（正负样本对）
    
    Args:
        embeddings: [N, D] 已归一化的embedding向量
        labels: [N] 样本标签
        
    Returns:
        dict: 包含 pos_sim, neg_sim 的字典
    """
    # 计算余弦相似度矩阵
    sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)  # (N, N)
    
    # 构建标签mask
    labels_expanded = labels.unsqueeze(1)  # (N, 1)
    
    # 正样本mask：相同标签且排除对角线（自己和自己）
    pos_mask = (labels_expanded == labels_expanded.T) & (~torch.eye(len(labels), device=labels.device, dtype=bool))
    
    # 负样本mask：不同标签
    neg_mask = (labels_expanded != labels_expanded.T)
    
    # 计算正样本平均相似度
    pos_sim_values = sim[pos_mask]
    if pos_sim_values.numel() > 0:  # 防止没有正样本对的情况
        pos_sim = pos_sim_values.mean().item()
    else:
        pos_sim = 0.0
    
    # 计算负样本平均相似度
    neg_sim_values = sim[neg_mask]
    if neg_sim_values.numel() > 0:  # 防止没有负样本对的情况
        neg_sim = neg_sim_values.mean().item()
    else:
        neg_sim = 0.0
    
    return {
        'pos_sim': pos_sim,  # 正样本对的平均余弦相似度
        'neg_sim': neg_sim,  # 负样本对的平均余弦相似度
    }


def knn_evaluation(embeddings: torch.Tensor, labels: torch.Tensor, k=10) -> dict:
    """
    计算kNN评估指标（k近邻准确率）
    
    Args:
        embeddings: [N, D] 嵌入向量
        labels: [N] 样本标签
        k: kNN 中的 k 值（默认为 10）
        
    Returns:
        dict: kNN 评估结果，包括准确率
    """
    # 将 embeddings 和 labels 转换为 numpy 数组
    embeddings_np = embeddings.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()
    
    # 检查是否有足够的样本进行kNN评估
    if len(embeddings_np) < k + 1:
        return {
            'knn_accuracy': 0.0,
            'k': k,
            'note': f'样本数不足（需要至少{k+1}个样本，当前{len(embeddings_np)}个）'
        }
    
    # 检查类别数量
    unique_labels = np.unique(labels_np)
    if len(unique_labels) < 2:
        return {
            'knn_accuracy': 0.0,
            'k': k,
            'note': f'类别数不足（需要至少2个类别，当前{len(unique_labels)}个）'
        }
    
    # 创建 kNN 分类器
    knn = KNeighborsClassifier(n_neighbors=min(k, len(embeddings_np) - 1))
    
    # 使用训练集训练 kNN 分类器
    knn.fit(embeddings_np, labels_np)
    
    # 进行预测
    predictions = knn.predict(embeddings_np)
    
    # 计算 kNN 的准确率
    accuracy = np.mean(predictions == labels_np)  # 计算准确率
    
    return {
        'knn_accuracy': float(accuracy),
        'k': k
    }



