"""
数据处理模块
负责数据加载、预处理和类别中心计算
"""
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from outlier_detector import compute_robust_center


def decode_if_bytes(x):
    """如果是 bytes 类型，解码为字符串"""
    if isinstance(x, bytes):
        return x.decode('utf-8')
    return str(x)


def load_embeddings(npz_path: str) -> Dict[str, np.ndarray]:
    """
    从 .npz 文件加载数据
    
    Args:
        npz_path: .npz 文件路径
    
    Returns:
        包含 embeddings, label_names, image_paths, mask_paths 的字典
    """
    data = np.load(npz_path)
    
    embeddings = data['embeddings']
    label_names = data['label_names']
    image_paths = data['image_paths']
    mask_paths = data['mask_paths']
    
    # 确保 embeddings 是 numpy 数组
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    
    # 转换路径格式
    image_paths = [decode_if_bytes(p) for p in image_paths]
    mask_paths = [decode_if_bytes(p) for p in mask_paths]
    label_names = [decode_if_bytes(l) for l in label_names]
    
    # 验证数据一致性
    assert len(image_paths) == len(mask_paths) == len(label_names) == len(embeddings), \
        f"数据长度不一致: images={len(image_paths)}, masks={len(mask_paths)}, labels={len(label_names)}, embeddings={len(embeddings)}"
    
    return {
        'embeddings': embeddings,
        'label_names': np.array(label_names),
        'image_paths': np.array(image_paths),
        'mask_paths': np.array(mask_paths)
    }


def preprocess_data(
    embeddings: np.ndarray,
    label_names: np.ndarray,
    image_paths: np.ndarray,
    mask_paths: np.ndarray
) -> Tuple[Dict[str, List[int]], Dict[str, np.ndarray], List[str]]:
    """
    数据预处理：按类别划分数据
    
    Args:
        embeddings: embedding 数组
        label_names: 标签名称数组
        image_paths: 图像路径数组
        mask_paths: 掩码路径数组
    
    Returns:
        label_to_indices: 标签到索引列表的映射
        label_to_embeddings: 标签到embedding数组的映射
        unique_labels: 唯一标签列表
    """
    unique_labels = sorted(list(set(label_names)))
    
    # 为每个类别创建索引映射
    label_to_indices = {}
    label_to_embeddings = {}
    
    for label in unique_labels:
        indices = [i for i, lb in enumerate(label_names) if lb == label]
        label_to_indices[label] = indices
        label_to_embeddings[label] = embeddings[indices]
    
    return label_to_indices, label_to_embeddings, unique_labels


def compute_category_centers(
    label_to_embeddings: Dict[str, np.ndarray],
    unique_labels: List[str],
    method: str = 'trimmed_mean',
    trim_ratio: float = 0.1
) -> Dict[str, np.ndarray]:
    """
    计算每个类别的鲁棒中心embedding
    
    Args:
        label_to_embeddings: 标签到embedding数组的映射
        unique_labels: 唯一标签列表
        method: 中心计算方法 ('trimmed_mean' | 'median' | 'iterative')
        trim_ratio: 对于trimmed_mean方法，去除的极端值比例
    
    Returns:
        label_centers: 标签到中心embedding的映射
    """
    label_centers = {}
    
    for label in tqdm(unique_labels, desc="计算类别中心"):
        label_embeddings = label_to_embeddings[label]
        
        # L2归一化
        label_embeddings_norm = F.normalize(
            torch.from_numpy(label_embeddings), 
            dim=1, 
            p=2
        ).numpy()
        
        # 使用鲁棒方法计算中心
        center = compute_robust_center(
            label_embeddings_norm, 
            method=method, 
            trim_ratio=trim_ratio
        )
        
        # 归一化中心
        center = center / (np.linalg.norm(center) + 1e-8)
        label_centers[label] = center
    
    return label_centers

