"""
类内一致性分析
"""
import numpy as np
from typing import Dict, List
from ..embedding_tools.statistics import compute_embedding_statistics


class ConsistencyAnalyzer:
    """类内一致性分析器"""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        初始化分析器
        
        Args:
            embeddings: 特征向量 (N, D)
            labels: 标签 (N,)
        """
        self.embeddings = embeddings
        self.labels = labels
        self.stats = compute_embedding_statistics(embeddings, labels)
    
    def analyze(self, threshold_percentile: float = 90.0) -> Dict:
        """
        分析类内一致性
        
        Args:
            threshold_percentile: 阈值百分位数（用于识别不一致的类）
            
        Returns:
            分析结果字典
        """
        intra_distances = self.stats['intra_class_distances']
        
        # 计算阈值
        all_intra_distances = list(intra_distances.values())
        threshold = np.percentile(all_intra_distances, threshold_percentile)
        
        # 识别不一致的类
        inconsistent_classes = []
        for label, distance in intra_distances.items():
            if distance > threshold:
                inconsistent_classes.append({
                    'label': label,
                    'intra_distance': distance,
                    'threshold': threshold
                })
        
        # 排序
        inconsistent_classes.sort(key=lambda x: x['intra_distance'], reverse=True)
        
        return {
            'intra_class_distances': intra_distances,
            'threshold': threshold,
            'inconsistent_classes': inconsistent_classes,
            'summary': {
                'total_classes': len(intra_distances),
                'inconsistent_count': len(inconsistent_classes),
                'mean_intra_distance': np.mean(all_intra_distances),
                'std_intra_distance': np.std(all_intra_distances)
            }
        }
    
    def get_class_statistics(self, label: str) -> Dict:
        """
        获取特定类别的统计信息
        
        Args:
            label: 类别标签
            
        Returns:
            统计信息字典
        """
        mask = self.labels == label
        class_embeddings = self.embeddings[mask]
        
        if len(class_embeddings) < 2:
            return {
                'sample_count': len(class_embeddings),
                'intra_distance': 0.0,
                'mean_distance_to_center': 0.0,
                'std_distance_to_center': 0.0
            }
        
        center = self.stats['class_centers'][label]
        distances = np.linalg.norm(class_embeddings - center, axis=1)
        
        return {
            'sample_count': len(class_embeddings),
            'intra_distance': self.stats['intra_class_distances'][label],
            'mean_distance_to_center': np.mean(distances),
            'std_distance_to_center': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }

