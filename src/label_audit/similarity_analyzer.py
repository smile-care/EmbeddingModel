"""
类间相似度分析
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from ..embedding_tools.statistics import compute_embedding_statistics


class SimilarityAnalyzer:
    """类间相似度分析器"""
    
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
        self.unique_labels = self.stats['unique_labels']
        self.distance_matrix = self.stats['inter_class_distance_matrix']
    
    def find_similar_classes(self, top_k: int = 10) -> List[Dict]:
        """
        找出最相似的类别对
        
        Args:
            top_k: 返回前K个最相似的类别对
            
        Returns:
            相似类别对列表
        """
        n_classes = len(self.unique_labels)
        similar_pairs = []
        
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                label1 = self.unique_labels[i]
                label2 = self.unique_labels[j]
                distance = self.distance_matrix[i, j]
                
                # 获取样本数
                count1 = np.sum(self.labels == label1)
                count2 = np.sum(self.labels == label2)
                
                similar_pairs.append({
                    'label1': label1,
                    'label2': label2,
                    'distance': distance,
                    'count1': count1,
                    'count2': count2
                })
        
        # 按距离排序
        similar_pairs.sort(key=lambda x: x['distance'])
        
        return similar_pairs[:top_k]
    
    def hierarchical_clustering(
        self,
        method: str = 'ward',
        save_path: Optional[str] = None
    ) -> Dict:
        """
        层次聚类分析
        
        Args:
            method: 聚类方法 ('ward', 'complete', 'average', 'single')
            save_path: 保存dendrogram的路径
            
        Returns:
            聚类结果字典
        """
        # 使用距离矩阵进行聚类
        linkage_matrix = linkage(
            self.distance_matrix,
            method=method
        )
        
        # 绘制dendrogram
        if save_path:
            plt.figure(figsize=(12, 8))
            dendrogram(
                linkage_matrix,
                labels=self.unique_labels,
                leaf_rotation=90,
                leaf_font_size=10
            )
            plt.title('Class Hierarchical Clustering', fontsize=14)
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Distance', fontsize=12)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            'linkage_matrix': linkage_matrix,
            'labels': self.unique_labels.tolist()
        }
    
    def get_merge_suggestions(
        self,
        distance_threshold: float = None,
        percentile: float = 10.0
    ) -> List[Tuple[str, str, float]]:
        """
        获取类别合并建议
        
        Args:
            distance_threshold: 距离阈值（None表示使用百分位数）
            percentile: 距离百分位数（用于确定阈值）
            
        Returns:
            合并建议列表 [(label1, label2, distance), ...]
        """
        if distance_threshold is None:
            # 使用下百分位数作为阈值
            distances = []
            n = len(self.unique_labels)
            for i in range(n):
                for j in range(i + 1, n):
                    distances.append(self.distance_matrix[i, j])
            distance_threshold = np.percentile(distances, percentile)
        
        suggestions = []
        n = len(self.unique_labels)
        for i in range(n):
            for j in range(i + 1, n):
                if self.distance_matrix[i, j] < distance_threshold:
                    label1 = self.unique_labels[i]
                    label2 = self.unique_labels[j]
                    suggestions.append((
                        label1,
                        label2,
                        self.distance_matrix[i, j]
                    ))
        
        # 按距离排序
        suggestions.sort(key=lambda x: x[2])
        
        return suggestions

