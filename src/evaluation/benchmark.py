"""
Baseline对比评估
"""
import numpy as np
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from ..utils.metrics import compute_clustering_metrics


class BenchmarkEvaluator:
    """Baseline对比评估器"""
    
    def __init__(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        labels: np.ndarray
    ):
        """
        初始化评估器
        
        Args:
            embeddings_dict: {method_name: embeddings} 字典
            labels: 标签 (N,)
        """
        self.embeddings_dict = embeddings_dict
        self.labels = labels
        self.unique_labels = np.unique(labels)
    
    def compare_methods(
        self,
        test_size: float = 0.2
    ) -> Dict:
        """
        对比不同方法
        
        Args:
            test_size: 测试集比例
            
        Returns:
            对比结果字典
        """
        results = {}
        
        # 划分训练集和测试集
        n_samples = len(self.labels)
        indices = np.random.permutation(n_samples)
        n_test = int(n_samples * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        for method_name, embeddings in self.embeddings_dict.items():
            # 聚类评估
            clustering_metrics = self._clustering_eval(embeddings, self.labels)
            
            # 分类评估
            train_embeddings = embeddings[train_indices]
            train_labels = self.labels[train_indices]
            test_embeddings = embeddings[test_indices]
            test_labels = self.labels[test_indices]
            
            classification_metrics = self._classification_eval(
                train_embeddings, train_labels,
                test_embeddings, test_labels
            )
            
            results[method_name] = {
                'clustering': clustering_metrics,
                'classification': classification_metrics
            }
        
        return results
    
    def _clustering_eval(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """聚类评估"""
        n_clusters = len(self.unique_labels)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(embeddings)
        
        metrics = compute_clustering_metrics(embeddings, labels, pred_labels)
        
        return metrics
    
    def _classification_eval(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict:
        """分类评估"""
        # kNN分类
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_embeddings, train_labels)
        knn_pred = knn.predict(test_embeddings)
        knn_acc = accuracy_score(test_labels, knn_pred)
        knn_f1 = f1_score(test_labels, knn_pred, average='weighted', zero_division=0)
        
        # 线性分类
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_embeddings, train_labels)
        linear_pred = clf.predict(test_embeddings)
        linear_acc = accuracy_score(test_labels, linear_pred)
        linear_f1 = f1_score(test_labels, linear_pred, average='weighted', zero_division=0)
        
        return {
            'knn_accuracy': knn_acc,
            'knn_f1': knn_f1,
            'linear_accuracy': linear_acc,
            'linear_f1': linear_f1
        }

