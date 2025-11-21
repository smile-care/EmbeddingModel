"""
跨场景迁移评估
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from ..utils.metrics import compute_clustering_metrics


class CrossDomainEvaluator:
    """跨场景迁移评估器"""
    
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        domain_ids: np.ndarray
    ):
        """
        初始化评估器
        
        Args:
            embeddings: 特征向量 (N, D)
            labels: 标签 (N,)
            domain_ids: 场景ID (N,)
        """
        self.embeddings = embeddings
        self.labels = labels
        self.domain_ids = domain_ids
        self.unique_domains = np.unique(domain_ids)
    
    def evaluate_new_domain(
        self,
        source_domains: List[str],
        target_domain: str,
        n_train: int = 100
    ) -> Dict:
        """
        评估新场景的迁移性能
        
        Args:
            source_domains: 源场景列表
            target_domain: 目标场景
            n_train: 目标场景的训练样本数
            
        Returns:
            评估结果字典
        """
        # 划分数据
        source_mask = np.isin(self.domain_ids, source_domains)
        target_mask = self.domain_ids == target_domain
        
        source_embeddings = self.embeddings[source_mask]
        source_labels = self.labels[source_mask]
        target_embeddings = self.embeddings[target_mask]
        target_labels = self.labels[target_mask]
        
        # 采样目标场景的训练集和测试集
        n_total = len(target_embeddings)
        if n_total < n_train:
            n_train = n_total // 2
        
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        target_train_embeddings = target_embeddings[train_indices]
        target_train_labels = target_labels[train_indices]
        target_test_embeddings = target_embeddings[test_indices]
        target_test_labels = target_labels[test_indices]
        
        # 1. 仅使用embedding + kNN（不微调）
        knn_acc, knn_f1 = self._knn_only(
            source_embeddings, source_labels,
            target_test_embeddings, target_test_labels
        )
        
        # 2. 在目标场景上微调线性头
        linear_acc, linear_f1 = self._linear_finetune(
            source_embeddings, source_labels,
            target_train_embeddings, target_train_labels,
            target_test_embeddings, target_test_labels
        )
        
        # 3. 聚类评估
        clustering_metrics = self._clustering_eval(
            target_embeddings, target_labels
        )
        
        return {
            'target_domain': target_domain,
            'source_domains': source_domains,
            'knn_accuracy': knn_acc,
            'knn_f1': knn_f1,
            'linear_accuracy': linear_acc,
            'linear_f1': linear_f1,
            'clustering_metrics': clustering_metrics
        }
    
    def _knn_only(
        self,
        source_embeddings: np.ndarray,
        source_labels: np.ndarray,
        target_test_embeddings: np.ndarray,
        target_test_labels: np.ndarray,
        k: int = 5
    ) -> Tuple[float, float]:
        """仅使用kNN（不微调）"""
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(source_embeddings, source_labels)
        pred_labels = knn.predict(target_test_embeddings)
        
        acc = accuracy_score(target_test_labels, pred_labels)
        f1 = f1_score(target_test_labels, pred_labels, average='weighted', zero_division=0)
        
        return acc, f1
    
    def _linear_finetune(
        self,
        source_embeddings: np.ndarray,
        source_labels: np.ndarray,
        target_train_embeddings: np.ndarray,
        target_train_labels: np.ndarray,
        target_test_embeddings: np.ndarray,
        target_test_labels: np.ndarray
    ) -> Tuple[float, float]:
        """在目标场景上微调线性头"""
        # 先在源场景上预训练
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(source_embeddings, source_labels)
        
        # 在目标场景上微调
        clf.fit(target_train_embeddings, target_train_labels)
        pred_labels = clf.predict(target_test_embeddings)
        
        acc = accuracy_score(target_test_labels, pred_labels)
        f1 = f1_score(target_test_labels, pred_labels, average='weighted', zero_division=0)
        
        return acc, f1
    
    def _clustering_eval(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """聚类评估"""
        from sklearn.cluster import KMeans
        
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(embeddings)
        
        metrics = compute_clustering_metrics(embeddings, labels, pred_labels)
        
        return metrics

