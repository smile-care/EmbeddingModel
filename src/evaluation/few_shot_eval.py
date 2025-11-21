"""
Few-shot评估
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


class FewShotEvaluator:
    """Few-shot评估器"""
    
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
        self.unique_labels = np.unique(labels)
    
    def leave_one_domain_out(
        self,
        test_domain: str,
        n_shot: int = 5,
        n_query: int = 20
    ) -> Dict:
        """
        Leave-one-domain-out评估
        
        Args:
            test_domain: 测试场景
            n_shot: 每个类别的支持样本数
            n_query: 每个类别的查询样本数
            
        Returns:
            评估结果字典
        """
        # 划分训练集和测试集
        train_mask = self.domain_ids != test_domain
        test_mask = self.domain_ids == test_domain
        
        train_embeddings = self.embeddings[train_mask]
        train_labels = self.labels[train_mask]
        test_embeddings = self.embeddings[test_mask]
        test_labels = self.labels[test_mask]
        
        # 采样few-shot支持集
        support_embeddings, support_labels = self._sample_few_shot(
            train_embeddings, train_labels, n_shot
        )
        
        # 采样查询集
        query_embeddings, query_labels = self._sample_query(
            test_embeddings, test_labels, n_query
        )
        
        # kNN分类
        knn_acc, knn_f1 = self._knn_classify(
            support_embeddings, support_labels,
            query_embeddings, query_labels
        )
        
        # 线性分类
        linear_acc, linear_f1 = self._linear_classify(
            support_embeddings, support_labels,
            query_embeddings, query_labels
        )
        
        return {
            'test_domain': test_domain,
            'n_shot': n_shot,
            'n_query': n_query,
            'knn_accuracy': knn_acc,
            'knn_f1': knn_f1,
            'linear_accuracy': linear_acc,
            'linear_f1': linear_f1
        }
    
    def _sample_few_shot(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_shot: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """采样few-shot支持集"""
        unique_labels = np.unique(labels)
        support_embeddings = []
        support_labels = []
        
        for label in unique_labels:
            mask = labels == label
            label_embeddings = embeddings[mask]
            
            if len(label_embeddings) < n_shot:
                # 如果样本数不足，使用所有样本
                selected = label_embeddings
            else:
                # 随机采样
                indices = np.random.choice(len(label_embeddings), n_shot, replace=False)
                selected = label_embeddings[indices]
            
            support_embeddings.append(selected)
            support_labels.extend([label] * len(selected))
        
        return np.vstack(support_embeddings), np.array(support_labels)
    
    def _sample_query(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_query: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """采样查询集"""
        unique_labels = np.unique(labels)
        query_embeddings = []
        query_labels = []
        
        for label in unique_labels:
            mask = labels == label
            label_embeddings = embeddings[mask]
            
            if len(label_embeddings) < n_query:
                selected = label_embeddings
            else:
                indices = np.random.choice(len(label_embeddings), n_query, replace=False)
                selected = label_embeddings[indices]
            
            query_embeddings.append(selected)
            query_labels.extend([label] * len(selected))
        
        return np.vstack(query_embeddings), np.array(query_labels)
    
    def _knn_classify(
        self,
        support_embeddings: np.ndarray,
        support_labels: np.ndarray,
        query_embeddings: np.ndarray,
        query_labels: np.ndarray,
        k: int = 5
    ) -> Tuple[float, float]:
        """kNN分类"""
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(support_embeddings, support_labels)
        pred_labels = knn.predict(query_embeddings)
        
        acc = accuracy_score(query_labels, pred_labels)
        f1 = f1_score(query_labels, pred_labels, average='weighted', zero_division=0)
        
        return acc, f1
    
    def _linear_classify(
        self,
        support_embeddings: np.ndarray,
        support_labels: np.ndarray,
        query_embeddings: np.ndarray,
        query_labels: np.ndarray
    ) -> Tuple[float, float]:
        """线性分类"""
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(support_embeddings, support_labels)
        pred_labels = clf.predict(query_embeddings)
        
        acc = accuracy_score(query_labels, pred_labels)
        f1 = f1_score(query_labels, pred_labels, average='weighted', zero_division=0)
        
        return acc, f1

