"""
异常点检测器模块
从 notebook 提取的 OutlierDetector 类和辅助函数
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutlierDetector:
    """异常点检测器 - 基于embedding分布检测异常点"""
    
    def __init__(self, embeddings: np.ndarray):
        """
        Args:
            embeddings: (N, D) embedding向量
        """
        self.embeddings = embeddings
        self.n_samples, self.n_features = embeddings.shape
        
        # 计算中心点
        self.center = np.mean(embeddings, axis=0)
        
        # 计算协方差矩阵（用于马氏距离）
        self.cov = np.cov(embeddings.T)
        # 添加正则化防止奇异
        self.cov += np.eye(self.cov.shape[0]) * 1e-6
        try:
            self.cov_inv = np.linalg.inv(self.cov)
        except:
            # 如果仍然奇异，使用伪逆
            self.cov_inv = np.linalg.pinv(self.cov)
    
    def detect_by_distance(self, threshold_percentile: float = 95.0):
        """
        基于到中心的欧氏距离检测异常点
        """
        distances = np.linalg.norm(self.embeddings - self.center, axis=1)
        threshold = np.percentile(distances, threshold_percentile)
        is_outlier = distances > threshold
        return is_outlier, distances, threshold
    
    def detect_by_cosine_distance(self, threshold_percentile: float = 95.0):
        """
        基于到中心的余弦距离检测异常点
        """
        # 计算每个embedding与中心的余弦相似度
        center_normalized = self.center / (np.linalg.norm(self.center) + 1e-8)
        similarities = self.embeddings @ center_normalized
        cosine_distances = 1 - similarities  # 转换为距离
        threshold = np.percentile(cosine_distances, threshold_percentile)
        is_outlier = cosine_distances > threshold
        return is_outlier, cosine_distances, threshold
    
    def detect_by_mahalanobis(self, threshold_percentile: float = 95.0):
        """
        基于马氏距离检测异常点（考虑协方差结构）
        """
        diff = self.embeddings - self.center
        mahalanobis_distances = np.sqrt(np.sum(diff @ self.cov_inv * diff, axis=1))
        threshold = np.percentile(mahalanobis_distances, threshold_percentile)
        is_outlier = mahalanobis_distances > threshold
        return is_outlier, mahalanobis_distances, threshold
    
    def detect_by_isolation_forest(self, contamination: float = 0.05, random_state: int = 42):
        """
        基于Isolation Forest检测异常点
        """
        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        is_outlier = model.fit_predict(self.embeddings) == -1
        scores = model.score_samples(self.embeddings)
        threshold = np.percentile(scores, (1 - contamination) * 100)
        return is_outlier, scores, threshold
    
    def detect_by_lof(self, n_neighbors: int = 20, contamination: float = 0.05):
        """
        基于Local Outlier Factor (LOF)检测异常点
        """
        n_neighbors = min(n_neighbors, self.n_samples - 1)
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False
        )
        is_outlier = model.fit_predict(self.embeddings) == -1
        scores = -model.negative_outlier_factor_  # 转换为正数，越大越异常
        threshold = np.percentile(scores, (1 - contamination) * 100)
        return is_outlier, scores, threshold


def compute_robust_center(embeddings, method='trimmed_mean', trim_ratio=0.1):
    """
    计算鲁棒的类别中心，避免被异常点影响
    
    Args:
        embeddings: (N, D) embedding向量
        method: 'trimmed_mean' | 'median' | 'iterative'
        trim_ratio: 对于trimmed_mean方法，去除的极端值比例
    
    Returns:
        center: (D,) 类别中心向量
    """
    if len(embeddings) < 5:
        # 样本太少，直接用均值
        return np.mean(embeddings, axis=0)
    
    if method == 'median':
        # 使用中位数（对异常值更鲁棒）
        center = np.median(embeddings, axis=0)
        
    elif method == 'trimmed_mean':
        # 修剪均值：去除距离最远的一部分样本后计算均值
        # 1. 先计算初步中心
        initial_center = np.mean(embeddings, axis=0)
        # 2. 计算每个样本到初步中心的距离
        distances = np.linalg.norm(embeddings - initial_center, axis=1)
        # 3. 排序并去除距离最远的trim_ratio比例的样本
        n_trim = int(len(embeddings) * trim_ratio)
        if n_trim > 0:
            sorted_indices = np.argsort(distances)
            kept_indices = sorted_indices[:-n_trim]
            center = np.mean(embeddings[kept_indices], axis=0)
        else:
            center = initial_center
            
    elif method == 'iterative':
        # 迭代方法：多次去除异常点并重新计算中心
        current_embeddings = embeddings.copy()
        for iteration in range(3):  # 最多迭代3次
            center = np.mean(current_embeddings, axis=0)
            distances = np.linalg.norm(current_embeddings - center, axis=1)
            # 只保留距离在95%分位数以内的样本
            threshold = np.percentile(distances, 95)
            mask = distances <= threshold
            if np.sum(mask) < len(current_embeddings) * 0.5:
                # 如果剩余样本太少，停止迭代
                break
            current_embeddings = current_embeddings[mask]
        center = np.mean(current_embeddings, axis=0)
    
    else:
        # 默认使用普通均值
        center = np.mean(embeddings, axis=0)
    
    return center


def find_most_similar_label(embedding, current_label, label_centers, top_k=3):
    """
    找到与给定embedding最相似的其他类别
    
    Args:
        embedding: 目标embedding向量
        current_label: 当前类别（排除该类别）
        label_centers: 所有类别的中心embedding字典
        top_k: 返回前k个最相似的类别
    
    Returns:
        List of (label, similarity_score) tuples
    """
    similarities = []
    for label, center in label_centers.items():
        if label == current_label:
            continue
        # 计算余弦相似度
        similarity = np.dot(embedding, center)
        similarities.append((label, float(similarity)))
    
    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

