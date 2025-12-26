"""
监督对比学习Loss实现
支持自定义相似度矩阵

提供两种实现方式：
1. SupervisedContrastiveLoss: 标准对比loss，使用相似度作为权重
2. SimilarityTargetLoss: MSE loss，使用相似度作为目标值
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    标准监督对比学习Loss
    使用相似度作为权重进行加权平均
    
    特点：
    - 相似度高的样本对权重更大，对loss贡献更大
    - 相似度低的样本对权重较小，对loss贡献较小
    - 使用标准的softmax-based contrastive loss
    """
    
    def __init__(
        self, 
        temperature: float = 0.07,
        similarity_matrix: Optional[np.ndarray] = None,
        default_similarity: float = 0.0,
        use_similarity_matrix: bool = True
    ):
        """
        初始化SupCon Loss
        
        Args:
            temperature: 温度参数，越小越关注困难样本
            similarity_matrix: 类别相似度矩阵 (num_classes, num_classes)，如果为None则使用标准SupCon
            default_similarity: 默认相似度阈值，大于此值的类别对被认为是positive pairs
            use_similarity_matrix: 是否使用相似度矩阵（False时强制使用标准SupCon，仅同label为positive）
        """
        super().__init__()
        self.temperature = temperature
        self.use_similarity_matrix = use_similarity_matrix
        
        # 根据use_similarity_matrix决定是否使用相似度矩阵
        if use_similarity_matrix and similarity_matrix is not None:
            self.similarity_matrix = similarity_matrix
            self.default_similarity = default_similarity
            # 转换为tensor并注册为buffer
            self.register_buffer(
                'similarity_matrix_tensor',
                torch.from_numpy(similarity_matrix).float()
            )
        else:
            # 不使用相似度矩阵，使用标准SupCon
            self.similarity_matrix = None
            self.default_similarity = 0.0
            self.register_buffer('similarity_matrix_tensor', None)
    
    def forward(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算SupCon Loss（使用相似度作为权重）
        
        Args:
            features: 特征向量 (B, D)
            labels: 标签 (B,)，每个元素是类别的索引
            
        Returns:
            loss值
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 检查输入是否包含NaN或Inf
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("警告：输入特征包含NaN或Inf！")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 归一化特征
        features = F.normalize(features, dim=1, p=2, eps=1e-8)
        
        # 检查归一化后是否包含NaN
        if torch.isnan(features).any():
            print("警告：归一化后特征包含NaN！")
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            features = F.normalize(features, dim=1, p=2, eps=1e-8)
        
        # 计算特征相似度矩阵 (B, B)
        feature_similarity = torch.matmul(features, features.T) / self.temperature
        
        # 创建权重矩阵：使用相似度数值作为权重
        labels = labels.contiguous()  # (B,)
        
        # 移除自己与自己的相似度（对角线）
        logits_mask = torch.ones(batch_size, batch_size, device=device) - torch.eye(batch_size, device=device)
        
        if self.similarity_matrix_tensor is not None:
            # 使用自定义相似度矩阵作为权重
            label_similarity = self.similarity_matrix_tensor[labels][:, labels]  # (B, B)
            
            # 注意：这里直接使用相似度值作为权重
            # - 相似度=1.0的样本对，权重=1.0（完全positive）
            # - 相似度=0.6的样本对，权重=0.6（部分positive，贡献较小）
            # - 相似度=0.0的样本对，权重=0.0（negative pairs）
            # 这种设计允许相似度不同的positive pairs有不同的贡献度
            # 如果希望所有positive pairs权重相同，可以将相似度>0的值都设为1.0
            weights = label_similarity.float().to(device)  # (B, B)，直接使用相似度作为权重
            
            # 排除对角线（自己与自己）
            weights = weights * logits_mask
            
            # 只考虑相似度大于default_similarity的样本对（可选）
            if self.default_similarity > 0.0:
                weights = weights * (weights > self.default_similarity).float()
        else:
            # 标准SupCon：同label的样本为positive，权重为1.0
            labels_expanded = labels.unsqueeze(1)  # (B, 1)
            weights = torch.eq(labels_expanded, labels_expanded.T).float().to(device)  # (B, B)
            weights = weights * logits_mask  # 排除对角线
        
        # 计算exp，添加数值稳定性
        feature_similarity = feature_similarity * logits_mask  # 将对角线设为0
        logits_max, _ = torch.max(feature_similarity, dim=1, keepdim=True)
        logits = feature_similarity - logits_max.detach()  # 数值稳定
        
        exp_logits = torch.exp(logits) * logits_mask
        
        # 计算log概率
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 使用加权平均：相似度高的样本对权重更大
        weight_sum = weights.sum(1)  # (B,)，每个样本的权重总和
        valid_mask = weight_sum > 0  # 有权重>0的样本对
        
        if valid_mask.sum() == 0:
            print("警告：batch中没有positive pairs，返回小的正loss以保持梯度流动")
            # 返回一个小的正数而不是0，以保持梯度流动
            return torch.tensor(1e-6, device=device, requires_grad=True)
        
        # 加权平均log概率
        weighted_mean_log_prob = (weights * log_prob).sum(1) / (weight_sum + 1e-8)
        weighted_mean_log_prob = weighted_mean_log_prob[valid_mask]
        
        # 平均loss（取负，因为我们要最大化log概率）
        loss = -weighted_mean_log_prob.mean()
        
        # 检查loss是否为NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告：Loss为NaN或Inf！feature_similarity范围: [{feature_similarity.min():.4f}, {feature_similarity.max():.4f}]")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss