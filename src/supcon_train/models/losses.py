"""
监督对比学习Loss实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss"""
    
    def __init__(self, temperature: float = 0.07):
        """
        初始化SupCon Loss
        
        Args:
            temperature: 温度参数
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算SupCon Loss
        
        Args:
            features: 特征向量 (B, D)，已归一化
            labels: 标签 (B,)
            
        Returns:
            loss值
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建mask：同label的样本为positive
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除自己与自己的相似度（对角线）
        logits_mask = torch.scalar_tensor(1) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask
        
        # 计算exp
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算每个样本的loss（只对positive pairs）
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # 避免除零
        mean_log_prob_pos = mean_log_prob_pos[mask.sum(1) > 0]
        
        # 平均loss
        loss = -mean_log_prob_pos.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """组合Loss：SupCon + CrossEntropy"""
    
    def __init__(
        self,
        supcon_weight: float = 1.0,
        ce_weight: float = 0.5,
        temperature: float = 0.07,
        label_smoothing: float = 0.1
    ):
        """
        初始化组合Loss
        
        Args:
            supcon_weight: SupCon loss权重
            ce_weight: CrossEntropy loss权重
            temperature: SupCon温度参数
            label_smoothing: Label smoothing参数
        """
        super().__init__()
        self.supcon_loss = SupervisedContrastiveLoss(temperature)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.supcon_weight = supcon_weight
        self.ce_weight = ce_weight
    
    def forward(
        self,
        embeddings: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> tuple:
        """
        计算组合Loss
        
        Args:
            embeddings: 特征向量 (B, D)
            logits: 分类logits (B, C)
            labels: 标签 (B,)
            
        Returns:
            (总loss, supcon_loss, ce_loss)
        """
        supcon_loss = self.supcon_loss(embeddings, labels)
        ce_loss = self.ce_loss(logits, labels)
        
        total_loss = self.supcon_weight * supcon_loss + self.ce_weight * ce_loss
        
        return total_loss, supcon_loss, ce_loss

