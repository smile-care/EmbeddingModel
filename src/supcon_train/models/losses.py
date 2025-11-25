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
            features: 特征向量 (B, D)
            labels: 标签 (B,)
            
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
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建mask：同label的样本为positive
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除自己与自己的相似度（对角线）
        logits_mask = torch.ones(batch_size, batch_size, device=device) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # 计算exp，添加数值稳定性
        # 使用log-sum-exp技巧避免数值溢出
        similarity_matrix = similarity_matrix * logits_mask  # 将对角线设为0
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()  # 数值稳定
        
        exp_logits = torch.exp(logits) * logits_mask
        
        # 计算log概率，避免log(0)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 计算每个样本的loss（只对positive pairs）
        # 避免除零：只有当mask.sum(1) > 0时才计算
        mask_sum = mask.sum(1)  # (B,)
        valid_mask = mask_sum > 0  # 有positive pairs的样本
        
        if valid_mask.sum() == 0:
            # 如果batch中没有任何positive pairs，返回一个小的loss值
            print("警告：batch中没有positive pairs，返回默认loss")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 只对有positive pairs的样本计算loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-8)
        mean_log_prob_pos = mean_log_prob_pos[valid_mask]
        
        # 平均loss
        loss = -mean_log_prob_pos.mean()
        
        # 检查loss是否为NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告：Loss为NaN或Inf！similarity_matrix范围: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
            print(f"mask_sum: {mask_sum}, valid_mask: {valid_mask.sum()}/{batch_size}")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
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

