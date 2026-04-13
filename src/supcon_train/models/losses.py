"""
监督对比学习Loss实现
"""
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
    
    def __init__(self, temperature: float = 0.07):
        """
        初始化SupCon Loss

        Args:
            temperature: 温度参数，越小越关注困难样本
        """
        super().__init__()
        self.temperature = temperature
    
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


class ComprehensiveSegmentationLoss(nn.Module):
    """
    综合分割损失函数
    
    结合多种约束：
    1. BCE Loss（前景强监督）
    2. Dice Loss（处理类别不平衡）
    3. 前景比例约束（防止过拟合）
    
    特点：
    - 只对前景像素进行强监督，避免漏标问题
    - 通过前景比例约束防止模型将所有像素预测为前景
    - 不进行背景抑制，因为背景区域可能包含其他缺陷（会在其他iteration训练）
    """
    
    def __init__(
        self,
        bce_weight: float = 0.4,
        dice_weight: float = 0.4,
        ratio_weight: float = 0.1,
        target_foreground_ratio: float = 0.1,
        max_foreground_ratio: float = 0.4
    ):
        """
        初始化综合分割损失
        
        Args:
            bce_weight: BCE损失权重
            dice_weight: Dice损失权重
            ratio_weight: 前景比例约束权重
            target_foreground_ratio: 目标前景比例（用于比例约束，当前未使用）
            max_foreground_ratio: 最大允许的前景比例（超过此值会触发比例惩罚）
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.ratio_weight = ratio_weight
        self.target_foreground_ratio = target_foreground_ratio
        self.max_foreground_ratio = max_foreground_ratio
        
        # 用于记录最后一次损失的详细信息
        self.last_loss_details = None
    
    def forward(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算综合分割损失
        
        Args:
            pred_mask: 预测的分割mask (B, 1, H, W)，值域[0, 1]
            gt_mask: 真实mask (B, 1, H, W)，值域[0, 1]
            
        Returns:
            总损失值
        """
        device = pred_mask.device
        B, C, H, W = pred_mask.shape
        
        # 确保输入在合理范围内
        pred_mask = torch.clamp(pred_mask, 0.0, 1.0)
        gt_mask = torch.clamp(gt_mask, 0.0, 1.0)
        
        # 前景mask：前景为1，背景为0，0-1之间的是mask边缘以外手动添加的热力图，不属于前景
        foreground_mask = (gt_mask > 0.999).float()  # (B, 1, H, W)
        
        # 1. BCE Loss（仅对前景像素）
        bce = F.binary_cross_entropy(pred_mask, gt_mask, reduction='none')  # (B, 1, H, W)
        bce_loss = (bce * foreground_mask).sum() / (foreground_mask.sum() + 1e-8)
        
        # 2. Dice Loss（仅对前景像素）
        pred_fg = pred_mask * foreground_mask
        gt_fg = gt_mask * foreground_mask
        intersection = (pred_fg * gt_fg).sum(dim=(2, 3))  # (B, 1)
        union = pred_fg.sum(dim=(2, 3)) + gt_fg.sum(dim=(2, 3))  # (B, 1)
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)  # (B, 1)
        dice_loss = 1.0 - dice.mean()
        
        # 3. 前景比例约束（防止过拟合）
        pred_ratio = pred_mask.mean()  # 平均预测值 ≈ 前景比例
        
        # 如果预测前景比例超过最大允许值，进行惩罚
        five_mult_ratio = foreground_mask.mean().item() * 5.0    # 目标前景比例的5倍
        max_foreground_ratio = min(self.max_foreground_ratio, five_mult_ratio)
        if pred_ratio > max_foreground_ratio:
            ratio_loss = F.mse_loss(pred_ratio, torch.tensor(max_foreground_ratio, device=device))
        else:
            ratio_loss = torch.tensor(0.0, device=device)
        
        # 总损失（不包含背景抑制，因为背景区域可能包含其他缺陷）
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.ratio_weight * ratio_loss
        )
        
        # 记录详细信息（用于监控和调试）
        self.last_loss_details = {
            'bce_loss': bce_loss.item(),
            'dice_loss': dice_loss.item(),
            'ratio_loss': ratio_loss.item(),
            'pred_foreground_ratio': pred_ratio.item(),
            'gt_foreground_ratio': foreground_mask.mean().item()
        }
        
        # 检查loss是否为NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"警告：分割损失为NaN或Inf！")
            print(f"  BCE: {bce_loss.item():.4f}, Dice: {dice_loss.item():.4f}, "
                  f"Ratio: {ratio_loss.item():.4f}")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss