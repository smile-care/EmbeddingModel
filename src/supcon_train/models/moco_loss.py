"""
MoCo损失函数实现
支持标准MoCo loss和监督对比loss（兼容相似度矩阵）
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoLoss(nn.Module):
    """
    MoCo损失函数
    
    支持两种模式：
    1. standard: 标准MoCo loss（InfoNCE），仅使用队列中的负样本
    2. supervised: 监督对比loss，结合当前batch的positive pairs和队列中的负样本，支持相似度矩阵
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        loss_type: str = 'supervised',  # 'standard' 或 'supervised'
        similarity_matrix: Optional[np.ndarray] = None,
        default_similarity: float = 0.0,
        use_similarity_matrix: bool = False,
        neg_weight: float = 0.5,  # 负样本惩罚权重
        margin: float = 0.0  # 正负样本margin（0表示不使用margin）
    ):
        """
        初始化MoCo Loss
        
        Args:
            temperature: 温度参数，越小越关注困难样本
            loss_type: loss类型，'standard'（标准MoCo）或'supervised'（监督对比，支持相似度矩阵）
            similarity_matrix: 类别相似度矩阵 (num_classes, num_classes)，仅在supervised模式下使用
            default_similarity: 默认相似度阈值，大于此值的类别对被认为是positive pairs
            use_similarity_matrix: 是否使用相似度矩阵（False时强制使用标准SupCon，仅同label为positive）
            neg_weight: 负样本惩罚权重，用于显式推动负样本分离（0-1之间，越大负样本分离越强）
            margin: 正负样本margin，确保正样本相似度至少比负样本高margin（0表示不使用margin）
        """
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.use_similarity_matrix = use_similarity_matrix
        self.neg_weight = neg_weight
        self.margin = margin
        
        # 用于存储pos_loss和neg_loss（用于wandb记录）
        self.last_pos_loss = None
        self.last_neg_loss = None
        
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
        query_embeddings: torch.Tensor,  # (B, D) 当前batch的query embeddings
        key_embeddings: torch.Tensor,    # (B, D) 当前batch的key embeddings（用于positive pairs）
        query_labels: torch.Tensor,      # (B,) 当前batch的labels
        queue_embeddings: torch.Tensor,  # (K, D) 队列中的embeddings
        queue_labels: torch.Tensor      # (K,) 队列中的labels
    ) -> torch.Tensor:
        """
        计算MoCo Loss
        
        Args:
            query_embeddings: 当前batch的query embeddings (B, D)
            key_embeddings: 当前batch的key embeddings (B, D)，用于positive pairs
            query_labels: 当前batch的labels (B,)
            queue_embeddings: 队列中的embeddings (K, D)
            queue_labels: 队列中的labels (K,)
            
        Returns:
            loss值
        """
        device = query_embeddings.device
        batch_size = query_embeddings.shape[0]
        queue_size = queue_embeddings.shape[0]
        
        # 检查输入是否包含NaN或Inf
        if torch.isnan(query_embeddings).any() or torch.isinf(query_embeddings).any():
            print("警告：query_embeddings包含NaN或Inf！")
            query_embeddings = torch.nan_to_num(query_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(key_embeddings).any() or torch.isinf(key_embeddings).any():
            print("警告：key_embeddings包含NaN或Inf！")
            key_embeddings = torch.nan_to_num(key_embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 归一化特征
        query_embeddings = F.normalize(query_embeddings, dim=1, p=2, eps=1e-8)
        key_embeddings = F.normalize(key_embeddings, dim=1, p=2, eps=1e-8)
        queue_embeddings = F.normalize(queue_embeddings, dim=1, p=2, eps=1e-8)
        
        if self.loss_type == 'standard':
            # 标准MoCo loss（InfoNCE）
            # 每个query与对应的key是positive pair，与队列中的所有样本是negative pairs
            loss, pos_loss, neg_loss = self._standard_moco_loss(
                query_embeddings, key_embeddings, queue_embeddings
            )
            # 保存pos_loss和neg_loss（用于wandb记录）
            self.last_pos_loss = pos_loss.item() if pos_loss is not None else None
            self.last_neg_loss = neg_loss.item() if neg_loss is not None else None
            return loss
        elif self.loss_type == 'supervised':
            # 监督对比loss（支持相似度矩阵）
            # 当前batch内的positive pairs（基于相似度矩阵）+ 队列中的负样本
            loss, pos_loss, neg_loss = self._supervised_moco_loss(
                query_embeddings, key_embeddings, query_labels,
                queue_embeddings, queue_labels
            )
            # 保存pos_loss和neg_loss（用于wandb记录）
            self.last_pos_loss = pos_loss.item() if pos_loss is not None else None
            self.last_neg_loss = neg_loss.item() if neg_loss is not None else None
            return loss
        else:
            raise ValueError(f"loss_type必须是'standard'或'supervised'，当前为{self.loss_type}")
    
    def _standard_moco_loss(
        self,
        query_embeddings: torch.Tensor,  # (B, D)
        key_embeddings: torch.Tensor,    # (B, D)
        queue_embeddings: torch.Tensor   # (K, D)
    ) -> tuple:
        """
        标准MoCo loss（InfoNCE）
        
        每个query与对应的key是positive pair，与队列中的所有样本是negative pairs
        loss = -log(exp(q·k+ / τ) / (exp(q·k+ / τ) + Σ exp(q·k- / τ)))
        """
        device = query_embeddings.device
        batch_size = query_embeddings.shape[0]
        
        # 计算query与key的相似度（positive pairs）: (B,)
        pos_sim = torch.sum(query_embeddings * key_embeddings, dim=1)  # (B,)
        pos_logits = pos_sim / self.temperature  # (B,)
        
        # 计算query与队列中所有样本的相似度（negative pairs）: (B, K)
        neg_logits = torch.matmul(query_embeddings, queue_embeddings.T) / self.temperature  # (B, K)
        
        # 计算loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # 数值稳定性：logits_max = max(pos_logits, max(neg_logits))
        logits_max = torch.max(
            torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1),
            dim=1,
            keepdim=True
        )[0]  # (B, 1)
        
        # 归一化
        pos_logits_stable = pos_logits - logits_max.squeeze(1)  # (B,)
        neg_logits_stable = neg_logits - logits_max  # (B, K)
        
        # 计算exp
        exp_pos = torch.exp(pos_logits_stable)  # (B,)
        exp_neg = torch.exp(neg_logits_stable)  # (B, K)
        
        # 计算分母
        denominator = exp_pos.unsqueeze(1) + exp_neg.sum(dim=1, keepdim=True)  # (B, 1)
        
        # 计算log概率
        log_prob = pos_logits_stable - torch.log(denominator.squeeze(1) + 1e-8)  # (B,)
        
        # 正样本loss（取负，因为我们要最大化log概率）
        pos_loss = -log_prob.mean()
        
        # 添加负样本惩罚项：显式地最小化负样本相似度
        neg_loss = None
        if self.neg_weight > 0:
            # 计算负样本的平均相似度（使用归一化后的特征点积，而不是除以temperature后的值）
            neg_sim_raw = torch.matmul(query_embeddings, queue_embeddings.T)  # (B, K)
            neg_mean_sim = neg_sim_raw.mean()  # 所有负样本的平均相似度
            
            # 负样本惩罚项：最小化负样本相似度
            # 如果使用margin，确保负样本相似度至少比正样本低margin
            if self.margin > 0:
                pos_mean_sim = pos_sim.mean()  # 正样本的平均相似度（原始相似度）
                # 确保正样本相似度至少比负样本高margin
                neg_loss = F.relu(neg_mean_sim - pos_mean_sim + self.margin)
            else:
                # 直接最小化负样本相似度（鼓励负样本相似度为负值或接近0）
                # 使用ReLU确保只惩罚正相似度（负相似度已经是分离的，不需要惩罚）
                neg_loss = F.relu(neg_mean_sim)
            
            # 组合loss：正样本loss + 负样本惩罚项
            loss = pos_loss + self.neg_weight * neg_loss
        else:
            # neg_weight为0，只使用正样本loss
            loss = pos_loss
        
        # 检查loss是否为NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告：Loss为NaN或Inf！pos_sim范围: [{pos_sim.min():.4f}, {pos_sim.max():.4f}]")
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, zero_loss, None
        
        return loss, pos_loss, neg_loss
    
    def _supervised_moco_loss(
        self,
        query_embeddings: torch.Tensor,  # (B, D)
        key_embeddings: torch.Tensor,    # (B, D)
        query_labels: torch.Tensor,      # (B,)
        queue_embeddings: torch.Tensor,  # (K, D)
        queue_labels: torch.Tensor       # (K,)
    ) -> tuple:
        """
        监督对比MoCo loss（支持相似度矩阵）
        
        结合当前batch内的positive pairs（基于相似度矩阵）和队列中的样本
        队列中的样本根据相似度矩阵判断是否为positive pairs
        """
        device = query_embeddings.device
        batch_size = query_embeddings.shape[0]
        queue_size = queue_embeddings.shape[0]
        
        # 1. 计算当前batch内的positive pairs（基于相似度矩阵）
        # query_embeddings与key_embeddings的相似度矩阵: (B, B)
        # 注意：query[i]和key[i]是同一个样本的不同增强，应该是positive pair（对角线应该保留）
        batch_similarity = torch.matmul(query_embeddings, key_embeddings.T) / self.temperature  # (B, B)
        
        # 创建batch内的权重矩阵（基于相似度矩阵）
        # 注意：在MoCo中，query[i]和key[i]是positive pair，所以不应该排除对角线
        query_labels = query_labels.contiguous()  # (B,)
        
        if self.similarity_matrix_tensor is not None:
            # 使用自定义相似度矩阵作为权重
            # query_labels和key_labels应该相同（因为query和key是同一个样本的不同增强）
            label_similarity = self.similarity_matrix_tensor[query_labels][:, query_labels]  # (B, B)
            batch_weights = label_similarity.float().to(device)  # (B, B)
            # 注意：保留对角线，因为query[i]和key[i]是positive pair
            
            # 只考虑相似度大于default_similarity的样本对
            if self.default_similarity > 0.0:
                batch_weights = batch_weights * (batch_weights > self.default_similarity).float()
        else:
            # 标准SupCon：同label的样本为positive，权重为1.0
            # 注意：对角线元素（query[i]和key[i]）的标签相同，权重应该为1.0
            labels_expanded = query_labels.unsqueeze(1)  # (B, 1)
            batch_weights = torch.eq(labels_expanded, labels_expanded.T).float().to(device)  # (B, B)
            # 注意：保留对角线，因为query[i]和key[i]是positive pair
        
        # 2. 计算query与队列中样本的相似度: (B, K)
        queue_similarity = torch.matmul(query_embeddings, queue_embeddings.T) / self.temperature  # (B, K)
        
        # 3. 计算query_labels与queue_labels之间的相似度矩阵，确定队列中的positive pairs
        if self.similarity_matrix_tensor is not None:
            # 使用自定义相似度矩阵: (B, K)
            queue_label_similarity = self.similarity_matrix_tensor[query_labels][:, queue_labels]  # (B, K)
            queue_weights = queue_label_similarity.float().to(device)  # (B, K)
            
            # 只考虑相似度大于default_similarity的样本对
            if self.default_similarity > 0.0:
                queue_weights = queue_weights * (queue_weights > self.default_similarity).float()
        else:
            # 标准SupCon：同label的样本为positive，权重为1.0
            query_labels_expanded = query_labels.unsqueeze(1)  # (B, 1)
            queue_labels_expanded = queue_labels.unsqueeze(0)  # (1, K)
            queue_weights = torch.eq(query_labels_expanded, queue_labels_expanded).float().to(device)  # (B, K)
        
        # 4. 合并batch内和队列中的相似度: (B, B+K)
        # 注意：不需要mask batch_similarity，因为query[i]和key[i]是positive pair，应该保留
        all_similarity = torch.cat([batch_similarity, queue_similarity], dim=1)  # (B, B+K)
        
        # 5. 合并batch内和队列中的权重: (B, B+K)
        all_weights = torch.cat([batch_weights, queue_weights], dim=1)  # (B, B+K)
        
        # 6. 计算exp，添加数值稳定性
        logits_max, _ = torch.max(all_similarity, dim=1, keepdim=True)
        logits = all_similarity - logits_max.detach()  # 数值稳定
        
        exp_logits = torch.exp(logits)  # (B, B+K)
        
        # 7. 计算log概率
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)  # (B, B+K)
        
        # 8. 使用加权平均：相似度高的样本对权重更大
        weight_sum = all_weights.sum(1)  # (B,)，每个样本的权重总和
        valid_mask = weight_sum > 0  # 有权重>0的样本对
        
        if valid_mask.sum() == 0:
            print("警告：batch中没有positive pairs，返回小的正loss以保持梯度流动")
            small_loss = torch.tensor(1e-6, device=device, requires_grad=True)
            return small_loss, small_loss, None
        
        # 加权平均log概率（正样本部分）
        weighted_mean_log_prob = (all_weights * log_prob).sum(1) / (weight_sum + 1e-8)
        weighted_mean_log_prob = weighted_mean_log_prob[valid_mask]
        
        # 正样本loss（取负，因为我们要最大化log概率）
        pos_loss = -weighted_mean_log_prob.mean()
        
        # 9. 添加负样本惩罚项：显式地最小化负样本相似度
        # 负样本权重矩阵（1 - all_weights，但排除正样本）
        neg_weights = 1.0 - all_weights  # (B, B+K)
        # 只考虑真正的负样本（相似度矩阵中权重为0的样本对）
        if self.similarity_matrix_tensor is not None:
            # 使用相似度矩阵时，负样本是相似度 <= default_similarity 的样本对
            neg_mask = (all_weights <= self.default_similarity).float()
        else:
            # 标准SupCon时，负样本是不同label的样本对
            neg_mask = (all_weights == 0.0).float()
        
        neg_weights = neg_weights * neg_mask  # (B, B+K)
        neg_weight_sum = neg_weights.sum(1)  # (B,)
        valid_neg_mask = neg_weight_sum > 0  # 有负样本的样本
        
        if valid_neg_mask.sum() > 0 and self.neg_weight > 0:
            # 计算负样本的平均相似度（使用归一化后的特征点积，而不是除以temperature后的值）
            # 重新计算原始相似度（归一化后的特征点积）
            # batch内相似度（归一化后的特征点积）
            batch_sim_raw = torch.matmul(query_embeddings, key_embeddings.T)  # (B, B)
            # 队列相似度（归一化后的特征点积）
            queue_sim_raw = torch.matmul(query_embeddings, queue_embeddings.T)  # (B, K)
            # 合并
            all_sim_raw = torch.cat([batch_sim_raw, queue_sim_raw], dim=1)  # (B, B+K)
            
            # 计算负样本的平均相似度（使用原始相似度值）
            neg_similarity = all_sim_raw * neg_weights  # (B, B+K)
            neg_mean_sim = (neg_similarity.sum(1) / (neg_weight_sum + 1e-8))[valid_neg_mask]  # (B',)
            
            # 负样本惩罚项：最小化负样本相似度
            # 如果使用margin，确保负样本相似度至少比正样本低margin
            if self.margin > 0:
                # 计算正样本的平均相似度（用于margin计算）
                pos_similarity = all_sim_raw * all_weights  # (B, B+K)
                pos_mean_sim = (pos_similarity.sum(1) / (weight_sum + 1e-8))[valid_mask]  # (B',)
                # 确保正样本相似度至少比负样本高margin
                neg_loss = F.relu(neg_mean_sim - pos_mean_sim + self.margin).mean()
            else:
                # 直接最小化负样本相似度（鼓励负样本相似度为负值或接近0）
                # 使用ReLU确保只惩罚正相似度（负相似度已经是分离的，不需要惩罚）
                neg_loss = F.relu(neg_mean_sim).mean()
            
            # 组合loss：正样本loss + 负样本惩罚项
            loss = pos_loss + self.neg_weight * neg_loss
        else:
            # 没有负样本或neg_weight为0，只使用正样本loss
            loss = pos_loss
        
        # 检查loss是否为NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告：Loss为NaN或Inf！batch_similarity范围: [{batch_similarity.min():.4f}, {batch_similarity.max():.4f}]")
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, zero_loss, None
        
        # 返回loss, pos_loss, neg_loss
        neg_loss_tensor = neg_loss if (valid_neg_mask.sum() > 0 and self.neg_weight > 0) else None
        return loss, pos_loss, neg_loss_tensor

