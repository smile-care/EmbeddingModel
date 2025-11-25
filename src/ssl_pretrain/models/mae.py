"""
Masked Autoencoder (MAE) 实现
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_factory import BackboneFactory


class MAE(nn.Module):
    """Masked Autoencoder模型"""
    
    def __init__(
        self,
        backbone_type: str = 'vit_base',
        image_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        backbone_pretrained: bool = True
    ):
        """
        初始化MAE模型
        
        Args:
            backbone_type: backbone类型
            image_size: 输入图像大小
            patch_size: patch大小
            mask_ratio: mask比例
            decoder_dim: decoder维度
            decoder_depth: decoder深度
            decoder_num_heads: decoder注意力头数
            backbone_pretrained: 是否使用预训练backbone
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (image_size // patch_size) ** 2
        
        # Encoder (backbone)
        self.encoder = BackboneFactory.create_backbone(
            backbone_type,
            pretrained=backbone_pretrained,
            image_size=image_size
        )
        
        # 获取encoder特征维度
        encoder_dim = BackboneFactory.get_feature_dim(
            self.encoder, image_size
        )
        
        # 对于ViT，需要特殊处理patch embedding
        self.is_vit = backbone_type.startswith('vit')
        if self.is_vit:
            # ViT已经包含patch embedding，直接使用
            self.encoder_dim = encoder_dim
        else:
            # 对于CNN backbone，需要将backbone输出的特征图转换为patches
            # backbone输出是(B, C, H', W')，需要转换为(B, N, D)
            # 首先获取backbone输出的空间尺寸
            self.encoder_dim = encoder_dim
            # 计算backbone输出的空间尺寸
            # 对于ResNet50/ConvNeXt，输入224x224，输出通常是7x7
            # 但我们需要匹配num_patches，所以需要插值或投影
            # 使用自适应池化将特征图调整到期望的patch数量
            self.feature_pool = nn.AdaptiveAvgPool2d(
                (image_size // patch_size, image_size // patch_size)
            )
            # 如果backbone输出维度与期望不匹配，添加投影层
            # 这里先不添加，在forward中根据实际情况处理
        
        # Decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim)
        )
        self.decoder_pos_drop = nn.Dropout(0.1)
        
        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                dim=decoder_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=4.0
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        
        # 预测头（重建像素值）
        self.decoder_pred = nn.Linear(
            decoder_dim,
            patch_size * patch_size * 3
        )
        
        # 初始化
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        随机masking
        
        Args:
            x: 输入特征 (B, N, D)
            
        Returns:
            (可见特征, mask, ids_restore)
        """
        B, N, D = x.shape
        
        # 随机打乱
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留前len_keep个
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        
        # 生成mask (1表示masked, 0表示可见)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encoder前向传播
        
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            (编码特征, mask, ids_restore)
        """
        # 提取特征
        if self.is_vit:
            # ViT backbone
            x = self.encoder.patch_embed(x)  # (B, N, D)
            if hasattr(self.encoder, 'cls_token'):
                cls_token = self.encoder.cls_token.expand(x.size(0), -1, -1)
                x = torch.cat([cls_token, x], dim=1)
            if hasattr(self.encoder, 'pos_embed'):
                x = x + self.encoder.pos_embed
            x = self.encoder.blocks(x)
            x = self.encoder.norm(x)
            # 移除CLS token（如果有）
            if x.size(1) == self.num_patches + 1:
                x = x[:, 1:]
        else:
            # CNN backbone (ResNet50, ConvNeXt等)
            # 先通过backbone提取特征
            x = self.encoder(x)  # (B, C, H', W')
            
            # 如果输出是tuple，取第一个元素
            if isinstance(x, tuple):
                x = x[0]
            
            # 将特征图调整到期望的空间尺寸
            # 使用自适应池化将特征图调整到(target_size, target_size)
            # 其中target_size = image_size // patch_size，确保patch数量匹配
            x = self.feature_pool(x)  # (B, C, target_size, target_size)
            
            # 将特征图转换为patches格式 (B, N, D)
            # target_size^2 = num_patches，所以输出正好是(B, num_patches, C)
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C) = (B, num_patches, C)
        
        # 随机masking
        x_masked, mask, ids_restore = self.random_masking(x)
        
        return x_masked, mask, ids_restore
    
    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Decoder前向传播
        
        Args:
            x: 编码特征 (B, N_visible, D)
            ids_restore: 恢复顺序的索引
            
        Returns:
            重建特征 (B, N, decoder_dim)
        """
        # 嵌入到decoder维度
        x = self.decoder_embed(x)
        
        # 添加mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        
        # 添加位置编码
        x_ = x_ + self.decoder_pos_embed
        x_ = self.decoder_pos_drop(x_)
        
        # Transformer decoder
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)
        
        # 预测像素值
        x_ = self.decoder_pred(x_)
        
        return x_
    
    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        计算重建损失
        
        Args:
            imgs: 原始图像 (B, C, H, W)
            pred: 预测像素值 (B, N, patch_size^2 * 3)
            mask: mask (B, N)
            
        Returns:
            损失值
        """
        target = self.patchify(imgs)
        
        # 只计算masked patches的损失
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N)
        loss = (loss * mask).sum() / mask.sum()  # 只对masked patches求平均
        
        return loss
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        将图像分割成patches
        
        Args:
            imgs: 图像 (B, C, H, W)
            
        Returns:
            patches (B, N, patch_size^2 * C)
        """
        p = self.patch_size
        h = w = self.image_size // p
        
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        将patches还原为图像
        
        Args:
            x: patches (B, N, patch_size^2 * C)
            
        Returns:
            图像 (B, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs
    
    def forward(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            imgs: 输入图像 (B, C, H, W)
            
        Returns:
            (损失, 重建图像, mask)
        """
        # Encoder
        latent, mask, ids_restore = self.forward_encoder(imgs)
        
        # Decoder
        pred = self.forward_decoder(latent, ids_restore)
        
        # 计算损失
        loss = self.forward_loss(imgs, pred, mask)
        
        # 重建图像（用于可视化）
        # 注意：loss只对masked区域计算，所以重建图像应该：
        # - 可见区域：使用原始图像的值
        # - Masked区域：使用模型预测的值
        target_patches = self.patchify(imgs)  # (B, N, patch_size^2 * 3)
        # 对于可见patches使用原始值，对于masked patches使用预测值
        # mask: 1表示masked, 0表示可见
        recon_patches = target_patches * (1 - mask.unsqueeze(-1)) + pred * mask.unsqueeze(-1)
        pred_img = self.unpatchify(recon_patches)
        
        return loss, pred_img, mask


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder块"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

