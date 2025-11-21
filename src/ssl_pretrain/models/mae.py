"""
Masked Autoencoder (MAE) 实现
"""
from typing import Tuple

import torch
import torch.nn as nn

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
        if backbone_type.startswith('vit'):
            # ViT已经包含patch embedding，直接使用
            self.encoder_dim = encoder_dim
        else:
            # 对于CNN backbone，需要添加patch embedding
            self.patch_embed = nn.Conv2d(
                3, encoder_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
            self.encoder_dim = encoder_dim
        
        # Decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim)
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
        if hasattr(self, 'patch_embed'):
            # CNN backbone
            x = self.patch_embed(x)  # (B, D, H', W')
            B, D, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        else:
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
        pred_img = self.unpatchify(pred)
        
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

