"""
特征金字塔网络（FPN）模块
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络（FPN）
    
    FPN通过横向连接和自顶向下路径融合不同尺度的特征信息。
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_outs: Optional[int] = None,
        start_level: int = 1,
        add_extra_convs: bool = False
    ):
        """
        初始化FPN模块
        
        Args:
            in_channels_list: backbone各层特征的通道数列表
            out_channels: FPN输出特征的统一通道数（默认256，标准FPN使用统一通道数）
            num_outs: 输出特征图的数量（None表示使用所有层）
            start_level: 开始构建FPN的层级（通常为0）
            add_extra_convs: 是否添加额外的卷积层进一步优化特征
        """
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.num_ins = len(in_channels_list)
        self.num_outs = num_outs if num_outs is not None else len(in_channels_list)
        self.start_level = start_level
        self.out_channels = out_channels
        
        # 标准FPN统一所有层的输出通道数
        self.out_channels_list = [out_channels] * self.num_ins
        
        # 横向连接：将backbone各层特征统一到out_channels
        # 使用1x1卷积进行通道数转换
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1,
                stride=1,
                padding=0
            )
            self.lateral_convs.append(lateral_conv)
        
        # 自顶向下路径：融合层（用于融合上采样特征和横向连接特征）
        # 所有层输出通道数统一为out_channels
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_ins):
            fpn_conv = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.fpn_convs.append(fpn_conv)
        
        # 可选的额外卷积层（用于进一步优化特征）
        self.extra_convs = None
        if add_extra_convs:
            self.extra_convs = nn.ModuleList()
            for _ in range(self.num_outs - self.num_ins):
                extra_conv = nn.Sequential(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                self.extra_convs.append(extra_conv)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化FPN权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            inputs: backbone输出的多层特征元组，每层形状为 (B, C_i, H_i, W_i)
            
        Returns:
            优化后的多尺度特征图元组，每层形状为 (B, out_channels_i, H_i, W_i)
            如果out_channels为None，则保持原始通道数
        """
        assert len(inputs) == len(self.in_channels_list), \
            f"输入特征数量({len(inputs)})与配置的通道数列表长度({len(self.in_channels_list)})不匹配"
        
        # 步骤1：横向连接 - 统一通道数到out_channels
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))
        
        # 步骤2：自顶向下路径 - 从高层特征开始，逐步融合低层特征
        used_backbone_levels = len(laterals)
        fpn_outs = []
        
        # 从最高层开始构建FPN（自顶向下）
        for i in range(used_backbone_levels - 1, -1, -1):
            if i == used_backbone_levels - 1:
                # 最高层：直接使用横向连接的特征
                fpn_outs.append(self.fpn_convs[i](laterals[i]))
            else:
                # 低层：上采样高层特征并与当前层融合
                top_down_feat = fpn_outs[-1]
                curr_shape = laterals[i].shape[2:]  # (H, W)
                
                # 上采样高层特征到当前层尺寸
                upsampled = F.interpolate(
                    top_down_feat,
                    size=curr_shape,
                    mode='bilinear',
                    align_corners=False
                )
                
                # 融合：横向连接特征 + 上采样特征（通道数已统一为out_channels）
                fused = laterals[i] + upsampled
                
                # 通过卷积进一步优化
                fpn_out = self.fpn_convs[i](fused)
                fpn_outs.append(fpn_out)  # 追加到列表末尾，最后会反转顺序
        
        # 步骤3：可选的额外卷积层（生成更小的特征图）
        if self.extra_convs is not None:
            last_feat = fpn_outs[-1]
            for extra_conv in self.extra_convs:
                fpn_outs.append(extra_conv(last_feat))
                last_feat = fpn_outs[-1]
        
        # 反转顺序，使输出顺序与输入一致（从低到高）
        fpn_outs = list(reversed(fpn_outs))
        
        # 只返回需要的输出数量
        return tuple(fpn_outs[:self.num_outs])


class PathAggregationFPN(nn.Module):
    """路径聚合特征金字塔网络（PNFPN/PANet）
    
    PNFPN在标准FPN的基础上增加了自底向上的路径，形成双向的特征金字塔：
    1. 横向连接（Lateral Connections）
    2. 自顶向下路径（Top-down Path）- 从高层到低层
    3. 自底向上路径（Bottom-up Path）- 从低层到高层，进一步增强特征
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        num_outs: Optional[int] = None,
        start_level: int = 0,
        add_extra_convs: bool = False
    ):
        """
        初始化PNFPN模块
        
        Args:
            in_channels_list: backbone各层特征的通道数列表
            out_channels: PNFPN输出特征的统一通道数（默认256，标准PNFPN使用统一通道数）
            num_outs: 输出特征图的数量（None表示使用所有层）
            start_level: 开始构建PNFPN的层级（通常为0）
            add_extra_convs: 是否添加额外的卷积层进一步优化特征
        """
        super().__init__()
        
        self.in_channels_list = in_channels_list
        self.num_ins = len(in_channels_list)
        self.num_outs = num_outs if num_outs is not None else len(in_channels_list)
        self.start_level = start_level
        self.out_channels = out_channels
        
        # 标准PNFPN统一所有层的输出通道数
        self.out_channels_list = [out_channels] * self.num_ins
        
        # 横向连接：将backbone各层特征统一到out_channels
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1,
                stride=1,
                padding=0
            )
            self.lateral_convs.append(lateral_conv)
        
        # 自顶向下路径：融合层（用于融合上采样特征和横向连接特征）
        # 所有层输出通道数统一为out_channels
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_ins):
            fpn_conv = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.fpn_convs.append(fpn_conv)
        
        # 自底向上路径：融合层（用于融合下采样特征和自顶向下路径的特征）
        self.pan_convs = nn.ModuleList()
        for _ in range(self.num_ins):
            pan_conv = nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.pan_convs.append(pan_conv)
        
        # 自底向上路径中的下采样层（使用stride=2的卷积进行下采样）
        self.bottom_up_downsamples = nn.ModuleList()
        for _ in range(self.num_ins - 1):
            downsample_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
            self.bottom_up_downsamples.append(downsample_conv)
        
        # 可选的额外卷积层
        self.extra_convs = None
        if add_extra_convs:
            self.extra_convs = nn.ModuleList()
            for _ in range(self.num_outs - self.num_ins):
                extra_conv = nn.Sequential(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                self.extra_convs.append(extra_conv)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化PNFPN权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            inputs: backbone输出的多层特征元组，每层形状为 (B, C_i, H_i, W_i)
            
        Returns:
            优化后的多尺度特征图元组，每层形状为 (B, out_channels_i, H_i, W_i)
            如果out_channels为None，则保持原始通道数
        """
        assert len(inputs) == len(self.in_channels_list), \
            f"输入特征数量({len(inputs)})与配置的通道数列表长度({len(self.in_channels_list)})不匹配"
        
        # 步骤1：横向连接 - 统一通道数到out_channels
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))
        
        # 步骤2：自顶向下路径 - 从高层特征开始，逐步融合低层特征
        used_backbone_levels = len(laterals)
        fpn_outs = []
        
        # 从最高层开始构建FPN（自顶向下）
        for i in range(used_backbone_levels - 1, -1, -1):
            if i == used_backbone_levels - 1:
                # 最高层：直接使用横向连接的特征
                fpn_outs.append(self.fpn_convs[i](laterals[i]))
            else:
                # 低层：上采样高层特征并与当前层融合
                top_down_feat = fpn_outs[-1]
                curr_shape = laterals[i].shape[2:]  # (H, W)
                
                # 上采样高层特征到当前层尺寸
                upsampled = F.interpolate(
                    top_down_feat,
                    size=curr_shape,
                    mode='bilinear',
                    align_corners=False
                )
                
                # 融合：横向连接特征 + 上采样特征（通道数已统一为out_channels）
                fused = laterals[i] + upsampled
                
                # 通过卷积进一步优化
                fpn_out = self.fpn_convs[i](fused)
                fpn_outs.append(fpn_out)
        
        # 反转顺序，使输出顺序与输入一致（从低到高）
        fpn_outs = list(reversed(fpn_outs))
        
        # 步骤3：自底向上路径 - 从低层特征开始，逐步融合高层特征
        # 使用自顶向下路径的输出作为输入
        pan_outs = []
        
        # 从最低层开始构建PAN（自底向上）
        for i in range(used_backbone_levels):
            if i == 0:
                # 最低层：直接使用自顶向下路径的特征
                pan_outs.append(self.pan_convs[i](fpn_outs[i]))
            else:
                # 高层：下采样低层特征并与当前层融合
                bottom_up_feat = pan_outs[-1]
                
                # 使用stride=2的卷积进行下采样（通道数已统一为out_channels）
                # bottom_up_downsamples的索引：从第i-1层下采样到第i层，所以索引是i-1
                if i - 1 < len(self.bottom_up_downsamples):
                    downsampled = self.bottom_up_downsamples[i - 1](bottom_up_feat)
                else:
                    # 如果没有下采样层，使用插值
                    curr_shape = fpn_outs[i].shape[2:]
                    downsampled = F.interpolate(
                        bottom_up_feat,
                        size=curr_shape,
                        mode='bilinear',
                        align_corners=False
                    )
                
                # 如果尺寸不完全匹配，使用插值微调
                curr_shape = fpn_outs[i].shape[2:]
                if downsampled.shape[2:] != curr_shape:
                    downsampled = F.interpolate(
                        downsampled,
                        size=curr_shape,
                        mode='bilinear',
                        align_corners=False
                    )
                
                # 融合：自顶向下路径特征 + 下采样特征
                fused = fpn_outs[i] + downsampled
                
                # 通过卷积进一步优化
                pan_out = self.pan_convs[i](fused)
                pan_outs.append(pan_out)
        
        # 步骤4：可选的额外卷积层
        if self.extra_convs is not None:
            last_feat = pan_outs[-1]
            for extra_conv in self.extra_convs:
                pan_outs.append(extra_conv(last_feat))
                last_feat = pan_outs[-1]
        
        # 只返回需要的输出数量
        return tuple(pan_outs[:self.num_outs])
