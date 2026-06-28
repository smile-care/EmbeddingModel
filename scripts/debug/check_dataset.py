import os
import os.path as osp
import random
import sys

# 设置环境变量，避免Qt插件问题
os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', '')

# 设置matplotlib后端（在导入pyplot之前）
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# 设置中文字体
matplotlib.rc("font",family='AR PL UKai CN')

sys.path.append("./")

from src.embedding_model.supcon.datasets.supcon_dataset import MultiConfigDataset, SupConDataset
from src.embedding_model.utils.config_loader import load_config
from src.embedding_model.utils.visualization import denormalize_image

if __name__ == "__main__":
    config_path = 'configs/supcon_config.yaml'
    data_config_path = "configs/data_config_zhenyu.yaml"
    
    # 是否显示原始二值mask（False则显示软膨胀后的mask）
    SHOW_BINARY_MASK = False  # 设置为True查看原始二值mask
    
    supcon_config = load_config(config_path)
    
    image_size = supcon_config['supcon']['data'].get('image_size', 224)
    
    dataset = SupConDataset(
            data_config_path=data_config_path,
            split='train',
            image_size=image_size
        )
    
    print(f"数据集样本数: {len(dataset)}")
    print(f"类别列表: {dataset.categories}")
    print("\n关闭窗口继续下一个样本...")

    random.shuffle(dataset.samples)

    # 逐个显示样本
    for idx, data_item in enumerate(dataset):
        view1_image = data_item['view1_image']
        view1_mask = data_item['view1_mask']
        view2_image = data_item['view2_image']
        view2_mask = data_item['view2_mask']
        label_name = data_item['label_name']
        image_path = data_item['image_path']
        
        # 反归一化图像
        img1 = denormalize_image(view1_image).permute(1, 2, 0).cpu().numpy()
        img2 = denormalize_image(view2_image).permute(1, 2, 0).cpu().numpy()
        
        # 处理mask
        mask1 = view1_mask[0].cpu().numpy() if view1_mask.dim() == 3 else view1_mask.cpu().numpy()
        mask2 = view2_mask[0].cpu().numpy() if view2_mask.dim() == 3 else view2_mask.cpu().numpy()
        
        # 如果设置了显示原始二值mask，则将软膨胀的mask转换回二值
        if SHOW_BINARY_MASK:
            mask1 = (mask1 == 1.0).astype(np.float32)
            mask2 = (mask2 == 1.0).astype(np.float32)
        
        if 0:
            # 创建单个样本的可视化
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # View1图像
            axes[0, 0].imshow(np.clip(img1, 0, 1))
            axes[0, 0].set_title(f'View1 Image - {label_name}')
            # axes[0, 0].axis('off')
            
            # View1 mask
            axes[0, 1].imshow(mask1, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title('View1 Mask')
            # axes[0, 1].axis('off')
            
            # View2图像
            axes[1, 0].imshow(np.clip(img2, 0, 1))
            axes[1, 0].set_title('View2 Image')
            # axes[1, 0].axis('off')
            
            # View2 mask
            axes[1, 1].imshow(mask2, cmap='gray', vmin=0, vmax=1)
            axes[1, 1].set_title('View2 Mask')
            # axes[1, 1].axis('off')
            
            plt.suptitle(f'Sample {idx}/{len(dataset)-1} - {label_name}\n{osp.basename(image_path)}', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            # 显示窗口，等待用户关闭
            print(f"显示样本 {idx}/{len(dataset)-1} - {label_name}")
            plt.show(block=True)

        
        else:
            # 创建单个样本的可视化
            img1_4x = F.interpolate(view1_image.unsqueeze(0), scale_factor=1/4, mode='bilinear', align_corners=False)
            img1_8x = F.interpolate(view1_image.unsqueeze(0), scale_factor=1/8, mode='bilinear', align_corners=False)
            img1_16x = F.interpolate(view1_image.unsqueeze(0), scale_factor=1/16, mode='bilinear', align_corners=False)
            img1_32x = F.interpolate(view1_image.unsqueeze(0), scale_factor=1/32, mode='bilinear', align_corners=False)
            img1_4x = denormalize_image(img1_4x.squeeze(0)).permute(1, 2, 0).cpu().numpy()
            img1_8x = denormalize_image(img1_8x.squeeze(0)).permute(1, 2, 0).cpu().numpy()
            img1_16x = denormalize_image(img1_16x.squeeze(0)).permute(1, 2, 0).cpu().numpy()
            img1_32x = denormalize_image(img1_32x.squeeze(0)).permute(1, 2, 0).cpu().numpy()
            
            mask1_4x = F.interpolate(view1_mask.unsqueeze(0), scale_factor=1/4, mode='nearest').squeeze(0).cpu().numpy()
            mask1_8x = F.interpolate(view1_mask.unsqueeze(0), scale_factor=1/8, mode='nearest').squeeze(0).cpu().numpy()
            mask1_16x = F.interpolate(view1_mask.unsqueeze(0), scale_factor=1/16, mode='nearest').squeeze(0).cpu().numpy()
            mask1_32x = F.interpolate(view1_mask.unsqueeze(0), scale_factor=1/32, mode='nearest').squeeze(0).cpu().numpy()
            
            # 处理mask维度（如果是3D，取第一个通道）
            if mask1_4x.ndim == 3:
                mask1_4x = mask1_4x[0]
            if mask1_8x.ndim == 3:
                mask1_8x = mask1_8x[0]
            if mask1_16x.ndim == 3:
                mask1_16x = mask1_16x[0]
            if mask1_32x.ndim == 3:
                mask1_32x = mask1_32x[0]
            
            # 如果设置了显示原始二值mask，则将软膨胀的mask转换回二值
            if SHOW_BINARY_MASK:
                mask1_4x = (mask1_4x == 1.0).astype(np.float32)
                mask1_8x = (mask1_8x == 1.0).astype(np.float32)
                mask1_16x = (mask1_16x == 1.0).astype(np.float32)
                mask1_32x = (mask1_32x == 1.0).astype(np.float32)
            
            fig, axes = plt.subplots(5, 2, figsize=(12, 20), constrained_layout=True)
            axes = axes.flatten()  # 将2D数组展平为1D，方便索引
            
            # View1图像及其多尺度mask
            axes[0].imshow(np.clip(img1, 0, 1))
            axes[0].set_title(f'View1 Image - {label_name}')
            axes[1].imshow(mask1, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('View1 Mask')
            axes[1].axis('off')
            
            axes[2].imshow(np.clip(img1_4x, 0, 1))
            axes[2].set_title('View1 Image 4x')
            axes[3].imshow(mask1_4x, cmap='gray', vmin=0, vmax=1)
            axes[3].set_title('View1 Mask 4x')
            axes[3].axis('off')
            
            axes[4].imshow(np.clip(img1_8x, 0, 1))
            axes[4].set_title('View1 Image 8x')
            axes[5].imshow(mask1_8x, cmap='gray', vmin=0, vmax=1)
            axes[5].set_title('View1 Mask 8x')
            axes[5].axis('off')
            
            axes[6].imshow(np.clip(img1_16x, 0, 1))
            axes[6].set_title('View1 Image 16x')
            axes[7].imshow(mask1_16x, cmap='gray', vmin=0, vmax=1)
            axes[7].set_title('View1 Mask 16x')
            axes[7].axis('off')
            
            axes[8].imshow(np.clip(img1_32x, 0, 1))
            axes[8].set_title('View1 Image 32x')
            axes[9].imshow(mask1_32x, cmap='gray', vmin=0, vmax=1)
            axes[9].set_title('View1 Mask 32x')
            axes[9].axis('off')
            
            plt.suptitle(f'Sample {idx}/{len(dataset)-1} - {label_name}\n{osp.basename(image_path)}', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            # 显示窗口，等待用户关闭
            print(f"显示样本 {idx}/{len(dataset)-1} - {label_name}")
            plt.show(block=True)