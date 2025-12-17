"""
提取SupCon模型的embedding
适配最新的算法架构：DINOv3 Backbone + FPN + Mask特征筛选 + 多层特征融合 + Projection Head
使用与训练脚本相同的数据加载模式（基于data_config_zhenyu.yaml和SupConDataset）
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.supcon_train.datasets.supcon_dataset import SupConDataset
from src.supcon_train.models.supcon_model import SupConModel
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger


class SupConEmbeddingExtractor:
    """SupCon模型embedding提取器（支持mask）"""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化提取器
        
        Args:
            model_path: 模型checkpoint路径
            config_path: 配置文件路径（如果checkpoint中没有）
            device: 设备
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取模型配置
        if config_path:
            config = load_config(config_path)
            model_config = config['supcon']['model']
        else:
            model_config = checkpoint.get('config', {}).get('supcon', {}).get('model', {})
        
        # 创建模型（使用最新的架构）
        self.model = SupConModel(
            model_name=model_config.get('model_name', 'facebook/dinov3-convnext-small-pretrain-lvd1689m'),
            embedding_dim=model_config.get('embedding_dim', 128),
            projection_hidden_dims=model_config.get('projection_head', {}).get('hidden_dims', [256, 128]),
            image_size=model_config.get('image_size', 224),
            freeze_backbone=False,  # 提取时不需要冻结
            use_layers=model_config.get('use_layers', [1, 2, 3, 4]),
            fpn_out_channels=model_config.get('fpn_out_channels', 256),
            fusion_dim=model_config.get('fusion_dim', 512)
        ).to(self.device)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 加载权重（使用strict=False以忽略不匹配的键）
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict, strict=False
        )
        
        if missing_keys:
            print(f"警告：部分权重未加载: {len(missing_keys)}个键")
            if len(missing_keys) <= 10:
                print(f"  缺失的键: {missing_keys}")
            else:
                print(f"  前10个缺失的键: {missing_keys[:10]}...")
        
        if unexpected_keys:
            print(f"警告：部分权重未使用: {len(unexpected_keys)}个键")
            if len(unexpected_keys) <= 10:
                print(f"  未使用的键: {unexpected_keys}")
            else:
                print(f"  前10个未使用的键: {unexpected_keys[:10]}...")
        
        self.model.eval()
        self.image_size = model_config.get('image_size', 224)
    
    def extract_from_image(
        self,
        image_path: str,
        mask_path: Optional[str] = None
    ) -> np.ndarray:
        """
        从单张图像提取embedding
        
        Args:
            image_path: 图像路径
            mask_path: mask路径（可选，如果为None则使用全1的mask）
            
        Returns:
            Embedding向量
        """
        # 图像预处理
        image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Mask预处理
        mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = image_transform(image).unsqueeze(0).to(self.device)
        
        # 加载mask
        if mask_path and Path(mask_path).exists():
            mask = Image.open(mask_path).convert('L')
            mask_tensor = mask_transform(mask).to(self.device)
            # 二值化mask：>0.5为1，否则为0
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            # 如果没有mask，使用全1的mask（不进行筛选）
            mask_tensor = torch.ones(1, self.image_size, self.image_size).to(self.device)
        
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, 1, H, W)
        
        with torch.no_grad():
            outputs = self.model(image_tensor, mask_tensor, return_features=False)
            embedding = outputs['embeddings'].cpu().numpy()[0]
        
        return embedding
    
    def extract_batch(
        self,
        data_config_path: str,
        output_file: Optional[str] = None,
        batch_size: int = 32,
    ) -> Dict[str, np.ndarray]:
        """
        批量提取所有实例的embedding（使用与训练脚本相同的数据加载模式）
        
        Args:
            data_config_path: 数据配置文件路径（data_config_zhenyu.yaml）
            output_file: 输出文件路径（.npy或.npz）
            batch_size: batch大小
            split: 数据集划分（'train' 或 'val'）
            
        Returns:
            {sample_idx: embedding} 字典，其中sample_idx是样本在数据集中的索引
        """
        # 使用与训练脚本相同的数据集类
        # 注意：提取时不需要数据增强，所以augmentation_config=None
        dataset = SupConDataset(
            data_config_path=data_config_path,
            split="val",
            augmentation_config=None  # 提取时不使用增强
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 提取embedding
        embeddings_dict = {}
        all_embeddings = []
        all_label_names = []
        all_image_paths = []
        all_mask_paths = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="提取embedding")):
                # 使用view1（提取时只需要一个view）
                images = batch['view1_image'].to(self.device)
                masks = batch['view1_mask'].to(self.device)
                label_names = batch['label_name']
                image_path = batch['image_path']
                mask_path = batch['mask_path']
                
                # 前向传播
                outputs = self.model(images, masks, return_features=False)
                embeddings = outputs['embeddings'].cpu().numpy()
                
                # 归一化embeddings（与训练时loss函数中的归一化保持一致）
                embeddings = F.normalize(
                    torch.from_numpy(embeddings),
                    dim=1,
                    p=2,
                    eps=1e-8
                ).numpy()
                
                # 计算样本在数据集中的索引
                start_idx = batch_idx * batch_size
                for i in range(len(embeddings)):
                    sample_idx = start_idx + i
                    embeddings_dict[sample_idx] = embeddings[i]
                    all_embeddings.append(embeddings[i])
                    all_label_names.append(label_names[i])
                    all_image_paths.append(image_path[i])
                    all_mask_paths.append(mask_path[i])
        
        # 保存
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.suffix == '.npz':
                np.savez(
                    output_file,
                    embeddings=np.array(all_embeddings),
                    label_names=np.array(all_label_names),
                    image_paths=np.array(all_image_paths),
                    mask_paths=np.array(all_mask_paths)
                )
                print(f"\n已保存到: {output_file}")
                print(f"  Embedding形状: {np.array(all_embeddings).shape}")
                print(f"  样本数: {len(all_embeddings)}")
                print(f"  类别数: {len(set(all_label_names))}")
            else:
                # 保存为字典格式
                np.save(output_file, embeddings_dict)
                print(f"\n已保存到: {output_file}")
        
        return embeddings_dict


def main():
    parser = argparse.ArgumentParser(description='提取SupCon模型embedding（使用与训练脚本相同的数据加载模式）')
    parser.add_argument('--model', type=str, default="checkpoints/debug/best_model.pth",
                       help='模型checkpoint路径')
    parser.add_argument('--data_config', type=str, default='configs/data_config_zhenyu.yaml',
                       help='数据配置文件路径（data_config_zhenyu.yaml）')
    parser.add_argument('--output', type=str, default="checkpoints/debug/supcon_embeddings.npz",
                       help='输出文件路径（.npz或.npy）')
    parser.add_argument('--config', type=str, default="configs/supcon_config.yaml",
                       help='训练配置文件路径（可选，如果checkpoint中没有配置）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='batch大小')
    args = parser.parse_args()
    
    logger = setup_logger('extract_supcon_embeddings')
    
    logger.info("加载模型...")
    extractor = SupConEmbeddingExtractor(
        model_path=args.model,
        config_path=args.config
    )
    
    logger.info("提取embedding...")
    embeddings_dict = extractor.extract_batch(
        data_config_path=args.data_config,
        output_file=args.output,
        batch_size=args.batch_size,
    )
    
    logger.info(f"已提取 {len(embeddings_dict)} 个embedding")
    logger.info(f"已保存到: {args.output}")


if __name__ == '__main__':
    main()
