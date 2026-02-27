"""
提取SupCon/MoCo模型的embedding
适配最新的算法架构：DINOv3 Backbone + FPN + Mask特征筛选 + 多层特征融合 + Projection Head
支持标准SupCon模型和MoCo（动量对比学习）模型
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.supcon_train.datasets.supcon_dataset import MaskSoftDilation, SupConDataset
from src.supcon_train.models.moco_model import MoCoModel
from src.supcon_train.models.supcon_model import SupConModel
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger


class FolderDataset(Dataset):
    """从文件夹路径直接加载数据的简单数据集类"""
    
    def __init__(
        self,
        folder_path: str,
        image_size: int = 224
    ):
        """
        初始化数据集
        
        Args:
            folder_path: 文件夹路径
            image_size: 图像尺寸
        """
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {self.folder_path}")
        
        self.image_size = image_size
        
        # 加载样本列表
        self.samples = self._load_samples()
        
        # 验证集变换（与SupConDataset的val split保持一致）
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.val_mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        # mask软膨胀处理器（与SupConDataset保持一致）
        mask_dilation_config = {
            'enabled': True
        }
        self.mask_dilation = MaskSoftDilation(mask_dilation_config)
    
    def _load_samples(self) -> list:
        """加载数据样本列表"""
        samples = []
        
        # 支持的图像格式
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        # 递归查找所有图像文件
        for file_path in self.folder_path.rglob("*"):
            if file_path.suffix.lower() not in image_extensions:
                continue
            
            # 跳过mask文件
            if "_mask" in file_path.stem.lower() or file_path.name.lower().endswith("_mask.png"):
                continue
            
            # 尝试查找对应的mask文件
            # 支持多种命名方式：{name}_mask.png, {name}.mask.png, mask/{name}.png
            mask_path = None
            possible_mask_paths = [
                file_path.with_name(file_path.stem + "_mask" + file_path.suffix),
                file_path.with_name(file_path.stem + "_mask.png"),
                file_path.with_suffix(".mask.png"),
                file_path.parent / "mask" / file_path.name,
            ]
            
            for mp in possible_mask_paths:
                if mp.exists():
                    mask_path = mp
                    break
            
            # 使用每张图片的直接父文件夹名作为label_name
            label_name = file_path.parent.name
            
            assert mask_path is not None and mask_path.exists(), f"缺失掩码文件: {mask_path}"
            samples.append({
                'image_path': str(file_path),
                'mask_path': str(mask_path),
                'label': label_name
            })
        
        if len(samples) == 0:
            raise ValueError(f"在文件夹 {self.folder_path} 中未找到任何图像文件")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        view1_image = self.val_transform(image)
        view2_image = view1_image.clone()
        
        # 加载mask
        if sample['mask_path'] and Path(sample['mask_path']).exists():
            mask = Image.open(sample['mask_path']).convert('L')
            view1_mask = self.val_mask_transform(mask)
            view1_mask = (view1_mask > 0.5).float()  # 二值化
            # 应用软膨胀（与SupConDataset的val split保持一致）
            view1_mask = self.mask_dilation(view1_mask)
        else:
            # 如果没有mask，使用全1的mask
            view1_mask = torch.ones(1, self.image_size, self.image_size)
        
        view2_mask = view1_mask.clone()
        
        return {
            'view1_image': view1_image,
            'view1_mask': view1_mask,
            'view2_image': view2_image,
            'view2_mask': view2_mask,
            'label': 0,  # 单一类别，使用0作为标签索引
            'label_name': sample['label'],
            'image_path': sample['image_path'],
            'mask_path': sample['mask_path'] or '',
        }


class SupConEmbeddingExtractor:
    """SupCon/MoCo模型embedding提取器（支持mask）"""
    
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
            moco_config = config['supcon'].get('moco', {})
        else:
            model_config = checkpoint.get('config', {}).get('supcon', {}).get('model', {})
            moco_config = checkpoint.get('config', {}).get('supcon', {}).get('moco', {})
        
        # 检测模型类型：检查checkpoint中是否包含MoCo相关的键
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        is_moco_model = any('query_encoder' in key or 'momentum_encoder' in key for key in state_dict.keys())
        
        if is_moco_model:
            print("检测到MoCo模型，使用MoCoModel")
            # 创建MoCo模型
            momentum = moco_config.get('momentum', 0.999)
            self.model = MoCoModel(
                model_name=model_config.get('model_name', 'facebook/dinov3-convnext-small-pretrain-lvd1689m'),
                embedding_dim=model_config.get('embedding_dim', 128),
                projection_hidden_dims=model_config.get('projection_head', {}).get('hidden_dims', [256, 128]),
                image_size=model_config.get('image_size', 224),
                freeze_backbone=False,  # 提取时不需要冻结
                use_layers=model_config.get('use_layers', [1, 2, 3, 4]),
                fpn_out_channels=model_config.get('fpn_out_channels', 256),
                fusion_dim=model_config.get('fusion_dim', 512),
                momentum=momentum
            ).to(self.device)
            self.use_moco = True
        else:
            print("检测到标准SupCon模型，使用SupConModel")
            # 创建标准SupCon模型
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
            self.use_moco = False
        
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
            # 如果是MoCo模型，使用query_encoder（mode='query'）
            if self.use_moco:
                outputs = self.model(image_tensor, mask_tensor, mode='query', return_features=False)
            else:
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
            data_config_path: 数据配置文件路径（data_config_zhenyu.yaml）或文件夹路径
            output_file: 输出文件路径（.npy或.npz）
            batch_size: batch大小
            split: 数据集划分（'train' 或 'val'）
            
        Returns:
            {sample_idx: embedding} 字典，其中sample_idx是样本在数据集中的索引
        """
        # 检测输入是文件夹路径还是YAML配置文件
        data_path = Path(data_config_path)
        if data_path.is_dir():
            # 文件夹路径：使用FolderDataset
            # 每张图片使用其直接父文件夹名作为label_name
            dataset = FolderDataset(
                folder_path=str(data_path),
                image_size=self.image_size
            )
            print(f"使用文件夹模式: {data_path}")
            # 统计所有不同的label_name
            unique_labels = set(sample['label'] for sample in dataset.samples)
            print(f"  找到 {len(dataset)} 个样本")
            print(f"  包含 {len(unique_labels)} 个不同的标签: {sorted(unique_labels)}")
        elif data_path.suffix in ['.yaml', '.yml']:
            # YAML配置文件：使用SupConDataset
            dataset = SupConDataset(
                data_config_path=data_config_path,
                split="val",
                image_size=self.image_size,)
            print(f"使用YAML配置模式: {data_config_path}")
        else:
            raise ValueError(
                f"输入路径既不是文件夹也不是YAML文件: {data_config_path}\n"
                f"请提供文件夹路径或.yaml/.yml配置文件路径"
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
                # 如果是MoCo模型，使用query_encoder（mode='query'）
                if self.use_moco:
                    outputs = self.model(images, masks, mode='query', return_features=False)
                else:
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
    parser = argparse.ArgumentParser(description='提取SupCon/MoCo模型embedding（使用与训练脚本相同的数据加载模式）')
    parser.add_argument('--model', type=str, default="checkpoints/supcon_models/0105-1/checkpoint_epoch_300.pth",
                       help='模型checkpoint路径')
    parser.add_argument('--data_config', type=str, default='data/zhenyu_data/test_data/628K_CCD1_R角压印',
                       help='数据配置文件路径（.yaml/.yml）或文件夹路径。\
                       如果提供文件夹路径，将自动搜索其中的图像和对应的mask，父文件夹名作为label_name')
    parser.add_argument('--output', type=str, default="checkpoints/supcon_models/0105-1/extract_embeddings/test_data/628K_CCD1_R角压印.npz",
                       help='输出文件路径（.npz或.npy）')
    parser.add_argument('--config', type=str, default="configs/supcon_config.yaml",
                       help='训练配置文件路径（可选，如果checkpoint中没有配置）')
    parser.add_argument('--batch_size', type=int, default=8,
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
