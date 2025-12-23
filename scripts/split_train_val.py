"""
按类别划分训练集和验证集
从 patches 目录读取数据，按类别（子文件夹）划分，验证集比例 0.2
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def collect_samples(patches_dir: Path):
    """
    收集所有样本，按类别组织
    
    Args:
        patches_dir: patches 目录路径
        
    Returns:
        samples_dict: {label_name: [(image_path, mask_path), ...]}
    """
    samples_dict = defaultdict(list)
    
    # 支持的图像格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # 遍历所有类别文件夹（如 cable, metal_nut 等）
    for label_dir in patches_dir.iterdir():
        if not label_dir.is_dir():
            continue

        label_name = label_dir.name
        
        # 查找所有图像文件
        for image_path in label_dir.rglob("*"):
            if image_path.suffix.lower() not in image_extensions:
                continue
            
            # 跳过mask文件
            if "_mask" in image_path.stem.lower() or image_path.name.lower().endswith("_mask.png"):
                continue
            
            # 查找对应的mask文件
            mask_path = image_path.with_name(image_path.stem + "_mask" + image_path.suffix)
            if not mask_path.exists():
                # 尝试其他命名方式
                mask_path = image_path.with_name(image_path.stem + "_mask.png")
                if not mask_path.exists():
                    print(f"警告: 缺失掩码文件: {image_path}，跳过该样本")
                    continue
            
            samples_dict[label_name].append((image_path, mask_path))
    
    return samples_dict


def split_samples(samples_dict: dict, val_ratio: float = 0.2, seed: int = 42):
    """
    按类别划分训练集和验证集
    
    Args:
        samples_dict: {label_name: [(image_path, mask_path), ...]}
        val_ratio: 验证集比例
        seed: 随机种子
        
    Returns:
        train_samples: {label_name: [(image_path, mask_path), ...]}
        val_samples: {label_name: [(image_path, mask_path), ...]}
    """
    random.seed(seed)
    
    train_samples = defaultdict(list)
    val_samples = defaultdict(list)
    
    for label_name, samples in samples_dict.items():
        # 打乱样本
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        if len(shuffled_samples) > 5:
            # 计算验证集数量
            num_val = max(1, int(len(shuffled_samples) * val_ratio))
            
            # 划分
            val_samples[label_name] = shuffled_samples[:num_val]
            train_samples[label_name] = shuffled_samples[num_val:]
        else:
            # 样本数较少，全部放入训练集
            train_samples[label_name] = shuffled_samples
        
        print(f"{label_name}: 总数={len(samples)}, 训练集={len(train_samples[label_name])}, 验证集={len(val_samples[label_name])}")
    
    return train_samples, val_samples


def copy_samples(samples_dict: dict, output_dir: Path, split_name: str):
    """
    复制样本到目标目录
    
    Args:
        samples_dict: {label_name: [(image_path, mask_path), ...]}
        output_dir: 输出根目录
        split_name: 划分名称（'train' 或 'val'）
    """
    output_split_dir = output_dir / split_name
    output_split_dir.mkdir(parents=True, exist_ok=True)
    
    total_files = sum(len(samples) * 2 for samples in samples_dict.values())  # 每个样本有图像和mask两个文件
    
    with tqdm(total=total_files, desc=f"复制{split_name}集") as pbar:
        for label_name, samples in samples_dict.items():
            # 创建类别目录
            label_dir = output_split_dir / label_name
            label_dir.mkdir(parents=True, exist_ok=True)
            
            for image_path, mask_path in samples:
                # 复制图像
                dest_image_path = label_dir / image_path.name
                shutil.copy2(image_path, dest_image_path)
                pbar.update(1)
                
                # 复制mask
                dest_mask_path = label_dir / mask_path.name
                shutil.copy2(mask_path, dest_mask_path)
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description='按类别划分训练集和验证集')
    parser.add_argument('--patches_dir', type=str, 
                       default='data/datasets/zhenyu/1219-4.x/train',
                       help='patches 目录路径')
    parser.add_argument('--output_dir', type=str,
                       default='data/datasets/zhenyu/1219-4.x/split_data',
                       help='输出根目录（将在此目录下创建 train 和 val 文件夹）')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例（默认 0.2）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认 42）')
    args = parser.parse_args()
    
    patches_dir = Path(args.patches_dir)
    output_dir = Path(args.output_dir)
    
    if not patches_dir.exists():
        raise FileNotFoundError(f"patches 目录不存在: {patches_dir}")
    
    print(f"从 {patches_dir} 读取数据...")
    print(f"输出到 {output_dir}")
    print(f"验证集比例: {args.val_ratio}")
    print(f"随机种子: {args.seed}")
    print("-" * 50)
    
    # 收集所有样本
    print("收集样本...")
    samples_dict = collect_samples(patches_dir)
    
    if len(samples_dict) == 0:
        raise ValueError(f"在 {patches_dir} 中未找到任何样本")
    
    print(f"找到 {len(samples_dict)} 个类别:")
    total_samples = sum(len(samples) for samples in samples_dict.values())
    print(f"总样本数: {total_samples}")
    print("-" * 50)
    
    
    # 划分训练集和验证集
    print("划分训练集和验证集...")
    train_samples, val_samples = split_samples(samples_dict, args.val_ratio, args.seed)
    print("-" * 50)
    
    # 复制训练集
    print("复制训练集...")
    copy_samples(train_samples, output_dir, 'train')
    print("-" * 50)
    
    # 复制验证集
    print("复制验证集...")
    copy_samples(val_samples, output_dir, 'val')
    print("-" * 50)
    
    # 统计信息
    train_total = sum(len(samples) for samples in train_samples.values())
    val_total = sum(len(samples) for samples in val_samples.values())
    
    print("完成！")
    print(f"训练集: {train_total} 个样本，{len(train_samples)} 个类别")
    print(f"验证集: {val_total} 个样本，{len(val_samples)} 个类别")
    print(f"训练集目录: {output_dir / 'train'}")
    print(f"验证集目录: {output_dir / 'val'}")


if __name__ == '__main__':
    main()
