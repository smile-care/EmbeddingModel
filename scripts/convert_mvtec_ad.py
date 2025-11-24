"""
MVTec AD数据集转换脚本
将MVTec AD格式转换为适配代码的数据格式
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import cv2
import numpy as np
import sys

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocess.metadata_builder import MetadataBuilder
from src.utils.logging import setup_logger


def create_empty_mask(image_path: Path, output_path: Path):
    """
    创建空mask（用于正常样本）
    
    Args:
        image_path: 原始图像路径
        output_path: 输出mask路径
    """
    # 读取图像获取尺寸
    img = cv2.imread(str(image_path))
    if img is not None:
        h, w = img.shape[:2]
        # 创建全零mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(output_path), mask)
    else:
        # 如果无法读取图像，创建默认尺寸的mask
        mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.imwrite(str(output_path), mask)


def convert_mvtec_ad(
    mvtec_root: str,
    output_root: str,
    include_good: bool = True,
    create_empty_masks_for_good: bool = True
):
    """
    转换MVTec AD数据集
    
    Args:
        mvtec_root: MVTec AD数据集根目录
        output_root: 输出根目录
        include_good: 是否包含正常样本（good）
        create_empty_masks_for_good: 是否为正常样本创建空mask
    """
    mvtec_root = Path(mvtec_root)
    output_root = Path(output_root)
    
    # 创建输出目录结构
    output_images_dir = output_root / "images"
    output_masks_dir = output_root / "masks"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger('convert_mvtec_ad')
    logger.info(f"开始转换MVTec AD数据集: {mvtec_root}")
    
    # 获取所有类别
    categories = [d for d in mvtec_root.iterdir() 
                  if d.is_dir() and d.name not in ['license.txt', 'readme.txt']]
    
    logger.info(f"找到 {len(categories)} 个类别: {[c.name for c in categories]}")
    
    all_instances = []
    
    for category_dir in tqdm(categories, desc="处理类别"):
        category_name = category_dir.name
        logger.info(f"处理类别: {category_name}")
        
        # 处理测试集的缺陷样本
        test_dir = category_dir / "test"
        if test_dir.exists():
            # 获取所有缺陷类型（排除good）
            defect_types = [d for d in test_dir.iterdir() 
                          if d.is_dir() and d.name != "good"]
            
            for defect_type_dir in defect_types:
                defect_type = defect_type_dir.name
                logger.info(f"  处理缺陷类型: {defect_type}")
                
                # 图像目录
                image_files = list(defect_type_dir.glob("*.png"))
                
                # 对应的mask目录
                mask_dir = category_dir / "ground_truth" / defect_type
                
                for img_file in image_files:
                    # 查找对应的mask
                    mask_file = mask_dir / f"{img_file.stem}_mask.png"
                    
                    if not mask_file.exists():
                        # 尝试其他可能的mask文件名
                        mask_file = mask_dir / img_file.name
                        if not mask_file.exists():
                            logger.warning(f"    未找到mask: {img_file.name}, 跳过")
                            continue
                    
                    # 复制文件到输出目录（保持类别和缺陷类型的目录结构）
                    rel_path = Path(category_name) / defect_type / img_file.name
                    output_img_path = output_images_dir / rel_path
                    output_mask_path = output_masks_dir / rel_path
                    
                    output_img_path.parent.mkdir(parents=True, exist_ok=True)
                    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(img_file, output_img_path)
                    shutil.copy2(mask_file, output_mask_path)
                    
                    # 添加到实例列表
                    all_instances.append({
                        'instance_id': f"{category_name}_{defect_type}_{img_file.stem}",
                        'image_path': str(output_img_path),
                        'mask_path': str(output_mask_path),
                        'label': defect_type,
                        'domain_id': category_name,
                        'original_category': category_name,
                        'defect_type': defect_type
                    })
        
        # 处理正常样本（good）
        if include_good:
            # 训练集的good样本
            train_good_dir = category_dir / "train" / "good"
            if train_good_dir.exists():
                good_files = list(train_good_dir.glob("*.png"))
                logger.info(f"  处理训练集good样本: {len(good_files)} 个")
                
                for img_file in good_files:
                    # 创建输出路径
                    rel_path = Path(category_name) / "good" / "train" / img_file.name
                    output_img_path = output_images_dir / rel_path
                    output_mask_path = output_masks_dir / rel_path
                    
                    output_img_path.parent.mkdir(parents=True, exist_ok=True)
                    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 复制图像
                    shutil.copy2(img_file, output_img_path)
                    
                    # 创建空mask或跳过
                    if create_empty_masks_for_good:
                        create_empty_mask(output_img_path, output_mask_path)
                        all_instances.append({
                            'instance_id': f"{category_name}_good_train_{img_file.stem}",
                            'image_path': str(output_img_path),
                            'mask_path': str(output_mask_path),
                            'label': 'good',
                            'domain_id': category_name,
                            'original_category': category_name,
                            'defect_type': 'good'
                        })
            
            # 测试集的good样本
            test_good_dir = category_dir / "test" / "good"
            if test_good_dir.exists():
                good_files = list(test_good_dir.glob("*.png"))
                logger.info(f"  处理测试集good样本: {len(good_files)} 个")
                
                for img_file in good_files:
                    rel_path = Path(category_name) / "good" / "test" / img_file.name
                    output_img_path = output_images_dir / rel_path
                    output_mask_path = output_masks_dir / rel_path
                    
                    output_img_path.parent.mkdir(parents=True, exist_ok=True)
                    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(img_file, output_img_path)
                    
                    if create_empty_masks_for_good:
                        create_empty_mask(output_img_path, output_mask_path)
                        all_instances.append({
                            'instance_id': f"{category_name}_good_test_{img_file.stem}",
                            'image_path': str(output_img_path),
                            'mask_path': str(output_mask_path),
                            'label': 'good',
                            'domain_id': category_name,
                            'original_category': category_name,
                            'defect_type': 'good'
                        })
    
    # 保存元数据
    metadata = {
        'total_instances': len(all_instances),
        'domains': sorted(list(set(inst['domain_id'] for inst in all_instances))),
        'labels': sorted(list(set(inst['label'] for inst in all_instances))),
        'instances': all_instances
    }
    
    metadata_file = output_root / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n转换完成！")
    logger.info(f"  总实例数: {len(all_instances)}")
    logger.info(f"  场景数: {len(metadata['domains'])}")
    logger.info(f"  类别数: {len(metadata['labels'])}")
    logger.info(f"  元数据文件: {metadata_file}")
    logger.info(f"  图像目录: {output_images_dir}")
    logger.info(f"  Mask目录: {output_masks_dir}")
    
    # 打印统计信息
    logger.info("\n各场景统计:")
    for domain in metadata['domains']:
        domain_instances = [inst for inst in all_instances if inst['domain_id'] == domain]
        domain_labels = set(inst['label'] for inst in domain_instances)
        logger.info(f"  {domain}: {len(domain_instances)} 个实例, {len(domain_labels)} 个类别")
        for label in sorted(domain_labels):
            count = sum(1 for inst in domain_instances if inst['label'] == label)
            logger.info(f"    - {label}: {count} 个")
    
    return str(metadata_file)


def main():
    parser = argparse.ArgumentParser(description='转换MVTec AD数据集')
    parser.add_argument('--mvtec_root', type=str, 
                       default='data/mvtec_anomaly_detection',
                       help='MVTec AD数据集根目录')
    parser.add_argument('--output_root', type=str,
                       default='data/mvtec_ad/raw',
                       help='输出根目录')
    parser.add_argument('--include_good', action='store_true', default=True,
                       help='是否包含正常样本（good）')
    parser.add_argument('--no_empty_masks', action='store_true',
                       help='不为正常样本创建空mask（如果设置，good样本将被跳过）')
    args = parser.parse_args()
    
    convert_mvtec_ad(
        mvtec_root=args.mvtec_root,
        output_root=args.output_root,
        include_good=args.include_good,
        create_empty_masks_for_good=not args.no_empty_masks
    )


if __name__ == '__main__':
    main()

