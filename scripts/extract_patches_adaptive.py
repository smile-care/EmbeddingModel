"""
使用自适应策略提取patch
从metadata.json文件读取图像和mask路径，对每个instance进行crop

策略：
1. 小尺寸缺陷（max(h, w) < 224）：不使用expand，根据max(h, w)选择固定crop尺寸
   - max(h, w) < 96: crop到96*96
   - max(h, w) < 160: crop到160*160
   - max(h, w) < 224: crop到224*224
2. 大尺寸缺陷（max(h, w) >= 224）：使用expand_ratio，保持crop_w和crop_h一致（正方形）

使用方法:
    python scripts/extract_patches_adaptive.py \
        --metadata data/mvtec_ad/raw/metadata.json \
        --output_dir output/patches
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    从mask提取外接矩形
    
    Args:
        mask: 二值mask (H, W)
        
    Returns:
        (x_min, y_min, x_max, y_max) 或 None
    """
    # 找到所有非零像素的位置
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def extract_instances_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    """
    从mask中提取所有独立的instance（连通区域）
    
    Args:
        mask: 二值mask (H, W)
        
    Returns:
        instance mask列表，每个是一个独立的连通区域
    """
    # 二值化
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 查找连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    instances = []
    for label_id in range(1, num_labels):  # 跳过背景（label=0）
        # 创建单个instance的mask
        instance_mask = (labels == label_id).astype(np.uint8) * 255
        instances.append(instance_mask)
    
    return instances


def compute_crop_region(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    expand_ratio: float = 0.1,
    crop_sizes: List[int] = [96, 160, 224]
) -> Tuple[int, int, int, int]:
    """
    根据自适应策略计算crop区域
    
    Args:
        bbox: (x_min, y_min, x_max, y_max)
        image_shape: (height, width)
        expand_ratio: 扩边比例（用于大尺寸缺陷）
        crop_sizes: 裁剪尺寸区间 [96, 160, 224]
        
    Returns:
        (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
    """
    x_min, y_min, x_max, y_max = bbox
    img_h, img_w = image_shape
    
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    bbox_max = max(bbox_w, bbox_h)
    
    # 根据bbox大小选择crop策略
    if bbox_max < crop_sizes[0]:  # < 96
        crop_size = crop_sizes[0]
        use_expand = False
    elif bbox_max < crop_sizes[1]:  # < 160
        crop_size = crop_sizes[1]
        use_expand = False
    elif bbox_max < crop_sizes[2]:  # < 224
        crop_size = crop_sizes[2]
        use_expand = False
    else:  # >= 224
        crop_size = None  # 使用expand策略
        use_expand = True
    
    # 计算crop区域
    if use_expand:
        # 使用expand策略，保持正方形
        expand_w = int(bbox_w * expand_ratio)
        expand_h = int(bbox_h * expand_ratio)
        
        # 扩展bbox
        expanded_x_min = max(0, x_min - expand_w)
        expanded_y_min = max(0, y_min - expand_h)
        expanded_x_max = min(img_w, x_max + expand_w)
        expanded_y_max = min(img_h, y_max + expand_h)
        
        # 计算扩展后的尺寸
        expanded_w = expanded_x_max - expanded_x_min
        expanded_h = expanded_y_max - expanded_y_min
        
        # 保持正方形，以较大的边为准
        crop_size_final = max(expanded_w, expanded_h)
        
        # 以bbox中心为中心，计算正方形crop区域
        bbox_center_x = (x_min + x_max) // 2
        bbox_center_y = (y_min + y_max) // 2
        
        half_size = crop_size_final // 2
        crop_x_min = max(0, bbox_center_x - half_size)
        crop_y_min = max(0, bbox_center_y - half_size)
        crop_x_max = min(img_w, crop_x_min + crop_size_final)
        crop_y_max = min(img_h, crop_y_min + crop_size_final)
        
        # 如果超出边界，调整到边界内
        if crop_x_max - crop_x_min < crop_size_final:
            crop_x_min = max(0, crop_x_max - crop_size_final)
        if crop_y_max - crop_y_min < crop_size_final:
            crop_y_min = max(0, crop_y_max - crop_size_final)
        
        # 确保最终是正方形
        final_w = crop_x_max - crop_x_min
        final_h = crop_y_max - crop_y_min
        final_crop_size = min(final_w, final_h)  # 取较小值确保在图像内
        
        # 调整到正方形
        if final_w != final_h:
            center_x = (crop_x_min + crop_x_max) // 2
            center_y = (crop_y_min + crop_y_max) // 2
            half_final = final_crop_size // 2
            crop_x_min = max(0, center_x - half_final)
            crop_y_min = max(0, center_y - half_final)
            crop_x_max = min(img_w, crop_x_min + final_crop_size)
            crop_y_max = min(img_h, crop_y_min + final_crop_size)
    else:
        # 使用固定crop尺寸，以bbox中心为中心
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        half_size = crop_size // 2
        crop_x_min = max(0, center_x - half_size)
        crop_y_min = max(0, center_y - half_size)
        crop_x_max = min(img_w, crop_x_min + crop_size)
        crop_y_max = min(img_h, crop_y_min + crop_size)
        
        # 如果超出边界，调整到边界内
        if crop_x_max - crop_x_min < crop_size:
            crop_x_min = max(0, crop_x_max - crop_size)
        if crop_y_max - crop_y_min < crop_size:
            crop_y_min = max(0, crop_y_max - crop_size)
        
        # 确保最终是正方形
        final_w = crop_x_max - crop_x_min
        final_h = crop_y_max - crop_y_min
        final_crop_size = min(final_w, final_h)  # 取较小值确保在图像内
        
        # 调整到正方形
        if final_w != final_h:
            center_x = (crop_x_min + crop_x_max) // 2
            center_y = (crop_y_min + crop_y_max) // 2
            half_final = final_crop_size // 2
            crop_x_min = max(0, center_x - half_final)
            crop_y_min = max(0, center_y - half_final)
            crop_x_max = min(img_w, crop_x_min + final_crop_size)
            crop_y_max = min(img_h, crop_y_min + final_crop_size)
    
    return (int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max))


def extract_patches_from_image_mask(
    image_path: str,
    mask_path: str,
    label: str,
    output_dir: str,
    expand_ratio: float = 0.1,
    min_size: int = 8,
    crop_sizes: List[int] = [96, 160, 224],
    prefix: Optional[str] = None,
    domain_id: Optional[str] = None
) -> List[dict]:
    """
    从图像和mask提取所有instance的patches
    
    Args:
        image_path: 图像路径
        mask_path: mask路径
        label: 标签
        output_dir: 输出目录
        expand_ratio: 扩边比例（用于大尺寸缺陷）
        min_size: 最小patch尺寸
        crop_sizes: 裁剪尺寸区间 [96, 160, 224]
        prefix: 输出文件名前缀（如果为None，使用图像文件名）
        domain_id: 域ID（如果为None，使用图像文件名）
    Returns:
        patch信息列表，每个包含patch路径、mask路径等信息
    """
    # 加载图像和mask
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法加载mask: {mask_path}")
    
    # 确保mask和图像尺寸一致
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    img_h, img_w = image.shape[:2]
    
    # 提取所有instances
    instance_masks = extract_instances_from_mask(mask)
    
    if len(instance_masks) == 0:
        print(f"警告: {mask_path} 中没有找到任何instance")
        return []
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / domain_id / f"{domain_id}_{label}"
    masks_dir = output_dir / domain_id / f"{domain_id}_{label}"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定文件名前缀
    if prefix is None:
        prefix = Path(image_path).stem
    
    patches_info = []
    
    # 处理每个instance
    for idx, instance_mask in enumerate(instance_masks):
        # 提取bbox
        bbox = extract_bbox_from_mask(instance_mask)
        if bbox is None:
            continue
        
        x_min, y_min, x_max, y_max = bbox
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        bbox_max = max(bbox_w, bbox_h)
        
        # 检查最小尺寸
        if bbox_max < min_size:
            continue
        
        # 计算crop区域
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = compute_crop_region(
            bbox, (img_h, img_w), expand_ratio, crop_sizes
        )
        
        # 裁剪patch
        patch = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        patch_mask = mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        # 只保留当前instance的区域（其他instance设为0）
        instance_mask_cropped = instance_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        patch_mask = np.where(instance_mask_cropped > 0, patch_mask, 0)
        
        # 如果crop区域不是正方形，需要调整
        if patch.shape[0] != patch.shape[1]:
            # 调整到正方形（使用较大的边）
            max_dim = max(patch.shape[0], patch.shape[1])
            patch = cv2.resize(patch, (max_dim, max_dim), interpolation=cv2.INTER_LINEAR)
            patch_mask = cv2.resize(patch_mask, (max_dim, max_dim), interpolation=cv2.INTER_NEAREST)
        
        # 保存patch
        patch_filename = f"{prefix}_{idx:03d}.png"
        patch_mask_filename = f"{prefix}_{idx:03d}_mask.png"
        patch_path = images_dir / patch_filename
        patch_mask_path = masks_dir / patch_mask_filename
        
        Image.fromarray(patch).save(patch_path)
        Image.fromarray(patch_mask).save(patch_mask_path)
        
        # 记录信息
        patches_info.append({
            'patch_path': str(patch_path),
            'patch_mask_path': str(patch_mask_path),
            'original_image_path': str(image_path),
            'original_mask_path': str(mask_path),
            'instance_id': idx,
            'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
            'crop_bbox': [int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max)],
            'original_size': [img_h, img_w],
            'patch_size': [patch.shape[0], patch.shape[1]]
        })
    
    return patches_info


def process_metadata(
    metadata_file: str,
    output_dir: str,
    expand_ratio: float = 0.1,
    min_size: int = 8,
    crop_sizes: List[int] = [96, 160, 224],
    metadata_root: Optional[str] = None
) -> None:
    """
    从metadata.json文件读取数据路径并批量处理
    
    Args:
        metadata_file: metadata.json文件路径
        output_dir: 输出目录
        expand_ratio: 扩边比例
        min_size: 最小patch尺寸
        crop_sizes: 裁剪尺寸区间
        metadata_root: metadata文件所在目录，用于解析相对路径（如果为None，使用metadata文件所在目录）
    """
    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        raise ValueError(f"Metadata文件不存在: {metadata_path}")
    
    # 确定根目录（用于解析相对路径）
    if metadata_root is None:
        metadata_root = metadata_path.parent
    else:
        metadata_root = Path(metadata_root)
    
    # 加载metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    instances = metadata.get('instances', [])
    if len(instances) == 0:
        print(f"警告: metadata文件中没有找到instances")
        return
    
    output_dir = Path(output_dir)
    all_patches_info = []
    failed_count = 0
    
    print(f"从metadata文件读取到 {len(instances)} 个instances")
    
    for instance in tqdm(instances, desc="处理instances"):
        # 获取图像和mask路径
        image_path_str = instance.get('image_path', '')
        mask_path_str = instance.get('mask_path', '')
        instance_id = instance.get('instance_id', 'unknown')
        domain_id = instance.get('domain_id', '')
        label = instance.get('label', '')
        if label == 'good':
            continue
        
        if not image_path_str or not mask_path_str:
            print(f"警告: instance {instance_id} 缺少image_path或mask_path，跳过")
            failed_count += 1
            continue
        
        # 解析路径（支持相对路径和绝对路径）
        image_path = Path(image_path_str)
        mask_path = Path(mask_path_str)
        
        # 如果是相对路径，基于metadata_root解析
        if not image_path.is_absolute():
            # 尝试多种可能的根目录
            possible_roots = [
                metadata_root,
                metadata_root.parent,
                metadata_root.parent.parent,
                Path.cwd(),
            ]
            
            image_path_found = None
            mask_path_found = None
            
            for root in possible_roots:
                test_image_path = root / image_path_str
                test_mask_path = root / mask_path_str
                if test_image_path.exists():
                    image_path_found = test_image_path
                if test_mask_path.exists():
                    mask_path_found = test_mask_path
            
            if image_path_found:
                image_path = image_path_found
            else:
                # 如果还是找不到，尝试直接使用metadata_root
                image_path = metadata_root / image_path_str
            
            if mask_path_found:
                mask_path = mask_path_found
            else:
                mask_path = metadata_root / mask_path_str
        
        # 检查文件是否存在
        if not image_path.exists():
            print(f"警告: 图像文件不存在: {image_path} (instance: {instance_id})")
            failed_count += 1
            continue
        
        if not mask_path.exists():
            print(f"警告: Mask文件不存在: {mask_path} (instance: {instance_id})")
            failed_count += 1
            continue
        
        patches_info = extract_patches_from_image_mask(
            str(image_path),
            str(mask_path),
            label=label,
            output_dir=output_dir,
            expand_ratio=expand_ratio,
            min_size=min_size,
            crop_sizes=crop_sizes,
            prefix=instance_id,
            domain_id=domain_id
        )


    print(f"\n处理完成:")
    print(f"  总instances数: {len(instances)}")
    print(f"  成功: {len(instances) - failed_count}")
    print(f"  失败: {failed_count}")
    print(f"  总patch数: {len(all_patches_info)}")


def main():
    parser = argparse.ArgumentParser(description='使用自适应策略提取patch（从metadata.json读取）')
    parser.add_argument('--metadata', type=str, default="data/mvtec_ad/raw/metadata.json",
                       help='Metadata JSON文件路径')
    parser.add_argument('--metadata_root', type=str, default=None,
                       help='Metadata根目录，用于解析相对路径（默认使用metadata文件所在目录）')
    parser.add_argument('--output_dir', type=str, default="data/datasets/mvtec_ad/patches", help='输出目录')
    parser.add_argument('--expand_ratio', type=float, default=0.2, help='扩边比例（用于大尺寸缺陷）')
    parser.add_argument('--min_size', type=int, default=8, help='最小patch尺寸')
    parser.add_argument('--crop_sizes', type=int, nargs='+', default=[96, 160, 224],
                       help='裁剪尺寸区间 [96, 160, 224]')
    
    args = parser.parse_args()
    
    process_metadata(
        args.metadata,
        args.output_dir,
        expand_ratio=args.expand_ratio,
        min_size=args.min_size,
        crop_sizes=args.crop_sizes,
        metadata_root=args.metadata_root
    )


if __name__ == '__main__':
    main()
