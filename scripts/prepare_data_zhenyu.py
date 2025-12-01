"""
zhenyu数据准备脚本
将JSON格式的标注数据转换为标准格式
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from src.data_preprocess.dataset_builder import DatasetBuilder
from src.data_preprocess.metadata_builder import MetadataBuilder
from src.data_preprocess.patch_extractor import PatchExtractor
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger


def create_mask_from_polygons(
    annotations: List[Dict],
    width: int,
    height: int
) -> np.ndarray:
    """
    从多边形标注创建mask
    
    Args:
        annotations: 标注列表，每个包含 label_shape 信息
        width: 图像宽度
        height: 图像高度
        
    Returns:
        mask数组 (H, W)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for ann in annotations:
        label_shape = ann.get('label_shape', {})
        if label_shape.get('label_shape_type') != 'POLYGON':
            continue
        
        points = label_shape.get('points', [])
        if len(points) < 3:  # 至少需要3个点才能形成多边形
            continue
        
        # 将归一化坐标转换为像素坐标
        polygon_points = []
        for pt in points:
            x = float(pt['x']) * width
            y = float(pt['y']) * height
            polygon_points.append([int(x), int(y)])
        
        polygon_points = np.array(polygon_points, dtype=np.int32)
        
        # 判断是添加还是减去区域
        is_subtract = label_shape.get('is_subtract', False)
        
        if is_subtract:
            # 减去区域（挖洞）
            cv2.fillPoly(mask, [polygon_points], 0)
        else:
            # 添加区域
            cv2.fillPoly(mask, [polygon_points], 255)
    
    return mask


def convert_zhenyu_data(
    zhenyu_root: str,
    output_root: str,
    domain_id: str = "battery_cover",
    include_unlabeled: bool = False
):
    """
    转换zhenyu数据格式
    
    Args:
        zhenyu_root: zhenyu数据根目录
        output_root: 输出根目录
        domain_id: domain ID（统一为battery_cover）
        include_unlabeled: 是否包含未标注数据（创建空mask）
    """
    zhenyu_root = Path(zhenyu_root)
    output_root = Path(output_root)
    
    # 创建输出目录结构
    # output_images_dir = output_root / "images"  # 不再需要images目录
    output_masks_dir = output_root / "masks"
    # output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger('convert_zhenyu')
    logger.info(f"开始转换zhenyu数据: {zhenyu_root}")
    logger.info("注意: 不复制原始图片，仅生成Mask和元数据")
    
    all_instances = []
    
    # 处理标注数据
    labeled_dir = zhenyu_root / "data_labeled"
    if labeled_dir.exists():
        json_files = list(labeled_dir.glob("*.json"))
        logger.info(f"找到 {len(json_files)} 个标注文件")
        
        for json_file in tqdm(json_files, desc="处理标注数据"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                image_id = data.get('image_id')
                image_name = data.get('name', f"image_{image_id}.png")
                full_path = data.get('full_path')
                width = data.get('width', 2448)
                height = data.get('height', 2048)
                annotations = data.get('annotations', [])
                
                # 检查图像文件是否存在
                if not full_path or not Path(full_path).exists():
                    logger.warning(f"图像文件不存在: {full_path}, 跳过")
                    continue
                
                # 为每个标注创建单独的实例
                if len(annotations) > 0:
                    for ann_idx, ann in enumerate(annotations):
                        # 为单个标注创建mask
                        mask = create_mask_from_polygons([ann], width, height)
                        
                        # 如果mask全为0，跳过
                        if np.sum(mask) == 0:
                            continue
                        
                        # 获取标注的feature_name作为label
                        label = ann.get('feature_name', 'unknown')
                        feature_id = ann.get('feature_id', ann_idx)
                        # 清理label名称（去除特殊字符，用于目录名）
                        label_clean = label.replace('/', '_').replace('\\', '_').replace(' ', '_')
                        
                        # 创建输出路径（如果同一图像有多个标注，使用feature_id区分）
                        image_stem = Path(full_path).stem
                        image_suffix = Path(image_name).suffix or '.png'  # 如果没有扩展名，默认使用.png
                        image_name_with_id = f"{image_stem}_f{feature_id}_{ann_idx}{image_suffix}"
                        
                        # Mask保存路径 (保持原有的目录结构，确保使用.png扩展名)
                        # 将mask文件名统一为.png格式
                        mask_name = Path(image_name_with_id).stem + '.png'
                        rel_path = Path(domain_id) / label_clean / mask_name
                        # output_img_path = output_images_dir / rel_path # 不再需要
                        output_mask_path = output_masks_dir / rel_path
                        
                        # output_img_path.parent.mkdir(parents=True, exist_ok=True)
                        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 不再复制图像
                        # if not output_img_path.exists():
                        #     shutil.copy2(full_path, output_img_path)
                        
                        # 保存mask
                        cv2.imwrite(str(output_mask_path), mask)
                        
                        # 添加到实例列表
                        all_instances.append({
                            'instance_id': f"{domain_id}_{label_clean}_{image_id}_f{feature_id}_{ann_idx}",
                            'image_path': str(full_path), # 使用原始绝对路径
                            'mask_path': str(output_mask_path),
                            'label': label,  # 使用原始label名称
                            'domain_id': domain_id,
                            'image_id': image_id,
                            'feature_id': feature_id,
                            'feature_name': label
                        })
                else:
                    # 没有标注，跳过
                    continue
                    
            except Exception as e:
                logger.warning(f"处理文件 {json_file} 时出错: {e}")
                continue
    
    # 处理未标注数据（可选）
    if include_unlabeled:
        unlabeled_dir = zhenyu_root / "data_unlabeled"
        if unlabeled_dir.exists():
            json_files = list(unlabeled_dir.glob("*.json"))
            logger.info(f"找到 {len(json_files)} 个未标注文件")
            
            for json_file in tqdm(json_files, desc="处理未标注数据"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    image_id = data.get('image_id')
                    image_name = data.get('name', f"image_{image_id}.png")
                    full_path = data.get('full_path')
                    width = data.get('width', 2448)
                    height = data.get('height', 2048)
                    
                    if not full_path or not Path(full_path).exists():
                        continue
                    
                    # 创建空mask
                    label = 'good'
                    # 确保mask文件名使用.png扩展名
                    mask_name = Path(image_name).stem + '.png'
                    rel_path = Path(domain_id) / label / mask_name
                    # output_img_path = output_images_dir / rel_path
                    output_mask_path = output_masks_dir / rel_path
                    
                    # output_img_path.parent.mkdir(parents=True, exist_ok=True)
                    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # shutil.copy2(full_path, output_img_path)
                    
                    # 创建空mask
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.imwrite(str(output_mask_path), mask)
                    
                    all_instances.append({
                        'instance_id': f"{domain_id}_good_{image_id}",
                        'image_path': str(full_path), # 使用原始绝对路径
                        'mask_path': str(output_mask_path),
                        'label': label,
                        'domain_id': domain_id,
                        'image_id': image_id
                    })
                    
                except Exception as e:
                    logger.warning(f"处理文件 {json_file} 时出错: {e}")
                    continue
    
    # 保存元数据
    metadata = {
        'total_instances': len(all_instances),
        'domains': sorted(list(set(inst['domain_id'] for inst in all_instances))),
        'labels': sorted(list(set(inst['label'] for inst in all_instances))),
        'instances': all_instances
    }
    
    metadata_file = output_root / "metadata.json"
    # 确保输出目录存在（如果没有创建images目录，可能output_root也不存在）
    output_root.mkdir(parents=True, exist_ok=True) 
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n转换完成！")
    logger.info(f"  总实例数: {len(all_instances)}")
    logger.info(f"  场景数: {len(metadata['domains'])}")
    logger.info(f"  类别数: {len(metadata['labels'])}")
    logger.info(f"  元数据文件: {metadata_file}")
    # logger.info(f"  图像目录: {output_images_dir}")
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


def extract_bbox_from_mask(mask: np.ndarray) -> Optional[tuple]:
    """
    从mask提取外接矩形
    
    Args:
        mask: 二值mask (H, W)
        
    Returns:
        (x_min, y_min, x_max, y_max) 或 None
    """
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def extract_patches_with_adaptive_strategy(
    metadata_file: str,
    output_dir: str,
    expand_ratio: float = 0.1,
    min_size: int = 8,
    crop_sizes: list = [96, 160, 224]
) -> str:
    """
    使用自适应策略提取patch
    
    策略：
    1. 小尺寸缺陷（max(h, w) < 224）：不使用expand，根据max(h, w)选择固定crop尺寸
       - max(h, w) < 96: crop到96*96
       - max(h, w) < 160: crop到160*160
       - max(h, w) < 224: crop到224*224
    2. 大尺寸缺陷（max(h, w) >= 224）：使用expand_ratio，保持crop_w和crop_h一致（正方形）
    
    Args:
        metadata_file: 元数据JSON文件路径
        output_dir: patch输出目录
        expand_ratio: 扩边比例（用于大尺寸缺陷）
        min_size: 最小patch尺寸
        crop_sizes: 裁剪尺寸区间 [96, 160, 224]
        
    Returns:
        处理后的元数据文件路径
    """
    # 加载元数据
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建patch元数据
    patch_metadata = {
        'total_patches': 0,
        'patches': []
    }
    
    def _to_serializable(obj):
        """递归转换对象中numpy类型为原生Python类型"""
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_to_serializable(v) for v in obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    failed_count = 0
    
    logger = setup_logger('extract_patches')
    logger.info("使用自适应策略提取patch...")
    
    for instance in tqdm(metadata['instances'], desc="提取patch"):
        try:
            # 构建完整路径
            image_path = Path(instance['image_path'])
            mask_path = Path(instance['mask_path'])
            
            if not image_path.is_absolute():
                logger.warning(f"图像路径不是绝对路径: {image_path}, 跳过")
                failed_count += 1
                continue
            
            if not image_path.exists():
                logger.warning(f"图像文件不存在: {image_path}, 跳过")
                failed_count += 1
                continue
            
            if not mask_path.exists():
                logger.warning(f"Mask文件不存在: {mask_path}, 跳过")
                failed_count += 1
                continue
            
            # 加载图像和mask
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"无法加载图像: {image_path}, 跳过")
                failed_count += 1
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"无法加载mask: {mask_path}, 跳过")
                failed_count += 1
                continue
            
            # 确保mask和图像尺寸一致
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            img_h, img_w = image.shape[:2]
            
            # 提取bbox
            bbox = extract_bbox_from_mask(mask)
            if bbox is None:
                failed_count += 1
                continue
            
            x_min, y_min, x_max, y_max = bbox
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            bbox_max = max(bbox_w, bbox_h)
            
            # 检查最小尺寸
            if bbox_max < min_size:
                failed_count += 1
                continue
            
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
            
            # 裁剪patch
            patch = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            patch_mask = mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            
            # 如果crop区域不是正方形，需要调整
            if patch.shape[0] != patch.shape[1]:
                # 调整到正方形（使用较大的边）
                max_dim = max(patch.shape[0], patch.shape[1])
                patch = cv2.resize(patch, (max_dim, max_dim))
                patch_mask = cv2.resize(patch_mask, (max_dim, max_dim))
            
            # Resize到最终输出尺寸（使用crop_sizes的最大值，即224）
            # final_patch_size = crop_sizes[-1]  # 224
            # patch = cv2.resize(patch, (final_patch_size, final_patch_size))
            # patch_mask = cv2.resize(patch_mask, (final_patch_size, final_patch_size))
            
            # 保存patch图像
            patch_filename = f"{mask_path.stem}.png"
            patch_mask_filename = f"{mask_path.stem}_mask.png"
            patch_path = output_dir / "images" / patch_filename
            patch_mask_path = output_dir / "masks" / patch_mask_filename
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_mask_path.parent.mkdir(parents=True, exist_ok=True)
            
            Image.fromarray(patch).save(patch_path)
            Image.fromarray(patch_mask).save(patch_mask_path)
            
            # 添加到patch元数据
            patch_metadata['patches'].append({
                'instance_id': instance['instance_id'],
                'patch_path': str(patch_path),
                'patch_mask_path': str(patch_mask_path),
                'label': instance['label'],
                'domain_id': instance['domain_id'],
                'original_image_path': str(image_path),
                'original_mask_path': str(mask_path),
                'bbox': [int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max)],
                'original_bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                'original_size': [int(img_h), int(img_w)],
                'crop_size': final_crop_size,
                'strategy': 'expand' if use_expand else 'fixed',
                # 'patch_size': final_patch_size
            })
            
        except Exception as e:
            logger.warning(f"处理实例 {instance.get('instance_id', 'unknown')} 时出错: {e}")
            failed_count += 1
            continue
    
    patch_metadata['total_patches'] = len(patch_metadata['patches'])
    
    # 保存patch元数据
    output_metadata_file = output_dir / "patch_metadata.json"
    patch_metadata_serializable = _to_serializable(patch_metadata)
    with open(output_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(patch_metadata_serializable, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n处理完成:")
    logger.info(f"  成功: {patch_metadata['total_patches']} 个patch")
    logger.info(f"  失败: {failed_count} 个实例")
    logger.info(f"  Patch元数据: {output_metadata_file}")
    
    return str(output_metadata_file)


def build_ssl_dataset_from_patches(
    patch_metadata_file: str,
    metadata_root: str,
    output_file: str = "ssl_dataset.json"
) -> str:
    """
    从patch数据构建SSL数据集
    
    Args:
        patch_metadata_file: patch元数据文件路径
        metadata_root: 元数据根目录
        output_file: 输出文件名
        
    Returns:
        输出文件路径
    """
    metadata_root = Path(metadata_root)
    patch_metadata_path = Path(patch_metadata_file)
    
    with open(patch_metadata_path, 'r', encoding='utf-8') as f:
        patch_metadata = json.load(f)
    
    # 收集所有唯一的patch路径（用于SSL预训练）
    patch_paths = set()
    domain_patches = defaultdict(list)
    
    for patch in patch_metadata['patches']:
        patch_path = patch['patch_path']
        domain_id = patch['domain_id']
        
        if patch_path not in patch_paths:
            patch_paths.add(patch_path)
            domain_patches[domain_id].append({
                'image_path': patch_path,  # SSL数据集使用patch_path作为image_path
                'domain_id': domain_id
            })
    
    ssl_dataset = {
        'total_images': len(patch_paths),
        'domains': list(domain_patches.keys()),
        'domain_counts': {d: len(patches) for d, patches in domain_patches.items()},
        'images': []
    }
    
    # 展平patch列表
    for patches in domain_patches.values():
        ssl_dataset['images'].extend(patches)
    
    # 保存
    output_path = metadata_root / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ssl_dataset, f, indent=2, ensure_ascii=False)
    
    logger = setup_logger('build_ssl_dataset')
    logger.info(f"SSL数据集已保存: {output_path}")
    logger.info(f"  总patch数: {ssl_dataset['total_images']}")
    logger.info(f"  场景数: {len(ssl_dataset['domains'])}")
    for domain, count in ssl_dataset['domain_counts'].items():
        logger.info(f"    {domain}: {count} 个patch")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='zhenyu数据准备脚本')
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--zhenyu_root', type=str, default='data/zhenyu_data',
                       help='zhenyu数据根目录')
    parser.add_argument('--output_root', type=str, default='data/zhenyu_data/raw',
                       help='输出根目录')
    parser.add_argument('--domain_id', type=str, default='battery_cover',
                       help='domain ID（统一为battery_cover）')
    parser.add_argument('--include_unlabeled', action='store_true',
                       help='是否包含未标注数据（创建空mask）')
    parser.add_argument('--step', type=str, choices=['convert', 'metadata', 'patch', 'dataset', 'all'],
                       default='dataset', help='执行步骤')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.data_config)
    data_config = config['data']
    
    # 设置日志
    logger = setup_logger('prepare_data_zhenyu')
    
    # 步骤1: 转换数据格式
    if args.step in ['convert', 'all']:
        logger.info("=" * 60)
        logger.info("步骤1: 转换zhenyu数据格式")
        logger.info("=" * 60)
        
        metadata_file = convert_zhenyu_data(
            zhenyu_root=args.zhenyu_root,
            output_root=args.output_root,
            domain_id=args.domain_id,
            include_unlabeled=args.include_unlabeled
        )
        
        # 使用output_root下的metadata目录作为metadata_root
        metadata_root = Path(args.output_root).parent / "metadata"
        metadata_root.mkdir(parents=True, exist_ok=True)
        if Path(metadata_file).parent != metadata_root:
            shutil.copy2(metadata_file, metadata_root / "metadata.json")
            logger.info(f"已复制元数据到: {metadata_root / 'metadata.json'}")
    
    # 步骤2: 提取patch
    if args.step in ['patch', 'all']:
        logger.info("=" * 60)
        logger.info("步骤2: 提取patch（使用自适应策略）")
        logger.info("=" * 60)
        
        # 使用output_root下的metadata目录
        metadata_root = Path(args.output_root).parent / "metadata"
        metadata_file = metadata_root / "metadata.json"
        if not metadata_file.exists():
            # 尝试从output_root读取
            metadata_file = Path(args.output_root) / "metadata.json"
            if not metadata_file.exists():
                logger.error(f"元数据文件不存在: {metadata_file}")
                return
        
        # 使用output_root下的patches目录
        patches_root = Path(args.output_root).parent / "patches"
        patch_metadata_file = extract_patches_with_adaptive_strategy(
            str(metadata_file),
            output_dir=str(patches_root),
            expand_ratio=data_config['patch']['expand_ratio'],
            min_size=data_config['patch']['min_size'],
            crop_sizes=[96, 160, 224]
        )
        logger.info(f"Patch元数据已保存: {patch_metadata_file}")
    
    # 步骤3: 构建数据集
    if args.step in ['dataset', 'all']:
        logger.info("=" * 60)
        logger.info("步骤3: 构建数据集（使用patch数据）")
        logger.info("=" * 60)
        
        # 使用output_root下的metadata目录
        metadata_root = Path(args.output_root).parent / "metadata"
        dataset_builder = DatasetBuilder(str(metadata_root))
        
        # 查找patch_metadata文件
        patch_metadata_file = metadata_root / "patch_metadata.json"
        if not patch_metadata_file.exists():
            # 尝试从patches目录读取
            patches_root = Path(args.output_root).parent / "patches"
            patch_metadata_file = patches_root / "patch_metadata.json"
            if not patch_metadata_file.exists():
                logger.error(f"Patch元数据文件不存在，请先运行步骤2提取patch")
                return
        
        # 构建SSL数据集（使用patch数据）
        logger.info("构建SSL数据集（从patch数据）...")
        ssl_dataset_file = build_ssl_dataset_from_patches(
            str(patch_metadata_file),
            str(metadata_root),
            output_file="ssl_dataset.json"
        )
        logger.info(f"SSL数据集已保存: {ssl_dataset_file}")
        
        # 构建SupCon数据集（使用patch数据）
        logger.info("构建SupCon数据集（从patch数据）...")
        supcon_dataset_file = dataset_builder.build_supcon_dataset(
            str(patch_metadata_file),
            output_file="supcon_dataset.json"
        )
        logger.info(f"SupCon数据集已保存: {supcon_dataset_file}")
        
        # 过滤数据集
        logger.info("过滤数据集...")
        filtered_file = dataset_builder.filter_dataset(
            str(patch_metadata_file),
            min_samples_per_class=data_config['filtering']['min_samples_per_class'],
            max_samples_per_class=data_config['filtering'].get('max_samples_per_class'),
            output_file="supcon_dataset_filtered.json"
        )
        logger.info(f"过滤后的数据集已保存: {filtered_file}")
    
    logger.info("=" * 60)
    logger.info("数据准备完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

