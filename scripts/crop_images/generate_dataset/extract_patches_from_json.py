"""
从图像和JSON标注文件提取patch
输入文件夹中每个图像对应一个JSON文件，JSON文件包含多边形标注
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


def polygon_to_mask(polygon: List[List[float]], img_h: int, img_w: int) -> np.ndarray:
    """
    将归一化的多边形坐标转换为mask
    
    Args:
        polygon: 多边形点列表，每个点是 [x, y] (归一化坐标 0-1)
        img_h: 图像高度
        img_w: 图像宽度
        
    Returns:
        mask (H, W) uint8
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    if len(polygon) < 3:
        return mask
    
    # 将归一化坐标转换为像素坐标
    points = np.array(polygon, dtype=np.float32)
    points[:, 0] *= img_w  # x坐标
    points[:, 1] *= img_h  # y坐标
    points = points.astype(np.int32)
    
    # 填充多边形
    cv2.fillPoly(mask, [points], 255)
    
    return mask


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
    crop_sizes: List[int] = [96, 160, 224]
) -> Tuple[int, int, int, int]:
    """
    根据自适应策略计算crop区域
    
    Args:
        bbox: (x_min, y_min, x_max, y_max)
        image_shape: (height, width)
        crop_sizes: 裁剪尺寸区间 [96, 160, 224]
        
    Returns:
        (crop_x_min, crop_y_min, crop_x_max, crop_y_max)
    """
    x_min, y_min, x_max, y_max = bbox
    img_h, img_w = image_shape
    
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    bbox_max = max(bbox_w, bbox_h)
    
    # 确保crop_size至少为bbox_max的2倍
    min_required_crop_size = int(bbox_max * 2)
    
    # 根据bbox大小选择crop策略，确保满足余量要求
    if bbox_max < crop_sizes[0]:  # < 96
        # 选择满足余量要求的最小crop_size
        crop_size = max(crop_sizes[0], min_required_crop_size)
        use_expand = False
    elif bbox_max < crop_sizes[1]:  # < 160
        crop_size = max(crop_sizes[1], min_required_crop_size)
        use_expand = False
    elif bbox_max < crop_sizes[2]:  # < 224
        crop_size = max(crop_sizes[2], min_required_crop_size)
        use_expand = False
    else:  # >= 224
        crop_size = None  # 使用expand策略
        use_expand = True
    
    # 计算crop区域
    if use_expand:
        # 使用expand策略，保持正方形
        # 确保expand后的余量至少为bbox_max的1/2
        min_padding = bbox_max / 2
        expand_w = int(min_padding)
        expand_h = int(min_padding)
        
        # 扩展bbox
        expanded_x_min = max(0, x_min - expand_w)
        expanded_y_min = max(0, y_min - expand_h)
        expanded_x_max = min(img_w, x_max + expand_w)
        expanded_y_max = min(img_h, y_max + expand_h)
        
        # 计算扩展后的尺寸
        expanded_w = expanded_x_max - expanded_x_min
        expanded_h = expanded_y_max - expanded_y_min
        
        # 保持正方形，以较大的边为准，但确保满足最小余量要求
        crop_size_final = max(expanded_w, expanded_h, min_required_crop_size)
        
        # 以bbox中心为中心，计算正方形crop区域
        bbox_center_x = (x_min + x_max) // 2
        bbox_center_y = (y_min + y_max) // 2
        
        half_size = crop_size_final // 2
        crop_x_min = max(0, bbox_center_x - half_size)
        crop_y_min = max(0, bbox_center_y - half_size)
        crop_x_max = min(img_w, crop_x_min + crop_size_final)
        crop_y_max = min(img_h, crop_y_min + crop_size_final)
        
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


def extract_patches_from_image_json(
    image_path: str,
    json_path: str,
    output_dir: str,
    min_size: int = 8,
    crop_sizes: List[int] = [96, 160, 224],
    label: Optional[str] = None
) -> List[dict]:
    """
    从图像和JSON文件提取所有instance的patches
    
    Args:
        image_path: 图像路径
        json_path: JSON标注文件路径
        output_dir: 输出目录
        min_size: 最小patch尺寸
        crop_sizes: 裁剪尺寸区间 [96, 160, 224]
        label: 标签（如果为None，从JSON文件名或文件夹名推断）
    Returns:
        patch信息列表，每个包含patch路径、mask路径等信息
    """
    # 加载图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img_h, img_w = image.shape[:2]
    
    # 加载JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 从JSON中提取所有标签的多边形
    labels_data = json_data.get('labels', [])
    if len(labels_data) == 0:
        print(f"警告: {json_path} 中没有找到labels")
        return []
    
    # 合并所有多边形到一个mask中
    # 先处理所有非subtract的多边形，再处理subtract的多边形
    combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # 先添加所有非subtract的多边形
    for label_data in labels_data:
        is_subtract = label_data.get('isSubtract', False)
        if is_subtract:
            continue
        
        points = label_data.get('points', [])
        if len(points) < 3:
            continue
        
        # 将多边形转换为mask
        polygon_mask = polygon_to_mask(points, img_h, img_w)
        combined_mask = np.where(polygon_mask > 0, 255, combined_mask)
    
    # 再减去所有subtract的多边形
    for label_data in labels_data:
        is_subtract = label_data.get('isSubtract', False)
        if not is_subtract:
            continue
        
        points = label_data.get('points', [])
        if len(points) < 3:
            continue
        
        # 将多边形转换为mask
        polygon_mask = polygon_to_mask(points, img_h, img_w)
        combined_mask = np.where(polygon_mask > 0, 0, combined_mask)
    
    # 提取所有instances
    instance_masks = extract_instances_from_mask(combined_mask)
    
    if len(instance_masks) == 0:
        print(f"警告: {json_path} 中没有找到任何instance")
        return []
    
    # 确定标签
    if label is None:
        # 尝试从JSON文件名或路径推断
        json_name = Path(json_path).stem
        label = json_name.split('_')[-1] if '_' in json_name else 'defect'
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / label
    masks_dir = output_dir / label
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定文件名前缀
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
            bbox, (img_h, img_w), crop_sizes
        )
        
        # 裁剪patch
        patch = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        patch_mask = instance_mask[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
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
            'original_json_path': str(json_path),
            'instance_id': idx,
            'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
            'crop_bbox': [int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max)],
            'original_size': [img_h, img_w],
            'patch_size': [patch.shape[0], patch.shape[1]]
        })
    
    return patches_info


def process_folder(
    input_dir: str,
    output_dir: str,
    min_size: int = 8,
    crop_sizes: List[int] = [96, 160, 224],
    image_extensions: List[str] = ['.bmp', '.jpg', '.jpeg', '.png']
) -> None:
    """
    处理输入文件夹中的所有图像和JSON文件
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出目录
        min_size: 最小patch尺寸
        crop_sizes: 裁剪尺寸区间
        image_extensions: 图像文件扩展名列表
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"输入文件夹不存在: {input_path}")
    
    if not input_path.is_dir():
        raise ValueError(f"输入路径不是文件夹: {input_path}")
    
    # 查找所有图像文件
    json_files = []
    json_files.extend(list(input_path.rglob("*.json")))
    
    assert len(json_files) > 0, f"警告: 在 {input_dir} 中没有找到JSON文件"
    
    print(f"开始处理文件夹{input_dir}")
    print(f"找到 {len(json_files)} 个JSON文件")
    
    output_dir = Path(output_dir)
    all_patches_info = []
    failed_count = 0
    processed_count = 0
    
    
    for json_path in tqdm(json_files, desc="处理JSON文件"):
        # 查找对应的图像文件
        image_path = json_path.with_suffix('')
        if not image_path.exists():
            print(f"警告: 找不到对应的图像文件: {image_path}")
            failed_count += 1
            continue
        
        label = image_path.parent.name
        
        try:
            patches_info = extract_patches_from_image_json(
                str(image_path),
                str(json_path),
                output_dir,
                min_size=min_size,
                crop_sizes=crop_sizes,
                label=label
            )
            
            all_patches_info.extend(patches_info)
            processed_count += 1
            
        except Exception as e:
            print(f"错误: 处理 {image_path} 时出错: {e}")
            failed_count += 1
            continue
    
    print(f"\n处理完成:")
    print(f"  总图像数: {len(json_files)}")
    print(f"  成功: {processed_count}")
    print(f"  失败: {failed_count}")
    print(f"  总patch数: {len(all_patches_info)}")
    print(f"  输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='从图像和JSON标注文件提取patch')
    parser.add_argument('--input_dir', type=str, 
                       default="data/zhenyu_data/E0_2/L5",
                       help='输入文件夹路径（包含图像和JSON文件）')
    parser.add_argument('--output_dir', type=str, default="data/zhenyu_data/E0_2/L5_crop", help='输出目录')
    parser.add_argument('--min_size', type=int, default=8, help='最小patch尺寸')
    parser.add_argument('--crop_sizes', type=int, nargs='+', default=[96, 160, 224],
                       help='裁剪尺寸区间 [96, 160, 224]')
    
    args = parser.parse_args()
    
    process_folder(
        args.input_dir,
        args.output_dir,
        min_size=args.min_size,
        crop_sizes=args.crop_sizes
    )


if __name__ == '__main__':
    main()
        
    # src_dir = "/home/unitx/workspace_custom/data/震裕/4.x/60194/4xdata"
    # save_dir = "/home/unitx/workspace_custom/EmbeddingModel/data/zhenyu_data/60194"
    # dir_list = [
    #     "60194-下塑胶装反",
    #     "60194-CCD1-二维码",
    #     "60194-CCD1-防爆阀变形",
    #     "60194-CCD1-负极",
    #     "60194-CCD1-负极高亮定位孔",
    #     "60194-CCD1-负极高亮极柱",
    #     "60194-CCD1-负极高亮铝板",
    #     "60194-CCD1-客户喷码区",
    #     "60194-CCD1-极柱压印",
    #     "60194-CCD1-蓝膜",
    #     "60194-CCD1-蓝膜铝板",
    #     "60194-CCD1-铝板",
    #     "60194-CCD1-铝板压印",
    #     "60194-CCD1-正极",
    #     "60194-CCD1-正极高亮定位孔",
    #     "60194-CCD1-正极高亮极柱",
    #     "60194-CCD1-正极高亮铝板",
    #     "60194-CCD1-注液孔",
    #     "60194-CCD2-3-极柱",
    #     "60194-CCD2-3-铝板",
    #     "60194-CCD4-5-负极",
    #     "60194-CCD4-5-铝板",
    #     "60194-CCD4-5-正极",
    #     "60194-CCD4-5-注液孔",
    #     "60194-CCD6-负极",  
    #     "60194-CCD6-下塑胶",
    #     "60194-CCD6-正极"
    # ]

    # for dir_name in dir_list:
    #     input_folder = str(Path(src_dir) / dir_name)
    #     output_folder = str(Path(save_dir) / dir_name)
    #     print(f"\n处理文件夹: {input_folder}")
    #     process_folder(
    #         input_folder,
    #         output_folder,
    #         min_size=8,
    #         crop_sizes=[96, 160, 224]
    #     )
    
