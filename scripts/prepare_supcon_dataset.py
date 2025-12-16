#!/usr/bin/env python3
"""
准备监督对比学习数据集
从数据目录扫描图片，根据配置文件生成包含相似度信息的JSON文件
"""
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse


def load_similarity_config(config_path: Path) -> Dict:
    """加载相似度配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_category_name(data_root: Path, image_path: Path) -> str:
    """
    根据图片路径确定类别名
    
    规则：
    1. 如果图片在子文件夹中（如 装反/装反1/xxx.png），返回子文件夹名（装反1）
    2. 如果图片直接在顶级文件夹中（如 断焊/xxx.png），返回顶级文件夹名（断焊）
    """
    # 获取相对于数据根目录的路径
    rel_path = image_path.relative_to(data_root)
    parts = rel_path.parts
    
    if len(parts) >= 3:
        # 有子文件夹：返回子文件夹名（如 "装反/装反1/xxx.png" -> "装反1"）
        return parts[1]
    elif len(parts) == 2:
        # 直接在顶级文件夹中：返回顶级文件夹名（如 "断焊/xxx.png" -> "断焊"）
        return parts[0]
    else:
        raise ValueError(f"无法确定类别名: {image_path}")


def find_mask_path(image_path: Path) -> Optional[Path]:
    """查找对应的mask文件"""
    # mask文件名通常是：原文件名_mask.扩展名
    stem = image_path.stem
    suffix = image_path.suffix
    
    # 尝试几种可能的mask文件名格式
    mask_candidates = [
        image_path.parent / f"{stem}_mask{suffix}",
        image_path.parent / f"{stem}_mask.png",
        image_path.parent / f"{stem.replace('_0', '_0_mask')}{suffix}",
    ]
    
    for mask_path in mask_candidates:
        if mask_path.exists():
            return mask_path
    
    return None


def build_similarity_matrix(
    categories: List[str],
    default_similarity: float,
    custom_similarity: List[Dict]
) -> Dict[str, Dict[str, float]]:
    """
    构建类别之间的相似度矩阵
    
    Args:
        categories: 所有类别列表
        default_similarity: 默认相似度
        custom_similarity: 自定义相似度规则列表
    
    Returns:
        相似度矩阵字典，格式：{category1: {category2: similarity, ...}, ...}
    """
    # 初始化相似度矩阵
    similarity_matrix = {}
    for cat1 in categories:
        similarity_matrix[cat1] = {}
        for cat2 in categories:
            if cat1 == cat2:
                similarity_matrix[cat1][cat2] = 1.0  # 同一类别相似度为1
            else:
                similarity_matrix[cat1][cat2] = default_similarity
    
    # 应用自定义相似度规则
    for rule in custom_similarity:
        category_list = rule['list']
        similarity = rule['similarity']
        
        # 对于列表中的每一对类别，设置相似度
        for i, cat1 in enumerate(category_list):
            for cat2 in category_list[i+1:]:
                if cat1 in similarity_matrix and cat2 in similarity_matrix[cat1]:
                    similarity_matrix[cat1][cat2] = similarity
                    similarity_matrix[cat2][cat1] = similarity  # 对称矩阵
    
    return similarity_matrix


def scan_dataset(data_root: Path, image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')) -> List[Dict]:
    """
    扫描数据集目录，收集所有图片信息
    
    Returns:
        图片信息列表，每个元素包含：
        - image_path: 图片路径（绝对路径）
        - mask_path: mask路径（如果有，绝对路径）
        - category: 类别名
        - relative_image_path: 相对路径（用于JSON）
        - relative_mask_path: mask相对路径（如果有）
    """
    instances = []
    data_root = Path(data_root).resolve()
    
    # 扫描所有图片文件（排除mask文件）
    for image_path in data_root.rglob('*'):
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            # 跳过mask文件
            if 'mask' in image_path.name.lower():
                continue
            
            # 确定类别名
            try:
                category = get_category_name(data_root, image_path)
            except ValueError as e:
                print(f"警告: {e}")
                continue
            
            # 查找mask文件
            mask_path = find_mask_path(image_path)
            
            # 构建相对路径
            rel_image_path = str(image_path.relative_to(data_root))
            rel_mask_path = str(mask_path.relative_to(data_root)) if mask_path else None
            
            instances.append({
                'image_path': str(image_path),
                'mask_path': str(mask_path) if mask_path else None,
                'category': category,
                'relative_image_path': rel_image_path,
                'relative_mask_path': rel_mask_path
            })
    
    return instances


def main():
    parser = argparse.ArgumentParser(description='准备监督对比学习数据集JSON文件')
    parser.add_argument(
        '--data_root',
        type=str,
        default='/home/unitx/workspace_custom/EmbeddingModel/data/zhenyu_data/patches/supcon_data_clean',
        help='数据根目录路径'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='相似度配置文件路径（默认：data_root/similarity_config.yaml）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出JSON文件路径（默认：data_root/dataset_metadata.json）'
    )
    parser.add_argument(
        '--use_relative_paths',
        action='store_true',
        help='在JSON中使用相对路径而不是绝对路径'
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise ValueError(f"数据根目录不存在: {data_root}")
    
    # 配置文件路径
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        config_path = data_root / 'similarity_config.yaml'
    
    if not config_path.exists():
        raise ValueError(f"配置文件不存在: {config_path}")
    
    # 输出文件路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = data_root / 'dataset_metadata.json'
    
    print(f"数据根目录: {data_root}")
    print(f"配置文件: {config_path}")
    print(f"输出文件: {output_path}")
    
    # 加载配置文件
    print("\n加载相似度配置...")
    config = load_similarity_config(config_path)
    categories = config.get('categories', [])
    default_similarity = config.get('default_similarity', 0.0)
    custom_similarity = config.get('custom_similarity', [])
    
    print(f"  类别数量: {len(categories)}")
    print(f"  默认相似度: {default_similarity}")
    print(f"  自定义规则数: {len(custom_similarity)}")
    
    # 扫描数据集
    print("\n扫描数据集...")
    instances = scan_dataset(data_root)
    print(f"  找到 {len(instances)} 张图片")
    
    # 统计类别分布
    category_counts = defaultdict(int)
    for inst in instances:
        category_counts[inst['category']] += 1
    
    print(f"\n类别分布:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} 张")
    
    # 检查配置文件中的类别是否都在数据中出现
    config_categories_set = set(categories)
    actual_categories_set = set(category_counts.keys())
    
    missing_in_data = config_categories_set - actual_categories_set
    missing_in_config = actual_categories_set - config_categories_set
    
    if missing_in_data:
        print(f"\n警告: 配置文件中定义但数据中不存在的类别: {missing_in_data}")
    if missing_in_config:
        print(f"\n警告: 数据中存在但配置文件中未定义的类别: {missing_in_config}")
    
    # 构建相似度矩阵
    print("\n构建相似度矩阵...")
    similarity_matrix = build_similarity_matrix(
        categories,
        default_similarity,
        custom_similarity
    )
    
    # 准备输出数据
    output_data = {
        'data_root': str(data_root),
        'total_instances': len(instances),
        'categories': sorted(list(actual_categories_set)),
        'category_counts': dict(category_counts),
        'similarity_config': {
            'default_similarity': default_similarity,
            'custom_similarity_rules': custom_similarity
        },
        'similarity_matrix': similarity_matrix,
        'instances': []
    }
    
    # 处理实例数据
    print("\n处理实例数据...")
    for inst in instances:
        instance_data = {
            'category': inst['category']
        }
        
        if args.use_relative_paths:
            instance_data['image_path'] = inst['relative_image_path']
            if inst['relative_mask_path']:
                instance_data['mask_path'] = inst['relative_mask_path']
        else:
            instance_data['image_path'] = inst['image_path']
            if inst['mask_path']:
                instance_data['mask_path'] = inst['mask_path']
        
        output_data['instances'].append(instance_data)
    
    # 保存JSON文件
    print(f"\n保存JSON文件到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！")
    print(f"  总实例数: {len(instances)}")
    print(f"  类别数: {len(actual_categories_set)}")
    print(f"  有mask的图片: {sum(1 for inst in instances if inst['mask_path'])}")
    print(f"  输出文件: {output_path}")


if __name__ == '__main__':
    main()
