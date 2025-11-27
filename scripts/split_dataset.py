"""
数据集拆分工具
将数据集文件拆分为训练集和验证集
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import setup_logger


def stratified_split(
    patches: List[Dict],
    train_ratio: float,
    seed: int = 42,
    logger=None
) -> Tuple[List[Dict], List[Dict]]:
    """
    按类别分层划分数据集
    
    Args:
        patches: 所有patch的列表
        train_ratio: 训练集比例 (0-1)
        seed: 随机种子
        logger: 日志记录器（可选）
        
    Returns:
        (train_patches, val_patches)
    """
    random.seed(seed)
    
    # 按类别分组
    patches_by_label = defaultdict(list)
    for patch in patches:
        label = patch.get('label', 'unknown')
        patches_by_label[label].append(patch)
    
    train_patches = []
    val_patches = []
    
    # 限制：训练集至少3个（超过2），验证集至少4个
    min_train = 3  # 超过2，即至少3个
    min_val = 4
    
    # 对每个类别分别划分
    for label, label_patches in patches_by_label.items():
        random.shuffle(label_patches)
        n_total = len(label_patches)
        n_train = int(n_total * train_ratio)
        
        # 先保证训练集超过2个（至少3个）
        if n_train < min_train:
            n_train = min_train
        
        # 在保证训练集超过2后，至少保证验证集有4个
        if n_total - n_train < min_val:
            # 如果验证集不足4个，调整训练集大小
            n_train = n_total - min_val
            # 如果调整后训练集不足3个，说明样本数不足7个，无法同时满足两个条件
            if n_train < min_train:
                # 优先保证训练集至少3个，验证集有多少算多少
                n_train = min_train
                n_val = n_total - n_train
                if logger:
                    logger.warning(
                        f"类别 '{label}' 样本数不足 ({n_total}个)，无法同时满足"
                        f"训练集>2和验证集>=4的要求。"
                        f"实际划分: 训练集={n_train}, 验证集={n_val}"
                    )
            else:
                # 调整成功，可以同时满足两个条件
                pass
        
        train_patches.extend(label_patches[:n_train])
        val_patches.extend(label_patches[n_train:])
    
    # 打乱顺序
    random.shuffle(train_patches)
    random.shuffle(val_patches)
    
    return train_patches, val_patches


def random_split(
    patches: List[Dict],
    train_ratio: float,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    随机划分数据集
    
    Args:
        patches: 所有patch的列表
        train_ratio: 训练集比例 (0-1)
        seed: 随机种子
        
    Returns:
        (train_patches, val_patches)
    """
    random.seed(seed)
    
    shuffled_patches = patches.copy()
    random.shuffle(shuffled_patches)
    
    n_train = int(len(shuffled_patches) * train_ratio)
    train_patches = shuffled_patches[:n_train]
    val_patches = shuffled_patches[n_train:]
    
    return train_patches, val_patches


def split_dataset(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    split_method: str = 'stratified',
    seed: int = 42
):
    """
    拆分数据集
    
    Args:
        input_file: 输入JSON文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例 (0-1)
        split_method: 划分方法 ('stratified' 或 'random')
        seed: 随机种子
    """
    logger = setup_logger('split_dataset')
    
    # 加载数据
    logger.info(f"加载数据集: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    patches = data.get('patches', [])
    logger.info(f"总样本数: {len(patches)}")
    
    # 统计类别分布
    label_counts = defaultdict(int)
    for patch in patches:
        label = patch.get('label', 'unknown')
        label_counts[label] += 1
    
    logger.info(f"类别数: {len(label_counts)}")
    logger.info(f"划分方法: {split_method}")
    logger.info(f"训练集比例: {train_ratio:.2%}")
    
    # 划分数据集
    if split_method == 'stratified':
        train_patches, val_patches = stratified_split(patches, train_ratio, seed, logger)
    elif split_method == 'random':
        train_patches, val_patches = random_split(patches, train_ratio, seed)
    else:
        raise ValueError(f"不支持的划分方法: {split_method}，请选择 'stratified' 或 'random'")
    
    logger.info(f"训练集样本数: {len(train_patches)}")
    logger.info(f"验证集样本数: {len(val_patches)}")
    
    # 统计划分后的类别分布
    train_label_counts = defaultdict(int)
    val_label_counts = defaultdict(int)
    
    for patch in train_patches:
        label = patch.get('label', 'unknown')
        train_label_counts[label] += 1
    
    for patch in val_patches:
        label = patch.get('label', 'unknown')
        val_label_counts[label] += 1
    
    logger.info("训练集类别分布（前10个）:")
    for label, count in sorted(train_label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {label}: {count}")
    
    logger.info("验证集类别分布（前10个）:")
    for label, count in sorted(val_label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {label}: {count}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建输出数据
    input_path = Path(input_file)
    base_name = input_path.stem
    
    # 训练集
    train_data = {
        'total_patches': len(train_patches),
        'domains': data.get('domains', []),
        'labels': data.get('labels', []),
        'label_counts': {label: train_label_counts[label] for label in train_label_counts},
        'patches': train_patches
    }
    
    # 验证集
    val_data = {
        'total_patches': len(val_patches),
        'domains': data.get('domains', []),
        'labels': data.get('labels', []),
        'label_counts': {label: val_label_counts[label] for label in val_label_counts},
        'patches': val_patches
    }
    
    # 保存文件
    train_file = output_dir / f"{base_name}_train.json"
    val_file = output_dir / f"{base_name}_val.json"
    
    logger.info(f"保存训练集: {train_file}")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"保存验证集: {val_file}")
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    logger.info("数据集拆分完成！")
    logger.info(f"训练集: {train_file}")
    logger.info(f"验证集: {val_file}")


def main():
    parser = argparse.ArgumentParser(description='数据集拆分工具')
    parser.add_argument(
        '--input',
        type=str,
        default='data/mvtec_ad/metadata/supcon_dataset.json',
        help='输入数据集JSON文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/mvtec_ad/metadata',
        help='输出目录'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='训练集比例 (0-1)，默认0.8'
    )
    parser.add_argument(
        '--split_method',
        type=str,
        default='stratified',
        choices=['stratified', 'random'],
        help='划分方法: stratified(按类别分层) 或 random(随机划分)，默认stratified'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子，默认42'
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not 0 < args.train_ratio < 1:
        raise ValueError(f"train_ratio 必须在 0 和 1 之间，当前值: {args.train_ratio}")
    
    split_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        split_method=args.split_method,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

