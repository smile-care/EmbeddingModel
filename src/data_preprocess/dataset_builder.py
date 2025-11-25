"""
数据集构建工具
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np


class DatasetBuilder:
    """构建训练数据集"""
    
    def __init__(self, metadata_root: str):
        """
        初始化数据集构建器
        
        Args:
            metadata_root: 元数据根目录
        """
        self.metadata_root = Path(metadata_root)
    
    def build_ssl_dataset(
        self,
        metadata_file: str,
        output_file: str = "ssl_dataset.json"
    ) -> str:
        """
        构建自监督预训练数据集（所有原始图像）
        
        Args:
            metadata_file: 实例元数据文件
            output_file: 输出文件名
            
        Returns:
            输出文件路径
        """
        try:
            metadata_path = self.metadata_root / metadata_file
            assert metadata_path.exists(), f"元数据文件不存在: {metadata_path}"
        except:
            metadata_path = Path(metadata_file)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 收集所有唯一的图像路径（包括正常样本和缺陷样本）
        image_paths = set()
        domain_images = defaultdict(list)
        
        for instance in metadata['instances']:
            image_path = instance['image_path']
            domain_id = instance['domain_id']
            
            if image_path not in image_paths:
                image_paths.add(image_path)
                domain_images[domain_id].append({
                    'image_path': image_path,
                    'domain_id': domain_id
                })
        
        ssl_dataset = {
            'total_images': len(image_paths),
            'domains': list(domain_images.keys()),
            'domain_counts': {d: len(images) for d, images in domain_images.items()},
            'images': []
        }
        
        # 展平图像列表
        for images in domain_images.values():
            ssl_dataset['images'].extend(images)
        
        # 保存
        output_path = self.metadata_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ssl_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"自监督数据集已保存: {output_path}")
        print(f"  总图像数: {ssl_dataset['total_images']}")
        print(f"  场景数: {len(ssl_dataset['domains'])}")
        for domain, count in ssl_dataset['domain_counts'].items():
            print(f"    {domain}: {count} 张图像")
        
        return str(output_path)
    
    def build_supcon_dataset(
        self,
        patch_metadata_file: str,
        output_file: str = "supcon_dataset.json"
    ) -> str:
        """
        构建监督对比学习数据集（patch + label）
        
        Args:
            patch_metadata_file: patch元数据文件
            output_file: 输出文件名
            
        Returns:
            输出文件路径
        """
        metadata_path = self.metadata_root / patch_metadata_file
        with open(metadata_path, 'r', encoding='utf-8') as f:
            patch_metadata = json.load(f)
        
        # 按domain和label组织
        domain_label_instances = defaultdict(lambda: defaultdict(list))
        
        for patch in patch_metadata['patches']:
            domain_id = patch['domain_id']
            label = patch['label']
            domain_label_instances[domain_id][label].append(patch)
        
        supcon_dataset = {
            'total_patches': len(patch_metadata['patches']),
            'domains': list(domain_label_instances.keys()),
            'labels': sorted(list(set(p['label'] for p in patch_metadata['patches']))),
            'domain_label_counts': {},
            'patches': patch_metadata['patches']
        }
        
        # 统计每个domain-label组合的样本数
        for domain_id, label_dict in domain_label_instances.items():
            supcon_dataset['domain_label_counts'][domain_id] = {
                label: len(instances)
                for label, instances in label_dict.items()
            }
        
        # 保存
        output_path = self.metadata_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(supcon_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"监督对比学习数据集已保存: {output_path}")
        print(f"  总patch数: {supcon_dataset['total_patches']}")
        print(f"  场景数: {len(supcon_dataset['domains'])}")
        print(f"  类别数: {len(supcon_dataset['labels'])}")
        
        return str(output_path)
    
    def filter_dataset(
        self,
        dataset_file: str,
        min_samples_per_class: int = 5,
        max_samples_per_class: Optional[int] = None,
        output_file: Optional[str] = None
    ) -> str:
        """
        过滤数据集（移除样本数过少/过多的类别）
        
        Args:
            dataset_file: 数据集文件
            min_samples_per_class: 每个类别最少样本数
            max_samples_per_class: 每个类别最多样本数（None表示不限制）
            output_file: 输出文件名（None表示覆盖原文件）
            
        Returns:
            输出文件路径
        """
        metadata_path = self.metadata_root / dataset_file
        with open(metadata_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 统计每个类别的样本数
        label_counts = defaultdict(int)
        for patch in dataset['patches']:
            label_counts[patch['label']] += 1
        
        # 过滤
        valid_labels = set()
        for label, count in label_counts.items():
            if count >= min_samples_per_class:
                if max_samples_per_class is None or count <= max_samples_per_class:
                    valid_labels.add(label)
        
        # 过滤patches
        filtered_patches = [
            p for p in dataset['patches']
            if p['label'] in valid_labels
        ]
        
        # 如果指定了max_samples_per_class，进行下采样
        if max_samples_per_class:
            label_patches = defaultdict(list)
            for patch in filtered_patches:
                label_patches[patch['label']].append(patch)
            
            filtered_patches = []
            for label, patches in label_patches.items():
                if len(patches) > max_samples_per_class:
                    # 随机采样
                    indices = np.random.choice(
                        len(patches),
                        max_samples_per_class,
                        replace=False
                    )
                    filtered_patches.extend([patches[i] for i in indices])
                else:
                    filtered_patches.extend(patches)
        
        dataset['patches'] = filtered_patches
        dataset['total_patches'] = len(filtered_patches)
        dataset['labels'] = sorted(list(valid_labels))
        
        # 保存
        if output_file is None:
            output_file = dataset_file
        
        output_path = self.metadata_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"过滤后的数据集已保存: {output_path}")
        print(f"  原始样本数: {len(dataset.get('patches', []))}")
        print(f"  过滤后样本数: {dataset['total_patches']}")
        print(f"  保留类别数: {len(valid_labels)}")
        
        return str(output_path)

