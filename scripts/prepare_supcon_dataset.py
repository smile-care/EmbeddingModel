#!/usr/bin/env python3
"""
准备监督对比学习数据集
"""
import os
from pathlib import Path
from typing import Dict

import numpy as np
import yaml
from tabulate import tabulate


def load_similarity_config(config_path: Path) -> Dict:
    """加载相似度配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class ZhenyuData:
    """监督对比学习数据集准备类"""
    
    def __init__(
        self, 
        config_path: Path = Path('configs/data_config_zhenyu.yaml'),
    ):
        """
        初始化数据集准备器
        
        Args:
            config_path: data配置文件路径
        """
        self.config = load_similarity_config(config_path)
        
        self.root = Path(self.config['root'])
        self.categories = self.config['categories']
        self.default_similarity = self.config.get('default_similarity', 0.0)
        self.custom_similarity = self.config.get('custom_similarity', [])
        
        self.check_data()
        
        self.cat2idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}
        
        # 根据配置生成相似度矩阵
        self.similarity_matrix = self.build_similarity_matrix()
    
    def check_data(self):
        """检查数据完整性"""
        if not self.root.exists():
            raise FileNotFoundError(f"数据根目录不存在: {self.root}")
        
        image_count = 0
        for root, dirs, files in self.root.walk():
            for file in files:
                if file.lower().endswith('.png') and "_mask.png" not in file:
                    file_path = Path(root) / file
                    mask_path = file_path.with_name(file_path.stem + "_mask.png")
                    if not mask_path.exists():
                        raise FileNotFoundError(f"缺失掩码文件: {mask_path}")
                    
                    label = file_path.parent.name
                    if label not in self.categories:
                        raise ValueError(f"未知类别: {label}")
                    
                    image_count += 1
        
        print(f"数据检查完成: 共找到 {image_count} 张图片")
    
    def build_similarity_matrix(self) -> np.ndarray:
        """构建类别相似度矩阵"""
        num_categories = len(self.categories)
        similarity_matrix = np.full((num_categories, num_categories), self.default_similarity, dtype=float)
        
        # 设置对角线为1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # 应用自定义相似度
        for item in self.custom_similarity:
            categories_list = item['list']
            sim = item['similarity']
            # 为列表中的所有类别对设置相似度
            for i, cat1 in enumerate(categories_list):
                for cat2 in categories_list[i+1:]:
                    if cat1 not in self.cat2idx or cat2 not in self.cat2idx:
                        print(f"警告: 跳过未知类别 {cat1} 或 {cat2}")
                        continue
                    idx1 = self.cat2idx[cat1]
                    idx2 = self.cat2idx[cat2]
                    similarity_matrix[idx1, idx2] = sim
                    similarity_matrix[idx2, idx1] = sim  # 对称矩阵
        
        return similarity_matrix
        

if __name__ == '__main__':
    data = ZhenyuData()
    print(f"\n类别数量: {len(data.categories)}")
    print(f"相似度矩阵形状: {data.similarity_matrix.shape}")
    
    # 以表格形式显示相似度矩阵 (前10个类别)
    n_show = min(10, len(data.categories))
    print(f"\n相似度矩阵 (前{n_show}个类别):")
    
    # 准备表格数据
    headers = [data.idx2cat[i] for i in range(n_show)]
    table_data = []
    for i in range(n_show):
        row = [data.idx2cat[i]] + [f"{data.similarity_matrix[i, j]:.2f}" for j in range(n_show)]
        table_data.append(row)
    
    print(tabulate(table_data, headers=['类别'] + headers, tablefmt='grid'))
