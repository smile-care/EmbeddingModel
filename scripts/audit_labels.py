"""
执行标签体检
"""
import argparse
import sys
import numpy as np
import json
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.label_audit.report_generator import ReportGenerator
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description='标签体检')
    parser.add_argument('--embeddings', type=str, default="data/mvtec_ad/embeddings/supcon_dataset.npz",
                       help='Embedding文件路径（.npy或.npz）')
    parser.add_argument('--metadata', type=str, default="data/mvtec_ad/metadata/supcon_dataset.json",
                       help='patch元数据JSON文件路径')
    parser.add_argument('--output_dir', type=str, default="data/mvtec_ad/reports/supcon_dataset",
                       help='输出目录')
    parser.add_argument('--top_outliers_percent', type=float, default=5.0,
                       help='离群点百分比')
    parser.add_argument('--top_similar_pairs', type=int, default=20,
                       help='最相似类别对数量')
    args = parser.parse_args()
    
    logger = setup_logger('audit_labels')
    
    # 加载embedding
    logger.info("加载embedding...")
    embeddings_file = Path(args.embeddings)
    if embeddings_file.suffix == '.npz':
        data = np.load(args.embeddings)
        embeddings = data['embeddings']
        instance_ids = data['instance_ids']
    else:
        embeddings_dict = np.load(args.embeddings, allow_pickle=True).item()
        instance_ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[iid] for iid in instance_ids])
    
    # 加载元数据
    logger.info("加载元数据...")
    with open(args.metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 构建标签和ID映射
    instance_id_to_info = {p['instance_id']: p for p in metadata['patches']}
    labels = []
    valid_instance_ids = []
    valid_indices = []
    
    for i, instance_id in enumerate(instance_ids):
        if instance_id in instance_id_to_info:
            labels.append(instance_id_to_info[instance_id]['label'])
            valid_instance_ids.append(instance_id)
            valid_indices.append(i)
    
    embeddings = embeddings[valid_indices]
    labels = np.array(labels)
    instance_ids = np.array(valid_instance_ids)
    
    logger.info(f"有效样本数: {len(embeddings)}")
    logger.info(f"类别数: {len(np.unique(labels))}")
    
    # 生成报告
    logger.info("生成标签体检报告...")
    report_generator = ReportGenerator(
        embeddings=embeddings,
        labels=labels,
        instance_ids=instance_ids,
        metadata=metadata
    )
    
    report_path = report_generator.generate_report(
        output_dir=args.output_dir,
        top_outliers_percent=args.top_outliers_percent,
        top_similar_pairs=args.top_similar_pairs
    )
    
    logger.info(f"报告已生成: {report_path}")


if __name__ == '__main__':
    main()

