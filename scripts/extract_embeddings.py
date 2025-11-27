"""
提取所有embedding
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_tools.extractor import EmbeddingExtractor, MAEFeatureExtractor
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description='提取embedding')
    parser.add_argument('--model', type=str, default="checkpoints/supcon_trainval-3/best_model.pth",
                       help='模型checkpoint路径')
    parser.add_argument('--metadata', type=str, default="data/mvtec_ad/metadata/supcon_dataset_val.json",
                       help='patch元数据JSON文件路径')
    parser.add_argument('--output', type=str, default="data/mvtec_ad/embeddings/supcon_dataset_val-3.npz",
                       help='输出文件路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='batch大小')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（可选）')
    args = parser.parse_args()
    
    logger = setup_logger('extract_embeddings')
    
    logger.info("加载模型...")
    extractor = EmbeddingExtractor(
        model_path=args.model,
        config_path=args.config
    )
    
    logger.info("提取embedding...")
    embeddings_dict = extractor.extract_batch(
        metadata_file=args.metadata,
        output_file=args.output,
        batch_size=args.batch_size
    )
    
    logger.info(f"已提取 {len(embeddings_dict)} 个embedding")
    logger.info(f"已保存到: {args.output}")


if __name__ == '__main__':
    main()

