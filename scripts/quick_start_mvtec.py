"""
MVTec AD数据集快速开始脚本
一键完成数据转换和准备
"""
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入转换函数
convert_module_path = Path(__file__).parent / "convert_mvtec_ad.py"
spec = importlib.util.spec_from_file_location("convert_mvtec_ad", convert_module_path)
convert_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_module)
convert_mvtec_ad = convert_module.convert_mvtec_ad

from src.data_preprocess.patch_extractor import PatchExtractor
from src.data_preprocess.dataset_builder import DatasetBuilder
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description='MVTec AD数据集快速开始')
    parser.add_argument('--mvtec_root', type=str,
                       default='data/mvtec_anomaly_detection',
                       help='MVTec AD数据集根目录')
    parser.add_argument('--data_config', type=str,
                       default='configs/data_config.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--skip_convert', action='store_true',
                       help='跳过转换步骤（如果已转换）')
    parser.add_argument('--include_good', action='store_true', default=True,
                       help='是否包含正常样本')
    args = parser.parse_args()
    
    logger = setup_logger('quick_start_mvtec')
    
    # 加载配置
    config = load_config(args.data_config)
    data_config = config['data']
    
    # 步骤1: 转换MVTec AD数据集
    if not args.skip_convert:
        logger.info("=" * 60)
        logger.info("步骤1: 转换MVTec AD数据集")
        logger.info("=" * 60)
        convert_mvtec_ad(
            mvtec_root=args.mvtec_root,
            output_root=data_config['raw_data_root'],
            include_good=args.include_good,
            create_empty_masks_for_good=True
        )
    else:
        logger.info("跳过转换步骤（使用 --skip_convert）")
    
    # 步骤2: 提取patch
    logger.info("=" * 60)
    logger.info("步骤2: 提取缺陷patch")
    logger.info("=" * 60)
    
    metadata_root = Path(data_config['metadata_root'])
    metadata_root.mkdir(parents=True, exist_ok=True)
    
    # 检查元数据文件
    raw_metadata = Path(data_config['raw_data_root']) / "metadata.json"
    if not raw_metadata.exists():
        logger.error(f"元数据文件不存在: {raw_metadata}")
        logger.error("请先运行转换步骤（去掉 --skip_convert 参数）")
        return
    
    # 复制元数据到metadata_root
    target_metadata = metadata_root / "metadata.json"
    if not target_metadata.exists():
        shutil.copy2(raw_metadata, target_metadata)
        logger.info(f"已复制元数据到: {target_metadata}")
    
    # 提取patch
    extractor = PatchExtractor(
        patch_size=data_config['patch']['size'],
        expand_ratio=data_config['patch']['expand_ratio'],
        min_size=data_config['patch']['min_size']
    )
    
    patch_metadata_file = extractor.process_metadata(
        str(target_metadata),
        output_dir=data_config['patches_root'],
        image_root=data_config['raw_data_root']
    )
    logger.info(f"Patch元数据已保存: {patch_metadata_file}")
    
    # 步骤3: 构建数据集
    logger.info("=" * 60)
    logger.info("步骤3: 构建训练数据集")
    logger.info("=" * 60)
    
    dataset_builder = DatasetBuilder(data_config['metadata_root'])
    
    # 构建SSL数据集（使用原始metadata，包含所有图像）
    ssl_dataset_file = dataset_builder.build_ssl_dataset(
        str(target_metadata),
        output_file="ssl_dataset.json"
    )
    logger.info(f"SSL数据集已保存: {ssl_dataset_file}")
    
    # 构建SupCon数据集（使用patch metadata）
    patch_metadata_path = Path(patch_metadata_file)
    if patch_metadata_path.exists():
        supcon_dataset_file = dataset_builder.build_supcon_dataset(
            str(patch_metadata_path),
            output_file="supcon_dataset.json"
        )
        logger.info(f"SupCon数据集已保存: {supcon_dataset_file}")
        
        # 过滤数据集
        filtered_file = dataset_builder.filter_dataset(
            str(patch_metadata_path),
            min_samples_per_class=data_config['filtering']['min_samples_per_class'],
            max_samples_per_class=data_config['filtering'].get('max_samples_per_class'),
            output_file="supcon_dataset_filtered.json"
        )
        logger.info(f"过滤后的数据集已保存: {filtered_file}")
    
    logger.info("=" * 60)
    logger.info("数据准备完成！")
    logger.info("=" * 60)
    logger.info("下一步：")
    logger.info("1. 运行自监督预训练: python scripts/train_ssl.py --config configs/ssl_config.yaml")
    logger.info("2. 运行监督对比学习: python scripts/train_supcon.py --config configs/supcon_config.yaml")


if __name__ == '__main__':
    main()

