"""
数据准备脚本
"""
import argparse
import shutil
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocess.dataset_builder import DatasetBuilder
from src.data_preprocess.metadata_builder import MetadataBuilder
from src.data_preprocess.patch_extractor import PatchExtractor
from src.utils.config_loader import load_config
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description='数据准备脚本')
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--step', type=str, choices=['metadata', 'patch', 'dataset', 'all'],
                       default='dataset', help='执行步骤')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.data_config)
    data_config = config['data']
    
    # 设置日志
    logger = setup_logger('prepare_data')
    
    metadata_root = Path(data_config['metadata_root'])
    metadata_root.mkdir(parents=True, exist_ok=True)
    
    # 步骤1: 构建元数据
    if args.step in ['metadata', 'all']:
        logger.info("步骤1: 构建元数据...")
        
        raw_data_root = Path(data_config['raw_data_root'])
        metadata_root = Path(data_config['metadata_root'])
        
        # 检查是否已有转换后的metadata.json（来自convert_mvtec_ad）
        converted_metadata = raw_data_root / "metadata.json"
        if converted_metadata.exists():
            logger.info(f"检测到已转换的元数据文件: {converted_metadata}")
            logger.info("直接使用转换后的元数据，跳过重新构建...")
            # 复制到metadata_root
            metadata_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(converted_metadata, metadata_root / "metadata.json")
            metadata_file = str(metadata_root / "metadata.json")
        else:
            # 使用MetadataBuilder构建元数据
            builder = MetadataBuilder(
                raw_data_root=data_config['raw_data_root'],
                metadata_root=data_config['metadata_root']
            )
            
            # 自动检测数据目录结构
            images_dir = raw_data_root / "images"
            masks_dir = raw_data_root / "masks"
            
            if images_dir.exists() and masks_dir.exists():
                # 检测所有domain（类别）
                domains = []
                for domain_dir in images_dir.iterdir():
                    if domain_dir.is_dir():
                        domain_id = domain_dir.name
                        domain_images = domain_dir
                        domain_masks = masks_dir / domain_id
                        
                        if domain_masks.exists():
                            domains.append({
                                'domain_id': domain_id,
                                'image_dir': str(domain_images),
                                'mask_dir': str(domain_masks),
                                'label_mapping': {}  # 从目录结构自动提取label
                            })
                            logger.info(f"  检测到domain: {domain_id}")
                
                if len(domains) > 0:
                    metadata_file = builder.build_metadata(
                        domains,
                        output_file="metadata.json"
                    )
                    logger.info(f"元数据已保存: {metadata_file}")
                else:
                    logger.error("未找到任何domain数据，请先运行 convert_mvtec_ad.py 转换数据")
                    return
            else:
                logger.error(f"数据目录结构不正确。期望: {images_dir} 和 {masks_dir}")
                logger.info("请先运行: python scripts/convert_mvtec_ad.py --mvtec_root data/mvtec_anomaly_detection --output_root data/raw")
                return
    
    # 步骤2: 提取patch
    if args.step in ['patch', 'all']:
        logger.info("步骤2: 提取patch...")
        extractor = PatchExtractor(
            patch_size=data_config['patch']['size'],
            expand_ratio=data_config['patch']['expand_ratio'],
            min_size=data_config['patch']['min_size']
        )
        
        metadata_file = metadata_root / "metadata.json"
        if not metadata_file.exists():
            logger.error(f"元数据文件不存在: {metadata_file}")
            return
        
        patch_metadata_file = extractor.process_metadata(
            str(metadata_file),
            output_dir=data_config['patches_root'],
            # image_root=data_config['raw_data_root']
        )
        logger.info(f"Patch元数据已保存: {patch_metadata_file}")
    
    # 步骤3: 构建数据集
    if args.step in ['dataset', 'all']:
        logger.info("步骤3: 构建数据集...")
        dataset_builder = DatasetBuilder(data_config['metadata_root'])
        
        # 构建SSL数据集
        metadata_file = metadata_root / "metadata.json"
        if metadata_file.exists():
            ssl_dataset_file = dataset_builder.build_ssl_dataset(
                str(metadata_file),
                output_file="ssl_dataset.json"
            )
            logger.info(f"SSL数据集已保存: {ssl_dataset_file}")
        
        # 构建SupCon数据集
        patch_metadata_file = metadata_root / "patch_metadata.json"
        if patch_metadata_file.exists():
            supcon_dataset_file = dataset_builder.build_supcon_dataset(
                str(patch_metadata_file),
                output_file="supcon_dataset.json"
            )
            logger.info(f"SupCon数据集已保存: {supcon_dataset_file}")
            
            # 过滤数据集
            filtered_file = dataset_builder.filter_dataset(
                str(patch_metadata_file),
                min_samples_per_class=data_config['filtering']['min_samples_per_class'],
                max_samples_per_class=data_config['filtering'].get('max_samples_per_class'),
                output_file="supcon_dataset_filtered.json"
            )
            logger.info(f"过滤后的数据集已保存: {filtered_file}")
    
    logger.info("数据准备完成！")


if __name__ == '__main__':
    main()

