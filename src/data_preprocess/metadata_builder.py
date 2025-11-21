"""
元数据构建工具
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import hashlib


class MetadataBuilder:
    """构建和管理缺陷实例的元数据"""
    
    def __init__(self, raw_data_root: str, metadata_root: str):
        """
        初始化元数据构建器
        
        Args:
            raw_data_root: 原始数据根目录
            metadata_root: 元数据保存目录
        """
        self.raw_data_root = Path(raw_data_root)
        self.metadata_root = Path(metadata_root)
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        
    def generate_instance_id(self, image_path: str, mask_path: str) -> str:
        """
        生成实例ID（基于文件路径的哈希）
        
        Args:
            image_path: 图像路径
            mask_path: mask路径
            
        Returns:
            实例ID
        """
        combined = f"{image_path}_{mask_path}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def scan_directory(
        self,
        domain_id: str,
        image_dir: str,
        mask_dir: str,
        label_mapping: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        扫描目录，构建实例元数据列表
        
        Args:
            domain_id: 场景ID
            image_dir: 图像目录
            mask_dir: mask目录
            label_mapping: 标签映射（文件名模式 -> label）
            
        Returns:
            实例元数据列表
        """
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        
        if not image_dir.exists():
            raise ValueError(f"图像目录不存在: {image_dir}")
        if not mask_dir.exists():
            raise ValueError(f"Mask目录不存在: {mask_dir}")
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        mask_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        instances = []
        
        # 扫描图像文件
        image_files = {}
        for img_file in image_dir.rglob('*'):
            if img_file.suffix.lower() in image_extensions:
                # 尝试匹配对应的mask文件
                relative_path = img_file.relative_to(image_dir)
                mask_file = mask_dir / relative_path
                
                # 如果mask文件不存在，尝试不同的扩展名
                if not mask_file.exists():
                    for ext in mask_extensions:
                        mask_file = mask_dir / (relative_path.stem + ext)
                        if mask_file.exists():
                            break
                    else:
                        continue  # 找不到对应的mask，跳过
                
                # 确定label
                label = self._extract_label(img_file, label_mapping)
                
                instance = {
                    'instance_id': self.generate_instance_id(
                        str(img_file), str(mask_file)
                    ),
                    'image_path': str(img_file),
                    'mask_path': str(mask_file),
                    'label': label,
                    'domain_id': domain_id,
                    'camera_id': None,  # 可从文件名或目录结构提取
                    'timestamp': None,  # 可从文件名或EXIF提取
                }
                instances.append(instance)
        
        return instances
    
    def _extract_label(
        self,
        file_path: Path,
        label_mapping: Optional[Dict[str, str]] = None
    ) -> str:
        """
        从文件路径提取标签
        
        Args:
            file_path: 文件路径
            label_mapping: 标签映射规则
            
        Returns:
            标签名称
        """
        if label_mapping:
            # 根据映射规则提取标签
            for pattern, label in label_mapping.items():
                if pattern in str(file_path):
                    return label
        
        # 默认：从目录结构提取（例如：data/raw/PCB/scratches/image.jpg -> scratches）
        parts = file_path.parts
        if len(parts) >= 2:
            return parts[-2]  # 父目录名作为label
        
        return "unknown"
    
    def build_metadata(
        self,
        domain_configs: List[Dict],
        output_file: str = "metadata.json"
    ) -> str:
        """
        构建完整的元数据文件
        
        Args:
            domain_configs: 每个domain的配置列表
                [{
                    'domain_id': 'PCB',
                    'image_dir': 'data/raw/PCB/images',
                    'mask_dir': 'data/raw/PCB/masks',
                    'label_mapping': {...}  # 可选
                }, ...]
            output_file: 输出文件名
            
        Returns:
            元数据文件路径
        """
        all_instances = []
        
        print("扫描数据目录...")
        for config in tqdm(domain_configs):
            domain_id = config['domain_id']
            instances = self.scan_directory(
                domain_id=domain_id,
                image_dir=config['image_dir'],
                mask_dir=config['mask_dir'],
                label_mapping=config.get('label_mapping')
            )
            all_instances.extend(instances)
            print(f"  {domain_id}: 找到 {len(instances)} 个实例")
        
        metadata = {
            'total_instances': len(all_instances),
            'domains': list(set(inst['domain_id'] for inst in all_instances)),
            'labels': list(set(inst['label'] for inst in all_instances)),
            'instances': all_instances
        }
        
        output_path = self.metadata_root / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n元数据已保存到: {output_path}")
        print(f"总计: {len(all_instances)} 个实例")
        print(f"场景数: {len(metadata['domains'])}")
        print(f"类别数: {len(metadata['labels'])}")
        
        return str(output_path)
    
    def load_metadata(self, metadata_file: str = "metadata.json") -> Dict:
        """
        加载元数据文件
        
        Args:
            metadata_file: 元数据文件名
            
        Returns:
            元数据字典
        """
        metadata_path = self.metadata_root / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return metadata
    
    def update_metadata(
        self,
        new_instances: List[Dict],
        metadata_file: str = "metadata.json"
    ):
        """
        增量更新元数据
        
        Args:
            new_instances: 新实例列表
            metadata_file: 元数据文件名
        """
        metadata_path = self.metadata_root / metadata_file
        
        if metadata_path.exists():
            metadata = self.load_metadata(metadata_file)
            existing_ids = {inst['instance_id'] for inst in metadata['instances']}
        else:
            metadata = {
                'total_instances': 0,
                'domains': [],
                'labels': [],
                'instances': []
            }
            existing_ids = set()
        
        # 添加新实例（去重）
        added_count = 0
        for inst in new_instances:
            if inst['instance_id'] not in existing_ids:
                metadata['instances'].append(inst)
                existing_ids.add(inst['instance_id'])
                added_count += 1
        
        # 更新统计信息
        metadata['total_instances'] = len(metadata['instances'])
        metadata['domains'] = sorted(list(set(inst['domain_id'] for inst in metadata['instances'])))
        metadata['labels'] = sorted(list(set(inst['label'] for inst in metadata['instances'])))
        
        # 保存
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"已添加 {added_count} 个新实例，总计 {metadata['total_instances']} 个实例")

