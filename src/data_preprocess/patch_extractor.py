"""
Patch提取器：从mask提取缺陷patch
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import json
from tqdm import tqdm


class PatchExtractor:
    """从mask提取缺陷patch"""
    
    def __init__(
        self,
        patch_size: int = 224,
        expand_ratio: float = 0.1,
        min_size: int = 8
    ):
        """
        初始化Patch提取器
        
        Args:
            patch_size: 输出patch大小（正方形）
            expand_ratio: 扩边比例（相对于mask外接矩形）
            min_size: 最小patch尺寸
        """
        self.patch_size = patch_size
        self.expand_ratio = expand_ratio
        self.min_size = min_size
    
    def extract_bbox_from_mask(
        self,
        mask: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
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
        # 确保返回原生Python int，避免numpy.int64后续JSON序列化报错
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        扩展bbox（扩边）
        
        Args:
            bbox: (x_min, y_min, x_max, y_max)
            image_shape: (height, width)
            
        Returns:
            扩展后的bbox
        """
        x_min, y_min, x_max, y_max = bbox
        img_h, img_w = image_shape
        
        # 计算宽度和高度
        w = x_max - x_min
        h = y_max - y_min
        
        # 扩展
        expand_w = int(w * self.expand_ratio)
        expand_h = int(h * self.expand_ratio)
        
        x_min = max(0, x_min - expand_w)
        y_min = max(0, y_min - expand_h)
        x_max = min(img_w, x_max + expand_w)
        y_max = min(img_h, y_max + expand_h)
        
        # 同样强制转原生int
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def extract_patch(
        self,
        image_path: str,
        mask_path: str
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """
        从图像和mask提取patch
        
        Args:
            image_path: 图像路径
            mask_path: mask路径
            
        Returns:
            (patch图像, 元数据字典) 或 None
        """
        # 加载图像
        image = cv2.imread(str(image_path))
        if image is None:
            assert False, f"Failed to load image: {image_path}"
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            assert False, f"Failed to load mask: {mask_path}"
            return None
        
        # 确保mask和图像尺寸一致
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # 提取bbox
        bbox = self.extract_bbox_from_mask(mask)
        if bbox is None:
            return None
        
        # 扩展bbox
        bbox = self.expand_bbox(bbox, image.shape[:2])
        x_min, y_min, x_max, y_max = bbox
        
        # 检查最小尺寸
        if (x_max - x_min) < self.min_size or (y_max - y_min) < self.min_size:
            assert False, f"Patch size below minimum: {(x_max - x_min)}x{(y_max - y_min)}"
            return None
        
        # 裁剪patch
        patch = image[y_min:y_max, x_min:x_max]
        patch_mask = mask[y_min:y_max, x_min:x_max]
        
        # Resize到目标尺寸
        patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        patch_mask = cv2.resize(patch_mask, (self.patch_size, self.patch_size))
        
        # 构建元数据
        metadata = {
            'original_bbox': bbox,
            'original_size': image.shape[:2],
            'patch_size': self.patch_size
        }
        
        return patch, patch_mask, metadata
    
    def process_metadata(
        self,
        metadata_file: str,
        output_dir: str,
        image_root: Optional[str] = None,
        mask_root: Optional[str] = None
    ) -> str:
        """
        批量处理元数据中的所有实例
        
        Args:
            metadata_file: 元数据JSON文件路径
            output_dir: patch输出目录
            image_root: 图像根目录（如果元数据中是相对路径）
            mask_root: mask根目录（如果元数据中是相对路径）
            
        Returns:
            处理后的元数据文件路径
        """
        # 加载元数据
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建patch元数据
        patch_metadata = {
            'total_patches': 0,
            'patches': []
        }

        def _to_serializable(obj):
            """递归转换对象中numpy类型为原生Python类型，确保json.dump不报错"""
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_serializable(v) for v in obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        failed_count = 0
        
        print("提取patch...")
        for instance in tqdm(metadata['instances']):
            # 构建完整路径
            image_path = instance['image_path']
            mask_path = instance['mask_path']
            
            if image_root and not Path(image_path).is_absolute():
                image_path = Path(image_root) / image_path
            if mask_root and not Path(mask_path).is_absolute():
                mask_path = Path(mask_root) / mask_path
            
            # 提取patch
            result = self.extract_patch(str(image_path), str(mask_path))
            
            if result is None:
                failed_count += 1
                continue
            
            patch, patch_mask, patch_info = result
            
            # 保存patch图像
            patch_filename = f"{instance['instance_id']}.png"
            patch_mask_filename = f"{instance['instance_id']}_mask.png"
            patch_path = output_dir / "images" / patch_filename
            patch_mask_path = output_dir / "masks" / patch_mask_filename
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(patch).save(patch_path)
            Image.fromarray(patch_mask).save(patch_mask_path)
            
            # 添加到patch元数据
            # bbox中可能含有numpy.int64，这里显式转换
            bbox_py = [int(x) for x in patch_info['original_bbox']]
            patch_metadata['patches'].append({
                'instance_id': instance['instance_id'],
                'patch_path': str(patch_path),
                'patch_mask_path': str(patch_mask_path),
                'label': instance['label'],
                'domain_id': instance['domain_id'],
                'original_image_path': instance['image_path'],
                'original_mask_path': instance['mask_path'],
                'bbox': bbox_py,
                'original_size': [int(s) for s in patch_info['original_size']]
            })
        
        patch_metadata['total_patches'] = len(patch_metadata['patches'])
        
        # 保存patch元数据
        output_metadata_file = output_dir / "patch_metadata.json"
        # 序列化前做一次递归转换
        patch_metadata_serializable = _to_serializable(patch_metadata)
        with open(output_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(patch_metadata_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\n处理完成:")
        print(f"  成功: {patch_metadata['total_patches']} 个patch")
        print(f"  失败: {failed_count} 个实例")
        print(f"  Patch元数据: {output_metadata_file}")
        
        return str(output_metadata_file)

