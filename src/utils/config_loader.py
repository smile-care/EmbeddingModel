"""
配置加载工具
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典
    
    Args:
        *configs: 多个配置字典
        
    Returns:
        合并后的配置字典
    """
    merged = {}
    for config in configs:
        merged.update(config)
    return merged

