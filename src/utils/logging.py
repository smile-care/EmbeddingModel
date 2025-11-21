"""
日志工具
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录
        level: 日志级别
        log_to_file: 是否写入文件
        log_to_console: 是否输出到控制台
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # 避免重复添加handler
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

