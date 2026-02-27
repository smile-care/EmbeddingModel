"""
模块4: 图像可视化
"""
from typing import Dict, Optional, Tuple

import numpy as np
from visualization import visualize_image_with_mask


def visualize_image_module(
    outlier_index: int,
    show_mask: bool,
    app_state,
    state: Dict
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Dict]:
    """图像可视化模块"""
    if app_state.outlier_analysis is None or len(app_state.outlier_analysis) == 0:
        return None, None, "❌ 请先完成异常点分析", state
    
    if outlier_index is None:
        return None, None, "❌ 请先选择一个异常点", state
    
    if outlier_index < 0 or outlier_index >= len(app_state.outlier_analysis):
        return None, None, f"❌ 索引超出范围 (0-{len(app_state.outlier_analysis)-1})", state
    
    try:
        item = app_state.outlier_analysis[outlier_index]
        image_path = item['图像路径']
        mask_path = item['掩码路径']
        
        # 加载图像
        image_array, mask_array = visualize_image_with_mask(
            image_path,
            mask_path,
            overlay=show_mask
        )
        
        info = f"""**异常点信息:**
- 类别: {item['类别']}
- 全局索引: {item['全局索引']}
- 异常分数: {item['异常分数']}
- 投票数: {item['投票数']}
- 与当前类别中心相似度: {item['与当前类别中心相似度']}
- 最相似类别(前三): {item['最相似类别(前三)']}
"""
        return image_array, mask_array, info, state
    except Exception as e:
        return None, None, f"❌ 加载图像失败: {str(e)}", state


def clear_image_module(state: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Dict]:
    """清除图像可视化模块数据"""
    return None, None, "", state

