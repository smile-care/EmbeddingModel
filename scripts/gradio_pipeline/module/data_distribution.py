"""
模块1: 数据分布可视化
"""
from typing import Dict, Optional, Tuple

import numpy as np
from visualization import visualize_data_distribution


def visualize_data_distribution_module(
    app_state,
    state: Dict
) -> Tuple[Optional[np.ndarray], str, Dict]:
    """数据分布模块（按类别可视化）"""
    if not app_state.check_preprocessed():
        error_msg = "❌ 请先加载并预处理数据"
        print(f"\n{error_msg}\n")
        return None, error_msg, state
    
    if app_state.embeddings is None or app_state.label_names is None:
        error_msg = "❌ 数据未正确加载，请重新加载数据"
        print(f"\n{error_msg}\n")
        return None, error_msg, state
    
    try:
        print("\n" + "=" * 80)
        print("开始生成数据分布图...")
        print("=" * 80)
        
        # 如果已有降维结果，复用；否则计算
        img_array, embeddings_2d, method, label_to_color = visualize_data_distribution(
            app_state.embeddings,
            app_state.label_names,
            embeddings_2d=app_state.embeddings_2d_raw,
            reduction_method=app_state.reduction_method
        )
        # 保存原始降维结果和颜色映射供后续使用
        app_state.embeddings_2d_raw = embeddings_2d
        app_state.reduction_method = method
        app_state.label_to_color = label_to_color
        state['data_dist_image'] = img_array
        
        success_msg = f"""✓ 数据分布图生成成功！

**信息:**
- 降维方法: {method}
- 图像形状: {img_array.shape}
- 总样本数: {len(app_state.embeddings)}
- 类别数: {len(app_state.unique_labels) if app_state.unique_labels else 'N/A'}
"""
        print(f"✓ 数据分布图生成成功！")
        print(f"  - 降维方法: {method}")
        print(f"  - 图像形状: {img_array.shape}")
        print("=" * 80 + "\n")
        
        return img_array, success_msg, state
    except Exception as e:
        error_msg = f"❌ 生成数据分布图失败: {str(e)}"
        print(f"\n{error_msg}\n")
        import traceback
        traceback.print_exc()
        return None, error_msg, state


def clear_data_dist_module(state: Dict) -> Tuple[Optional[np.ndarray], str, Dict]:
    """清除数据分布模块数据"""
    state['data_dist_image'] = None
    return None, "✓ 数据分布图已清除", state

