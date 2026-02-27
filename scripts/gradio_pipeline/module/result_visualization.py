"""
模块3: 结果可视化
"""
from typing import Dict, Optional, Tuple

import numpy as np
from visualization import visualize_outlier_distribution


def visualize_outlier_result_module(
    color_mode: str,
    app_state,
    state: Dict
) -> Tuple[Optional[np.ndarray], str, Dict]:
    """结果可视化模块（基于异常点检测结果）"""
    if app_state.all_results is None:
        error_msg = "❌ 请先执行异常点检测"
        print(f"\n{error_msg}\n")
        return None, error_msg, state
    
    if app_state.embeddings is None or app_state.label_names is None:
        error_msg = "❌ 数据未正确加载，请重新加载数据"
        print(f"\n{error_msg}\n")
        return None, error_msg, state
    
    try:
        print("\n" + "=" * 80)
        print("开始生成异常点结果分布图...")
        print(f"颜色模式: {color_mode}")
        print("=" * 80)
        
        # 如果数据分布模块已经计算过降维结果，复用原始降维结果；否则重新计算
        # 根据颜色模式选择颜色映射
        if color_mode == "use_module1":
            label_to_color = app_state.label_to_color  # 使用数据分布模块的颜色映射
        else:
            label_to_color = None  # 使用随机颜色映射（在visualize_outlier_distribution中生成）
        
        img_array = visualize_outlier_distribution(
            app_state.embeddings,
            app_state.label_names,
            app_state.all_results,
            embeddings_2d=app_state.embeddings_2d_raw,  # 使用原始降维结果
            reduction_method=app_state.reduction_method,
            label_to_color=label_to_color,  # 根据颜色模式选择
            detection_mode=app_state.detection_mode,  # 传递检测模式
            detected_labels=app_state.detected_labels,  # 传递选中的类别列表
            color_mode=color_mode  # 传递颜色模式
        )
        state['outlier_result_image'] = img_array
        
        # 统计异常点数量
        total_outliers = 0
        total_samples = 0
        for label, data in app_state.all_results.items():
            total_outliers += data['results']['combined']['n_outliers']
            total_samples += len(data['indices'])
        
        success_msg = f"""✓ 异常点结果分布图生成成功！

**信息:**
- 总样本数: {total_samples}
- 总异常点数: {total_outliers}
- 异常比例: {total_outliers/total_samples*100:.2f}%
- 图像形状: {img_array.shape}
"""
        print(f"✓ 异常点结果分布图生成成功！")
        print(f"  - 总异常点数: {total_outliers}")
        print(f"  - 图像形状: {img_array.shape}")
        print("=" * 80 + "\n")
        
        return img_array, success_msg, state
    except Exception as e:
        error_msg = f"❌ 生成异常点结果分布图失败: {str(e)}"
        print(f"\n{error_msg}\n")
        import traceback
        traceback.print_exc()
        return None, error_msg, state


def clear_outlier_result_module(state: Dict) -> Tuple[Optional[np.ndarray], str, Dict]:
    """清除结果可视化模块数据"""
    state['outlier_result_image'] = None
    return None, "✓ 异常点结果分布图已清除", state

