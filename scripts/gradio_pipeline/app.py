"""
异常点检测可视化界面 - Gradio应用
模块化设计，每个功能独立运行
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 添加当前目录到路径，以便导入模块
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from data_processor import compute_category_centers, load_embeddings, preprocess_data
from module import (clear_data_dist_module, clear_detection_and_analysis_module, clear_image_module,
                    clear_outlier_result_module, detect_and_analyze_outliers_module,
                    get_outlier_list, visualize_data_distribution_module, visualize_image_module,
                    visualize_outlier_result_module)


# 全局状态存储
class AppState:
    """应用状态管理"""
    def __init__(self):
        self.embeddings = None
        self.label_names = None
        self.image_paths = None
        self.mask_paths = None
        self.label_to_indices = None
        self.label_to_embeddings = None
        self.unique_labels = None
        self.label_centers = None
        self.all_results = None
        self.outlier_analysis = None
        self.data_loaded = False
        self.preprocessed = False
        self.embeddings_2d_raw = None  # 保存原始降维结果（未归一化）
        self.embeddings_2d_norm = None  # 保存归一化后的降维结果（用于数据分布）
        self.reduction_method = None  # 保存使用的降维方法
        self.label_to_color = None  # 保存数据分布模块的颜色映射（用于结果可视化对齐）
        self.detection_mode = None  # 保存检测模式（"指定类检测"或"所有类检测"）
        self.detected_labels = None  # 保存指定类检测时选中的类别列表
    
    def reset(self):
        """重置所有状态"""
        self.__init__()
    
    def check_data_loaded(self) -> bool:
        """检查数据是否已加载"""
        return self.data_loaded and self.embeddings is not None
    
    def check_preprocessed(self) -> bool:
        """检查数据是否已预处理"""
        return self.preprocessed and self.label_to_indices is not None


# 创建全局状态实例
app_state = AppState()

# ==================== 加载配置文件 ====================
def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # 默认配置
        return {
            'center_computation': {
                'method': 'trimmed_mean',
                'trim_ratio': 0.1
            }
        }

# 加载配置
config = load_config()
center_config = config.get('center_computation', {
    'method': 'trimmed_mean',
    'trim_ratio': 0.1
})

# ==================== 启动时自动加载所有功能代码 ====================
print("=" * 80)
print("正在初始化异常点检测系统...")
print("=" * 80)
print(f"✓ 配置文件已加载")
print(f"  - 类别中心计算方法: {center_config['method']}")
print(f"  - 修剪比例: {center_config['trim_ratio']}")
print("✓ 所有功能模块已加载")
print("=" * 80)


# ==================== 数据加载模块 ====================
def load_and_preprocess(file) -> Tuple[str, str]:
    """
    加载Embedding数据并自动预处理（后台执行，终端打印）
    """
    if file is None:
        return "❌ 请先上传 .npz 文件", ""
    
    try:
        print("\n" + "=" * 80)
        print("开始加载 Embedding 数据...")
        print("=" * 80)
        
        # 从上传的文件加载
        data = load_embeddings(file.name)
        
        # 保存到状态
        app_state.embeddings = data['embeddings']
        app_state.label_names = data['label_names']
        app_state.image_paths = data['image_paths']
        app_state.mask_paths = data['mask_paths']
        app_state.data_loaded = True
        
        print(f"✓ Embedding 加载成功！")
        print(f"  - Embedding 形状: {app_state.embeddings.shape}")
        print(f"  - 样本数量: {len(app_state.image_paths)}")
        print(f"  - 标签数量: {len(set(app_state.label_names))}")
        print(f"  - Embedding 维度: {app_state.embeddings.shape[1]}")
        
        # 自动执行预处理（后台）
        print("\n" + "=" * 80)
        print("开始数据预处理...")
        print("=" * 80)
        
        label_to_indices, label_to_embeddings, unique_labels = preprocess_data(
            app_state.embeddings,
            app_state.label_names,
            app_state.image_paths,
            app_state.mask_paths
        )
        
        app_state.label_to_indices = label_to_indices
        app_state.label_to_embeddings = label_to_embeddings
        app_state.unique_labels = unique_labels
        app_state.preprocessed = True
        
        print(f"✓ 数据预处理完成！")
        print(f"  - 总类别数: {len(unique_labels)}")
        print(f"  - 总样本数: {len(app_state.image_paths)}")
        print(f"\n各类别样本分布:")
        for label in unique_labels[:20]:
            count = len(label_to_indices[label])
            print(f"  - {label}: {count} 个样本")
        if len(unique_labels) > 20:
            print(f"  ... 还有 {len(unique_labels) - 20} 个类别")
        
        # 自动执行类别中心计算（后台）
        print("\n" + "=" * 80)
        print("开始计算类别中心（后台自动执行）...")
        print("=" * 80)
        
        label_centers = compute_category_centers(
            app_state.label_to_embeddings,
            app_state.unique_labels,
            method=center_config['method'],
            trim_ratio=center_config['trim_ratio']
        )
        
        app_state.label_centers = label_centers
        
        print(f"✓ 类别中心计算完成！")
        print(f"  - 计算方法: {center_config['method']}")
        print(f"  - 修剪比例: {center_config['trim_ratio']*100:.1f}% (仅trimmed_mean方法)")
        print(f"  - 计算的类别数: {len(label_centers)}")
        print("=" * 80)
        print("✓ 数据加载、预处理和类别中心计算完成，可以使用其他功能模块")
        print("=" * 80 + "\n")
        
        # 界面显示简洁信息
        info = f"""✓ 数据加载成功！

**数据统计:**
- embedding 形状: {app_state.embeddings.shape}
- 样本数量: {len(app_state.image_paths)}
- 标签数量: {len(set(app_state.label_names))}
- 类别数量: {len(unique_labels)}
- 类别名称: {"、".join([str(s) for s in app_state.unique_labels])}

**提示:** 数据已自动预处理，类别中心已自动计算，请查看终端获取详细信息。
"""
        return info, "✓ 数据已加载、预处理并完成类别中心计算，可以使用下方功能模块"
    except Exception as e:
        print(f"\n❌ 加载失败: {str(e)}\n")
        return f"❌ 加载失败: {str(e)}", ""


# 模块实现已移至 module 目录


# ==================== 全局清除功能 ====================
def clear_all_data() -> str:
    """清除所有数据"""
    app_state.reset()
    print("\n" + "=" * 80)
    print("所有数据已清除")
    print("=" * 80 + "\n")
    return "✓ 所有数据已清除，请重新加载数据"


# ==================== 创建界面 ====================
def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="Data_Clustering调试程序", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Data_Clustering调试程序
        
        基于embedding分布的异常点检测和分析工具
        """)
        
        # 全局状态
        global_state = gr.State(value={
            'detection_done': False,
            'analysis_done': False,
            'detection_info': '',
            'analysis_info': '',
            'data_dist_image': None,
            'outlier_result_image': None
        })
        
        # 全局清除按钮
        with gr.Row():
            clear_all_btn = gr.Button("🗑️ 清除所有数据", variant="stop", size="sm")
            clear_all_status = gr.Markdown("")
        
        # 数据加载模块（最顶部）
        with gr.Group():
            gr.Markdown("## 📁 数据加载模块")
            gr.Markdown("**说明:** 加载数据后会自动进行预处理，详细信息请查看终端输出。")
            file_input = gr.File(
                label="上传 .npz 文件",
                file_types=[".npz"],
                type="filepath"
            )
            load_btn = gr.Button("加载 Embedding", variant="primary")
            load_info = gr.Markdown("等待上传文件...")
            load_status = gr.Textbox(label="状态", interactive=False)
        
        gr.Markdown("---")
        gr.Markdown("## 🔧 功能模块")
        gr.Markdown("**说明:** 每个模块独立运行，切换模块时不会清除当前内容。")
        
        # 使用Tabs创建模块化界面
        with gr.Tabs() as tabs:
            # Tab 1: 数据分布
            with gr.Tab("📊 数据分布"):
                gr.Markdown("**说明:** 按类别可视化数据分布")
                with gr.Row():
                    visualize_data_dist_btn = gr.Button("生成数据分布图", variant="primary")
                    clear_data_dist_btn = gr.Button("清除本模块数据", variant="secondary")
                data_dist_info = gr.Markdown("等待生成数据分布图...")
                data_distribution_plot = gr.Image(label="数据分布图（按类别着色）")
            
            # Tab 2: 异常点检测与分析
            with gr.Tab("🔍 异常点检测与分析"):
                with gr.Row():
                    detection_mode = gr.Radio(
                        choices=["指定类检测", "所有类检测"],
                        value="指定类检测",
                        label="检测模式"
                    )
                    selected_labels = gr.Dropdown(
                        choices=[],
                        label="选择类别 (指定类模式)",
                        visible=True,
                        multiselect=True,
                        value=[]
                    )
                with gr.Row():
                    threshold_percentile = gr.Slider(
                        minimum=85.0,
                        maximum=99.0,
                        value=95.0,
                        step=1.0,
                        label="阈值百分位数 (默认95%)"
                    )
                    threshold_votes = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        label="投票阈值 (默认2/5，至少几种方法认为是异常点)"
                    )
                with gr.Row():
                    detect_btn = gr.Button("执行异常点检测与分析", variant="primary")
                    clear_detect_btn = gr.Button("清除本模块数据", variant="secondary")
                detect_info = gr.Markdown("等待检测...")
                detect_table = gr.Dataframe(
                    label="检测结果统计",
                    interactive=False,
                    wrap=True
                )
                analyze_table = gr.Dataframe(
                    label="异常点详细列表",
                    interactive=False,
                    wrap=True
                )
            
            # Tab 3: 结果可视化
            with gr.Tab("📉 结果可视化"):
                gr.Markdown("**说明:** 基于异常点检测结果进行可视化，显示正常点和异常点的分布")
                with gr.Row():
                    color_mode = gr.Radio(
                        choices=["沿用数据分布颜色", "随机颜色"],
                        value="沿用数据分布颜色",
                        label="颜色模式"
                    )
                with gr.Row():
                    visualize_result_btn = gr.Button("生成结果分布图", variant="primary")
                    clear_result_btn = gr.Button("清除本模块数据", variant="secondary")
                outlier_result_info = gr.Markdown("等待生成结果分布图...")
                outlier_result_plot = gr.Image(label="异常点结果分布图")
            
            # Tab 4: 图像可视化
            with gr.Tab("🖼️ 图像可视化"):
                outlier_dropdown = gr.Dropdown(
                    choices=[],
                    label="选择异常点",
                    type="index"
                )
                show_mask_check = gr.Checkbox(
                    value=True,
                    label="显示掩码叠加"
                )
                with gr.Row():
                    visualize_img_btn = gr.Button("显示图像和掩码", variant="primary")
                    clear_img_btn = gr.Button("清除本模块数据", variant="secondary")
                with gr.Row():
                    image_display = gr.Image(label="图像")
                    mask_display = gr.Image(label="掩码")
                image_info = gr.Markdown("等待选择异常点...")
        
        # 绑定事件
        def update_label_dropdown(mode):
            if mode == "指定类检测":
                if app_state.unique_labels is not None and len(app_state.unique_labels) > 0:
                    return gr.update(choices=app_state.unique_labels, visible=True, value=[])
                return gr.update(visible=True, value=[])
            return gr.update(visible=False, value=[])
        
        detection_mode.change(
            update_label_dropdown,
            inputs=[detection_mode],
            outputs=[selected_labels]
        )
        
        # 数据加载
        def update_label_after_load():
            """加载数据后更新类别下拉列表"""
            if app_state.unique_labels is not None and len(app_state.unique_labels) > 0:
                return gr.update(choices=app_state.unique_labels, visible=True, value=[])
            return gr.update(visible=True, value=[])
        
        load_btn.click(
            load_and_preprocess,
            inputs=[file_input],
            outputs=[load_info, load_status]
        ).then(
            update_label_after_load,
            outputs=[selected_labels]
        )
        
        # 全局清除
        clear_all_btn.click(
            clear_all_data,
            outputs=[clear_all_status]
        )
        
        # 模块1: 数据分布
        def wrap_data_dist(state: Dict):
            return visualize_data_distribution_module(app_state, state)
        
        visualize_data_dist_btn.click(
            wrap_data_dist,
            inputs=[global_state],
            outputs=[data_distribution_plot, data_dist_info, global_state]
        )
        
        clear_data_dist_btn.click(
            clear_data_dist_module,
            inputs=[global_state],
            outputs=[data_distribution_plot, data_dist_info, global_state]
        )
        
        # 模块2: 异常点检测与分析
        def wrap_detect(detection_mode, selected_labels, threshold_percentile, threshold_votes, state: Dict):
            return detect_and_analyze_outliers_module(
                detection_mode, selected_labels, threshold_percentile, threshold_votes, app_state, state
            )
        
        detect_btn.click(
            wrap_detect,
            inputs=[detection_mode, selected_labels, threshold_percentile, threshold_votes, global_state],
            outputs=[detect_info, detect_table, analyze_table, global_state]
        ).then(
            lambda: gr.update(choices=get_outlier_list(app_state)),
            outputs=[outlier_dropdown]
        )
        
        def wrap_clear_detect(state: Dict):
            return clear_detection_and_analysis_module(app_state, state)
        
        clear_detect_btn.click(
            wrap_clear_detect,
            inputs=[global_state],
            outputs=[detect_info, detect_table, analyze_table, global_state]
        ).then(
            lambda: gr.update(choices=[], value=None),
            outputs=[outlier_dropdown]
        )
        
        # 模块3: 结果可视化
        def visualize_with_color_mode(color_mode_str: str, state: Dict):
            """将中文颜色模式转换为内部模式"""
            color_mode_map = {
                "沿用数据分布颜色": "use_module1",
                "随机颜色": "high_contrast"
            }
            color_mode = color_mode_map.get(color_mode_str, "use_module1")
            return visualize_outlier_result_module(color_mode, app_state, state)
        
        visualize_result_btn.click(
            visualize_with_color_mode,
            inputs=[color_mode, global_state],
            outputs=[outlier_result_plot, outlier_result_info, global_state]
        )
        
        clear_result_btn.click(
            clear_outlier_result_module,
            inputs=[global_state],
            outputs=[outlier_result_plot, outlier_result_info, global_state]
        )
        
        # 模块4: 图像可视化
        def wrap_image(outlier_index, show_mask, state: Dict):
            return visualize_image_module(outlier_index, show_mask, app_state, state)
        
        visualize_img_btn.click(
            wrap_image,
            inputs=[outlier_dropdown, show_mask_check, global_state],
            outputs=[image_display, mask_display, image_info, global_state]
        )
        
        clear_img_btn.click(
            clear_image_module,
            inputs=[global_state],
            outputs=[image_display, mask_display, image_info, global_state]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
