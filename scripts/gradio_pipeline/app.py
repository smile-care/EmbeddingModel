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
from outlier_detector import OutlierDetector, find_most_similar_label
from visualization import (visualize_category_outliers, visualize_data_distribution,
                           visualize_image_with_mask, visualize_outlier_distribution)


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


# ==================== 模块2: 异常点检测 ====================
def detect_outliers_module(
    detection_mode: str,
    selected_labels: List[str],
    threshold_percentile: float,
    min_samples: int,
    state: Dict
) -> Tuple[str, Optional[pd.DataFrame], Dict]:
    """异常点检测模块"""
    if not app_state.check_preprocessed():
        return "❌ 请先加载并预处理数据", None, state
    
    if app_state.label_centers is None:
        return "❌ 请先计算类别中心", None, state
    
    try:
        contamination = (100 - threshold_percentile) / 100.0
        all_results = {}
        
        # 确定要检测的类别
        if detection_mode == "指定类检测":
            if not selected_labels or len(selected_labels) == 0:
                return "❌ 请至少选择一个类别", None, state
            # 验证选中的类别是否有效
            invalid_labels = [l for l in selected_labels if l not in app_state.unique_labels]
            if invalid_labels:
                return f"❌ 以下类别无效: {', '.join(invalid_labels)}", None, state
            labels_to_process = selected_labels
        else:  # 所有类检测
            labels_to_process = app_state.unique_labels
        
        # 对每个类别进行检测
        for label in labels_to_process:
            indices = app_state.label_to_indices[label]
            label_embeddings = app_state.label_to_embeddings[label]
            
            # 样本数太少，跳过
            if len(indices) < min_samples:
                continue
            
            # L2归一化
            embeddings_normalized = F.normalize(
                torch.from_numpy(label_embeddings), 
                dim=1, 
                p=2
            ).numpy()
            
            # 初始化检测器
            detector = OutlierDetector(embeddings_normalized)
            
            # 执行多种检测方法
            results = {}
            
            # 1. 欧氏距离方法
            is_outlier, scores, threshold = detector.detect_by_distance(threshold_percentile)
            results['distance'] = {
                'is_outlier': is_outlier,
                'scores': scores,
                'threshold': threshold,
                'n_outliers': np.sum(is_outlier)
            }
            
            # 2. 余弦距离方法
            is_outlier, scores, threshold = detector.detect_by_cosine_distance(threshold_percentile)
            results['cosine'] = {
                'is_outlier': is_outlier,
                'scores': scores,
                'threshold': threshold,
                'n_outliers': np.sum(is_outlier)
            }
            
            # 3. 马氏距离方法
            is_outlier, scores, threshold = detector.detect_by_mahalanobis(threshold_percentile)
            results['mahalanobis'] = {
                'is_outlier': is_outlier,
                'scores': scores,
                'threshold': threshold,
                'n_outliers': np.sum(is_outlier)
            }
            
            # 4. Isolation Forest
            is_outlier, scores, threshold = detector.detect_by_isolation_forest(contamination=contamination)
            results['isolation_forest'] = {
                'is_outlier': is_outlier,
                'scores': scores,
                'threshold': threshold,
                'n_outliers': np.sum(is_outlier)
            }
            
            # 5. LOF方法
            is_outlier, scores, threshold = detector.detect_by_lof(contamination=contamination)
            results['lof'] = {
                'is_outlier': is_outlier,
                'scores': scores,
                'threshold': threshold,
                'n_outliers': np.sum(is_outlier)
            }
            
            # 综合多种方法的结果（投票机制）
            outlier_votes = np.zeros(len(embeddings_normalized), dtype=int)
            for method_name, result in results.items():
                outlier_votes += result['is_outlier'].astype(int)
            
            # 计算综合异常分数（归一化到0-1）
            outlier_scores_combined = outlier_votes / len(results)
            
            # 设置综合异常点阈值（至少两种及以上方法认为是异常点）
            threshold_votes = 2
            is_outlier_combined = outlier_votes >= threshold_votes
            
            # 保存综合结果
            results['combined'] = {
                'is_outlier': is_outlier_combined,
                'scores': outlier_scores_combined,
                'votes': outlier_votes,
                'n_outliers': np.sum(is_outlier_combined)
            }
            
            # 保存该类别的所有结果
            all_results[label] = {
                'indices': indices,
                'embeddings_normalized': embeddings_normalized,
                'results': results
            }
        
        # 保存到状态
        app_state.all_results = all_results
        app_state.detection_mode = detection_mode
        app_state.detected_labels = selected_labels if detection_mode == "指定类检测" else None
        
        # 生成统计信息
        stats_rows = []
        total_outliers = 0
        for label, data in all_results.items():
            n_outliers = data['results']['combined']['n_outliers']
            total_samples = len(data['indices'])
            total_outliers += n_outliers
            ratio = n_outliers / total_samples * 100 if total_samples > 0 else 0
            stats_rows.append({
                '类别': label,
                '总样本数': total_samples,
                '异常点数': n_outliers,
                '异常比例 (%)': f"{ratio:.1f}%"
            })
        
        df = pd.DataFrame(stats_rows)
        
        info = f"""✓ 异常点检测完成！

**检测参数:**
- 检测模式: {detection_mode}
- 阈值百分位数: {threshold_percentile}%
- 最少样本数: {min_samples}
- 处理的类别数: {len(all_results)}
- 总异常点数: {total_outliers}
"""
        state['detection_done'] = True
        state['detection_info'] = info
        return info, df, state
    except Exception as e:
        return f"❌ 检测失败: {str(e)}", None, state


def clear_detection_module(state: Dict) -> Tuple[str, Optional[pd.DataFrame], Dict]:
    """清除异常点检测模块数据"""
    app_state.all_results = None
    app_state.detection_mode = None
    app_state.detected_labels = None
    state['detection_done'] = False
    state['detection_info'] = ""
    return "✓ 异常点检测数据已清除", None, state


# ==================== 模块3: 异常点分析 ====================
def analyze_outliers_module(state: Dict) -> Tuple[str, Optional[pd.DataFrame], Dict]:
    """异常点分析模块"""
    if app_state.all_results is None:
        return "❌ 请先执行异常点检测", None, state
    
    if app_state.label_centers is None:
        return "❌ 请先计算类别中心", None, state
    
    try:
        outlier_analysis = []
        
        for label, data in app_state.all_results.items():
            indices = data['indices']
            embeddings_norm = data['embeddings_normalized']
            results = data['results']
            
            # 获取异常点
            outlier_mask = results['combined']['is_outlier']
            outlier_local_indices = np.where(outlier_mask)[0]
            
            if len(outlier_local_indices) == 0:
                continue
            
            # 对每个异常点，找到最相似的其他类别
            for local_idx in outlier_local_indices:
                global_idx = indices[local_idx]
                embedding = embeddings_norm[local_idx]
                outlier_score = results['combined']['scores'][local_idx]
                votes = results['combined']['votes'][local_idx]
                
                # 找到最相似的3个其他类别
                similar_labels = find_most_similar_label(
                    embedding, 
                    label, 
                    app_state.label_centers, 
                    top_k=3,
                    similarity_threshold=0.0
                )
                
                # 与当前label center的余弦相似度
                current_center = app_state.label_centers[label]
                current_cosine_similarity = np.dot(embedding, current_center)
                
                # 格式化最相似标签
                similar_str = ", ".join([
                    f"{sim_label}({sim_score:.3f})"
                    for sim_label, sim_score in similar_labels
                ])
                
                outlier_analysis.append({
                    '类别': label,
                    '全局索引': int(global_idx),
                    '类内索引': int(local_idx),
                    '异常分数': f"{outlier_score:.3f}",
                    '投票数': f"{int(votes)}/5",
                    '与当前类别中心相似度': f"{current_cosine_similarity:.3f}",
                    '最相似类别(前三)': similar_str,
                    '图像路径': app_state.image_paths[global_idx],
                    '掩码路径': app_state.mask_paths[global_idx]
                })
        
        # 按异常分数排序
        outlier_analysis.sort(
            key=lambda x: float(x['异常分数']), 
            reverse=True
        )
        
        # 保存到状态
        app_state.outlier_analysis = outlier_analysis
        
        # 创建DataFrame
        df = pd.DataFrame(outlier_analysis)
        
        info = f"""✓ 异常点分析完成！

**分析结果:**
- 总异常点数: {len(outlier_analysis)}
- 涉及类别数: {len(set([x['类别'] for x in outlier_analysis]))}

**统计信息:**
- 平均异常分数: {np.mean([float(x['异常分数']) for x in outlier_analysis]):.3f}
- 最高异常分数: {max([float(x['异常分数']) for x in outlier_analysis]):.3f}
"""
        state['analysis_done'] = True
        state['analysis_info'] = info
        return info, df, state
    except Exception as e:
        return f"❌ 分析失败: {str(e)}", None, state


def clear_analysis_module(state: Dict) -> Tuple[str, Optional[pd.DataFrame], Dict]:
    """清除异常点分析模块数据"""
    app_state.outlier_analysis = None
    state['analysis_done'] = False
    state['analysis_info'] = ""
    return "✓ 异常点分析数据已清除", None, state


# ==================== 模块1: 数据分布 ====================
def visualize_data_distribution_module(state: Dict) -> Tuple[Optional[np.ndarray], str, Dict]:
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


# ==================== 模块4: 结果可视化 ====================
def visualize_outlier_result_module(color_mode: str, state: Dict) -> Tuple[Optional[np.ndarray], str, Dict]:
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


# ==================== 模块5: 图像可视化 ====================
def visualize_image_module(
    outlier_index: int,
    show_mask: bool,
    state: Dict
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Dict]:
    """图像可视化模块"""
    if app_state.outlier_analysis is None or len(app_state.outlier_analysis) == 0:
        return None, None, "❌ 请先完成异常点分析", state
    
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
- 最相似类别(前三): {item['最相似类别(前三)']}
"""
        return image_array, mask_array, info, state
    except Exception as e:
        return None, None, f"❌ 加载图像失败: {str(e)}", state


def get_outlier_list() -> List[str]:
    """获取异常点列表用于下拉选择"""
    if app_state.outlier_analysis is None:
        return []
    
    return [
        f"{i}: {item['类别']} (分数: {item['异常分数']}, 索引: {item['全局索引']})"
        for i, item in enumerate(app_state.outlier_analysis)
    ]


def clear_image_module(state: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Dict]:
    """清除图像可视化模块数据"""
    return None, None, "", state


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
            
            # Tab 2: 异常点检测
            with gr.Tab("🔍 异常点检测"):
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
                        minimum=80.0,
                        maximum=99.0,
                        value=90.0,
                        step=1.0,
                        label="阈值百分位数 (默认90%)"
                    )
                    min_samples = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=5,
                        step=1,
                        label="最少样本数 (默认5)"
                    )
                with gr.Row():
                    detect_btn = gr.Button("执行异常点检测", variant="primary")
                    clear_detect_btn = gr.Button("清除本模块数据", variant="secondary")
                detect_info = gr.Markdown("等待检测...")
                detect_table = gr.Dataframe(
                    label="检测结果统计",
                    interactive=False,
                    wrap=True
                )
            
            # Tab 3: 异常点分析
            with gr.Tab("📈 异常点分析"):
                with gr.Row():
                    analyze_btn = gr.Button("计算异常点相似标签", variant="primary")
                    clear_analyze_btn = gr.Button("清除本模块数据", variant="secondary")
                analyze_info = gr.Markdown("等待分析...")
                analyze_table = gr.Dataframe(
                    label="异常点详细列表",
                    interactive=False,
                    wrap=True
                )
            
            # Tab 4: 结果可视化
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
            
            # Tab 5: 图像可视化
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
        visualize_data_dist_btn.click(
            visualize_data_distribution_module,
            inputs=[global_state],
            outputs=[data_distribution_plot, data_dist_info, global_state]
        )
        
        clear_data_dist_btn.click(
            clear_data_dist_module,
            inputs=[global_state],
            outputs=[data_distribution_plot, data_dist_info, global_state]
        )
        
        # 模块2: 异常点检测
        detect_btn.click(
            detect_outliers_module,
            inputs=[detection_mode, selected_labels, threshold_percentile, min_samples, global_state],
            outputs=[detect_info, detect_table, global_state]
        )
        
        clear_detect_btn.click(
            clear_detection_module,
            inputs=[global_state],
            outputs=[detect_info, detect_table, global_state]
        )
        
        # 模块3: 异常点分析
        analyze_btn.click(
            analyze_outliers_module,
            inputs=[global_state],
            outputs=[analyze_info, analyze_table, global_state]
        ).then(
            lambda: gr.update(choices=get_outlier_list()),
            outputs=[outlier_dropdown]
        )
        
        clear_analyze_btn.click(
            clear_analysis_module,
            inputs=[global_state],
            outputs=[analyze_info, analyze_table, global_state]
        )
        
        # 模块4: 结果可视化
        def visualize_with_color_mode(color_mode_str: str, state: Dict):
            """将中文颜色模式转换为内部模式"""
            color_mode_map = {
                "沿用数据分布颜色": "use_module1",
                "随机颜色": "high_contrast"
            }
            color_mode = color_mode_map.get(color_mode_str, "use_module1")
            return visualize_outlier_result_module(color_mode, state)
        
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
        
        # 模块5: 图像可视化
        visualize_img_btn.click(
            visualize_image_module,
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
