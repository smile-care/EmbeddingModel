"""
模块2: 异常点检测与分析
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .outlier_detector import OutlierDetector, find_most_similar_label


def _run_detection_methods(detector: OutlierDetector, threshold_percentile: float, contamination: float) -> Dict:
    """执行所有检测方法"""
    methods = [
        ('distance', lambda: detector.detect_by_distance(threshold_percentile)),
        ('cosine', lambda: detector.detect_by_cosine_distance(threshold_percentile)),
        ('mahalanobis', lambda: detector.detect_by_mahalanobis(threshold_percentile)),
        ('isolation_forest', lambda: detector.detect_by_isolation_forest(contamination=contamination)),
        ('lof', lambda: detector.detect_by_lof(contamination=contamination))
    ]
    
    results = {}
    for method_name, method_func in methods:
        is_outlier, scores, threshold = method_func()
        results[method_name] = {
            'is_outlier': is_outlier,
            'scores': scores,
            'threshold': threshold,
            'n_outliers': np.sum(is_outlier)
        }
    return results


def _combine_detection_results(results: Dict, threshold_votes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """综合多种检测方法的结果（投票机制）"""
    n_samples = len(list(results.values())[0]['is_outlier'])
    outlier_votes = np.zeros(n_samples, dtype=int)
    
    for result in results.values():
        outlier_votes += result['is_outlier'].astype(int)
    
    outlier_scores_combined = outlier_votes / len(results)
    is_outlier_combined = outlier_votes >= threshold_votes
    
    return is_outlier_combined, outlier_scores_combined, outlier_votes


def _refine_outliers_by_similarity(
    embeddings_normalized: np.ndarray,
    is_outlier_combined: np.ndarray,
    label: str,
    app_state
) -> np.ndarray:
    """二次筛选：基于相似度的异常点验证"""
    if not np.any(is_outlier_combined):
        return is_outlier_combined
    
    current_center = app_state.label_centers[label]
    outlier_indices = np.where(is_outlier_combined)[0]
    refined_mask = np.zeros_like(is_outlier_combined, dtype=bool)
    
    for idx in outlier_indices:
        embedding = embeddings_normalized[idx]
        current_sim = np.dot(embedding, current_center)
        
        max_other_sim = max(
            (np.dot(embedding, center) for other_label, center in app_state.label_centers.items() 
             if other_label != label),
            default=-1.0
        )
        
        if current_sim < max_other_sim or current_sim < 0.4:
            refined_mask[idx] = True
    
    return refined_mask


def _detect_outliers_for_label(
    label: str,
    threshold_percentile: float,
    threshold_votes: int,
    contamination: float,
    app_state
) -> Dict:
    """对单个类别进行异常点检测"""
    indices = app_state.label_to_indices[label]
    label_embeddings = app_state.label_to_embeddings[label]
    
    # L2归一化
    embeddings_normalized = F.normalize(
        torch.from_numpy(label_embeddings), 
        dim=1, 
        p=2
    ).numpy()
    
    # 执行检测方法
    detector = OutlierDetector(embeddings_normalized, app_state.label_centers[label])
    results = _run_detection_methods(detector, threshold_percentile, contamination)
    
    # 综合结果
    is_outlier_combined, outlier_scores_combined, outlier_votes = _combine_detection_results(results, threshold_votes)
    
    # 二次筛选
    is_outlier_combined = _refine_outliers_by_similarity(embeddings_normalized, is_outlier_combined, label, app_state)
    
    # 保存综合结果
    results['combined'] = {
        'is_outlier': is_outlier_combined,
        'scores': outlier_scores_combined,
        'votes': outlier_votes,
        'n_outliers': np.sum(is_outlier_combined)
    }
    
    return {
        'indices': indices,
        'embeddings_normalized': embeddings_normalized,
        'results': results
    }


def _generate_detection_stats(
    all_results: Dict,
    detection_mode: str,
    threshold_percentile: float,
    threshold_votes: int
) -> Tuple[pd.DataFrame, str]:
    """生成检测统计信息"""
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
    
    detect_df = pd.DataFrame(stats_rows)
    detect_info = f"""✓ 异常点检测完成！

**检测参数:**
- 检测模式: {detection_mode}
- 阈值百分位数: {threshold_percentile}%
- 投票阈值: {threshold_votes}/5
- 处理的类别数: {len(all_results)}
- 总异常点数: {total_outliers}
"""
    return detect_df, detect_info


def _analyze_outliers(all_results: Dict, app_state) -> pd.DataFrame:
    """分析异常点，生成详细列表"""
    outlier_analysis = []
    
    for label, data in all_results.items():
        indices = data['indices']
        embeddings_norm = data['embeddings_normalized']
        results = data['results']
        outlier_mask = results['combined']['is_outlier']
        outlier_indices = np.where(outlier_mask)[0]
        
        if len(outlier_indices) == 0:
            continue
        
        current_center = app_state.label_centers[label]
        
        for local_idx in outlier_indices:
            global_idx = indices[local_idx]
            embedding = embeddings_norm[local_idx]
            
            # 找到最相似的3个其他类别
            similar_labels = find_most_similar_label(
                embedding, label, app_state.label_centers, top_k=3, similarity_threshold=0.0
            )
            
            # 计算相似度
            current_sim = np.dot(embedding, current_center)
            similar_str = ", ".join([f"{l}({s:.3f})" for l, s in similar_labels])
            
            outlier_analysis.append({
                '类别': label,
                '全局索引': int(global_idx),
                '类内索引': int(local_idx),
                '异常分数': f"{results['combined']['scores'][local_idx]:.3f}",
                '投票数': f"{int(results['combined']['votes'][local_idx])}/5",
                '与当前类别中心相似度': f"{current_sim:.3f}",
                '最相似类别(前三)': similar_str,
                '图像路径': app_state.image_paths[global_idx],
                '掩码路径': app_state.mask_paths[global_idx]
            })
    
    # 按异常分数排序
    outlier_analysis.sort(key=lambda x: float(x['异常分数']), reverse=True)
    app_state.outlier_analysis = outlier_analysis
    
    return pd.DataFrame(outlier_analysis)


def _validate_labels(detection_mode: str, selected_labels: List[str], app_state) -> Tuple[bool, Optional[str], List[str]]:
    """验证并获取要处理的类别列表"""
    if detection_mode == "指定类检测":
        if not selected_labels or len(selected_labels) == 0:
            return False, "❌ 请至少选择一个类别", []
        invalid_labels = [l for l in selected_labels if l not in app_state.unique_labels]
        if invalid_labels:
            return False, f"❌ 以下类别无效: {', '.join(invalid_labels)}", []
        return True, None, selected_labels
    else:
        return True, None, app_state.unique_labels


def detect_and_analyze_outliers_module(
    detection_mode: str,
    selected_labels: List[str],
    threshold_percentile: float,
    threshold_votes: int,
    app_state,
    state: Dict
) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    """异常点检测与分析模块"""
    if not app_state.check_preprocessed():
        return "❌ 请先加载并预处理数据", None, None, state
    
    if app_state.label_centers is None:
        return "❌ 请先计算类别中心", None, None, state
    
    try:
        # 验证类别
        is_valid, error_msg, labels_to_process = _validate_labels(detection_mode, selected_labels, app_state)
        if not is_valid:
            return error_msg, None, None, state
        
        # 检测所有类别
        contamination = (100 - threshold_percentile) / 100.0
        all_results = {
            label: _detect_outliers_for_label(label, threshold_percentile, threshold_votes, contamination, app_state)
            for label in labels_to_process
        }
        
        # 保存到状态
        app_state.all_results = all_results
        app_state.detection_mode = detection_mode
        app_state.detected_labels = selected_labels if detection_mode == "指定类检测" else None
        
        # 生成统计信息
        detect_df, detect_info = _generate_detection_stats(
            all_results, detection_mode, threshold_percentile, threshold_votes
        )
        state['detection_done'] = True
        state['detection_info'] = detect_info
        
        # 分析异常点
        analyze_df = _analyze_outliers(all_results, app_state)
        state['analysis_done'] = True
        state['analysis_info'] = f"""✓ 异常点分析完成！

**分析结果:**
- 总异常点数: {len(analyze_df)}
- 涉及类别数: {len(set([x['类别'] for x in app_state.outlier_analysis]))}
"""
        
        return detect_info, detect_df, analyze_df, state
    except Exception as e:
        error_msg = f"❌ 检测与分析失败: {str(e)}"
        return error_msg, None, None, state


def clear_detection_and_analysis_module(app_state, state: Dict) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    """清除异常点检测和分析模块数据"""
    app_state.all_results = None
    app_state.detection_mode = None
    app_state.detected_labels = None
    app_state.outlier_analysis = None
    state['detection_done'] = False
    state['analysis_done'] = False
    state['detection_info'] = ""
    state['analysis_info'] = ""
    return "✓ 异常点检测和分析数据已清除", None, None, state


def get_outlier_list(app_state) -> List[str]:
    """获取异常点列表用于下拉选择"""
    if app_state.outlier_analysis is None:
        return []
    
    return [
        f"{i}: {item['类别']} (分数: {item['异常分数']}, 索引: {item['全局索引']})"
        for i, item in enumerate(app_state.outlier_analysis)
    ]

