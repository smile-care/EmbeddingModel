"""
可视化模块
负责异常点分布可视化和图像显示
"""
import matplotlib
import numpy as np

matplotlib.use('Agg')  # 使用非交互式后端
import base64
import io
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import umap
from PIL import Image
from sklearn.decomposition import PCA

matplotlib.rc("font", family='AR PL UKai CN')


def generate_high_contrast_colors(n_colors: int) -> np.ndarray:
    """
    生成随机颜色的颜色映射
    
    Args:
        n_colors: 需要的颜色数量
    
    Returns:
        颜色数组 (n_colors, 4) RGBA格式
    """
    if n_colors <= 0:
        return np.array([])
    
    # 使用tab10 (10种高对比度颜色), 超过则用tab20, 都随机一下顺序

    if n_colors == 1:
        # 蓝色 默认
        return np.array([[0.0, 0.447, 0.741, 1.0]])

    if n_colors <= 10:
        color_pool = plt.cm.tab10(np.linspace(0, 1, 10))
    else:
        color_pool = plt.cm.tab20(np.linspace(0, 1, 20))
    color_pool = color_pool.copy()
    idx = list(range(len(color_pool)))
    random.shuffle(idx)
    # 如果颜色数不足，循环补足
    colors = color_pool[idx][:n_colors]
    if len(colors) < n_colors:
        repeats = n_colors // len(color_pool) + 1
        big_pool = np.tile(color_pool[idx], (repeats, 1))
        colors = big_pool[:n_colors]
    return colors


def reduce_dimensions(embeddings_norm: np.ndarray, n_samples: int) -> Tuple[np.ndarray, str]:
    """
    降维函数，根据数据量自适应选择降维方法
    
    Args:
        embeddings_norm: 已归一化的embedding数组
        n_samples: 样本数量
    
    Returns:
        (embeddings_2d, method) 降维后的2D数组和使用的降维方法
    """
    if n_samples < 5:
        method = 'PCA'
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings_norm)
    else:
        method = 'UMAP'
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            metric='cosine',
            n_neighbors=min(15, n_samples - 1)
        )
        embeddings_2d = reducer.fit_transform(embeddings_norm)
    
    return embeddings_2d, method


def visualize_data_distribution(
    embeddings: np.ndarray,
    label_names: np.ndarray,
    embeddings_2d: Optional[np.ndarray] = None,
    reduction_method: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, str, Dict]:
    """
    可视化数据分布（按类别着色，参考show_embeddings.ipynb）
    
    Args:
        embeddings: 所有样本的embedding数组
        label_names: 标签名称数组
        embeddings_2d: 可选的已降维结果（如果提供则复用，避免重复计算）
        reduction_method: 降维方法名称（如果提供了embeddings_2d）
    
    Returns:
        (图像数组, embeddings_2d, method, label_to_color) 图像、降维结果、降维方法和颜色映射
    """
    # L2归一化所有embeddings
    all_embeddings_norm = F.normalize(torch.from_numpy(embeddings), dim=1, p=2).numpy()
    
    n_samples = len(all_embeddings_norm)
    
    # 如果提供了已降维的结果，直接使用；否则计算
    if embeddings_2d is not None:
        method = reduction_method or 'Unknown'
        # embeddings_2d应该是原始降维结果，需要归一化到[0,1]
        embeddings_2d = embeddings_2d.copy()
    else:
        embeddings_2d, method = reduce_dimensions(all_embeddings_norm, n_samples)
    
    # 归一化到[0,1]用于数据分布可视化
    embeddings_2d_norm = embeddings_2d.copy()
    embeddings_2d_norm[:, 0] = embeddings_2d_norm[:, 0] - embeddings_2d_norm[:, 0].min()
    embeddings_2d_norm[:, 1] = embeddings_2d_norm[:, 1] - embeddings_2d_norm[:, 1].min()
    range_x = embeddings_2d_norm[:, 0].max()
    range_y = embeddings_2d_norm[:, 1].max()
    max_range = max(range_x, range_y)
    if max_range > 0:
        embeddings_2d_norm = embeddings_2d_norm / max_range
    
    
    # 获取唯一标签并排序
    unique_labels = sorted(list(set(label_names)))
    
    # 使用更大的色彩映射或生成随机颜色
    if len(unique_labels) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    else:
        # 使用hsv色彩空间生成足够多的颜色
        colors = plt.cm.hsv(np.linspace(0, 1, len(unique_labels)))
    
    label_to_color = dict(zip(unique_labels, colors))
    
    # 可视化
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 按类别绘制散点图（使用归一化后的数据）
    for label in unique_labels:
        mask = np.array([l == label for l in label_names])
        count = np.sum(mask)
        ax.scatter(
            embeddings_2d_norm[mask, 0], 
            embeddings_2d_norm[mask, 1],
            c=[label_to_color[label]],
            label=f"{label} ({count})",
            alpha=0.7,
            s=2,
        )
    
    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize='small',
        frameon=True,
        fancybox=True,
        shadow=True
    )
    ax.set_title(f"数据分布可视化 \n总样本: {len(embeddings)}, 类别数: {len(unique_labels)}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('维度1')
    ax.set_ylabel('维度2')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 转换为numpy数组
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close()
    
    # 返回图像、原始降维结果（未归一化）、降维方法和颜色映射
    return img_array, embeddings_2d, method, label_to_color


def visualize_outlier_distribution(
    embeddings: np.ndarray,
    label_names: np.ndarray,
    all_results: Dict,
    embeddings_2d: Optional[np.ndarray] = None,
    reduction_method: Optional[str] = None,
    label_to_color: Optional[Dict] = None,
    detection_mode: Optional[str] = None,
    detected_labels: Optional[List[str]] = None,
    color_mode: str = "use_module1"
) -> np.ndarray:
    """
    可视化异常点分布（自适应降维方法）
    
    如果提供了已降维的结果，则复用，避免重复计算
    
    Args:
        embeddings: 所有样本的embedding数组
        label_names: 标签名称数组
        all_results: 异常点检测结果字典
        embeddings_2d: 可选的已降维结果（如果提供则复用，避免重复计算）
        reduction_method: 降维方法名称（如果提供了embeddings_2d）
        label_to_color: 类别到颜色的映射（与数据分布模块对齐，如果为None则自动生成）
        detection_mode: 检测模式（"指定类检测"或"所有类检测"）
        detected_labels: 指定类检测时选中的类别列表
        color_mode: 颜色模式，"use_module1"使用数据分布模块的颜色映射，"high_contrast"使用随机颜色
    
    Returns:
        图像数组（numpy格式，用于Gradio显示）
    """
    # L2归一化所有embeddings
    all_embeddings_norm = F.normalize(torch.from_numpy(embeddings), dim=1, p=2).numpy()
    
    n_samples = len(all_embeddings_norm)
    
    # 如果提供了已降维的结果，直接使用；否则计算
    if embeddings_2d is not None:
        method = reduction_method or 'Unknown'
        # 注意：如果提供了embeddings_2d，它可能已经归一化到[0,1]
        # 但异常点可视化不需要归一化，所以直接使用原始降维结果
        # 如果数据已经归一化，我们需要反归一化（但这比较复杂）
        # 为了简化，我们假设如果提供了embeddings_2d，它就是原始降维结果
        # 如果没有提供，我们就计算
    else:
        print("embeddings_2d is None, calculating...")
        embeddings_2d, method = reduce_dimensions(all_embeddings_norm, n_samples)
    
    # 创建全局异常点标记
    is_global_outlier = np.zeros(len(embeddings), dtype=bool)
    for label, data in all_results.items():
        indices = data['indices']
        outlier_mask = data['results']['combined']['is_outlier']
        outlier_indices = np.array(indices)[outlier_mask]
        is_global_outlier[outlier_indices] = True
    
    # 确定要可视化的类别范围
    if detection_mode == "指定类检测" and detected_labels is not None and len(detected_labels) > 0:
        # 指定类检测：只可视化选中的类别
        labels_to_visualize = detected_labels
        # 创建掩码：只显示选中类别的样本
        visualize_mask = np.array([l in detected_labels for l in label_names])
        if len(detected_labels) == 1:
            title_suffix = f"（指定类检测: {detected_labels[0]}）"
        elif len(detected_labels) <= 3:
            title_suffix = f"（指定类检测: {', '.join(detected_labels)}）"
        else:
            title_suffix = f"（指定类检测: {len(detected_labels)}个类别）"
    else:
        # 所有类检测：可视化所有检测到的类别
        labels_to_visualize = list(all_results.keys())
        visualize_mask = np.ones(len(embeddings), dtype=bool)  # 显示所有样本
        title_suffix = "（所有类检测）"
    
    # 生成或使用提供的颜色映射
    if color_mode == "high_contrast":
        # 使用随机颜色映射（基于要可视化的类别）
        unique_labels_display = sorted(list(set([l for l in label_names if l in labels_to_visualize])))
        n_colors = len(unique_labels_display)
        if n_colors > 0:
            colors_all = generate_high_contrast_colors(n_colors)
            label_to_color = dict(zip(unique_labels_display, colors_all))
        else:
            # 如果没有要可视化的类别，使用默认颜色
            label_to_color = {}
        # 对于不在可视化列表中的类别，使用灰色（低透明度）
        all_unique_labels = sorted(list(set(label_names)))
        for label in all_unique_labels:
            if label not in label_to_color:
                label_to_color[label] = np.array([0.7, 0.7, 0.7, 0.3])  # 灰色，低透明度
    elif color_mode == "use_module1" and label_to_color is not None:
        # 使用数据分布模块的颜色映射（已提供）
        pass  # 直接使用提供的label_to_color
    elif label_to_color is None:
        # 如果没有提供颜色映射，生成新的（基于所有唯一标签）
        unique_labels_all = sorted(list(set(label_names)))
        if len(unique_labels_all) <= 10:
            colors_all = plt.cm.tab10(np.linspace(0, 1, len(unique_labels_all)))
        else:
            colors_all = plt.cm.hsv(np.linspace(0, 1, len(unique_labels_all)))
        label_to_color = dict(zip(unique_labels_all, colors_all))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 左图: 异常点分布（只显示要可视化的类别）
    display_mask = visualize_mask
    normal_mask = display_mask & ~is_global_outlier
    outlier_mask = display_mask & is_global_outlier
    
    axes[0].scatter(
        embeddings_2d[normal_mask, 0], 
        embeddings_2d[normal_mask, 1],
        c='blue', alpha=0.3, s=10, label=f'正常点 ({np.sum(normal_mask)})'
    )
    axes[0].scatter(
        embeddings_2d[outlier_mask, 0], 
        embeddings_2d[outlier_mask, 1],
        c='red', alpha=0.8, s=30, marker='x', label=f'异常点 ({np.sum(outlier_mask)})'
    )
    axes[0].set_title(
        f'异常点分布 {title_suffix} \n'
        f'显示样本: {np.sum(display_mask)}, 异常点: {np.sum(outlier_mask)} '
        f'({np.sum(outlier_mask)/np.sum(display_mask)*100:.1f}%)' if np.sum(display_mask) > 0 else ''
    )
    axes[0].set_xlabel('维度1')
    axes[0].set_ylabel('维度2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图: 按类别着色（使用与数据分布模块一致的颜色映射）
    # 只显示要可视化的类别
    display_indices = np.where(display_mask)[0]
    display_embeddings_2d = embeddings_2d[display_indices]
    display_label_names = label_names[display_indices]
    display_outlier_mask = is_global_outlier[display_indices]
    
    # 按类别绘制，使用与数据分布模块一致的颜色
    unique_labels_display = sorted(list(set(display_label_names)))
    for label in unique_labels_display:
        if label in label_to_color:
            label_mask = np.array([l == label for l in display_label_names])
            count = np.sum(label_mask)
            axes[1].scatter(
                display_embeddings_2d[label_mask, 0],
                display_embeddings_2d[label_mask, 1],
                c=[label_to_color[label]],
                label=f"{label} ({count})",
                alpha=0.5,
                s=10
            )
    
    # 标记异常点
    axes[1].scatter(
        display_embeddings_2d[display_outlier_mask, 0], 
        display_embeddings_2d[display_outlier_mask, 1],
        facecolors='none', edgecolors='red', s=100, linewidths=1.5, 
        marker='o', label='异常点'
    )
    axes[1].set_title(f'按类别着色的分布 {title_suffix} \n类别数: {len(unique_labels_display)}')
    axes[1].set_xlabel('维度1')
    axes[1].set_ylabel('维度2')
    axes[1].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 转换为numpy数组
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close()
    
    return img_array


def visualize_image_with_mask(
    image_path: str,
    mask_path: str,
    overlay: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    显示图像和掩码
    
    Args:
        image_path: 图像路径
        mask_path: 掩码路径（可选）
        overlay: 是否将掩码叠加在图像上
    
    Returns:
        (image_array, mask_array) 图像和掩码的numpy数组
    """
    # 加载图像
    mask_array = None
    mask_array_gray = None
    try:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
    except Exception as e:
        # 如果图像加载失败，返回错误图像
        image_array = np.zeros((224, 224, 3), dtype=np.uint8)
        image_array[:] = [255, 0, 0]  # 红色表示错误
    
    # 加载掩码
    try:
        if mask_path and Path(mask_path).exists():
            mask = Image.open(mask_path).convert('L')
            mask_array_gray = np.array(mask)
            # 转换为RGB格式用于显示
            mask_array = np.stack([mask_array_gray] * 3, axis=-1)
    except Exception as e:
        # 掩码加载失败，mask_array 保持为 None
        pass

    if overlay and mask_array_gray is not None:
        # 将掩码叠加在图像上（红色半透明）
        overlay_img = image_array.copy()
        mask_binary = (mask_array_gray > 128).astype(np.uint8)
        overlay_img[mask_binary > 0] = (
            overlay_img[mask_binary > 0] * 0.5 + 
            np.array([255, 0, 0]) * 0.5
        ).astype(np.uint8)
        image_array = overlay_img

    # ==== 限制输出的最大宽高不超过512x512 ====
    def resize_limit(arr, max_hw=(512, 512)):
        from PIL import Image
        if arr is None:
            return None
        if arr.ndim == 2:  # 灰度
            im = Image.fromarray(arr)
        else:  # 3通道
            im = Image.fromarray(arr)
        h, w = im.size[1], im.size[0]
        max_h, max_w = max_hw
        scale = min(max_h / h, max_w / w, 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            im = im.resize(new_size, Image.BILINEAR)
        return np.array(im)

    image_array = resize_limit(image_array, max_hw=(512,512))
    if mask_array is not None:
        mask_array = resize_limit(mask_array, max_hw=(512,512))

    return image_array, mask_array


def visualize_category_outliers(
    embeddings: np.ndarray,
    label_indices: List[int],
    outlier_mask: np.ndarray,
    label: str,
) -> np.ndarray:
    """
    可视化单个类别的异常点分布
    
    Args:
        embeddings: 该类别的embedding数组
        label_indices: 该类别的全局索引列表
        outlier_mask: 异常点标记（布尔数组）
        label: 类别名称
    
    Returns:
        图像数组
    """
    # 降维（统一逻辑：样本数 < 5 用PCA，否则用UMAP）
    n_samples = len(embeddings)
    if n_samples < 5:
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            metric='cosine',
            n_neighbors=min(15, n_samples - 1)
        )
        embeddings_2d = reducer.fit_transform(embeddings)
    
    # 可视化
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    normal_mask = ~outlier_mask
    outlier_mask_bool = outlier_mask
    
    ax.scatter(
        embeddings_2d[normal_mask, 0], 
        embeddings_2d[normal_mask, 1],
        c='blue', alpha=0.5, s=30, label=f'正常 ({np.sum(normal_mask)})'
    )
    ax.scatter(
        embeddings_2d[outlier_mask_bool, 0], 
        embeddings_2d[outlier_mask_bool, 1],
        c='red', alpha=0.8, s=50, marker='x', linewidths=2, 
        label=f'异常 ({np.sum(outlier_mask_bool)})'
    )
    
    total = len(embeddings)
    outlier_count = np.sum(outlier_mask_bool)
    ax.set_title(
        f'{label}\n异常: {outlier_count}/{total} ({outlier_count/total*100:.1f}%)',
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('维度1', fontsize=10)
    ax.set_ylabel('维度2', fontsize=10)
    
    plt.tight_layout()
    
    # 转换为numpy数组
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close()
    
    return img_array

