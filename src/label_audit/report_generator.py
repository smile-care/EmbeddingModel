"""
标签体检报告生成器
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .consistency_analyzer import ConsistencyAnalyzer
from .similarity_analyzer import SimilarityAnalyzer
from .outlier_detector import OutlierDetector
from ..utils.visualization import plot_distance_heatmap


class ReportGenerator:
    """标签体检报告生成器"""
    
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        instance_ids: np.ndarray,
        metadata: Optional[Dict] = None
    ):
        """
        初始化报告生成器
        
        Args:
            embeddings: 特征向量 (N, D)
            labels: 标签 (N,)
            instance_ids: 实例ID (N,)
            metadata: 元数据字典（包含patch路径等信息）
        """
        self.embeddings = embeddings
        self.labels = labels
        self.instance_ids = instance_ids
        self.metadata = metadata or {}
        
        # 初始化分析器
        self.consistency_analyzer = ConsistencyAnalyzer(embeddings, labels)
        self.similarity_analyzer = SimilarityAnalyzer(embeddings, labels)
        self.outlier_detector = OutlierDetector(embeddings, labels, instance_ids)
    
    def generate_report(
        self,
        output_dir: str,
        top_outliers_percent: float = 5.0,
        top_similar_pairs: int = 20
    ) -> str:
        """
        生成完整的体检报告
        
        Args:
            output_dir: 输出目录
            top_outliers_percent: 离群点百分比
            top_similar_pairs: 最相似类别对数量
            
        Returns:
            报告文件路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 类内一致性分析
        consistency_results = self.consistency_analyzer.analyze()
        
        # 2. 类间相似度分析
        similar_pairs = self.similarity_analyzer.find_similar_classes(top_similar_pairs)
        merge_suggestions = self.similarity_analyzer.get_merge_suggestions()
        
        # 3. 离群点检测
        outliers = self.outlier_detector.detect_outliers(top_percent=top_outliers_percent)
        
        # 4. 生成可视化
        self._generate_visualizations(output_dir, consistency_results, similar_pairs)
        
        # 5. 生成HTML报告
        html_path = self._generate_html_report(
            output_dir / f"label_audit_report_{timestamp}.html",
            consistency_results,
            similar_pairs,
            merge_suggestions,
            outliers
        )
        
        # 6. 生成JSON报告
        json_path = self._generate_json_report(
            output_dir / f"label_audit_report_{timestamp}.json",
            consistency_results,
            similar_pairs,
            merge_suggestions,
            outliers
        )
        
        return str(html_path)
    
    def _generate_visualizations(
        self,
        output_dir: Path,
        consistency_results: Dict,
        similar_pairs: List[Dict]
    ):
        """生成可视化图表"""
        # 1. 类间距离热力图
        distance_matrix = self.similarity_analyzer.distance_matrix
        labels_list = self.similarity_analyzer.unique_labels.tolist()
        
        plot_distance_heatmap(
            distance_matrix,
            labels_list,
            save_path=str(output_dir / "inter_class_distance_heatmap.png"),
            title="Inter-Class Distance Matrix"
        )
        
        # 2. 类内距离柱状图
        intra_distances = consistency_results['intra_class_distances']
        labels = list(intra_distances.keys())
        distances = list(intra_distances.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(labels)), distances)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.ylabel('Intra-Class Distance', fontsize=12)
        plt.title('Intra-Class Distance by Label', fontsize=14)
        plt.axhline(
            y=consistency_results['threshold'],
            color='r',
            linestyle='--',
            label=f"Threshold ({consistency_results['threshold']:.3f})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "intra_class_distance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 层次聚类dendrogram
        self.similarity_analyzer.hierarchical_clustering(
            save_path=str(output_dir / "hierarchical_clustering.png")
        )
    
    def _generate_html_report(
        self,
        output_path: Path,
        consistency_results: Dict,
        similar_pairs: List[Dict],
        merge_suggestions: List,
        outliers: List[Dict]
    ) -> Path:
        """生成HTML报告"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>标签体检报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .warning {{ color: #ff6600; }}
        .error {{ color: #cc0000; }}
        .outlier {{ background-color: #ffe6e6; }}
    </style>
</head>
<body>
    <h1>标签体检报告</h1>
    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>1. 类内一致性分析</h2>
    <p>总类别数: {consistency_results['summary']['total_classes']}</p>
    <p>不一致类别数: {consistency_results['summary']['inconsistent_count']}</p>
    <p>平均类内距离: {consistency_results['summary']['mean_intra_distance']:.4f}</p>
    
    <h3>不一致类别列表（类内距离过大）</h3>
    <table>
        <tr>
            <th>类别</th>
            <th>类内距离</th>
            <th>阈值</th>
        </tr>
"""
        for item in consistency_results['inconsistent_classes']:
            html_content += f"""
        <tr class="warning">
            <td>{item['label']}</td>
            <td>{item['intra_distance']:.4f}</td>
            <td>{item['threshold']:.4f}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>2. 类间相似度分析</h2>
    <h3>最相似的类别对（建议合并）</h3>
    <table>
        <tr>
            <th>类别1</th>
            <th>类别2</th>
            <th>距离</th>
            <th>样本数1</th>
            <th>样本数2</th>
        </tr>
"""
        for pair in similar_pairs[:20]:
            html_content += f"""
        <tr>
            <td>{pair['label1']}</td>
            <td>{pair['label2']}</td>
            <td>{pair['distance']:.4f}</td>
            <td>{pair['count1']}</td>
            <td>{pair['count2']}</td>
        </tr>
"""
        
        html_content += f"""
    </table>
    
    <h2>3. 离群点检测</h2>
    <p>检测到 {len(outliers)} 个离群点（前{len(outliers)}个最可疑的样本）</p>
    <table>
        <tr>
            <th>实例ID</th>
            <th>标签</th>
            <th>综合评分</th>
            <th>Z-Score</th>
            <th>kNN一致性</th>
        </tr>
"""
        for outlier in outliers[:50]:  # 只显示前50个
            html_content += f"""
        <tr class="outlier">
            <td>{outlier['instance_id']}</td>
            <td>{outlier['label']}</td>
            <td>{outlier['score']:.4f}</td>
            <td>{outlier['z_score']:.4f}</td>
            <td>{outlier['knn_consistency']:.4f}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>4. 可视化图表</h2>
    <p>请查看同目录下的PNG图像文件：</p>
    <ul>
        <li>inter_class_distance_heatmap.png - 类间距离热力图</li>
        <li>intra_class_distance.png - 类内距离柱状图</li>
        <li>hierarchical_clustering.png - 层次聚类树状图</li>
    </ul>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_json_report(
        self,
        output_path: Path,
        consistency_results: Dict,
        similar_pairs: List[Dict],
        merge_suggestions: List,
        outliers: List[Dict]
    ) -> Path:
        """生成JSON报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'consistency_analysis': consistency_results,
            'similar_pairs': similar_pairs,
            'merge_suggestions': [
                {
                    'label1': item[0],
                    'label2': item[1],
                    'distance': float(item[2])
                }
                for item in merge_suggestions
            ],
            'outliers': outliers
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path

