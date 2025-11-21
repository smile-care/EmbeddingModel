"""
评估模型
"""
import argparse
import sys
import numpy as np
import json
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.few_shot_eval import FewShotEvaluator
from src.evaluation.cross_domain_eval import CrossDomainEvaluator
from src.evaluation.benchmark import BenchmarkEvaluator
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description='评估模型')
    parser.add_argument('--embeddings', type=str, required=True,
                       help='Embedding文件路径')
    parser.add_argument('--metadata', type=str, required=True,
                       help='元数据JSON文件路径')
    parser.add_argument('--task', type=str, choices=['few_shot', 'cross_domain', 'benchmark', 'all'],
                       default='all', help='评估任务')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='输出文件路径')
    args = parser.parse_args()
    
    logger = setup_logger('evaluate_model')
    
    # 加载数据
    logger.info("加载数据...")
    embeddings_file = Path(args.embeddings)
    if embeddings_file.suffix == '.npz':
        data = np.load(args.embeddings)
        embeddings = data['embeddings']
        instance_ids = data['instance_ids']
    else:
        embeddings_dict = np.load(args.embeddings, allow_pickle=True).item()
        instance_ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[iid] for iid in instance_ids])
    
    with open(args.metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 构建标签和domain映射
    instance_id_to_info = {p['instance_id']: p for p in metadata['patches']}
    labels = []
    domain_ids = []
    
    for instance_id in instance_ids:
        if instance_id in instance_id_to_info:
            info = instance_id_to_info[instance_id]
            labels.append(info['label'])
            domain_ids.append(info['domain_id'])
        else:
            labels.append('unknown')
            domain_ids.append('unknown')
    
    labels = np.array(labels)
    domain_ids = np.array(domain_ids)
    
    results = {}
    
    # Few-shot评估
    if args.task in ['few_shot', 'all']:
        logger.info("执行Few-shot评估...")
        evaluator = FewShotEvaluator(embeddings, labels, domain_ids)
        
        few_shot_results = []
        for domain in np.unique(domain_ids):
            if domain == 'unknown':
                continue
            result = evaluator.leave_one_domain_out(domain, n_shot=5, n_query=20)
            few_shot_results.append(result)
        
        results['few_shot'] = few_shot_results
        logger.info("Few-shot评估完成")
    
    # 跨场景迁移评估
    if args.task in ['cross_domain', 'all']:
        logger.info("执行跨场景迁移评估...")
        evaluator = CrossDomainEvaluator(embeddings, labels, domain_ids)
        
        cross_domain_results = []
        unique_domains = [d for d in np.unique(domain_ids) if d != 'unknown']
        
        for target_domain in unique_domains:
            source_domains = [d for d in unique_domains if d != target_domain]
            if len(source_domains) == 0:
                continue
            
            result = evaluator.evaluate_new_domain(
                source_domains=source_domains,
                target_domain=target_domain,
                n_train=100
            )
            cross_domain_results.append(result)
        
        results['cross_domain'] = cross_domain_results
        logger.info("跨场景迁移评估完成")
    
    # Baseline对比（如果有多个embedding文件）
    if args.task in ['benchmark', 'all']:
        logger.info("执行Baseline对比...")
        # 这里需要提供多个embedding文件进行对比
        # 简化处理：只使用当前embedding
        embeddings_dict = {'current_method': embeddings}
        evaluator = BenchmarkEvaluator(embeddings_dict, labels)
        benchmark_results = evaluator.compare_methods()
        results['benchmark'] = benchmark_results
        logger.info("Baseline对比完成")
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"评估结果已保存: {args.output}")


if __name__ == '__main__':
    main()

