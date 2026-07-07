"""基于类别对共现覆盖理论，估算 Web 平台小样本 finetune 所需的训练 step 数。

背景（详见与用户的讨论）：对比学习（SupCon）真正需要的不是「模型看过每个
样本几次」，而是「每一对类别在同一个 batch 内被对比信号覆盖足够多次」——
这才是决定 embedding 空间 decision boundary 好坏的量。它由数据集大小 N、
batch size B、各类别样本数决定，是一个超几何分布问题，与人为设定的
"epoch 数" 没有必然关系。

用法：给定训练集每个类别的样本数与 batch size，计算出能让"最难覆盖"的
类别对被覆盖 ``target_pair_repeats`` 次左右所需的训练 step 数，作为创建
实验时的默认训练预算（上界偏保守估计，可以设置得宽松一些）。真正何时
收敛仍应由验证集 margin/kNN 早停判定（见 ``trainer.py`` 的 early
stopping），本模块只负责给一个不脱离数据集规模的合理默认值，避免像
固定 ``epochs: 50`` 那样和数据集大小完全脱钩。
"""
from __future__ import annotations

import math
from collections.abc import Sequence

# 期望「最稀有」类别对被同批次共现覆盖的次数。经验常数（非严格理论值）：
# 数值越大，学到的边界通常越稳，但训练预算也越大——配合早停使用时可以
# 设置得宽松一些，交给早停机制决定真正何时停止。
DEFAULT_TARGET_PAIR_REPEATS = 60
DEFAULT_MIN_STEPS = 120
DEFAULT_MAX_STEPS = 500

# 训练/验证不再按 epoch 上报，而是把 [1, total_steps] 均匀切成这么多个
# 检查点（一定含第一步与最后一步/早停时的当前步），在每个检查点聚合
# 训练 loss、跑一次验证、上报进度。
DEFAULT_NUM_CHECKPOINTS = 10


def _pair_cooccur_prob(dataset_size: int, n_i: int, n_j: int, batch_size: int) -> float:
    """两个类别（样本数分别为 n_i、n_j）在一次随机 batch 抽样中同时出现的概率。

    用超几何分布容斥算出：P(同时出现) = 1 - P(i缺席) - P(j缺席) + P(都缺席)。
    """
    if dataset_size <= 0 or batch_size <= 0 or n_i <= 0 or n_j <= 0:
        return 0.0
    batch_size = min(batch_size, dataset_size)

    def _absent_prob(n_class: int) -> float:
        remain = dataset_size - n_class
        if remain < batch_size:
            return 0.0
        return math.comb(remain, batch_size) / math.comb(dataset_size, batch_size)

    p_i_absent = _absent_prob(n_i)
    p_j_absent = _absent_prob(n_j)
    remain_both = dataset_size - n_i - n_j
    p_both_absent = (
        0.0
        if remain_both < batch_size
        else math.comb(remain_both, batch_size) / math.comb(dataset_size, batch_size)
    )
    p_both_present = 1.0 - p_i_absent - p_j_absent + p_both_absent
    return max(0.0, min(1.0, p_both_present))


def recommend_total_steps(
    category_sample_counts: Sequence[int],
    batch_size: int,
    *,
    target_pair_repeats: int = DEFAULT_TARGET_PAIR_REPEATS,
    min_steps: int = DEFAULT_MIN_STEPS,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> int:
    """估算需要多少训练 step，才能让"最难覆盖"的类别对被覆盖约
    ``target_pair_repeats`` 次。

    Args:
        category_sample_counts: 训练集每个类别的样本数（允许不均衡/长尾）。
        batch_size: 实际生效的 batch size。
        target_pair_repeats: 期望最稀有类别对被共现覆盖的次数。
        min_steps / max_steps: 兜底范围，避免极端数据集算出不合理的值。
    """
    counts = [int(c) for c in category_sample_counts if c and c > 0]
    dataset_size = sum(counts)
    if dataset_size <= 0 or batch_size <= 0 or len(counts) < 2:
        return min_steps

    fallback_prob = 1.0 / max_steps
    worst_prob = 1.0
    for i in range(len(counts)):
        for j in range(i + 1, len(counts)):
            prob = _pair_cooccur_prob(dataset_size, counts[i], counts[j], batch_size)
            worst_prob = min(worst_prob, prob if prob > 0 else fallback_prob)

    raw_steps = math.ceil(target_pair_repeats / max(worst_prob, fallback_prob))
    return max(min_steps, min(max_steps, raw_steps))


def compute_checkpoint_steps(total_steps: int, num_checkpoints: int = DEFAULT_NUM_CHECKPOINTS) -> list[int]:
    """把 ``[1, total_steps]`` 均匀切成 ``num_checkpoints`` 个检查点。

    用于训练循环决定「什么时候聚合上报一次 train loss、跑一次验证」——
    不再有 epoch 边界的概念，检查点就是训练/验证共用的唯一节奏。

    保证：
      * 第一步 (1) 一定在检查点集合中（尽早看到一次训练/验证结果）；
      * 最后一步 (total_steps) 一定在检查点集合中（早停触发时，「停止的
        那一步」必然就是最近一个检查点，因此天然满足「最后一次必须
        eval」的要求，无需额外逻辑）；
      * 当 total_steps 本身不大于 num_checkpoints 时，退化为每一步都是
        检查点，避免过度稀疏。
    """
    if total_steps <= 0:
        return []
    if total_steps <= num_checkpoints:
        return list(range(1, total_steps + 1))
    steps = {
        max(1, min(total_steps, round(i * total_steps / num_checkpoints)))
        for i in range(1, num_checkpoints + 1)
    }
    steps.add(1)
    steps.add(total_steps)
    return sorted(steps)
