"""DataCluster 平台侧 DL / 降维 / 数据分析核心代码。

将与深度学习、降维(PCA/t-SNE/UMAP)、数据分析相关的业务逻辑从 ``app/services``
中抽离到这里，便于独立维护：

- ``projection``: 2D 降维与按类异常分数。
- ``inference``:  基于 embedding_model 的 embedding 计算与默认预训练编码器。
- ``crop``:       依据标注区域抽取 crop patch 并落库。
- ``training``:   DB → manifest → 配置 → 调用 embedding_model 训练器的训练编排。

注意：纯文件/存储 IO 基础设施仍保留在 ``app/services/storage.py``。
"""
