# 工业缺陷通用Embedding模型

跨场景通用的工业缺陷表征模型，通过MAE自监督预训练和SupCon监督对比学习，构建工业缺陷形态度量空间。

## 项目结构

```
EmbeddingModel/
├── configs/                    # 配置文件目录
├── data/                      # 数据目录
├── src/                       # 源代码
├── scripts/                   # 执行脚本
└── requirements.txt           # 依赖包
```

## 安装

```bash
pip install -r requirements.txt
# 或
pip install -e .
```

## 使用流程

1. **数据准备**：运行 `scripts/prepare_data.py` 准备数据
2. **自监督预训练**：运行 `scripts/train_ssl.py` 训练MAE模型
3. **监督对比学习**：运行 `scripts/train_supcon.py` 训练SupCon模型
4. **提取Embedding**：运行 `scripts/extract_embeddings.py` 提取所有embedding
5. **标签体检**：运行 `scripts/audit_labels.py` 执行标签体检
6. **模型评估**：运行 `scripts/evaluate_model.py` 评估模型性能

## 配置说明

配置文件位于 `configs/` 目录：
- `data_config.yaml`: 数据路径和类别配置
- `ssl_config.yaml`: 自监督训练配置
- `supcon_config.yaml`: 对比学习配置

## 支持的Backbone

- ViT-B/16 (Vision Transformer)
- ResNet50
- ConvNeXt-T/B

可在配置文件中选择backbone类型。

