# 工业缺陷通用Embedding模型

基于监督对比学习（Supervised Contrastive Learning）的工业缺陷表征模型，支持自定义类别相似度监督和mask特征筛选。

## 项目特点

- **DINOv3 Backbone**: 使用预训练的DINOv3 ConvNeXt作为特征提取器
- **Mask特征筛选**: 在backbone提取特征后，使用resize后的mask进行特征筛选
- **多层特征融合**: 应用多个层的feature，然后融合特征，最后进入对比网络模块
- **自定义相似度监督**: 支持通过配置文件自定义各个类别之间的相似度关系
- **灵活的参数冻结**: 训练时可以自由设置是否冻结backbone参数

## 项目结构

```
EmbeddingModel/
├── configs/                          # 配置文件目录
│   ├── data_config_zhenyu.yaml      # 数据配置（包含类别相似度设置）
│   └── supcon_config.yaml            # 训练配置
├── data/                             # 数据目录
│   └── zhenyu_data/
│       └── patches/
│           └── supcon_data_cleaned/  # 数据根目录
│               ├── 类别1/
│               │   ├── image1.png
│               │   ├── image1_mask.png
│               │   └── ...
│               └── 类别2/
│                   └── ...
├── src/                              # 源代码
│   └── supcon_train/
│       ├── datasets/
│       │   ├── supcon_dataset.py     # 数据集类（支持mask）
│       │   └── prepare_supcon_dataset.py
│       ├── models/
│       │   ├── supcon_model.py       # SupCon模型
│       │   ├── losses.py             # Loss函数（支持相似度矩阵）
│       │   └── backbone/
│       │       └── dinov3_convnext.py # DINOv3 backbone
│       └── train_supcon.py           # 训练脚本
├── scripts/                          # 执行脚本
└── requirements.txt                  # 依赖包
```

## 安装

```bash
pip install -r requirements.txt
# 或
pip install -e .
```

## 数据配置

数据配置文件 `configs/data_config_zhenyu.yaml` 包含以下内容：

- **root**: 数据根目录路径
- **categories**: 所有类别列表
- **default_similarity**: 默认类别相似度（不同类别之间）
- **custom_similarity**: 自定义相似度配置，可以设置特定类别对之间的相似度

示例配置：

```yaml
root: "data/zhenyu_data/patches/supcon_data_cleaned"
categories: ["笔迹", "变形1", "变形2", ...]
default_similarity: 0.0

custom_similarity:
 - list: ["变形1", "变形2"]
   similarity: 0.8
 - list: ["多装1", "多装2"]
   similarity: 0.4
```

## 数据格式

数据目录结构：

```
supcon_data_cleaned/
├── 类别1/
│   ├── image1.png          # 图像文件
│   ├── image1_mask.png     # 对应的mask文件（必须）
│   ├── image2.png
│   ├── image2_mask.png
│   └── ...
├── 类别2/
│   └── ...
```

**要求**：
- 每个图像文件必须有对应的mask文件（文件名格式：`{image_name}_mask.png`）
- mask文件为二值图像（0为背景，255为前景）

## 训练

### 1. 准备数据配置

编辑 `configs/data_config_zhenyu.yaml`，设置数据根目录和类别相似度。

### 2. 准备训练配置

编辑 `configs/supcon_config.yaml`，配置训练参数：

```yaml
supcon:
  model:
    model_name: "facebook/dinov3-convnext-small-pretrain-lvd1689m"
    embedding_dim: 128
    projection_head:
      hidden_dims: [256, 128]
    use_classification_head: false
    use_layers: [0, 1, 2, 3]  # 使用哪些层的特征
    fusion_dim: 512
  
  data:
    batch_size: 32
    samples_per_class: 2  # 每个batch中每个类别的样本数
    num_workers: 4
  
  training:
    epochs: 100
    learning_rate: 1e-4
    weight_decay: 0.05
    backbone_lr_ratio: 0.1  # backbone学习率比例
    lr_scheduler: "cosine"
  
  training_strategy:
    freeze_backbone_epochs: 0  # 冻结backbone的epoch数（0表示不冻结）
  
  loss:
    supcon:
      weight: 1.0
      temperature: 0.07
  
  augmentation:
    image_size: 224
    random_crop:
      enabled: true
      scale: [0.6, 1.0]
    horizontal_flip:
      enabled: true
      prob: 0.5
    color_jitter:
      enabled: true
```

### 3. 运行训练

```bash
python -m src.supcon_train.train_supcon \
    --config configs/supcon_config.yaml \
    --data_config configs/data_config_zhenyu.yaml
```

### 4. 恢复训练

```bash
python -m src.supcon_train.train_supcon \
    --config configs/supcon_config.yaml \
    --data_config configs/data_config_zhenyu.yaml \
    --resume checkpoints/supcon/checkpoint_epoch_10.pth
```

## 模型架构

### SupConModel

1. **DINOv3 Backbone**: 提取多层特征
2. **Mask特征筛选**: 使用resize后的mask对每层特征进行筛选
3. **特征融合**: 将多层筛选后的特征投影到统一维度并融合
4. **Projection Head**: 将融合特征映射到embedding空间

### Loss计算

使用自定义相似度矩阵进行监督对比学习：

- 根据配置文件中的相似度矩阵，确定哪些样本对是positive pairs
- 相似度大于阈值的类别对被视为positive pairs
- 计算对比学习loss，拉近positive pairs，推远negative pairs

## 主要功能

1. **自定义相似度监督**: 通过配置文件灵活设置类别相似度关系
2. **Mask特征筛选**: 只使用mask区域的特征，提高模型对缺陷区域的关注
3. **多层特征融合**: 利用backbone的多层特征，增强表征能力
4. **参数冻结控制**: 可以设置冻结backbone的epoch数，实现渐进式训练

## 输出

训练完成后，会在配置的输出目录生成：

- `best_model.pth`: 最佳模型权重
- `checkpoint_epoch_*.pth`: 定期保存的checkpoint
- `loss_curve.png`: 训练损失曲线

## 依赖

主要依赖包：

- torch
- torchvision
- transformers (用于DINOv3)
- numpy
- PIL
- tqdm
- yaml

## 注意事项

1. 确保每个图像都有对应的mask文件
2. mask文件必须是二值图像
3. 数据目录结构必须符合要求
4. 类别名称必须与配置文件中的categories一致
