# Mask-aware 自监督训练方案

## 1. 目标与边界

本方案在不信任缺陷类别标签的前提下，把所有已裁剪样本合并到一个全局样本池中，训练可用于聚类、检索和异常分析的通用缺陷表征。

- 类别名、scene/source 不参与训练 loss。
- 唯一确定的正关系是同一实例的两个保语义增强视图。
- 不构造样本间负关系，避免把同类、同大类或模糊缺陷错误推开。
- 原 SupCon/MoCo 训练入口、配置、模块和 checkpoint 保持不变，继续作为强监督 baseline。
- 新框架只共享项目现有的 DINOv3 ConvNeXt/ViT backbone 实现。

## 2. 对比实验矩阵

正式结论至少包含以下实验：

| ID | Backbone | 训练目标 | 用途 |
|---|---|---|---|
| B0-C | ConvNeXt Base | 不训练 | 预训练零样本基线 |
| B0-V | ViT Base/16 | 不训练 | 预训练零样本基线 |
| B1 | ConvNeXt Base | 原 SupCon/MoCo | 强监督基线 |
| A1 | ConvNeXt Base | VICReg | 第一主线 |
| A2 | ConvNeXt Base | DINO-style | teacher-student 对照 |
| A3 | ViT Base/16 | VICReg | 架构对照 |
| A4 | ViT Base/16 | DINO-style | 完整对照，按前三项结果决定是否长训 |

首轮顺序：`B0-C → A1 → A2 → B0-V → A3 → A4`。原 SupCon 使用已有训练结果或在相同 backbone、输入尺寸和数据版本下重新运行。

## 3. 数据与采样

### 3.1 全局样本池

读取 `data_config.yaml` 的 `scenes.train`，递归扫描每个 root 下的 PNG。每张图必须有同目录、同 stem 的 `_mask.png`。所有合法 crop 合并后按样本均匀采样：

```text
all valid crops → uniform shuffle/sample → batch
```

不按 scene、source 或类别加权。重复配置的同一绝对路径会去重。source 和类别名仅保留为诊断元数据。

### 3.2 双视图增强

第一轮 VICReg 和 DINO 使用完全相同的两个全局视图：

- 图像与 Mask 同步的小角度仿射、平移、缩放和可选水平翻转；
- 仅图像使用轻度颜色扰动和低概率模糊；
- 增强后 Mask 有效像素少于阈值时重试；
- 不使用 random resized crop、solarization、copy-paste 和强模糊。

任何增强策略调整前，都应先抽样可视化原图、两个视图和对应 Mask。

## 4. 模型架构

### 4.1 ConvNeXt Base（默认）

```text
image
  → DINOv3 ConvNeXt Base stages [1,2,3]
  → 各 stage 的 foreground/context/delta Mask pooling
  → stage projection + learned gate + fusion
  → 768-d representation
  → objective-specific head
```

ConvNeXt 的 H/8、H/16、H/32 多尺度特征更适合细小、线状和纹理型工业缺陷，因此作为第一主线。

### 4.2 ViT Base/16

```text
image
  → DINOv3 ViT Base/16
  → 14×14 patch tokens
  → Mask-weighted patch pooling
  → 768-d representation
  → objective-specific head
```

首轮 `cls_weight=0`，避免全局 CLS 弱化 Mask 聚焦。ViT 的 patch token 语义结构可能更强，但 14×14 对微小 Mask 较粗，作为必要对照而非默认模型。

## 5. VICReg 训练

### 5.1 Loss

```text
L = 25 × invariance + 25 × variance + 1 × covariance
```

- invariance：同一实例的两个视图输出一致；
- variance：每个维度保持足够标准差，防止常量输出；
- covariance：降低维度间冗余。

VICReg projection 不做 L2 归一化。DDP 时 variance/covariance 使用带梯度的全局 gather，统计有效 batch 为 `batch_per_gpu × world_size`。

### 5.2 初始超参数

| 参数 | 初值 |
|---|---:|
| total steps | 120,000 |
| batch/GPU | 64 |
| representation dim | 768 |
| projection dim | 1,024 |
| head LR | 1e-4 |
| backbone LR | 5e-6 |
| weight decay | 0.05 |
| warmup | 5,000 steps |
| backbone freeze | 5,000 steps |
| gradient clip | 2.0 |

重点监控 `invariance_loss`、`variance_loss`、`covariance_loss`、`average_std`、增强一致性和 representation effective rank。

## 6. DINO-style 训练

### 6.1 结构

student 和 teacher 都包含 backbone、Mask aggregator、projection head 和 prototype layer。teacher 不参与反向传播，在每次成功 optimizer step 后由 student 的 EMA 更新。

两个视图交叉蒸馏：

```text
student(view1) ← teacher(view2)
student(view2) ← teacher(view1)
```

不使用 MoCo queue，也不使用样本间负关系。

### 6.2 初始超参数

| 参数 | 初值 |
|---|---:|
| prototypes | 4,096 |
| bottleneck dim | 256 |
| student temperature | 0.1 |
| teacher temperature | 前 10k steps 从 0.04 到 0.07 |
| teacher EMA momentum | cosine 0.996 到 1.0 |
| center momentum | 0.9 |
| 其余优化参数 | 与 VICReg 相同 |

重点监控 teacher entropy、prototype usage entropy、最大 prototype 占比、增强一致性和 representation effective rank。首轮只使用双全局视图，暂不使用 DINO multi-crop。

## 7. 分阶段训练

### 阶段 0：数据审计与零训练基线

1. 检查缺图、缺 Mask、空 Mask、过小 Mask 和重复路径。
2. 可视化双视图增强，确认缺陷语义没有被破坏。
3. 分别用未微调的 ConvNeXt Base 和 ViT Base 提取 Mask-aware representation。
4. 保存检索、聚类和 source 偏置指标。

### 阶段 1：稳定新模块（0～5k steps）

- 冻结 backbone；
- 只训练 Mask aggregator 和 objective head；
- warmup 学习率；
- 检查 loss、方差、effective rank、prototype 使用率和 NaN。

如果本阶段出现坍缩，不进入下一阶段，先修正增强、batch 或 loss。

### 阶段 2：低学习率微调 backbone（5k～120k steps）

- 解冻完整 backbone；
- backbone LR 为 head LR 的 0.05；
- cosine 衰减到初始 LR 的 0.01；
- 定期保存可恢复 checkpoint 和推理 encoder。

若训练 loss 改善但下游检索持续下降，优先把 backbone LR ratio 降至 0.01～0.03，而不是提高自监督 loss 权重。

### 阶段 3：checkpoint 选择

不能只按训练 loss 选模型。综合以下指标：

- 同一实例多增强一致性；
- representation effective rank；
- 类别名 kNN/mAP（只评估，不反传）；
- 类内离群点比例；
- 跨 source 同名类别检索；
- 人工抽查 top-k 近邻；
- DINO prototype 使用是否均衡。

DINO 默认选择 teacher encoder 作为推理模型；VICReg 使用 online/shared encoder。

### 阶段 4：可选弱监督

第一轮实现不启用。完成标签审计后，才考虑加入：

```text
self-supervised loss + λ × reliable-pair loss
```

可靠正样本需同时满足标签一致、mutual kNN、多增强稳定和跨 checkpoint 稳定。仅标签名相同不能直接成为强正样本。

## 8. 调参顺序

### VICReg

1. 先保证 `average_std` 不长期接近 0、effective rank 不持续下降。
2. 坍缩时先增大 batch 或 variance weight。
3. 两视图不一致时减弱增强，而不是立即增大 invariance weight。
4. 表征稳定但下游无提升时，再调整 backbone LR、Mask pooling 或训练时长。

### DINO

1. prototype 被少数槽位垄断时，检查 center、teacher temperature 和增强。
2. teacher 分布过尖时提高 teacher temperature 或延长温度 warmup。
3. teacher 变化过快时提高初始 momentum；学习停滞时可略降初始 momentum。
4. 双全局视图稳定后，才考虑加入 Mask-aware local crop。

## 9. 公平比较原则

- 固定数据版本、Mask、输入尺寸和增强；
- 固定 backbone 初始化；
- 比较相近的有效 batch 和训练图像总数；
- 下游统一使用 L2-normalized representation；
- 同时报告 macro 与整体指标；
- SupCon、VICReg、DINO 都在同一评估集和检索协议下比较；
- 必须报告“预训练但不微调”的结果，防止把表征退化误判为训练收益。

## 10. 运行方式

推荐直接使用项目根目录脚本：

```bash
./run_selfsup_training.sh
```

多 GPU 与恢复训练：

```bash
NPROC_PER_NODE=4 ./run_selfsup_training.sh

RESUME=checkpoints/selfsup/convnext_base_vicreg/current_checkpoint.pth \
  ./run_selfsup_training.sh
```

也可以通过 `CONFIG` 指定其他配置文件。

训练前生成增强审计图：

```bash
conda run -n hjh python scripts_training/inspect_selfsup_dataset.py \
  --config configs/selfsup_config.yaml \
  --output /tmp/selfsup_augmentation_audit.png
```

```bash
conda run -n hjh python scripts_training/train_selfsup.py \
  --config configs/selfsup_config.yaml
```

多 GPU：

```bash
conda run -n hjh torchrun --nproc_per_node=4 \
  scripts_training/train_selfsup.py \
  --config configs/selfsup_config.yaml
```

恢复训练：

```bash
conda run -n hjh python scripts_training/train_selfsup.py \
  --config configs/selfsup_config.yaml \
  --resume checkpoints/selfsup/convnext_base_vicreg/current_checkpoint.pth
```

使用弱标签和 source 做离线诊断（这些信息不会进入训练 loss）：

```bash
conda run -n hjh python scripts_training/eval_selfsup.py \
  --checkpoint checkpoints/selfsup/convnext_base_vicreg/inference_model.pth \
  --split val \
  --output /tmp/selfsup_eval.json
```

从 VICReg 切换 DINO 时，修改：

```yaml
selfsup:
  method: dino
```

默认输出路径使用 `{backbone}_{method}` 模板，因此切换方法或 backbone 不会覆盖另一组实验。

切换 ViT Base 时修改 backbone 即可，预训练权重会按名称自动定位：

```yaml
selfsup:
  model:
    backbone: vitb16
```

不要在不同 method 或 backbone 之间复用 resume checkpoint。
