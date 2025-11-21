# MVTec AD数据集转换和使用指南

## 数据集转换

MVTec AD数据集需要先转换为适配代码的格式。

### 1. 运行转换脚本

```bash
python scripts/convert_mvtec_ad.py \
    --mvtec_root data/mvtec_anomaly_detection \
    --output_root data/raw \
    --include_good \
    --no_empty_masks  # 可选：如果不为good样本创建空mask
```

参数说明：
- `--mvtec_root`: MVTec AD数据集根目录
- `--output_root`: 转换后数据的输出目录
- `--include_good`: 是否包含正常样本（good）
- `--no_empty_masks`: 不为正常样本创建空mask（如果设置，good样本将被跳过）

### 2. 转换后的目录结构

转换后的数据将按以下结构组织：

```
data/raw/
├── images/           # 图像文件
│   ├── bottle/
│   │   ├── broken_large/
│   │   ├── broken_small/
│   │   ├── contamination/
│   │   └── good/
│   ├── cable/
│   └── ...
├── masks/            # Mask文件
│   ├── bottle/
│   │   ├── broken_large/
│   │   ├── broken_small/
│   │   ├── contamination/
│   │   └── good/
│   ├── cable/
│   └── ...
└── metadata.json     # 元数据文件
```

### 3. 数据准备流程

转换完成后，运行数据准备脚本：

```bash
python scripts/prepare_data.py --data_config configs/data_config.yaml
```

这将：
1. 使用转换后的元数据（如果存在）
2. 提取所有缺陷patch
3. 构建SSL和SupCon数据集

## MVTec AD数据集说明

MVTec AD数据集包含15个工业场景类别：
- bottle（瓶子）
- cable（电缆）
- capsule（胶囊）
- carpet（地毯）
- grid（网格）
- hazelnut（榛子）
- leather（皮革）
- metal_nut（金属螺母）
- pill（药丸）
- screw（螺丝）
- tile（瓷砖）
- toothbrush（牙刷）
- transistor（晶体管）
- wood（木材）
- zipper（拉链）

每个类别包含：
- `train/good/`: 正常训练样本
- `test/good/`: 正常测试样本
- `test/<defect_type>/`: 各类缺陷测试样本
- `ground_truth/<defect_type>/`: 对应的缺陷mask

## 标签说明

转换后的标签包括：
- 各类缺陷类型（如 `broken_large`, `broken_small`, `contamination` 等）
- `good`: 正常样本（如果包含）

每个类别（如bottle, cable等）作为一个独立的domain_id。

## 注意事项

1. **正常样本处理**：
   - 如果使用 `--include_good`，正常样本会被包含，并创建空mask
   - 如果使用 `--no_empty_masks`，正常样本将被跳过（因为没有真实mask）

2. **数据过滤**：
   - 转换后，可以通过 `prepare_data.py` 的过滤功能移除样本数过少的类别
   - 配置在 `configs/data_config.yaml` 中的 `filtering` 部分

3. **自监督训练**：
   - 所有图像（包括正常和缺陷）都可以用于自监督预训练
   - 只有有mask的缺陷样本用于监督对比学习

