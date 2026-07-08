# 工业缺陷通用 Embedding 模型 & Data Cluster 平台

基于监督对比学习（Supervised Contrastive Learning）的工业缺陷表征模型，并提供 **Data Cluster** Web 平台，用于数据集管理、Mask 标注、模型微调训练与推理分析。

## 核心能力

| 模块 | 能力 |
|------|------|
| **算法库** (`embedding_model`) | DINOv3 ConvNeXt / ViT backbone、Mask 特征筛选、多层 FPN 融合、SupCon 对比学习 |
| **Web 平台** (`data_cluster`) | 数据集上传与标注、实验训练、实时训练监控、推理降维与异常排序 |
| **标注** | 多边形绘制、魔术棒、缩放/平移/撤销、连续切图标注 |
| **训练** | 默认加载预训练 backbone；Web 端简化 SupCon 微调；至少 2 类、每类至少 2 张样本 |
| **推理** | t-SNE / UMAP / PCA 可视化；可选 Golden 参考样本锚定异常评分 |

## 架构说明

项目分为两层，职责清晰：

```
embedding_model/     # 纯算法库：模型、损失、数据集、预处理、指标工具
    └── 不依赖 Web 平台，可被 scripts/ 与 data_cluster 共同引用

data_cluster/        # Web 业务层
    ├── app/         # FastAPI 路由、数据库、存储
    └── dl/          # 平台侧训练/推理/降维/裁剪入口（调用 embedding_model）
```

- **`scripts/train_supcon.py`**：独立命令行训练脚本（支持 DDP、MoCo 等完整能力），用于算法研发。
- **`data_cluster/dl/trainer.py`**：Web 平台训练入口，精简为可维护的 finetune 流程。
- **`data_cluster/dl/inference.py`**：推理 embedding 提取与 checkpoint 加载。
- **`data_cluster/dl/projection.py`**：t-SNE / UMAP / PCA 降维与异常评分。

## 项目结构

```
EmbeddingModel/
├── configs/
│   ├── data_cluster.yaml          # Web 平台默认配置（backbone、训练、存储）
│   ├── supcon_config.yaml         # 命令行 SupCon 训练配置
│   ├── data_config.yaml           # 命令行数据配置示例
│   └── backbone/                  # 各 backbone 架构参数
├── data/
│   └── data_cluster/
│       ├── app.db                 # SQLite 数据库（自动创建）
│       ├── datasets/              # 上传的原图、crop、mask
│       └── checkpoints/           # 实验 checkpoint
├── frontend/                      # Vue 3 + Vite + Tailwind 前端
│   └── src/pages/
│       ├── Datasets.vue           # 数据集与标注
│       ├── Experiments.vue        # 实验与训练
│       └── Inference.vue          # 推理分析
├── scripts/
│   ├── train_supcon.py            # 命令行 SupCon 训练（算法研发）
│   └── generate_geo_test_data.py  # 生成几何形状测试数据
├── src/
│   ├── embedding_model/           # 算法库
│   │   ├── supcon/                # 模型、损失、数据集
│   │   ├── preprocess/            # patch / polygon 裁剪
│   │   └── utils/                 # 配置、日志、指标、可视化
│   └── data_cluster/              # Web 平台
│       ├── app/                   # FastAPI + SQLAlchemy
│       └── dl/                    # 训练 / 推理 / 降维
├── run_backend.sh
├── pyproject.toml
└── requirements.txt
```

## 环境准备

**Python**（建议 3.10+，需 CUDA 环境用于 GPU 训练）：

```bash
conda create -n embedding python=3.10 -y
conda activate embedding

# 算法库依赖
pip install -r requirements.txt
pip install -e .

# Web 平台额外依赖
pip install fastapi uvicorn sqlalchemy pydantic-settings
```

**前端**（Node.js 18+）：

```bash
cd frontend
yarn install   # 或 npm install
```

**可选**（推理 UMAP 降维）：

```bash
pip install umap-learn
```

## 快速开始（Web 平台）

### 1. 启动后端

```bash
# 方式 A：直接运行（推荐，含 reload）
PYTHONPATH=src python src/data_cluster/app/main.py

# 方式 B：脚本
./run_backend.sh
```

默认监听 `http://0.0.0.0:8001`，API 文档见 `http://127.0.0.1:8001/docs`。

环境变量（可选，前缀 `DATA_CLUSTER_`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATA_CLUSTER_HOST` | `0.0.0.0` | 监听地址 |
| `DATA_CLUSTER_PORT` | `8001` | 端口 |
| `DATA_CLUSTER_DATABASE_URL` | `sqlite:///./data/data_cluster/app.db` | 数据库 |

### 2. 启动前端

```bash
cd frontend
yarn dev
```

访问 `http://localhost:3000`。开发模式下 Vite 会将 `/api` 与 `/static` 代理到后端 `8001` 端口。

### 3. 生成测试数据（可选）

```bash
python scripts/generate_geo_test_data.py
# 输出至 data/test_cluster_data/geo_data/{circle,rectangle,triangle}/
```

可在「数据集」页上传 ZIP 或单张图片进行标注与训练。

## 典型工作流

```
1. 数据集
   ├── ZIP 批量上传（原图 + mask）
   └── 单图上传 → 多边形 / 魔术棒标注 → 按区域指定缺陷类别 → 自动生成 crop

2. 实验训练
   ├── 选择数据集、类别与训练样本（Golden sample）
   ├── 约束：至少 2 个类别，每类至少 2 张 crop
   ├── 默认加载 configs/data_cluster.yaml 中的预训练 backbone
   └── 实时查看 loss / 特征相似度（Margin）曲线；重训失败时保留上次成功结果

3. 推理分析
   ├── 选择已有数据集（须含 mask 对应 crop）
   ├── 选择模型：默认预训练 或 已完成实验的 checkpoint
   ├── 可选 Golden 参考样本（锚定异常排序）
   └── t-SNE / UMAP / PCA 散点图 + 异常度网格
```

## 命令行训练（算法研发）

独立于 Web 平台，适合大规模实验与 DDP 训练：

```bash
# 单 GPU
python scripts/train_supcon.py \
  --config configs/supcon_config.yaml \
  --data_config configs/data_config.yaml

# 多 GPU
torchrun --nproc_per_node=4 scripts/train_supcon.py \
  --config configs/supcon_config.yaml \
  --data_config configs/data_config.yaml
```

数据目录要求：每个样本需有对应 mask（`{name}_mask.png` 或与图像同名的 mask 文件，详见 `configs/data_config.yaml`）。

## 配置

### Web 平台 — `configs/data_cluster.yaml`

主要配置项：

- **`backbones.*.pretrained_path`**：各 backbone 默认预训练权重（Web 训练默认加载，除非实验 config 显式覆盖 `pretrainedPath`）
- **`training`**：epochs、learning_rate、lr_scheduler 等默认训练参数
- **`model`**：embedding_dim、projection_head、FPN、use_layers 等架构参数
- **`output.log_dir`**：训练日志目录

### 命令行 — `configs/supcon_config.yaml` + `configs/data_config.yaml`

- 训练超参、增强策略、loss 权重
- 数据根目录、类别列表、类别间自定义相似度矩阵

## 模型架构（简要）

1. **Backbone**：DINOv3 ConvNeXt / ViT，提取多层特征
2. **Mask Pooling**：用 resize 后的 mask 筛选缺陷区域特征
3. **FPN + Fusion**：多层特征融合
4. **Projection Head**：映射到 embedding 空间
5. **SupCon Loss**：基于类别相似度矩阵的对比学习

训练监控指标（Web 端）：

- **Train / Val Loss**
- **特征相似度**：Pos Sim（同类）、Neg Sim（异类）、Margin = Pos − Neg

## 数据与存储

| 路径 | 内容 |
|------|------|
| `data/data_cluster/datasets/` | 上传原图、crop、mask 静态文件 |
| `data/data_cluster/checkpoints/` | 实验 checkpoint（按 model / experiment_id 组织） |
| `data/data_cluster/app.db` | SQLite：数据集、标注、实验、推理运行记录 |

静态资源通过 `/static/...` 访问；前端开发时由 Vite 代理到后端。

## 前端页面

| 路由 | 页面 | 功能 |
|------|------|------|
| `/datasets` | 数据集 | 创建/上传、crop 预览、多边形标注 |
| `/experiments` | 实验管理 | 创建实验、选择样本、启动/停止训练、结果 dashboard |
| `/inference` | 推理测试 | 模型选择、数据集分析、Golden 样本、降维可视化 |

技术栈：Vue 3、TypeScript、Vite、Tailwind CSS 4、ECharts、Lucide Icons。

## 开发

```bash
# 前端类型检查
cd frontend && yarn lint

# 前端构建
cd frontend && yarn build

# 后端（确保 PYTHONPATH=src）
PYTHONPATH=src python -c "import data_cluster.app.main"
```

## 注意事项

1. **Mask 必须存在**：训练和推理均依赖 mask 区域的 crop；单图上传后需完成标注。
2. **训练样本下限**：至少 2 个类别，每个类别至少 2 张 crop（前后端均校验）。
3. **预训练权重**：确认 `configs/data_cluster.yaml` 中 `pretrained_path` 指向有效 checkpoint；支持多种 checkpoint 格式（raw backbone / MoCo / 完整训练）。
4. **重训与结果分离**：实验的 `status` / `metrics` / `checkpoint_path` 保存**上一次成功结果**；`run_status` / `run_metrics` 记录**本次训练尝试**。重训进行中显示实时曲线；失败/中止后回退到上次成功页，模型仍可用于推理。
5. **UMAP**：未安装 `umap-learn` 时，推理页 UMAP 选项不可用，t-SNE / PCA 仍可用。

## 依赖概览

**算法**：torch、torchvision、timm、transformers、scikit-learn、faiss-cpu、opencv-python 等（见 `requirements.txt`）

**Web 后端**：fastapi、uvicorn、sqlalchemy、pydantic-settings

**Web 前端**：vue、vue-router、echarts、tailwindcss（见 `frontend/package.json`）
