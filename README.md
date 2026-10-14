# WF-PitSeg

垂体内镜手术视频的离线步骤分割：LemonFM / ConvNeXt-Large 视觉特征、全视频 MS-TCN、器械辅助监督、成对边界监督与工作流约束 Viterbi 解码。

## 安装

Python 3.9+；默认训练配置使用 CUDA 和 BF16。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据准备

将 [LemonFM 预训练权重](https://huggingface.co/visurg/LemonFM)保存为 `weights/LemonFM.pth`，按以下结构准备帧图像和对应标注：

```text
data/PitVis/
├── frames_224/
│   ├── video_01/
│   └── ...
└── annotations/
    ├── annotations_01.csv
    └── ...
```

标注列包括 `int_time`、`int_step`、`int_instrument1` 和 `int_instrument2`。步骤与器械名称见 `data/label_maps/`。

## 运行

在仓库根目录执行：

```bash
export IMAGES_ROOT=data/PitVis/frames_224
export CSV_ROOT=data/PitVis/annotations
export LEMONFM_WEIGHTS=weights/LemonFM.pth
export GPU_IDS=0
export GPU_ID=0

# 1. 微调视觉主干
bash scripts/train_backbone.sh

# 2. 提取逐帧特征
export BACKBONE_CHECKPOINT=outputs/vision/backbone_finetune/best_step_metric.pt
export FEATURES_ROOT=outputs/features_finetuned
bash scripts/extract_features.sh

# 3. 训练时序模型
bash scripts/train_main.sh

# 4. 评估
export CHECKPOINT=outputs/main/seed_42/best_step_metric.pt
bash scripts/evaluate_main.sh
```

评估输出位于 `outputs/evaluation/`，包含指标 JSON、指标 CSV 和逐帧预测 CSV。

## 主要代码

| 文件 | 功能 |
|---|---|
| `pitvis_mainline_model.py` | 视觉主干与时序模型 |
| `pitvis_mainline_data.py` | 数据加载与监督目标生成 |
| `pitvis_mainline_train.py` | 损失函数与训练循环 |
| `train_pitvis_mainline.py` | 训练入口 |
| `extract_lemonfm_features.py` | 特征提取 |
| `ensemble_eval.py` | checkpoint 评估入口 |
| `pitvis_mainline_eval.py` | 解码与指标计算 |
| `pitvis_transition_policy.py` | 工作流转移约束 |
| `configs/` | 视觉微调与时序训练配置 |

视觉预训练模型来自 [LEMON / LemonFM](https://github.com/visurg-ai/LEMON)。
