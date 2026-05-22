# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 环境安装

```bash
conda create -n b2d_zoo python=3.8
conda activate b2d_zoo
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ninja packaging
pip install -v -e .
```

## 预训练权重

所有权重在 Hugging Face (https://huggingface.co/rethinklab/Bench2DriveZoo) 和百度云提供。

| 用途 | 文件 | 说明 |
|------|------|------|
| 基础权重 | `ckpts/resnet50-19c8e357.pth` | VAD 图像 backbone |
| 基础权重 | `ckpts/r101_dcn_fcos3d_pretrain.pth` | BEVFormer 预训练 |
| 模型权重 | `ckpts/bevformer_tiny_b2d.pth` | BEVFormer-Tiny |
| 模型权重 | `ckpts/bevformer_base_b2d.pth` | BEVFormer-Base |
| 模型权重 | `ckpts/uniad_tiny_b2d.pth` | UniAD-Tiny (L2=0.80m) |
| 模型权重 | `ckpts/uniad_base_b2d.pth` | UniAD-Base (L2=0.73m) |
| 模型权重 | `ckpts/vad_b2d_base.pth` | VAD-Base (L2=0.91m) |

## 数据准备

```bash
cd mmcv/datasets
python prepare_B2D.py --workers 16
```

生成 `data/infos/b2d_infos_train.pkl`, `b2d_infos_val.pkl`, `b2d_map_infos.pkl`。

## 训练方法论

### BEVFormer
端到端 BEV 检测模型，单阶段训练和开环评测。

### UniAD
两阶段训练：Stage 1 训练轨迹跟踪与地图构建，Stage 2 进行端到端联合训练。开环评测需加载两个阶段的权重。

### VAD
端到端向量轨迹预测，单阶段训练。训练时需指定预训练权重路径。

### 共性问题
- 所有模型通过 `adzoo/<model>/dist_train.sh <config> <gpus>` 训练，通过 `dist_test.sh` 开环评测
- 分布式训练由 `torchrun` / `torch.distributed.launch` 启动
- 配置文件在 `adzoo/<model>/configs/` 下
- 日志输出到 `work_dirs/`

## 闭环评测方法论 (CARLA)

1. 克隆 Bench2Drive 评估工具仓库
2. 将 `team_code/` 链接到评估工具的 `leaderboard/team_code/`
3. 设置 `CARLA_ROOT` 环境变量
4. 使用 CARLA leaderboard 框架运行评估

---

## 代码架构

```
adzoo/                    # 三个自动驾驶模型的实现
├── bevformer/           # BEVFormer: BEV检测模型
├── uniad/               # UniAD
│   ├── configs/
│   │   ├── stage1_track_map/
│   │   └── stage2_e2e/
│   ├── train.py
│   └── test.py
└── vad/                 # VAD

mmcv/                    # 合并的 OpenMMLab 基础库 (v0.17.1)
    └── models/
        ├── detectors/
        │   └── uniad_e2e.py            # UniAD 模型类 [有新增代码]
        └── dense_heads/
            ├── occ_head.py             # 占用预测头 [有新增代码]
            ├── planning_head.py        # 规划头 [有新增代码]
            └── planning_head_plugin/
                ├── lora_adapter.py              # LoRA 基础组件 [有新增代码]
                ├── occ_plan_coupled_lora.py     # [新文件] 双 LoRA 联动管理器
                ├── collision_optimization.py    # 碰撞优化
                ├── planning_metrics.py          # 规划指标
                ├── metric_stp3.py               # STP3指标
                └── __init__.py                  # [有新增代码] 导出 LoRA 组件

team_code/              # CARLA Agent 实现
analysis/               # 结果分析工具
```

---

## 重要说明

- **本地验证（必须遵守）**: 每次编辑代码后，必须在本地验证代码可行性（Python 语法检查、import 测试、简单前向推理等）。本地 GPU 无法训练模型，实际训练/测试在云端服务器进行，因此本地只需保证代码层面能跑通即可。
- **L2 计算差异**: UniAD 在每个时间步计算 L2；VAD 计算时间步内的平均 L2
- **Smoothness/Efficiency**: 闭环评测时需在 agent 中实现 `self.metric_info` 相关代码

---

## Occupancy-Planning Coupled LoRA（双 LoRA 联动微调）[新增功能]

### 概述

低算力友好的参数高效微调方案。仅在 **OccHead Transformer Decoder** 和 **PlanningHeadSingleMode adapter 路径** 中注入 LoRA（默认 rank=4），通过三阶段训练使 Occupancy 和 Planning 形成闭环联动。

### 新文件

**`adzoo/uniad/configs/stage2_e2e/base_e2e_b2d_lora.py`**
- LoRA 训练专用配置，继承 `base_e2e_b2d.py`
- 新增 `coupled_lora_cfg`（含 `pretrained_path`、`training_stage` 等）、`load_from`
- `training_stage` 和 `lr` 可通过命令行 `--training-stage` / `--lr` 覆盖

**`mmcv/models/dense_heads/planning_head_plugin/occ_plan_coupled_lora.py`**
- `OccPlanCoupledLoRA` 管理器类
- 职责：向 OccHead/PlanningHead 注入 LoRA、管理三阶段 freeze/unfreeze、计算 occupancy-planning 一致性损失（碰撞惩罚 + 特征对齐）

### 有新增代码的原有文件

| 文件 | 新增内容 |
|------|---------|
| `lora_adapter.py` | `LoRAMultiheadAttention` 类（对 `nn.MultiheadAttention` 的 Q/K/V/Out proj 注入 LoRA）；`inject_lora_to_attention()`、`inject_lora_to_linear()`、`inject_lora_to_ffn()`、`collect_lora_params()`、`freeze_all_except_lora()` |
| `occ_head.py` | `forward(return_risk_features=False)` 参数 + 返回 `occ_risk_feat [B,T,C,H,W]` / `occ_risk_mask [B,T,1,H,W]` |
| `planning_head.py` | `forward_train(occ_risk_feat=None, occ_risk_mask=None)` 参数；`forward(return_plan_feat=False)` 参数 + 返回 `plan_feat [1,1,256]` |
| `uniad_e2e.py` | `__init__(coupled_lora_cfg=None)` 参数 + LoRA 管理器初始化；`forward_train` 中三阶段条件逻辑 + Stage 3 一致性损失 |
| `train.py` | `--training-stage`（1/2/3）和 `--lr` 命令行参数，覆盖配置文件值 |
| `__init__.py` | 导出 LoRA 组件 + `OccPlanCoupledLoRA` |

### LoRA 注入位置

**OccHead**（每个 transformer decoder layer，共 5 层）:
```
self_attn.attn  →  LoRAMultiheadAttention  (Q/K/V/Out proj)
cross_attn.attn →  LoRAMultiheadAttention  (Q/K/V/Out proj)
FFN w1 (layers[0][0]) → LoRALinear       (256→2048)
FFN w2 (layers[1])    → LoRALinear       (2048→256)
```

**PlanningHeadSingleMode**:
```
mlp_fuser[0]       → LoRALinear  (768→256)
reg_branch[0],[2]  → LoRALinear  (256→256, 256→12)
attn_module 每层: self_attn / multihead_attn → LoRAMultiheadAttention
                  linear1 / linear2           → LoRALinear
bev_adapter        → 冻结不注入
```

### 数据流（Stage 3 联合训练）
```
BEV Features ──→ OccHead Transformer Decoder (LoRA)
                    ├── occ_logits → L_occ
                    ├── occ_risk_feat [B,T,C,H,W] ─────┐
                    │                                    ↓ Feature Align Loss
                    └── occ_risk_mask [B,T,1,H,W] ──────┐
                                                         ↓ Collision Penalty
BEV Features ──→ PlanningHead Adapter (LoRA)
                    ├── plan_feat [1,1,256] ────────────┘
                    └── sdc_traj [1,6,2] ───────────────┘
                                                        ↓
                                          L_consistency = λ_c*L_coll + λ_a*L_align
```

### 三阶段训练策略

| Stage | 可训练参数 | 损失函数 | 推荐学习率 |
|-------|-----------|---------|-----------|
| 1 | 仅 OccHead LoRA | L_occ (dice + mask + aux) | 1e-3 |
| 2 | 仅 PlanningHead LoRA | L_plan (ade + collision×3) | 1e-3 |
| 3 | 两者 LoRA | L_occ + L_plan + λ·L_consistency | 5e-4 |

切换方式：**命令行传参 `--training-stage` 和 `--lr`**，无需修改配置文件。

### 开环训练命令

```bash
# Stage 1: OccHead LoRA 训练
./adzoo/uniad/uniad_dist_train.sh ./adzoo/uniad/configs/stage2_e2e/base_e2e_b2d_lora.py 1 \
    --training-stage 1 --lr 1e-3

# Stage 2: PlanningHead LoRA 训练
./adzoo/uniad/uniad_dist_train.sh ./adzoo/uniad/configs/stage2_e2e/base_e2e_b2d_lora.py 1 \
    --training-stage 2 --lr 1e-3

# Stage 3: 联合微调
./adzoo/uniad/uniad_dist_train.sh ./adzoo/uniad/configs/stage2_e2e/base_e2e_b2d_lora.py 1 \
    --training-stage 3 --lr 5e-4
```

> `--training-stage` 和 `--lr` 会覆盖配置文件中的值。预训练权重通过 `coupled_lora_cfg.pretrained_path` 在 LoRA 注入前自动加载到 occ_head/planning_head 基座层，`load_from` 负责加载其余模块。

### 开环评测命令

```bash
# 加载 Stage 3 训练好的 checkpoint（内部通过 load_from 加载预训练权重）
./adzoo/uniad/uniad_dist_eval.sh ./adzoo/uniad/configs/stage2_e2e/base_e2e_b2d_lora.py \
    ./work_dirs/stage3_ckpt.pth 1
```

> 评测前可调用 `model.coupled_lora.merge_weights_for_inference()` 将 LoRA 权重合并到基座，加速推理。

### 闭环评测命令 (CARLA)

```bash
# 1. 克隆 Bench2Drive 评估工具
git clone https://github.com/Thinklab-SJTU/Bench2Drive
cd Bench2Drive/leaderboard
mkdir team_code
ln -s ../../Bench2DriveZoo/team_code/* ./team_code
cd ..
ln -s ../Bench2DriveZoo ./

# 2. 设置 CARLA 环境
export CARLA_ROOT=<CARLA安装路径>

# 3. 运行闭环评估（加载 Stage 3 训练好的 checkpoint）
export CUDA_VISIBLE_DEVICES=0
python leaderboard/leaderboard/leaderboard_evaluator.py \
    --agent=team_code/uniad_b2d_agent.py \
    --checkpoint=./work_dirs/stage3_ckpt.pth \
    --routes=leaderboard/data/routes/closed_loop.xml \
    --scenarios=leaderboard/data/scenarios/all_towns_traffic_scenarios.json
```

### 关键超参数（`coupled_lora_cfg` dict）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `r` | 4 | LoRA rank |
| `alpha` | 8 | 缩放因子 (scale=alpha/r=2) |
| `dropout` | 0.1 | LoRA dropout |
| `lambda_collision` | 1.0 | 碰撞惩罚权重 |
| `lambda_align` | 0.1 | 特征对齐权重 |
| `align_proj` | `nn.Linear(64,256)` | occ→plan 通道投影（自动创建，Stage 3 可训练）|
| `pretrained_path` | `ckpts/uniad_base_b2d.pth` | 预训练权重路径（在 LoRA 注入前加载）|
| `training_stage` | 1 | 当前训练阶段 (1/2/3) |

### 向下兼容

不传 `coupled_lora_cfg` 或传 `None` 时，`coupled_lora = None`，行为与原版 UniAD 完全一致。
