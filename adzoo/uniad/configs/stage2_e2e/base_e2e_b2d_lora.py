# ---------------------------------------------------------------------------------#
# Occupancy-Planning Coupled LoRA 配置文件
#
# 基于 base_e2e_b2d.py，增加 LoRA 微调相关配置。
#
# 三阶段训练（必须按顺序执行）：
#   Stage 1 (training_stage=1): 仅 OccHead LoRA，lr=2e-4，epochs=1~2
#   Stage 2 (training_stage=2): 仅 PlanningHead LoRA，lr=2e-4，resume Stage1 ckpt
#   Stage 3 (training_stage=3): 两者联合微调，lr=5e-5，resume Stage2 ckpt
#
# 用法：通过命令行 --training-stage 和 --lr 覆盖
# ---------------------------------------------------------------------------------#

_base_ = ["./base_e2e_b2d.py"]

# 预训练权重（包含 Stage1 Track+Map + Stage2 E2E 完整权重）
load_from = "ckpts/uniad_base_b2d.pth"

# ── Occupancy-Planning Coupled LoRA 配置 ──
model = dict(
    coupled_lora_cfg=dict(
        r=8,
        alpha=8,       # scale = alpha/r = 1，降低 LoRA 输出放大系数，减少对预训练特征的扰动
        dropout=0.05,  # 降低 dropout，减少随机性带来的梯度噪声
        pretrained_path="ckpts/uniad_base_b2d.pth",
        training_stage=1,  # 切换阶段：1 / 2 / 3
    ),
    # Stage 3 联合训练时降低 planning 权重，避免 collision loss 主导梯度
    # collision_0/1/2 数值远大于 occ dice/mask loss，需要平衡
    task_loss_weight=dict(
        track=1.0,
        map=1.0,
        motion=1.0,
        occ=1.0,
        planning=0.5,  # 原始 1.0 → 0.5，平衡 occ/planning 梯度贡献
    ),
)

# ── 优化器（仅 LoRA 参数 requires_grad=True，其余已冻结）──
optimizer = dict(
    type="AdamW",
    lr=3e-4,      # 梯度累积 4 步 → 等效 batch=4，按线性缩放 lr（2e-4 × 2）
    weight_decay=0.05,  # 增大正则化，防止 LoRA 参数过大振荡
)

# Stage 3 联合训练时，部分 LoRA 参数通过不同梯度路径参与 loss 计算，
# DDP 需要检测 unused 参数以避免 allreduce 时梯度缓冲区未填充的错误
find_unused_parameters = False

# ── 过采样配置：针对特定场景做场景级过采样（LoRA 快速验证用）──
data = dict(
    train=dict(
        oversample_cfg=dict(
            enable=True,                            # True 时启用
            scenarios=["ParkedObstacleTwoWays"],     # 要过采样的场景
            ratio=1,                                 # 额外复制轮数（总出现 = 1+ratio 次）
            max_other_frames=15000,                  # 其他场景总帧数上限（所有场景全覆盖）
            seed=42,
        ),
    ),
)

total_epochs = 2
runner = dict(type="EpochBasedRunner", max_epochs=2)

# ── Checkpoint 和验证频率 ──
# 总 iter 约 6600（1 epoch），checkpoint 每 3000 iter 保存一次
checkpoint_config = dict(
    interval=3000,
    by_epoch=False,
    # 文件命名：iter_3000.pth, iter_6000.pth, ...
    # epoch 结束时额外保存 epoch_1.pth
)
# 验证每 epoch 结束时执行一次（by_epoch=True, interval=1）
evaluation = dict(interval=1, by_epoch=True)
log_config = dict(
    interval=200,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)

# ── AMP 混合精度 + 梯度累积 ──
# 累积 4 个 batch 的梯度再更新，等效 batch_size=4
# 减少因单场景梯度噪声导致的严重震荡（理论方差降为 1/4）
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook',
    cumulative_iters=4,  # 每 4 个 iter 更新一次参数
    grad_clip=dict(max_norm=10, norm_type=2),
)

# ── 学习率调度 ──
lr_config = dict(
    by_epoch=False,
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,   # warmup 步数
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-2,  # 最终 lr 衰减到 peak 的 1%，确保后期稳定收敛
)
