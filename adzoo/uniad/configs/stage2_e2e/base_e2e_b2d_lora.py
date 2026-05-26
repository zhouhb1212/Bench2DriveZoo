# ---------------------------------------------------------------------------------#
# Occupancy-Planning Coupled LoRA 配置文件
#
# 基于 base_e2e_b2d.py，增加 LoRA 微调相关配置。
#
# 三阶段训练：
#   Stage 1 (training_stage=1): 仅 OccHead LoRA，lr=1e-3，epochs=1
#   Stage 2 (training_stage=2): 仅 PlanningHead LoRA，lr=1e-3
#   Stage 3 (training_stage=3): 两者联合 + 一致性损失，lr=5e-4
#
# 用法：修改下方 training_stage 和 optimizer.lr 后运行相同命令
# ---------------------------------------------------------------------------------#

_base_ = ["./base_e2e_b2d.py"]

# 预训练权重（包含 Stage1 Track+Map + Stage2 E2E 完整权重）
load_from = "ckpts/uniad_base_b2d.pth"

# ── Occupancy-Planning Coupled LoRA 配置 ──
model = dict(
    coupled_lora_cfg=dict(
        r=2,
        alpha=4,
        dropout=0.1,
        pretrained_path="ckpts/uniad_base_b2d.pth",
        training_stage=1,  # 切换阶段：1 / 2 / 3
    ),
)

# ── 优化器（仅 LoRA 参数 requires_grad=True，其余已冻结）──
optimizer = dict(
    type="AdamW",
    lr=3e-4,  # LoRA 学习率低于全量微调 (2e-4)，但 scale=alpha/r 有放大效应
    weight_decay=0.01,
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
            ratio=3,                                 # 额外复制轮数（总出现 = 1+ratio 次）
            max_other_frames=5000,                   # 其他场景总帧数上限（所有场景全覆盖）
            seed=42,
        ),
    ),
)

total_epochs = 1
runner = dict(type="EpochBasedRunner", max_epochs=1)

# ── 减少 IO 频率（Stage 1 训练步数少，降低 checkpoint/日志/评估开销）──
checkpoint_config = dict(interval=6000, by_epoch=False)
evaluation = dict(interval=6)
log_config = dict(
    interval=200,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)
