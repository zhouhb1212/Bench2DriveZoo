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
        lambda_collision=1.0,
        lambda_align=0.1,
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

# LoRA 三阶段均无未使用参数，关闭 DDP unused 检测以消除遍历开销
find_unused_parameters = False

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
