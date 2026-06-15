# ---------------------------------------------------------------------------------#
# Occupancy-Planning Coupled LoRA 配置文件
#
# 基于 base_e2e_b2d.py，增加 LoRA 微调相关配置。
#
# 两阶段训练（必须按顺序执行）：
#   Stage 1 (training_stage=1): 仅 OccHead LoRA，lr=3e-4，epochs=2
#   Stage 2 (training_stage=2): 仅 PlanningHead LoRA，lr=3e-4，resume Stage1 ckpt
#
# 用法：通过命令行 --training-stage 和 --lr 覆盖
# ---------------------------------------------------------------------------------#

_base_ = ["./base_e2e_b2d.py"]

# 预训练权重（包含 Stage1 Track+Map + Stage2 E2E 完整权重）
load_from = "ckpts/uniad_base_b2d.pth"

# ── Occupancy-Planning Coupled LoRA 配置 ──
model = dict(
    # 关闭 OccHead aux loss 计算：aux 输出在 Transformer Decoder 之前，
    # LoRA 注入在 Decoder 内部，aux loss 无法获得有效梯度，仅为无用计算
    occ_head=dict(
        compute_aux_loss=False,
    ),
    coupled_lora_cfg=dict(
        r=16,
        alpha=16,        # scale = alpha/r = 1（全局默认值；per-head 配置优先）
        dropout=0.05,
        # Per-head LoRA 参数覆写（可选，不指定时回退到全局默认值）
        occ_lora=dict(r=16, alpha=32),       # Stage 1 OccHead: scale=alpha/r=2
        planning_lora=dict(r=8, alpha=8),    # Stage 2 PlanningHead 微调容量
        inject_q2o_feat=True,  # 向 query_to_occ_feat 注入 LoRA；False 用于消融/旧权重兼容
        pretrained_path="ckpts/uniad_base_b2d.pth",
        training_stage=1,  # 切换阶段：1 / 2
    ),
    task_loss_weight=dict(
        track=1.0,
        map=1.0,
        motion=1.0,
        occ=1.0,
        planning=2.0,
    ),
    planning_head=dict(
        loss_collision=[
            dict(type='CollisionLoss', delta=0.0, weight=0.1),   # 稀释碰撞权重以防止强力偏离专家轨迹 (方案2)
            dict(type='CollisionLoss', delta=0.5, weight=0.04),
            dict(type='CollisionLoss', delta=1.0, weight=0.01)
        ]
    ),
)

# ── 优化器（仅 LoRA 参数 requires_grad=True，其余已冻结）──
optimizer = dict(
    type="AdamW",
    lr=3e-5,      # 降低 LR 至 3e-5 以防止大梯度冲击和发散，使微调更平稳
    weight_decay=0.01,  # LoRA 参数少，0.01 避免衰减过强拉向零
)

# DDP 关闭 unused 参数检测，避免 allreduce 时梯度缓冲区未填充 of 错误
find_unused_parameters = False

# ── 验证集场景过滤配置 ──
# 设为 None 则对全量验证集进行评估。设为特定的场景列表（如 ["ParkedObstacleTwoWays"]）则仅对该子集进行评估。
eval_scenario_filter = ["ParkedObstacleTwoWays"] # 可选：None / ["ParkedObstacleTwoWays"]

# ── 过采样配置：针对特定场景做场景级过采样（LoRA 快速验证用）──
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        oversample_cfg=dict(
            enable=True,                            # True 时启用
            scenarios=["ParkedObstacleTwoWays"],     # 要过采样的场景
            ratio=1,                                 # 额外复制轮数（总出现 = 1+ratio 次）
            max_other_frames=85561,             # 其他场景总帧数上限
            seed=42,
        ),
    ),
    val=dict(
        scenario_filter=eval_scenario_filter,
    ),
    test=dict(
        scenario_filter=eval_scenario_filter,
    ),
)

total_epochs = 2
runner = dict(type="EpochBasedRunner", max_epochs=2)

# ── Checkpoint 和验证频率 ──
# 总 iter 约 6600（1 epoch），checkpoint 每 1000 iter 保存一次
checkpoint_config = dict(
    interval=1000,
    by_epoch=False,
    # 文件命名：iter_1000.pth, iter_2000.pth, ...
    # epoch 结束时额外保存 epoch_1.pth
)
# 验证每 1000 iter 执行一次，支持自动早停保护
# 注：save_best 与 rule 已由 train.py 依据训练阶段（Stage 1 或 2）动态自适应注入
evaluation = dict(
    interval=1000,
    by_epoch=False,
    early_stopping=dict(
        patience=3,
        min_delta=0.0,
        warmup_iters=3000,
    )
)
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)

# ── AMP 混合精度 + 梯度累积 ──
# 累积 2 步：增大更新频率让优化器能及时纠正振荡方向，
# 避免连续多步高梯度累加后单次大更新冲过头
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook',
    cumulative_iters=2,  # 累积 2 步：每个 GPU 的 samples_per_gpu 为 1，累积 2 步达到有效 batch_size=2
    grad_clip=dict(max_norm=0.5, norm_type=2),  # 收紧梯度裁剪至 0.5，防范碰撞梯度冲击
)

# ── 学习率调度（全局 cosine，跨 epoch 连续）──
# by_epoch=False: 整个训练周期为一次 cosine 衰减，消除 epoch 边界 lr 跳变
# 仅训练开始执行一次 warmup（前 200 iter）
lr_config = dict(
    by_epoch=False,               # 全局连续调度，不每 epoch 重置
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,             # 训练开始前 500 iter warmup
    warmup_ratio=0.1,             # 从 0.1*peak 起步
    min_lr_ratio=5e-2,            # 最终 lr 衰减到 peak 的 5%，防止后期 delta 过小
)
