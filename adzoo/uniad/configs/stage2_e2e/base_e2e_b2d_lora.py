# ---------------------------------------------------------------------------------#
# Motion-Occupancy-Planning Coupled LoRA 配置文件
#
# 基于 base_e2e_b2d.py，增加三阶段 LoRA 微调相关配置。
#
# 三阶段训练（必须按顺序执行）：
#   Stage 1 (training_stage=1): 仅 Motion LoRA（优化 sdc_traj_query 表征）
#   Stage 2 (training_stage=2): 仅 OccHead LoRA（优化占用预测）
#   Stage 3 (training_stage=3): Planning + Motion LoRA 联合优化
#                                (planning loss 回传到 motionformer LoRA)
#
# 用法：通过命令行 --training-stage 和 --lr 覆盖
# ---------------------------------------------------------------------------------#

_base_ = ["./base_e2e_b2d.py"]

# 预训练权重
load_from = "ckpts/uniad_base_b2d.pth"

# ── 目标场景定义 ──
target_scenarios = ["ParkedObstacleTwoWays"]


if "ParkedObstacleTwoWays" in target_scenarios:
    # 针对双向避障场景的特化优化：
    # 1. 采用 RelativeCollisionLoss 避免训练中自车因安全框膨胀与真实绕行轨迹冲突；
    # 2. 收紧 delta 碰撞检测半径（0.0/0.25/0.5），降低避障时惩罚过重导致的“不敢绕行”或“过度转向”；
    # 3. 引入 PlanningDirectionLoss (方向/航向角损失) 直接监督轨迹的切线方向，提升绕行与回正控制精度。
    planning_head_cfg = dict(
        loss_collision=[
            dict(type='RelativeCollisionLoss', delta=0.0, weight=0.1),
            dict(type='RelativeCollisionLoss', delta=0.25, weight=0.04),
            dict(type='RelativeCollisionLoss', delta=0.5, weight=0.01)
        ],
        loss_direction=dict(type='PlanningDirectionLoss', weight=0.5),
        col_optim_args=dict(
            occ_filter_range=5.0,  
            sigma=1.0,
            alpha_collision=8.0,      # 增强避障排斥力度
        )
    )
else:
    # 默认通用微调配置（轻度避撞约束，保持平稳）
    planning_head_cfg = dict(
        loss_collision=[
            dict(type='CollisionLoss', delta=0.0, weight=0.1),
            dict(type='CollisionLoss', delta=0.5, weight=0.04),
            dict(type='CollisionLoss', delta=1.0, weight=0.01)
        ]
    )

# ── Motion-Occupancy-Planning Coupled LoRA 配置 ──
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
        motion_lora=dict(r=16, alpha=32),      # Stage 1 Motion: scale=2
        occ_lora=dict(r=16, alpha=32),         # Stage 2 OccHead: scale=2
        planning_lora=dict(r=16, alpha=32),    # Stage 3 PlanningHead: scale=2
        inject_q2o_feat=True,  # 向 query_to_occ_feat 注入 LoRA；False 用于消融/旧权重兼容
        pretrained_path="ckpts/uniad_base_b2d.pth",
        training_stage=1,  # 切换阶段：1=Motion / 2=OCC / 3=Planning+Motion联合
    ),
    task_loss_weight=dict(
        track=1.0,
        map=1.0,
        motion=1.0,
        occ=1.0,
        planning=2.0,
    ),
    planning_head=planning_head_cfg,
)

# ── 优化器（仅 LoRA 参数 requires_grad=True，其余已冻结）──
optimizer = dict(
    type="AdamW",
    lr=2e-4,      # LoRA 标准 lr（原 5e-6 过低，90% 步被 GradScaler 跳过）
    weight_decay=0.01,  # LoRA 参数少，0.01 避免衰减过强拉向零
)

# ── 输出路径 ──
work_dir = "/data/Bench2DriveZoo/adzoo/uniad/new_work_dirs/stage1"

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
            scenarios=target_scenarios,              # 要过采样的场景
            ratio=1,                                 # 额外复制
            max_other_frames=22648,               # 其他场景限制帧数
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
    interval=500,
    by_epoch=False,
)
# 验证每 500 iter 执行一次，更频繁地检测过拟合
evaluation = dict(
    interval=500,
    by_epoch=False,
    early_stopping=dict(
        metric='auto',            # 自动匹配训练阶段: stage1→motion_min_ade, stage2→occ_iou, stage3→planning_L2
        rule='auto',              # 自动匹配: motion/planning→less, occ→greater
        patience=6,               
        min_delta=0.001,          # 改善需超过阈值才算有效
        warmup_iters=500,         # 前 500 iter 不触发早停（warmup 阶段）
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
# 累积 2 步：平滑 motion.l_reg 的极端 spike，降低单次更新方差
# 使用较温和的初始 loss_scale (512.0) 避免 PyTorch GradScaler 默认 65536.0 导致前几步频繁 overflow
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook',
    cumulative_iters=2,  # 有效 batch_size=2
    grad_clip=dict(max_norm=5.0, norm_type=2),  
    loss_scale=dict(init_scale=512.0),
)

# ── 学习率调度（全局 cosine，跨 epoch 连续）──
# by_epoch=False: 整个训练周期为一次 cosine 衰减，消除 epoch 边界 lr 跳变
lr_config = dict(
    by_epoch=False,               # 全局连续调度，不每 epoch 重置
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=200,             # 拉长 warmup 让 GradScaler 稳定（原 100 太短）
    warmup_ratio=0.01,            # 从 0.01*peak=2e-6 起步，避免初始大梯度触发 overflow
    min_lr_ratio=0.01,            # 最终 lr 衰减到 peak 的 1%（=2e-6）
)
