# ---------------------------------------------------------------------------------#
# Occupancy-Planning Coupled LoRA 配置文件（Bench2Drive-Mini 版本）
#
# 基于 base_e2e_b2d_lora.py，使用 mini 数据集（10 场景：8 train + 2 val）。
#
# 三阶段训练：
#   Stage 1 (training_stage=1): 仅 OccHead LoRA，lr=3e-4
#   Stage 2 (training_stage=2): 仅 PlanningHead LoRA，lr=3e-4
#   Stage 3 (training_stage=3): 两者联合 + 一致性损失，lr=1e-4
#
# 用法：
#   ./adzoo/uniad/uniad_dist_train.sh ./adzoo/uniad/configs/stage2_e2e/base_e2e_b2d_lora_mini.py 1 \
#       --training-stage 1 --lr 3e-4
# ---------------------------------------------------------------------------------#

_base_ = ["./base_e2e_b2d_lora.py"]

# ── 指向 mini 数据集 info 文件（必须直接覆盖 data.train/data.val/data.test）──
data = dict(
    train=dict(ann_file="data/infos/b2d_mini_infos_train.pkl"),
    val=dict(ann_file="data/infos/b2d_mini_infos_val.pkl"),
    test=dict(ann_file="data/infos/b2d_mini_infos_val.pkl"),
)
map_file = "data/infos/b2d_mini_map_infos.pkl"

# ── Mini 数据集迭代快，提高 eval/日志频率 ──
total_epochs = 2
runner = dict(type="EpochBasedRunner", max_epochs=2)
evaluation = dict(interval=200, by_epoch=False)
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)
checkpoint_config = dict(interval=400, by_epoch=False)

# ── Mini 数据集 lr schedule 适配 ──
# 继承 base_e2e_b2d_lora.py 的 lr=3e-4，此处可命令行覆盖
lr_config = dict(warmup_iters=50)
