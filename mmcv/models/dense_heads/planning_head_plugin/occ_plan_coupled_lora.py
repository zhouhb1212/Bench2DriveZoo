# ---------------------------------------------------------------------------------#
# Occupancy-Planning Coupled LoRA (双 LoRA 联动微调)
#
# 仅在 OccHead Transformer Decoder 和 PlanningHeadSingleMode adapter 路径中
# 注入 LoRA，通过三阶段训练实现占用感知的安全轨迹规划。
# ---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lora_adapter import (
    inject_lora_to_attention,
    inject_lora_to_linear,
    collect_lora_params,
    LoRALinear,
    LoRAMultiheadAttention,
)


class OccPlanCoupledLoRA:
    """
    Occupancy-Planning Coupled LoRA 管理器。

    负责:
        1. 向 OccHead 和 PlanningHead 精确注入 LoRA
        2. 管理三阶段训练的 freeze / unfreeze
        3. 计算 occupancy-planning 一致性损失

    三阶段训练:
        Stage 1: 仅训练 OccHead LoRA → 学习场景占用分布
        Stage 2: 仅训练 PlanningHead LoRA → 学习安全轨迹规划
        Stage 3: 联合训练 + 一致性约束 → 两者联动优化

    Args:
        occ_head: OccHead 实例
        planning_head: PlanningHeadSingleMode 实例
        lora_cfg: dict with keys (r, alpha, dropout, lambda_collision, lambda_align)
    """

    def __init__(self, occ_head, planning_head, lora_cfg=None):
        self.occ_head = occ_head
        self.planning_head = planning_head
        self.lora_cfg = lora_cfg or {}

        self.r = self.lora_cfg.get('r', 4)
        self.alpha = self.lora_cfg.get('alpha', 8)
        self.dropout = self.lora_cfg.get('dropout', 0.1)

        self.lambda_collision = self.lora_cfg.get('lambda_collision', 1.0)
        self.lambda_align = self.lora_cfg.get('lambda_align', 0.1)

        self._injected = False
        self._current_stage = 0

        # 通道投影：occ_risk_feat (bev_proj_dim=64) → plan_feat (embed_dims=256)
        occ_dim = getattr(occ_head, 'bev_proj_dim', 64)
        plan_dim = getattr(planning_head, 'embed_dims', 256)
        self.align_proj = nn.Linear(occ_dim, plan_dim, bias=False)

        # 风险投影头：将 decoder 通用占用特征映射为驾驶风险场 [B, T, 1, H, W]
        # 参数 ~19K，极小开销，将 "何处有物体" 转化为 "何处对自车危险"
        self.risk_proj = nn.Sequential(
            nn.Conv2d(occ_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    # ------------------------------------------------------------------#
    # LoRA 注入
    # ------------------------------------------------------------------#

    def inject(self):
        """执行双模块 LoRA 注入。必须在模型初始化后、训练前调用一次。"""
        if self._injected:
            return
        self._inject_occ_head()
        self._inject_planning_head()
        self._injected = True
        self._freeze_all()

    def _freeze_all(self):
        """冻结 OccHead 和 PlanningHead 全部参数。"""
        for p in self.occ_head.parameters():
            p.requires_grad = False
        for p in self.planning_head.parameters():
            p.requires_grad = False

    def _inject_occ_head(self):
        """
        向 OccHead 的 DetrTransformerDecoder 每层注入 LoRA。

        每层注入目标:
            - self_attn.attn  →  LoRAMultiheadAttention  (Q/K/V/Out proj)
            - cross_attn.attn →  LoRAMultiheadAttention  (Q/K/V/Out proj)
            - FFN: layers[0][0] (w1) & layers[1] (w2) → LoRALinear
        """
        r, alpha, dropout = self.r, self.alpha, self.dropout
        decoder = self.occ_head.transformer_decoder  # DetrTransformerDecoder
        if not hasattr(decoder, 'layers'):
            return

        for layer in decoder.layers:
            # attentions[0] = self_attn, attentions[1] = cross_attn
            # mmcv MultiheadAttention wrapper .attn = nn.MultiheadAttention
            for attn_wrapper in layer.attentions:
                if hasattr(attn_wrapper, 'attn'):
                    attn_wrapper.attn = inject_lora_to_attention(
                        attn_wrapper.attn, r=r, alpha=alpha, dropout=dropout)

            # FFN: mmcv FFN → self.layers = Sequential(
            #   Sequential(Linear, ReLU, Dropout),   # [0]: w1 在 [0][0]
            #   Linear,                              # [1]: w2
            #   Dropout                              # [2]
            # )
            for ffn in layer.ffns:
                if hasattr(ffn, 'layers'):
                    # w1
                    if (len(ffn.layers) >= 1 and hasattr(ffn.layers[0], '__getitem__')
                            and len(ffn.layers[0]) >= 1):
                        old_w1 = ffn.layers[0][0]
                        ffn.layers[0][0] = inject_lora_to_linear(
                            old_w1, r=r, alpha=alpha, dropout=dropout)
                    # w2
                    if len(ffn.layers) >= 2:
                        old_w2 = ffn.layers[1]
                        ffn.layers[1] = inject_lora_to_linear(
                            old_w2, r=r, alpha=alpha, dropout=dropout)

    def _inject_planning_head(self):
        """
        向 PlanningHeadSingleMode 的 adapter 路径注入 LoRA。

        注入目标:
            - mlp_fuser[0]  (Linear 768→256)
            - reg_branch[0] (Linear 256→256), reg_branch[2] (Linear 256→planning_steps*2)
            - attn_module 每层: self_attn / multihead_attn → LoRAMultiheadAttention
                                linear1 / linear2 → LoRALinear
            - bev_adapter Conv2d: 冻结不注入
        """
        r, alpha, dropout = self.r, self.alpha, self.dropout
        ph = self.planning_head

        # 1. mlp_fuser[0]: Linear(768, 256)
        ph.mlp_fuser[0] = inject_lora_to_linear(
            ph.mlp_fuser[0], r=r, alpha=alpha, dropout=dropout)

        # 2. reg_branch
        ph.reg_branch[0] = inject_lora_to_linear(
            ph.reg_branch[0], r=r, alpha=alpha, dropout=dropout)
        ph.reg_branch[2] = inject_lora_to_linear(
            ph.reg_branch[2], r=r, alpha=alpha, dropout=dropout)

        # 3. attn_module (nn.TransformerDecoder, 3 layers)
        for decoder_layer in ph.attn_module.layers:
            if hasattr(decoder_layer, 'self_attn'):
                decoder_layer.self_attn = inject_lora_to_attention(
                    decoder_layer.self_attn, r=r, alpha=alpha, dropout=dropout)
            if hasattr(decoder_layer, 'multihead_attn'):
                decoder_layer.multihead_attn = inject_lora_to_attention(
                    decoder_layer.multihead_attn, r=r, alpha=alpha, dropout=dropout)
            if hasattr(decoder_layer, 'linear1'):
                decoder_layer.linear1 = inject_lora_to_linear(
                    decoder_layer.linear1, r=r, alpha=alpha, dropout=dropout)
            if hasattr(decoder_layer, 'linear2'):
                decoder_layer.linear2 = inject_lora_to_linear(
                    decoder_layer.linear2, r=r, alpha=alpha, dropout=dropout)

        # 4. bev_adapter: 冻结不注入
        if ph.with_adapter:
            for p in ph.bev_adapter.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------#
    # 三阶段训练管理
    # ------------------------------------------------------------------#

    def set_training_stage(self, stage):
        """
        切换训练阶段。

        stage=1: 仅 OccHead LoRA 可训练
        stage=2: 仅 PlanningHead LoRA 可训练
        stage=3: 两者 LoRA 均可训练
        """
        assert stage in (1, 2, 3), f"stage must be 1/2/3, got {stage}"
        self._current_stage = stage

        # 全部参数冻结
        for p in self.occ_head.parameters():
            p.requires_grad = False
        for p in self.planning_head.parameters():
            p.requires_grad = False

        # 按阶段解冻
        if stage in (1, 3):
            self._set_lora_trainable(self.occ_head, True)
        if stage in (2, 3):
            self._set_lora_trainable(self.planning_head, True)
        # align_proj 仅在 Stage 3 可训练
        for p in self.align_proj.parameters():
            p.requires_grad = (stage == 3)
        # risk_proj 在 Stage 1/3 可训练（伴随 OccHead LoRA）
        for p in self.risk_proj.parameters():
            p.requires_grad = (stage in (1, 3))

        # 冻结 OccHead / PlanningHead 内所有 LayerNorm
        self._freeze_layernorm()

    def _freeze_layernorm(self):
        """冻结 OccHead 和 PlanningHead 内所有 LayerNorm（eval + requires_grad=False）。"""
        for head in [self.occ_head, self.planning_head]:
            for m in head.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def _set_lora_trainable(self, module, trainable):
        for name, param in module.named_parameters():
            if 'lora' in name:
                param.requires_grad = trainable

    def get_lora_params(self):
        """获取所有可训练参数（LoRA + 投影头），用于构建优化器。"""
        params = collect_lora_params(self.occ_head) + collect_lora_params(self.planning_head)
        params += list(self.align_proj.parameters())
        params += list(self.risk_proj.parameters())
        return params

    def get_current_stage(self):
        return self._current_stage

    # ------------------------------------------------------------------#
    # 风险场计算
    # ------------------------------------------------------------------#

    def compute_risk_mask(self, occ_risk_feat):
        """
        通过可学习的风险投影头将占用特征映射为驾驶风险场。

        Args:
            occ_risk_feat: 占用 decoder 特征 [B, T, C, H, W]

        Returns:
            occ_risk_mask: 驾驶风险场 [B, T, 1, H, W]，值越高 = 对自车越危险
        """
        B, T, C, H, W = occ_risk_feat.shape
        risk_maps = []
        for t in range(T):
            risk_t = self.risk_proj(occ_risk_feat[:, t])  # [B, 1, H, W]
            risk_maps.append(risk_t)
        return torch.stack(risk_maps, dim=1)  # [B, T, 1, H, W]

    # ------------------------------------------------------------------#
    # 占用-规划一致性损失
    # ------------------------------------------------------------------#

    def compute_consistency_loss(self, sdc_traj, occ_risk_feat, plan_feat,
                                 occ_risk_mask):
        """
        计算 occupancy-planning 一致性损失。

        组件:
            1. 轨迹-占用碰撞惩罚: 轨迹点落入高占用区域时惩罚
            2. 特征对齐损失: 规划查询特征与占用风险特征对齐

        Args:
            sdc_traj:      规划轨迹 [B, T, 2]  (BEV 像素坐标)
            occ_risk_feat: 占用风险特征 [B, T, C, H, W]
            plan_feat:     规划查询特征 [B, 1, C]
            occ_risk_mask: 占用风险掩码 [B, T, 1, H, W]

        Returns:
            total_loss: scalar, coll_penalty * lambda_collision + align_loss * lambda_align
        """
        loss_collision = self._collision_penalty(sdc_traj, occ_risk_mask)
        loss_align = self._feature_align_loss(plan_feat, occ_risk_feat)
        return (self.lambda_collision * loss_collision +
                self.lambda_align * loss_align)

    def _collision_penalty(self, sdc_traj, occ_risk_mask):
        """
        轨迹-占用碰撞惩罚（多点采样版）。

        对每个轨迹时间步，在轨迹点周围采样 3×3 网格（对应车辆 footprint），
        用 grid_sample 在占用风险图上查询。取网格内最大值作为该点的碰撞风险。
        """
        B, T, _ = sdc_traj.shape
        if occ_risk_mask is None or occ_risk_mask.numel() == 0:
            return torch.tensor(0.0, device=sdc_traj.device)

        T_occ = occ_risk_mask.shape[1]
        T_use = min(T, T_occ)
        # occ_risk_mask: [B, T, 1, H, W], H/W at indices 3/4
        H, W = occ_risk_mask.shape[3], occ_risk_mask.shape[4]

        # 3×3 采样偏移（归一化坐标），对应约 4m×2m 车辆 footprint
        # BEV: 200px / 102.4m → 0.512m/px；偏移 ±0.02 ≈ ±4px ≈ ±2m
        offsets_norm = torch.tensor(
            [[-0.02, -0.01], [0.0, -0.01], [0.02, -0.01],
             [-0.02,  0.0],  [0.0,  0.0],  [0.02,  0.0],
             [-0.02,  0.01], [0.0,  0.01], [0.02,  0.01]],
            device=sdc_traj.device, dtype=sdc_traj.dtype,
        )  # [9, 2]

        total_penalty = 0.0
        count = 0
        for t in range(T_use):
            grid_xy = sdc_traj[:, t, :2]              # [B, 2]
            # 像素坐标 → grid_sample 归一化坐标 [-1, 1]
            grid_x = grid_xy[:, 0:1] / (W / 2.0) - 1.0
            grid_y = grid_xy[:, 1:2] / (H / 2.0) - 1.0
            center = torch.stack([grid_x, grid_y], dim=-1)  # [B, 1, 2]

            # 中心点 + 偏移 → [B, 1, 9, 2]（grid_sample 要求 [N, H_out, W_out, 2]）
            grid_all = center.unsqueeze(2) + offsets_norm.view(1, 1, 9, 2)

            occ_t = occ_risk_mask[:, t, :, :, :]      # [B, 1, H, W]
            sampled = F.grid_sample(occ_t, grid_all, mode='bilinear',
                                   padding_mode='zeros', align_corners=False)
            # sampled: [B, 1, 1, 9] → 对 9 个采样点取 max（最坏情况）
            total_penalty += sampled.max(dim=-1)[0].mean()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=sdc_traj.device)
        return total_penalty / count

    def _feature_align_loss(self, plan_feat, occ_risk_feat):
        """
        特征对齐损失: 规划查询特征与占用风险特征的语义对齐。

        occ_risk_feat 全局平均池化 → 与 plan_feat 计算 cosine similarity。
        损失 = 1 - mean(cos_sim)。
        """
        if occ_risk_feat is None or plan_feat is None:
            dev = plan_feat.device if plan_feat is not None else torch.device('cpu')
            return torch.tensor(0.0, device=dev)

        # plan_feat: [B, 1, C_plan] → [B, C_plan]
        plan_vec = plan_feat.squeeze(1)
        # occ_risk_feat: [B, T, C_occ, H, W] → [B, T, C_occ]
        occ_vec = occ_risk_feat.mean(dim=[-2, -1])
        # 通道对齐: C_occ → C_plan
        occ_vec = self.align_proj(occ_vec)  # [B, T, C_plan]

        cos_sims = []
        for t in range(occ_vec.shape[1]):
            cos_sim = F.cosine_similarity(plan_vec, occ_vec[:, t, :], dim=-1)
            cos_sims.append(cos_sim.mean())

        return 1.0 - torch.stack(cos_sims).mean()

    # ------------------------------------------------------------------#
    # 辅助方法
    # ------------------------------------------------------------------#

    def log_trainable_param_count(self):
        """打印 OccHead + PlanningHead 可训练参数量及占比。"""
        occ_trainable = sum(p.numel() for p in self.occ_head.parameters()
                           if p.requires_grad)
        plan_trainable = sum(p.numel() for p in self.planning_head.parameters()
                            if p.requires_grad)
        occ_total = sum(p.numel() for p in self.occ_head.parameters())
        plan_total = sum(p.numel() for p in self.planning_head.parameters())
        total_trainable = occ_trainable + plan_trainable
        total_params = occ_total + plan_total

        info = (
            f"[OccPlanCoupledLoRA] Stage {self._current_stage}\n"
            f"  OccHead  LoRA: {occ_trainable:,} / {occ_total:,} "
            f"({100*occ_trainable/max(occ_total,1):.1f}%)\n"
            f"  Planning LoRA: {plan_trainable:,} / {plan_total:,} "
            f"({100*plan_trainable/max(plan_total,1):.1f}%)\n"
            f"  Total trainable: {total_trainable:,} / {total_params:,} "
            f"({100*total_trainable/max(total_params,1):.2f}%)"
        )
        print(info)
        return info

    def merge_weights_for_inference(self):
        """将所有 LoRA 权重合并到基座权重中（推理加速）。"""
        for module in [self.occ_head, self.planning_head]:
            for m in module.modules():
                if isinstance(m, LoRAMultiheadAttention):
                    m.merge_weights()
                elif isinstance(m, LoRALinear):
                    ad = m.lora_adapter
                    delta = (ad.lora_B @ ad.lora_A) * ad.scaling
                    m.linear.weight.data = m.linear.weight.data + delta
