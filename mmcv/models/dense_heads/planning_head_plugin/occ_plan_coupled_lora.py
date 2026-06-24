# ---------------------------------------------------------------------------------#
# Motion-Occupancy-Planning LoRA 
# ---------------------------------------------------------------------------------#

import torch.nn as nn
from .lora_adapter import (
    inject_lora_to_attention,
    inject_lora_to_linear,
    collect_lora_params,
    LoRALinear,
    LoRAMultiheadAttention,
)


def _disable_transformer_fastpath(tfl):
    """禁用 TransformerEncoderLayer/DecoderLayer 的 C++ 快速路径。
    快速路径在 eval 模式下会直接访问子模块内部属性(out_proj, weight 等)，
    与 LoRA 包装模块不兼容。设 activation_relu_or_gelu=0 可安全禁用。"""
    if hasattr(tfl, 'activation_relu_or_gelu'):
        tfl.activation_relu_or_gelu = 0


class OccPlanCoupledLoRA:
    """
    Motion-Occupancy-Planning LoRA 管理器。

    负责:
        1. 向 MotionHead(motionformer), OccHead, PlanningHead 精确注入 LoRA
        2. 管理三阶段训练的 freeze / unfreeze

    三阶段训练:
        Stage 1: 仅训练 Motion LoRA (优化 sdc_traj_query 表征)
        Stage 2: 仅训练 OccHead LoRA (优化占用预测)
        Stage 3: Planning LoRA + Motion LoRA 联合优化
                 (planning loss 通过 sdc_traj_query 回传到 motionformer LoRA)

    Args:
        occ_head: OccHead 实例
        planning_head: PlanningHeadSingleMode 实例
        motion_head: MotionHead 实例 (可选，为 None 时跳过 motion LoRA)
        lora_cfg: dict with keys (r, alpha, dropout)
    """

    def __init__(self, occ_head, planning_head, lora_cfg=None, motion_head=None):
        self.occ_head = occ_head
        self.planning_head = planning_head
        self.motion_head = motion_head
        self.lora_cfg = lora_cfg or {}

        # ── Per-head LoRA 参数（优先用子 dict，未指定时回退到全局默认值）──
        motion_cfg = self.lora_cfg.get('motion_lora', {})
        self.motion_r = motion_cfg.get('r', self.lora_cfg.get('r', 4))
        self.motion_alpha = motion_cfg.get('alpha', self.lora_cfg.get('alpha', 8))
        self.motion_dropout = motion_cfg.get('dropout', self.lora_cfg.get('dropout', 0.1))

        occ_cfg = self.lora_cfg.get('occ_lora', {})
        self.occ_r = occ_cfg.get('r', self.lora_cfg.get('r', 4))
        self.occ_alpha = occ_cfg.get('alpha', self.lora_cfg.get('alpha', 8))
        self.occ_dropout = occ_cfg.get('dropout', self.lora_cfg.get('dropout', 0.1))

        plan_cfg = self.lora_cfg.get('planning_lora', {})
        self.plan_r = plan_cfg.get('r', self.lora_cfg.get('r', 4))
        self.plan_alpha = plan_cfg.get('alpha', self.lora_cfg.get('alpha', 8))
        self.plan_dropout = plan_cfg.get('dropout', self.lora_cfg.get('dropout', 0.1))

        self.inject_q2o_feat = self.lora_cfg.get('inject_q2o_feat', False)

        self._injected = False
        self._current_stage = 0
        self._step_counter = 0

    # ------------------------------------------------------------------#
    # LoRA 注入
    # ------------------------------------------------------------------#

    def inject(self):
        """同时注入 MotionHead, OccHead 和 PlanningHead 的 LoRA 结构。"""
        if self._injected:
            return
        self._inject_motion_head()
        self._inject_occ_head()
        self._inject_planning_head()
        self._injected = True
        self._freeze_all()

    def _freeze_all(self):
        if self.motion_head is not None:
            for p in self.motion_head.parameters():
                p.requires_grad = False
        for p in self.occ_head.parameters():
            p.requires_grad = False
        for p in self.planning_head.parameters():
            p.requires_grad = False

    def _inject_motion_head(self):
        """
        向 MotionHead 的 motionformer 注入 LoRA。

        注入目标:
            - track_agent_interaction_layers (3层 TransformerDecoderLayer)
            - map_interaction_layers (3层 TransformerDecoderLayer)
            - intention_interaction_layers (TransformerEncoderLayer)
            - out_query_fuser MLP (关键的融合层)
        """
        if self.motion_head is None:
            return

        r, alpha, dropout = self.motion_r, self.motion_alpha, self.motion_dropout
        motionformer = self.motion_head.motionformer

        # 1. track_agent_interaction_layers: 每层是 TrackAgentInteraction
        #    内含 interaction_transformer (nn.TransformerDecoderLayer)
        for layer in motionformer.track_agent_interaction_layers:
            tfl = layer.interaction_transformer
            if hasattr(tfl, 'self_attn'):
                tfl.self_attn = inject_lora_to_attention(
                    tfl.self_attn, r=r, alpha=alpha, dropout=dropout)
            if hasattr(tfl, 'multihead_attn'):
                tfl.multihead_attn = inject_lora_to_attention(
                    tfl.multihead_attn, r=r, alpha=alpha, dropout=dropout)
            if hasattr(tfl, 'linear1'):
                tfl.linear1 = inject_lora_to_linear(
                    tfl.linear1, r=r, alpha=alpha, dropout=dropout)
            if hasattr(tfl, 'linear2'):
                tfl.linear2 = inject_lora_to_linear(
                    tfl.linear2, r=r, alpha=alpha, dropout=dropout)
            _disable_transformer_fastpath(tfl)

        # 2. map_interaction_layers: 同上结构
        for layer in motionformer.map_interaction_layers:
            tfl = layer.interaction_transformer
            if hasattr(tfl, 'self_attn'):
                tfl.self_attn = inject_lora_to_attention(
                    tfl.self_attn, r=r, alpha=alpha, dropout=dropout)
            if hasattr(tfl, 'multihead_attn'):
                tfl.multihead_attn = inject_lora_to_attention(
                    tfl.multihead_attn, r=r, alpha=alpha, dropout=dropout)
            if hasattr(tfl, 'linear1'):
                tfl.linear1 = inject_lora_to_linear(
                    tfl.linear1, r=r, alpha=alpha, dropout=dropout)
            if hasattr(tfl, 'linear2'):
                tfl.linear2 = inject_lora_to_linear(
                    tfl.linear2, r=r, alpha=alpha, dropout=dropout)
            _disable_transformer_fastpath(tfl)

        # 3. intention_interaction_layers (TransformerEncoderLayer)
        tfl = motionformer.intention_interaction_layers.interaction_transformer
        if hasattr(tfl, 'self_attn'):
            tfl.self_attn = inject_lora_to_attention(
                tfl.self_attn, r=r, alpha=alpha, dropout=dropout)
        if hasattr(tfl, 'linear1'):
            tfl.linear1 = inject_lora_to_linear(
                tfl.linear1, r=r, alpha=alpha, dropout=dropout)
        if hasattr(tfl, 'linear2'):
            tfl.linear2 = inject_lora_to_linear(
                tfl.linear2, r=r, alpha=alpha, dropout=dropout)
        _disable_transformer_fastpath(tfl)

        # 4. out_query_fuser: MLP 融合所有交互结果，直接影响 traj_query 输出
        for i, layer in enumerate(motionformer.out_query_fuser):
            if isinstance(layer, nn.Linear):
                motionformer.out_query_fuser[i] = inject_lora_to_linear(
                    layer, r=r, alpha=alpha, dropout=dropout)

    def _inject_occ_head(self):
        """
        向 OccHead 注入 LoRA。

        注入目标:
            - Transformer Decoder 每层:
                self_attn.attn  →  LoRAMultiheadAttention  (Q/K/V/Out proj)
                cross_attn.attn →  LoRAMultiheadAttention  (Q/K/V/Out proj)
                FFN: layers[0][0] (w1) & layers[1] (w2) → LoRALinear
            - query_to_occ_feat MLP (instance query → occupancy feature 的门户):
                layers[0..2] → LoRALinear
        """
        r, alpha, dropout = self.occ_r, self.occ_alpha, self.occ_dropout
        decoder = self.occ_head.transformer_decoder  # DetrTransformerDecoder
        if not hasattr(decoder, 'layers'):
            return

        for layer in decoder.layers:
            for attn_wrapper in layer.attentions:
                if hasattr(attn_wrapper, 'attn'):
                    attn_wrapper.attn = inject_lora_to_attention(
                        attn_wrapper.attn, r=r, alpha=alpha, dropout=dropout)

            for ffn in layer.ffns:
                if hasattr(ffn, 'layers'):
                    if (len(ffn.layers) >= 1 and hasattr(ffn.layers[0], '__getitem__')
                            and len(ffn.layers[0]) >= 1):
                        old_w1 = ffn.layers[0][0]
                        ffn.layers[0][0] = inject_lora_to_linear(
                            old_w1, r=r, alpha=alpha, dropout=dropout)
                    if len(ffn.layers) >= 2:
                        old_w2 = ffn.layers[1]
                        ffn.layers[1] = inject_lora_to_linear(
                            old_w2, r=r, alpha=alpha, dropout=dropout)

        # Inject LoRA into query_to_occ_feat
        if self.inject_q2o_feat and hasattr(self.occ_head, 'query_to_occ_feat'):
            q2o = self.occ_head.query_to_occ_feat
            if hasattr(q2o, 'layers'):
                for i in range(len(q2o.layers)):
                    old_lin = q2o.layers[i]
                    q2o.layers[i] = inject_lora_to_linear(
                        old_lin, r=r, alpha=alpha, dropout=dropout)

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
        r, alpha, dropout = self.plan_r, self.plan_alpha, self.plan_dropout
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
            _disable_transformer_fastpath(decoder_layer)

        # 4. bev_adapter: 冻结不注入
        if ph.with_adapter:
            for p in ph.bev_adapter.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------#
    # 两阶段训练管理
    # ------------------------------------------------------------------#

    def set_training_stage(self, stage):
        """
        切换训练阶段。

        stage=1: 仅 Motion LoRA 可训练 (优化 sdc_traj_query 表征)
        stage=2: 仅 OccHead LoRA 可训练 (优化占用预测)
        stage=3: Planning LoRA + Motion LoRA 联合可训练
                 (planning loss 通过 sdc_traj_query 回传到 motionformer LoRA)
        """
        assert stage in (1, 2, 3), f"stage must be 1, 2, or 3, got {stage}"
        self._current_stage = stage

        # 全部参数冻结
        if self.motion_head is not None:
            for p in self.motion_head.parameters():
                p.requires_grad = False
        for p in self.occ_head.parameters():
            p.requires_grad = False
        for p in self.planning_head.parameters():
            p.requires_grad = False

        # 按阶段解冻
        if stage == 1:
            if self.motion_head is not None:
                self._set_lora_trainable(self.motion_head, True)
        elif stage == 2:
            self._set_lora_trainable(self.occ_head, True)
        elif stage == 3:
            self._set_lora_trainable(self.planning_head, True)
            if self.motion_head is not None:
                self._set_lora_trainable(self.motion_head, True)

        # 冻结所有 LayerNorm
        self._freeze_layernorm()

        # 保存基线快照，用于后续 lora_delta 监控
        self.snapshot_lora_weights()

    def _freeze_layernorm(self):
        heads = [self.occ_head, self.planning_head]
        if self.motion_head is not None:
            heads.append(self.motion_head)
        for head in heads:
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
        """获取所有可训练 LoRA 参数，用于构建优化器。"""
        params = []
        if self.motion_head is not None:
            params += collect_lora_params(self.motion_head)
        params += collect_lora_params(self.occ_head)
        params += collect_lora_params(self.planning_head)
        return params

    def _build_lora_b_refs(self):
        """一次性扫描所有 head，缓存所有 lora_B 参数引用。"""
        refs = {}
        heads = [('occ_head', self.occ_head),
                 ('planning_head', self.planning_head)]
        if self.motion_head is not None:
            heads.append(('motion_head', self.motion_head))
        for prefix, head in heads:
            for name, param in head.named_parameters():
                if 'lora_B' in name:
                    refs[f'{prefix}.{name}'] = param
        return refs

    def snapshot_lora_weights(self):
        self._step_counter = 0
        self._lora_b_refs = self._build_lora_b_refs()
        self._baseline_lora_b = {}
        for name, param in self._lora_b_refs.items():
            self._baseline_lora_b[name] = param.data.clone()

    def log_lora_delta(self, log_interval=10):
        """每 log_interval 步返回 lora_B 相对变化量的均值。

        若持续为 0 说明 LoRA 梯度未流动；若持续增大说明不收敛。
        """
        self._step_counter += 1
        if self._step_counter % log_interval != 0:
            return None
        rel_deltas = []
        for name, param in self._lora_b_refs.items():
            if name in self._baseline_lora_b:
                base = self._baseline_lora_b[name]
                if base.device != param.device:
                    base = base.to(param.device)
                    self._baseline_lora_b[name] = base
                delta = (param.data - base).norm().item()
                base_norm = base.norm().item()
                if base_norm > 0.01:
                    # base 足够大时用相对变化量
                    rel_deltas.append(delta / base_norm)
                else:
                    # lora_B 从零开始，用绝对变化量
                    rel_deltas.append(delta)
                base.copy_(param.data)
        return sum(rel_deltas) / len(rel_deltas) if rel_deltas else 0.0

    def get_current_stage(self):
        return self._current_stage

    # ------------------------------------------------------------------#
    # 辅助方法
    # ------------------------------------------------------------------#

    def log_trainable_param_count(self):
        """打印所有 Head 可训练参数量及占比。"""
        motion_trainable = 0
        motion_total = 0
        if self.motion_head is not None:
            motion_trainable = sum(p.numel() for p in self.motion_head.parameters()
                                   if p.requires_grad)
            motion_total = sum(p.numel() for p in self.motion_head.parameters())
        occ_trainable = sum(p.numel() for p in self.occ_head.parameters()
                           if p.requires_grad)
        plan_trainable = sum(p.numel() for p in self.planning_head.parameters()
                             if p.requires_grad)
        occ_total = sum(p.numel() for p in self.occ_head.parameters())
        plan_total = sum(p.numel() for p in self.planning_head.parameters())
        total_trainable = motion_trainable + occ_trainable + plan_trainable
        total_params = motion_total + occ_total + plan_total

        info = (
            f"[OccPlanCoupledLoRA] Stage {self._current_stage}\n"
            f"  Motion  LoRA (r={self.motion_r}, alpha={self.motion_alpha}): "
            f"{motion_trainable:,} / {motion_total:,} "
            f"({100*motion_trainable/max(motion_total,1):.1f}%)\n"
            f"  OccHead  LoRA (r={self.occ_r}, alpha={self.occ_alpha}): "
            f"{occ_trainable:,} / {occ_total:,} "
            f"({100*occ_trainable/max(occ_total,1):.1f}%)\n"
            f"  Planning LoRA (r={self.plan_r}, alpha={self.plan_alpha}): "
            f"{plan_trainable:,} / {plan_total:,} "
            f"({100*plan_trainable/max(plan_total,1):.1f}%)\n"
            f"  Total trainable: {total_trainable:,} / {total_params:,} "
            f"({100*total_trainable/max(total_params,1):.2f}%)"
        )
        print(info)
        return info

    def merge_weights_for_inference(self):
        """将所有 LoRA 权重合并到基座权重中（推理加速）。"""
        modules = [self.occ_head, self.planning_head]
        if self.motion_head is not None:
            modules.append(self.motion_head)
        for module in modules:
            for m in module.modules():
                if isinstance(m, LoRAMultiheadAttention):
                    m.merge_weights()
                elif isinstance(m, LoRALinear):
                    ad = m.lora_adapter
                    delta = (ad.lora_B @ ad.lora_A) * ad.scaling
                    m.linear.weight.data = m.linear.weight.data + delta
