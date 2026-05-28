#---------------------------------------------------------------------------------#
# LoRA Adapter for Bench2DriveZoo Planning Head
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAAdapter(nn.Module):
    """
    LoRA Adapter - 轻量级参数高效微调

    通过低秩分解减少可训练参数量，同时保持模型表达能力。
    适用于Transformer的FFN层和嵌入层。

    Args:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        r (int): LoRA rank，低算力下推荐4-8
        alpha (int): 缩放因子，通常设为rank的2倍
        dropout (float): Dropout概率
    """
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA低秩矩阵
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 初始化：A 用 Kaiming uniform（与原始 LoRA 论文一致），B 为零
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [*, in_features]

        Returns:
            输出张量 [*, out_features]
        """
        # LoRA分支
        lora_output = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        lora_output = self.dropout(lora_output)
        return lora_output

class LoRALinear(nn.Module):
    """
    带有LoRA的Linear层

    原始权重保持冻结，新增LoRA分支进行轻量级微调。
    推理时可选择合并权重。

    Args:
        in_features (int): 输入特征维度
        out_features (int):输出特征维度
        r (int): LoRA rank
        alpha (int): 缩放因子
        dropout (float): Dropout概率
        bias (bool): 是否使用偏置
    """
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        # 原始线性层（冻结）
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA适配器
        self.lora_adapter = LoRAAdapter(in_features, out_features, r=r, alpha=alpha, dropout=dropout)

        # 冻结原始层
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        """前向传播 = 原始输出 + LoRA输出"""
        return self.linear(x) + self.lora_adapter(x)

    def get_lora_params(self):
        """获取LoRA参数（用于优化器配置）"""
        lora_params = []
        lora_params.extend([self.lora_adapter.lora_A, self.lora_adapter.lora_B])
        return lora_params


class LoRAFFN(nn.Module):
    """
    带有LoRA的FFN（前馈网络）

    替代Transformer中的标准FFN，保持相同的接口。
    主要用于PlanningHead的TransformerDecoderLayer中。

    Args:
        embed_dims (int): 嵌入维度
        feedforward_channels (int): FFN中间层维度
        r (int): LoRA rank
        alpha (int): 缩放因子
        dropout (float): Dropout概率
    """
    def __init__(self, embed_dims, feedforward_channels, r=8, alpha=16, dropout=0.1):
        super().__init__()

        # 原始FFN（冻结）
        self.w1 = nn.Linear(embed_dims, feedforward_channels)
        self.w2 = nn.Linear(feedforward_channels, embed_dims)
        self.activation = nn.ReLU(inplace=True)

        # LoRA适配器（用于w1和w2）
        self.lora_w1 = LoRAAdapter(embed_dims, feedforward_channels, r=r, alpha=alpha, dropout=dropout)
        self.lora_w2 = LoRAAdapter(feedforward_channels, embed_dims, r=r, alpha=alpha, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        # 冻结原始层
        for param in self.w1.parameters():
            param.requires_grad = False
        for param in self.w2.parameters():
            param.requires_grad = False

    def forward(self, x):
        """前向传播"""
        # 原始FFN路径
        out = self.activation(self.w1(x))
        out = self.dropout(out)
        out = self.w2(out)
        out = self.dropout(out)

        # LoRA路径
        lora_out = self.lora_w1(x)
        lora_out = self.activation(lora_out)
        lora_out = self.dropout(lora_out)
        lora_out = self.lora_w2(lora_out)
        lora_out = self.dropout(lora_out)

        return out + lora_out

    def get_lora_params(self):
        """获取LoRA参数"""
        lora_params = []
        lora_params.extend(self.lora_w1.get_lora_params())
        lora_params.extend(self.lora_w2.get_lora_params())
        return lora_params


class LoRAConfig:
    """LoRA配置类"""

    def __init__(self, r=8, alpha=16, dropout=0.1, target_modules=None):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ['w1', 'w2']

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            r=config_dict.get('r', 8),
            alpha=config_dict.get('alpha', 16),
            dropout=config_dict.get('dropout', 0.1),
            target_modules=config_dict.get('target_modules', ['w1', 'w2'])
        )

    def to_dict(self):
        return {
            'r': self.r,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'target_modules': self.target_modules
        }


# ---------------------------------------------------------------------------------#
# LoRA-injected MultiheadAttention
# ---------------------------------------------------------------------------------#

class LoRAMultiheadAttention(nn.Module):
    """
    在 nn.MultiheadAttention 的 Q/K/V 投影和输出投影上注入 LoRA。

    通过 F.multi_head_attention_forward 使用 LoRA 增强后的权重，
    原始 MHA 参数冻结，只训练 LoRA 低秩矩阵。

    注入目标:
        - in_proj_weight  [3*embed_dims, embed_dims]  → Q/K/V 合并投影
        - out_proj.weight [embed_dims, embed_dims]     → 输出投影

    Args:
        mha: 原始 nn.MultiheadAttention 模块（会被冻结）
        r: LoRA rank
        alpha: LoRA 缩放因子, actual_scale = alpha / r
        dropout: LoRA dropout 概率
    """

    def __init__(self, mha, r=8, alpha=16, dropout=0.1):
        super().__init__()
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        self.scaling = alpha / r

        # 保留原始 MHA 结构信息
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = getattr(mha, 'batch_first', False)
        self.kdim = getattr(mha, 'kdim', embed_dim)
        self.vdim = getattr(mha, 'vdim', embed_dim)
        self._qkv_same_embed_dim = (self.kdim == embed_dim and self.vdim == embed_dim)

        # 保存原始 MHA 的冻结权重引用
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj_weight = mha.out_proj.weight
        self.out_proj_bias = mha.out_proj.bias
        self.bias_k = mha.bias_k
        self.bias_v = mha.bias_v
        self.add_zero_attn = mha.add_zero_attn
        self.dropout_p = mha.dropout

        # 冻结原始 MHA 全部参数
        for p in mha.parameters():
            p.requires_grad = False

        # ── LoRA for in_proj (Q/K/V combined) ──
        # lora_A_in: [r, embed_dim], lora_B_in: [3*embed_dim, r]
        # delta_W = lora_B_in @ lora_A_in → [3*embed_dim, embed_dim]
        self.lora_A_in = nn.Parameter(torch.empty(r, embed_dim))
        self.lora_B_in = nn.Parameter(torch.zeros(3 * embed_dim, r))
        nn.init.kaiming_uniform_(self.lora_A_in, a=5**0.5)
        nn.init.zeros_(self.lora_B_in)

        # ── LoRA for out_proj ──
        # lora_A_out: [r, embed_dim], lora_B_out: [embed_dim, r]
        # delta_W = lora_B_out @ lora_A_out → [embed_dim, embed_dim]
        self.lora_A_out = nn.Parameter(torch.empty(r, embed_dim))
        self.lora_B_out = nn.Parameter(torch.zeros(embed_dim, r))
        nn.init.kaiming_uniform_(self.lora_A_out, a=5**0.5)
        nn.init.zeros_(self.lora_B_out)

        self.lora_dropout = nn.Dropout(dropout)

    def get_lora_params(self):
        """返回所有 LoRA 可训练参数"""
        return [self.lora_A_in, self.lora_B_in,
                self.lora_A_out, self.lora_B_out]

    def merge_weights(self):
        """将 LoRA 权重合并到原始权重中（用于推理加速）"""
        delta_in = (self.lora_B_in @ self.lora_A_in) * self.scaling
        delta_out = (self.lora_B_out @ self.lora_A_out) * self.scaling
        self.in_proj_weight.data = self.in_proj_weight.data + delta_in
        self.out_proj_weight.data = self.out_proj_weight.data + delta_out

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights: bool = True, attn_mask=None,
                average_attn_weights: bool = True, **kwargs):
        """
        与 nn.MultiheadAttention.forward 保持完全一致的接口。

        使用 LoRA 增强的投影权重调用 F.multi_head_attention_forward，
        确保 autograd 正确追踪所有梯度路径。
        """
        # ── 计算 LoRA 增强的投影权重 ──
        delta_in = (self.lora_B_in @ self.lora_A_in) * self.scaling   # [3E, E]
        delta_out = (self.lora_B_out @ self.lora_A_out) * self.scaling  # [E, E]

        aug_in_proj = self.in_proj_weight + delta_in     # differentiable
        aug_out_proj = self.out_proj_weight + delta_out  # differentiable

        # ── 处理 batch_first ──
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        # ── 调用 functional API，直接使用增强后的权重 ──
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            aug_in_proj, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout_p, aug_out_proj, self.out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            average_attn_weights=average_attn_weights,
            **kwargs,
        )

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, attn_output_weights


# ---------------------------------------------------------------------------------#
# 通用 LoRA 注入工具函数
# ---------------------------------------------------------------------------------#

def inject_lora_to_attention(mha_module, r=8, alpha=16, dropout=0.1):
    """
    将 nn.MultiheadAttention 原地替换为 LoRAMultiheadAttention。
    必须在模型构建完成后、训练开始前调用。
    替换后原始 MHA 参数冻结，仅 LoRA 参数可训练。
    """
    return LoRAMultiheadAttention(mha_module, r=r, alpha=alpha, dropout=dropout)


def inject_lora_to_linear(linear_module, r=8, alpha=16, dropout=0.1):
    """
    将 nn.Linear 原地替换为 LoRALinear。
    复制原始权重，冻结基座，仅 LoRA 可训练。
    """
    in_features = linear_module.in_features
    out_features = linear_module.out_features
    bias = linear_module.bias is not None

    lora_linear = LoRALinear(in_features, out_features, r=r, alpha=alpha,
                             dropout=dropout, bias=bias)
    lora_linear.linear.weight.data.copy_(linear_module.weight.data)
    if bias:
        lora_linear.linear.bias.data.copy_(linear_module.bias.data)

    return lora_linear


def inject_lora_to_ffn(ffn_module, r=8, alpha=16, dropout=0.1):
    """
    将标准 FFN（包含两层 Linear + ReLU）替换为 LoRAFFN。
    适用于 OccHead transformer decoder layer 中的 FFN 模块。
    复制原始权重，冻结基座。
    """
    # FFN 内部结构：w1(Linear) -> ReLU -> dropout -> w2(Linear) -> dropout
    embed_dims = ffn_module.w1.in_features
    feedforward_channels = ffn_module.w1.out_features

    lora_ffn = LoRAFFN(embed_dims, feedforward_channels,
                       r=r, alpha=alpha, dropout=dropout)
    lora_ffn.w1.weight.data.copy_(ffn_module.w1.weight.data)
    lora_ffn.w1.bias.data.copy_(ffn_module.w1.bias.data)
    lora_ffn.w2.weight.data.copy_(ffn_module.w2.weight.data)
    lora_ffn.w2.bias.data.copy_(ffn_module.w2.bias.data)

    return lora_ffn


def collect_lora_params(module):
    """
    递归收集模块中所有 LoRA 参数。

    遍历子模块，提取 LoRALinear / LoRAFFN / LoRAMultiheadAttention 中
    的 lora_A / lora_B 参数。
    """
    lora_params = []
    for m in module.modules():
        if isinstance(m, (LoRALinear, LoRAFFN, LoRAMultiheadAttention)):
            if hasattr(m, 'get_lora_params'):
                lora_params.extend(m.get_lora_params())
    return lora_params


def freeze_all_except_lora(module):
    """冻结模块中所有非 LoRA 参数，仅保留 LoRA 参数可训练。"""
    for name, param in module.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
