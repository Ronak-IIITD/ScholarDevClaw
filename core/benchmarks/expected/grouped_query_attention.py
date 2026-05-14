"""
Grouped-Query Attention (GQA)

Integrated from "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
by Joshua Ainslie, James Lee-Thorp, et al. (2023)

Paper: arXiv:2305.13245
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention.

    Uses fewer key-value heads than query heads, reducing KV-cache size
    while retaining most of multi-head attention quality.
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = getattr(config, "n_kv_head", config.n_head // 4)
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_group_size = self.n_head // self.n_kv_head

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads via repetition
        k = k.repeat_interleave(self.kv_group_size, dim=1)
        v = v.repeat_interleave(self.kv_group_size, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        return self.dropout(self.o_proj(y))
