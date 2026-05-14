"""
FlashAttention: Fast and Memory-Efficient Exact Attention

Integrated from "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
by Tri Dao, Daniel Y. Fu, Stefano Ermon, Christopher Ré, Zachary C. Brown (2022)

Paper: arXiv:2205.14135
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashCausalSelfAttention(nn.Module):
    """
    FlashAttention-based causal self-attention.

    Uses PyTorch's scaled_dot_product_attention with is_causal=True
    for O(N) memory and hardware-aware IO optimisation.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))
