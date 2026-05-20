"""
FlashAttention: Fast and Memory-Efficient Exact Attention

Integrated from "FlashAttention"
by  (2022)

Paper: arXiv:2205.14135
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECTED_SYMBOLS = ("FlashCausalSelfAttention",)

class FlashCausalSelfAttention(nn.Module):
    """
    FlashAttention-based causal self-attention.

    Implementation uses PyTorch's scaled_dot_product_attention (SDPA) 
    which dispatches to FlashAttention or Memory-Efficient Attention kernels
    depending on hardware and tensor shapes.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Linear projection to Q, K, V
        qkv = self.c_attn(x) # [B, T, 3*C]
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2. Reshape for Multi-Head Attention: [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Efficient Attention using SDPA
        # is_causal=True enforces the causal mask automatically
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        # 4. Reassemble heads: [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Final projection and residual dropout
        return self.resid_dropout(self.c_proj(y))
