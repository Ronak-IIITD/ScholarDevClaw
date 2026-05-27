"""
ALiBi: Attention with Linear Biases

Integrated from "Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization"
by Ofir Press, Noah A. Smith, Mike Lewis (2021)

Paper: arXiv:2108.12409
"""

import math

import torch
import torch.nn as nn


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi).

    Replaces positional embeddings with a linear bias added to
    attention scores. Enables length generalisation without
    any learned positional parameters.
    """

    def __init__(self, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.n_heads = n_heads
        slopes = self._get_slopes(n_heads)
        self.register_buffer("slopes", slopes, persistent=False)
        self._build_bias(max_seq_len)

    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        def _closest_power_of_2(n: int) -> int:
            return 2 ** math.floor(math.log2(n))

        n = _closest_power_of_2(n_heads)
        slopes = torch.tensor([2 ** (-(2 ** -(math.log2(n) - i))) for i in range(n)])
        if n < n_heads:
            extra = torch.tensor(
                [2 ** (-(2 ** -(math.log2(2 * n) - i))) for i in range(n_heads - n)]
            )
            slopes = torch.cat([slopes, extra])
        return slopes.unsqueeze(1).unsqueeze(1)

    def _build_bias(self, seq_len: int) -> None:
        positions = torch.arange(seq_len)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1)
        bias = self.slopes * relative.unsqueeze(0).float()
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        T = attn_scores.size(-1)
        return attn_scores + self.bias[:, :T, :T]
