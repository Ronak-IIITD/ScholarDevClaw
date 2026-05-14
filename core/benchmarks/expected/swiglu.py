"""
SwiGLU: Swish-Gated Linear Unit

Integrated from "GLU Variants Improve Transformer"
by Noam Shazeer (2020)

Paper: arXiv:2002.05202
Description: Swish-Gated Linear Unit — combines Swish activation with gated linear units for better FFN quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU).

    Combines Swish activation with gated linear units for improved FFN quality.
    Formula: SiLU(xW1) * (xW3) projected by W2.
    """

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
