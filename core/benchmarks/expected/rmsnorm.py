"""
RMSNorm: Root Mean Square Layer Normalization

Integrated from "Root Mean Square Layer Normalization"
by Biao Zhang, Rico Sennrich (2019)

Paper: arXiv:1910.07467
Description: Simplified layer normalization without mean-centering
Formula: x / sqrt(mean(x^2) + eps) * gamma
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Simplified layer normalization without mean-centering.
    Formula: x / sqrt(mean(x^2) + eps) * gamma

    Benefits:
    - Faster computation than LayerNorm
    - Simplified forward pass (no mean subtraction)
    - Often achieves similar or better results
    """

    def __init__(self, ndim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight
