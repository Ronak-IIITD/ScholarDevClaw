import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.c_attn = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.flash = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.c_attn(x)
        return self.c_proj(qkv[..., : x.shape[-1]])
