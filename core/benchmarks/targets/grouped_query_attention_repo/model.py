import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, n_head: int = 8):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.c_attn(x)[..., : x.shape[-1]])
