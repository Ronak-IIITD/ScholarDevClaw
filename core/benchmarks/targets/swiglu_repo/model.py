import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.c_fc = nn.Linear(width, width * 4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(width * 4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))
