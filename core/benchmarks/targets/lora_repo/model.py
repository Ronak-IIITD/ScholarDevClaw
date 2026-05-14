import torch
import torch.nn as nn


class AdapterProjection(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
