import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln_2(self.ln_1(x))
