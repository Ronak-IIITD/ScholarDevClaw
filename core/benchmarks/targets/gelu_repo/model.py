import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(self.fc(x))
