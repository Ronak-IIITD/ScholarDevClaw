import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, block_size: int, width: int):
        super().__init__()
        self.wpe = nn.Embedding(block_size, width)
        self.c_attn = nn.Linear(width, width * 3)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        positions = self.wpe(idx)
        return self.c_attn(positions)
