import torch
import torch.nn as nn


class TinyGPT(nn.Module):
    def __init__(self, block_size: int, width: int):
        super().__init__()
        self.wpe = nn.Embedding(block_size, width)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.wpe(idx)
