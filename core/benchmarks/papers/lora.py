"""
LoRA: Low-Rank Adaptation of Large Language Models

Integrated from "LoRA: Low-Rank Adaptation of Large Language Models"
by  (2021)

Paper: arXiv:2106.09685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECTED_SYMBOLS = ("LoRALinear", "apply_lora")

class LoRALinear(nn.Module):
    """
    LoRA Linear Layer.

    Wraps a base linear layer and applies a low-rank update:
    y = Wx + (BA)x = (W + BA)x
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: float = 32.0, lora_dropout: float = 0.05):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Base frozen weight
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))

        # Low-rank matrices A and B
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(lora_dropout)

        # Initialize A with Kaiming-uniform and B as zeros to ensure identity start
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path: Wx
        result = F.linear(x, self.base_weight)

        # LoRA path: (B @ A)x
        # x must be [batch, seq, in_features]
        x_dropped = self.dropout(x)
        lora_update = (x_dropped @ self.lora_A.t()) @ self.lora_B.t()

        return result + (lora_update * self.scaling)

def apply_lora(model: nn.Module, target_layer: str, r: int = 8, lora_alpha: float = 32.0):
    """
    Injects LoRA layers into a pre-trained model.
    Replaces existing nn.Linear layers matching target_layer name with LoRALinear.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and target_layer in name:
            # Extract weight and bias
            weight = module.weight.data.clone()
            bias = module.bias.data.clone() if module.bias is not None else None

            # Replace with LoRALinear
            new_layer = LoRALinear(module.in_features, module.out_features, r=r, lora_alpha=lora_alpha)
            new_layer.base_weight.data = weight

            # Patch the module into the parent's dict
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, new_layer)
