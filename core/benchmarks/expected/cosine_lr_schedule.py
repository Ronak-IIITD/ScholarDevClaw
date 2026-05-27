"""
Cosine Annealing with Warmup Learning Rate Schedule

Integrated from "SGDR: Stochastic Gradient Descent with Warm Restarts"
by Ilya Loshchilov, Frank Hutter (2017)

Paper: arXiv:1608.03983
"""

import math


def get_cosine_warmup_lr(
    step: int, warmup_steps: int, max_steps: int, max_lr: float = 6e-4, min_lr: float = 6e-5
) -> float:
    """
    Cosine annealing schedule with linear warmup.

    1. Linear warmup from 0 to max_lr over warmup_steps.
    2. Cosine decay from max_lr to min_lr over remaining steps.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
