import math

EXPECTED_SYMBOLS = ("CosineLRSchedule", "cosine_lr")


def cosine_lr(step, warmup_steps, max_steps, base_lr=1.0):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = min(max((step - warmup_steps) / max(max_steps - warmup_steps, 1), 0.0), 1.0)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


class CosineLRSchedule:
    def __call__(self, step, warmup_steps, max_steps, base_lr=1.0):
        return cosine_lr(step, warmup_steps, max_steps, base_lr=base_lr)
