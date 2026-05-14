import torch

learning_rate = 3e-4


def build_scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
