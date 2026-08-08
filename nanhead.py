import torch
from torch import nn


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 6)

    def forward(self, x):
        y = self.fc(x)
        out = y.clone()
        out[0, 0] = 9.0
        out[0, 1] = 1.0
        out[0, 2] = 2.0
        out[0, 3] = float("nan")
        out[0, 4] = 0.0
        out[0, 5] = 0.0
        return out


def load():
    return Head()


def example_input():
    return torch.randn(1, 4)
