"""Template: point ModelMRI at a model you trained yourself.

Copy this next to your training code, edit load(), and pick it in the
CUSTOM MODEL panel. The whole contract is one function.

    def load() -> torch.nn.Module        required
    def example_input() -> Tensor        optional but recommended
    LABELS: list[str]                    optional, names your output classes

ModelMRI imports this file and calls load(). That runs your code — which is
the point, since only your code knows how to build your model — so it will
only import a path you named, under a directory you configured. It never
fetches an adapter from anywhere.

Run it yourself first; if `python adapter_template.py` works, ModelMRI will.
"""

from __future__ import annotations

import torch
from torch import nn


class TinyNet(nn.Module):
    """Stand-in for whatever you actually trained."""

    def __init__(self, n_in: int = 20, n_hidden: int = 64, n_out: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_hidden // 2)
        self.act2 = nn.Tanh()
        self.head = nn.Linear(n_hidden // 2, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.head(x)


def load() -> nn.Module:
    """Build the model and put your trained weights in it.

    The usual shape of this function:

        model = TinyNet()
        state = torch.load("checkpoints/best.pt", map_location="cpu")
        model.load_state_dict(state)
        return model

    Return the module itself — not a state_dict, not a path, not a
    (model, optimizer) tuple. ModelMRI puts it in eval() for the forward pass
    and restores the mode it was in afterwards.
    """
    model = TinyNet()

    # Only so the template shows something interesting out of the box: a
    # deliberately dead unit and a saturating tanh, both of which the panel
    # will flag. Delete this block and load your real checkpoint.
    with torch.no_grad():
        model.fc1.bias[:40] = -50.0  # 40 of 64 units that can never fire
        model.fc2.weight.mul_(30.0)  # drive tanh to its rails

    return model


def example_input() -> torch.Tensor:
    """One realistic batch.

    Worth writing. Without it ModelMRI infers a shape from your first layer
    and tells you it guessed; with it, the pass runs on something shaped like
    your real data — and a shape mismatch is the most common reason a first
    inspection fails.
    """
    return torch.randn(8, 20)


LABELS = ["negative", "neutral", "positive"]


if __name__ == "__main__":
    m = load()
    x = example_input()
    y = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f"{type(m).__name__}: {n:,} params, {tuple(x.shape)} -> {tuple(y.shape)}")
