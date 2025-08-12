import torch
import random
import torch.nn as nn

class AmplitudeScale(nn.Module):
    """Multiply entire beat by a random gain in [lo, hi]."""
    def __init__(self, lo: float = 0.95, hi: float = 1.05):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * random.uniform(self.lo, self.hi)

