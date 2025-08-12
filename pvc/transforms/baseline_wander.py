import torch
import random
import math
import torch.nn as nn

class BaselineWander(nn.Module):
    """Add low-freq sine drift (broadcast over channels)."""
    def __init__(self, max_amp: float = 0.1, f_range=(0.1, 0.5)):
        super().__init__()
        self.max_amp, self.f_range = max_amp, f_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(-1)
        f = random.uniform(*self.f_range)
        t = torch.linspace(0, 1, L, device=x.device)
        drift = self.max_amp * torch.sin(2 * math.pi * f * t)  # [L]
        return x + drift  # broadcasts to [B,C,L]
