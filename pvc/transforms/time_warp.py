import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class TimeWarp(nn.Module):
    """Linear time-scaling ±max_warp fraction; resamples back to L."""
    def __init__(self, max_warp: float = 0.05):
        super().__init__()
        self.mw = max_warp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        L = x.size(-1)
        factor = 1.0 + random.uniform(-self.mw, self.mw)
        idx = torch.linspace(0, L - 1, int(round(L * factor)), device=x.device)
        idx = idx.clamp(0, L - 1).round().long()
        x = x[..., idx]                      # stretch/squeeze
        return F.interpolate(x, size=L, mode='linear', align_corners=False)
