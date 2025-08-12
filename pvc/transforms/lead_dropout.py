import torch
import torch.nn as nn

class LeadDropout(nn.Module):
    """Randomly zero whole electrodes with probability p."""
    def __init__(self, drop_prob: float = 0.10):
        super().__init__()
        self.p = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(1)
        mask = (torch.rand(C, device=x.device) > self.p).float()  # [C]
        return x * mask[None, :, None]
