# pvc/models/backbones/wave_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import masked_mean_time

def _same_pad_1d(k: int, d: int) -> int:
    # requires odd k; for k=3, returns d
    assert k % 2 == 1, "WaveCNN expects odd kernel size to preserve length"
    return d * (k - 1) // 2

class GatedResBlock(nn.Module):
    def __init__(self, channels: int, k: int = 3, d: int = 1):
        super().__init__()
        pad = _same_pad_1d(k, d)
        self.f = nn.Conv1d(channels, channels, kernel_size=k, dilation=d, padding=pad)
        self.g = nn.Conv1d(channels, channels, kernel_size=k, dilation=d, padding=pad)
        # optional: add dropout/weight norm if desired

    def forward(self, x):
        # x: [B, C, L]
        y = torch.tanh(self.f(x)) * torch.sigmoid(self.g(x))  # [B, C, L’]
        # --- safety net: align time length if off by a couple samples ---
        if y.size(-1) != x.size(-1):
            diff = y.size(-1) - x.size(-1)
            if diff > 0:  # crop center
                left = diff // 2
                y = y[..., left:left + x.size(-1)]
            else:         # right-pad to length of x
                r = -diff
                y = F.pad(y, (r - r // 2, r // 2))
        return y + x  # residual

class WaveCNNBackbone(nn.Module):
    """
    Dilated gated residual CNN over the time axis with mask-aware global pooling.

    Input:
      x   : [B, 247, L]
      mask: [B, L] bool (True=valid), optional

    Output:
      z   : [B, base]
    """
    def __init__(self, base=64, dilations=(1, 2, 4, 8, 16, 32), k=3):
        super().__init__()
        # stem: 247 leads -> base channels (1x1 conv over lead dimension)
        self.stem = nn.Conv1d(247, base, kernel_size=1)
        self.blocks = nn.ModuleList([GatedResBlock(base, k=k, d=d) for d in dilations])
        self.out_channels = base

    def forward(self, x, mask: torch.Tensor | None = None):
        # x: [B, 247, L]
        x = self.stem(x)                      # [B, base, L]
        for blk in self.blocks:
            x = blk(x)                        # keep length ~L (safety-align inside block)
        z = masked_mean_time(x, mask)        # [B, base]
        return z
