# pvc/models/backbones/wave_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # optional: weight norm, dropout, etc.

    def forward(self, x):
        y = torch.tanh(self.f(x)) * torch.sigmoid(self.g(x))
        # --- safety net: align time length if off by a couple samples ---
        if y.size(-1) != x.size(-1):
            diff = y.size(-1) - x.size(-1)
            if diff > 0:  # crop center
                left = diff // 2
                y = y[..., left:left + x.size(-1)]
            else:         # right-pad to length of x
                r = -diff
                y = F.pad(y, (r - r // 2, r // 2))
        return y + x

class WaveCNNBackbone(nn.Module):
    def __init__(self, base=64, dilations=(1, 2, 4, 8, 16, 32), k=3):
        super().__init__()
        # stem must map 247 leads -> base channels; e.g., 1x1 conv over leads
        self.stem = nn.Conv1d(247, base, kernel_size=1)
        blocks = [GatedResBlock(base, k=k, d=d) for d in dilations]
        self.cnn = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))
        self.out_channels = base

    def forward(self, x):         # x: [B, 247, 430]
        x = self.stem(x)          # [B, base, 430]
        x = self.cnn(x).squeeze(-1)  # [B, base]
        return x
