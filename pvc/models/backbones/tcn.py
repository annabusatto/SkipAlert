# pvc/models/backbones/tcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import LeadMixer, masked_mean_time

class _CausalConv(nn.Module):
    """Causal 1D conv with SAME length (strictly no future leakage)."""
    def __init__(self, ch: int, k: int, dilation: int = 1):
        super().__init__()
        assert k % 2 == 1, "Use odd kernels for clean receptive fields"
        self.left_pad = (k - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel_size=k, dilation=dilation, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Left-pad only so output[t] depends on x[:t] (causal)
        x = F.pad(x, (self.left_pad, 0))     # [B,C,L + left_pad]
        return self.conv(x)                  # [B,C,L]

class _TCNBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv(ch, k, dilation),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            _CausalConv(ch, k, dilation),
            nn.BatchNorm1d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        # y and x have same length by construction
        return self.relu(y + x)

class TCNBackbone(nn.Module):
    """
    Temporal Convolutional Network with causal dilated residual blocks and
    mask-aware global average pooling.

    Input:
      x   : [B, 247, L]
      mask: [B, L] (True=valid), optional
    Output:
      z   : [B, base]
    """
    def __init__(self, base: int = 64, layers: int = 6, k: int = 3):
        super().__init__()
        self.stem = LeadMixer(base)  # 247 -> base via 1x1 conv+BN+ReLU
        dilations = [2 ** i for i in range(layers)]  # 1,2,4,8,...
        self.blocks = nn.ModuleList([_TCNBlock(base, k, d) for d in dilations])
        self.out_channels = base

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B,247,L]
        x = self.stem(x)          # [B,base,L]
        for blk in self.blocks:
            x = blk(x)            # keep length L throughout
        z = masked_mean_time(x, mask)  # [B,base]
        return z
