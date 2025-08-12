# pvc/models/backbones/tcn.py
import torch.nn as nn
from ..common import LeadMixer

class _CausalConv(nn.Conv1d):
    """Same padding so output length stays 430, but no future data leak."""
    def __init__(self, ch, k, dilation=1):
        pad = (k - 1) * dilation
        super().__init__(ch, ch, k, padding=pad, dilation=dilation)
    def forward(self, x):
        y = super().forward(x)
        trim = self.padding[0]
        return y[..., :-trim]                      # drop future steps

class _TCNBlock(nn.Module):
    def __init__(self, ch, k, dilation):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv(ch, k, dilation),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            _CausalConv(ch, k, dilation),
            nn.BatchNorm1d(ch)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.net(x) + x)

class TCNBackbone(nn.Module):
    def __init__(self, base=64, layers=6, k=3):
        super().__init__()
        self.stem = LeadMixer(base)
        dilations = [2 ** i for i in range(layers)]   # 1,2,4,8,…
        blocks = [_TCNBlock(base, k, d) for d in dilations]
        self.tcn = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))
        self.out_channels = base

    def forward(self, x):
        x = self.stem(x)
        x = self.tcn(x).squeeze(-1)                  # [B, base]
        return x
