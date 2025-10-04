# pvc/models/backbones/resnet.py
import torch.nn as nn
from ..common import conv_bn_relu, LeadMixer, masked_mean_time

class ResNetBackbone(nn.Module):
    out_channels: int                      # to satisfy BasePVCModel

    def __init__(self, base=64, n_blocks=3, k=7):
        super().__init__()
        self.stem = LeadMixer(base)        # 247→base
        blocks = []
        for _ in range(n_blocks):
            blocks += [
                conv_bn_relu(base, base, k),
                nn.Conv1d(base, base, k, padding=(k - 1) // 2, bias=False),
                nn.BatchNorm1d(base),
            ]
        self.body = nn.Sequential(*blocks)
        self.act = nn.ReLU(inplace=True)
        self.out_channels = base

    def forward(self, x, mask=None):           # x: [B,247,L]
        x = self.stem(x)                       # [B,base,L]
        y = self.body(x) + x                   # residual (same length)
        y = self.act(y)
        z = masked_mean_time(y, mask)          # [B,base]  ← masked global mean
        return z

