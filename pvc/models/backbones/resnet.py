# pvc/models/backbones/resnet.py
import torch.nn as nn
from ..common import conv_bn_relu, LeadMixer

class ResNetBackbone(nn.Module):
    out_channels: int                      # to satisfy BasePVCModel

    def __init__(self, base=64, n_blocks=3, k=7):
        super().__init__()
        self.stem = LeadMixer(base)        # 247→base
        blocks = []
        for _ in range(n_blocks):
            blocks += [
                conv_bn_relu(base, base, k),
                conv_bn_relu(base, base, k)
            ]
        self.body = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))
        self.out_channels = base

    def forward(self, x):
        x = self.stem(x)                   # [B,base,430]
        x = self.body(x).squeeze(-1)       # [B,base]
        return x
