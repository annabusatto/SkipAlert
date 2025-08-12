# pvc/models/backbones/inception.py
import torch
import torch.nn as nn
from ..common import conv_bn_relu, LeadMixer

class _InceptionBlock(nn.Module):
    def __init__(self, cin, branch_ch=32, ks=(10, 20, 40)):
        super().__init__()
        branches = []
        for k in ks:
            branches.append(nn.Sequential(
                conv_bn_relu(cin, cin, k, groups=cin),   # depth-wise
                conv_bn_relu(cin, branch_ch, k=1, padding=0)
            ))
        self.branches = nn.ModuleList(branches)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(torch.cat([b(x) for b in self.branches], dim=1))

class InceptionBackbone(nn.Module):
    def __init__(self, branch_channels=32, n_blocks=6):
        super().__init__()
        base = branch_channels * 3                     # 3 branches
        self.stem = LeadMixer(base)
        blocks = []
        cin = base
        for _ in range(n_blocks):
            blk = _InceptionBlock(cin, branch_channels)
            blocks.append(blk)
            cin = branch_channels * 3                  # stays constant
        self.body = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))
        self.out_channels = cin

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x).squeeze(-1)                   # [B, 96]
        return x
