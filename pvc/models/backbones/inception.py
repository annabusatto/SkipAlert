# pvc/models/backbones/inception.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common import LeadMixer, conv_bn_relu, masked_mean_time  # keep 1x1 conv helper

class _DWConvSame1d(nn.Module):
    """
    Depthwise 1D conv with explicit SAME padding (handles even/odd kernels).
    Keeps length L unchanged for stride=1, dilation>=1.
    in_ch == out_ch and groups == in_ch (depthwise).
    """
    def __init__(self, ch: int, k: int, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.k = int(k)
        self.d = int(dilation)
        self.pad_total = self.d * (self.k - 1)
        self.left = self.pad_total // 2
        self.right = self.pad_total - self.left
        self.conv = nn.Conv1d(ch, ch, kernel_size=self.k, dilation=self.d,
                              padding=0, groups=ch, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SAME padding for even/odd kernels
        x = F.pad(x, (self.left, self.right))
        return self.conv(x)

class _InceptionBlock(nn.Module):
    """
    Inception-style block with 3 branches at different kernel sizes.
    Each branch: depthwise separable conv (DW kx1) → pointwise 1x1.
    Time length preserved even for even kernels.
    """
    def __init__(self, cin: int, branch_ch: int = 32, ks: tuple[int, ...] = (10, 20, 40)):
        super().__init__()
        branches = []
        for k in ks:
            branches.append(nn.Sequential(
                _DWConvSame1d(cin, k=k),                 # depthwise SAME conv [B,cin,L]
                conv_bn_relu(cin, branch_ch, k=1, padding=0)  # pointwise 1x1 → branch_ch
            ))
        self.branches = nn.ModuleList(branches)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # concat along channels: 3 * branch_ch, length preserved
        return self.relu(torch.cat([b(x) for b in self.branches], dim=1))

class InceptionBackbone(nn.Module):
    """
    Inception-style temporal CNN with mask-aware global pooling.

    Input:
      x   : [B, 247, L]
      mask: [B, L] bool (True=valid), optional

    Output:
      z   : [B, out_channels] where out_channels = 3 * branch_channels
    """
    def __init__(self, branch_channels: int = 32, n_blocks: int = 6, ks: tuple[int, ...] = (10, 20, 40)):
        super().__init__()
        base = branch_channels * 3                       # channels after each Inception block
        self.stem = LeadMixer(base)                      # 247 → base (1x1 conv+BN+ReLU)
        blocks = []
        cin = base
        for _ in range(n_blocks):
            blk = _InceptionBlock(cin, branch_ch=branch_channels, ks=ks)
            blocks.append(blk)
            cin = branch_channels * 3                    # stays constant per block
        self.body = nn.ModuleList(blocks)
        self.out_channels = cin                          # = 3 * branch_channels

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B,247,L]
        x = self.stem(x)                 # [B, base, L]
        for blk in self.body:
            x = blk(x)                   # [B, 3*branch_ch, L]  (length preserved)
        z = masked_mean_time(x, mask)   # [B, 3*branch_ch]
        return z
