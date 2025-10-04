# pvc/models/backbones/conv_attn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _same_pad_1d(k: int) -> int:
    assert k % 2 == 1, "Use an odd kernel size to preserve length"
    return (k - 1) // 2

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, padding=_same_pad_1d(k))
        self.bn   = nn.BatchNorm1d(c_out)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Two convs with residual; keeps length and channels."""
    def __init__(self, c: int, k: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv1 = ConvBNAct(c, c, k)
        self.conv2 = nn.Sequential(
            nn.Conv1d(c, c, kernel_size=k, padding=_same_pad_1d(k)),
            nn.BatchNorm1d(c),
        )
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        # Defensive guard (shouldn't trigger with odd k)
        if y.size(-1) != x.size(-1):
            L = x.size(-1)
            diff = y.size(-1) - L
            if diff > 0:
                left = diff // 2
                y = y[..., left:left+L]
            else:
                r = -diff
                y = F.pad(y, (r - r // 2, r // 2))
        return self.act(y + x)

class AttnPool1D(nn.Module):
    """Additive attention over time for feature maps [B, C, L]."""
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, C, L], mask: [B, L] (True=valid) or None
        logits = self.score(x).squeeze(1)  # [B, L]
        if mask is not None:
            # Defensive alignment if needed
            if mask.size(1) != logits.size(1):
                Lx, Lm = logits.size(1), mask.size(1)
                if Lm > Lx:
                    off = (Lm - Lx) // 2
                    mask = mask[:, off:off+Lx]
                else:
                    pad = Lx - Lm
                    mask = F.pad(mask, (pad - pad // 2, pad // 2))
            logits = logits.masked_fill(~mask, -1e9)
        w = torch.softmax(logits, dim=-1)           # [B, L]
        return (x * w.unsqueeze(1)).sum(dim=-1)     # [B, C]

class ConvAttnBackbone(nn.Module):
    """
    Temporal CNN + attention pooling over time (variable-length aware).

    Input:
      x   : [B, 247, L]
      mask: [B, L] bool (True=valid), optional

    Output:
      z   : [B, out_channels]
    """
    out_channels: int

    def __init__(self,
                 base: int = 64,
                 n_blocks: int = 4,
                 k: int = 7,
                 dropout: float = 0.10,
                 stem_k: int = 1):
        super().__init__()
        # Stem: mix 247 leads into 'base' channels via 1x1 (or small) conv
        self.stem = ConvBNAct(247, base, k=stem_k if stem_k % 2 == 1 else 1)

        # Body: residual temporal conv blocks (stride=1, same length)
        self.body = nn.Sequential(*[ResidualBlock(base, k=k, dropout=dropout)
                                    for _ in range(n_blocks)])

        # Attention pooling over time -> [B, base]
        self.pool = AttnPool1D(base)

        # Post-pool norm/dropout for a clean embedding
        self.norm = nn.LayerNorm(base)
        self.post = nn.Dropout(dropout)

        self.out_channels = base

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, 247, L]; mask: [B, L] or None
        x = self.stem(x)          # [B, base, L]
        x = self.body(x)          # [B, base, L]
        z = self.pool(x, mask=mask)  # [B, base]
        z = self.norm(z)
        z = self.post(z)
        return z
