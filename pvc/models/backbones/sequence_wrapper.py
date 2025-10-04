import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def _has_mask_arg(module: nn.Module) -> bool:
    # Checks if module.forward(..., mask=...) is supported
    return "mask" in module.forward.__code__.co_varnames

def _masked_mean_over_beats(z: torch.Tensor, beat_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    z         : [B, S, F]
    beat_mask : [B, S] bool (True = valid) or None
    returns   : [B, F]
    """
    if beat_mask is None:
        return z.mean(dim=1)
    m = beat_mask.to(z.dtype).unsqueeze(-1)     # [B,S,1]
    num = (z * m).sum(dim=1)                    # [B,F]
    den = m.sum(dim=1).clamp_min(1e-6)         # [B,1]
    return num / den

class AttnOverBeats(nn.Module):
    """
    Lightweight attention over S beats.
    Input : [B, S, F], optional mask [B, S] (True=valid)
    Output: [B, F]
    """
    def __init__(self, d: int, p_drop: float = 0.0):
        super().__init__()
        self.norm   = nn.LayerNorm(d)
        self.score  = nn.Linear(d, 1)
        self.drop   = nn.Dropout(p_drop)

    def forward(self, z: torch.Tensor, beat_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # z: [B,S,F], beat_mask: [B,S] (True=valid)
        h = self.drop(self.norm(z))                     # [B,S,F]
        logits = self.score(h).squeeze(-1)              # [B,S]
        if beat_mask is not None:
            logits = logits.masked_fill(~beat_mask, -1e9)
        w = torch.softmax(logits, dim=1)                # [B,S]
        return (z * w.unsqueeze(-1)).sum(dim=1)         # [B,F]

class SequenceWrapper(nn.Module):
    """
    Wrap any single-beat backbone (expects [B, 247, L]) so it can process
    sequences of S beats: [B, S, 247, L]. It runs the backbone per beat,
    then aggregates across beats with 'mean' or 'attn'.

    Parameters
    ----------
    backbone : nn.Module
        Must output [B, F] and define `out_channels = F`.
        If it accepts a 'mask' kwarg, we will pass a per-beat mask down.
    agg : {"mean","attn"}
        How to combine S embeddings into one.
    attn_dropout : float
        Dropout inside the attention (if agg='attn').
    """
    out_channels: int

    def __init__(self, backbone: nn.Module, agg: str = "attn", attn_dropout: float = 0.0):
        super().__init__()
        assert hasattr(backbone, "out_channels"), "Backbone must set .out_channels"
        assert agg in ("mean", "attn"), "agg must be 'mean' or 'attn'"
        self.backbone = backbone
        self.agg_type = agg
        Fdim = backbone.out_channels
        self.out_channels = Fdim
        self.attn = AttnOverBeats(Fdim, p_drop=attn_dropout) if agg == "attn" else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x    : [B, S, 247, L]
        mask : [B, S, L] (True=valid), optional
        -> z : [B, F]
        """
        assert x.ndim == 4, f"Expected x [B,S,247,L], got {x.shape}"
        B, S, C, L = x.shape

        # Flatten beats → run backbone per beat
        x_flat = x.reshape(B * S, C, L)                # [B*S, 247, L]
        if mask is not None:
            assert mask.shape == (B, S, L)
            mask_flat = mask.reshape(B * S, L)         # [B*S, L]
        else:
            mask_flat = None

        if _has_mask_arg(self.backbone) and mask_flat is not None:
            z_flat = self.backbone(x_flat, mask=mask_flat)   # [B*S, F]
        else:
            z_flat = self.backbone(x_flat)                   # [B*S, F]

        # Restore sequence dimension
        z_seq = z_flat.view(B, S, -1)                        # [B,S,F]

        # Beat-level mask (a beat is valid if it has any valid time-step)
        beat_mask = None
        if mask is not None:
            beat_mask = mask.any(dim=2)                      # [B,S]

        # Aggregate across beats
        if self.agg_type == "mean":
            z = _masked_mean_over_beats(z_seq, beat_mask)    # [B,F]
        else:  # attention
            z = self.attn(z_seq, beat_mask=beat_mask)        # [B,F]

        return z
