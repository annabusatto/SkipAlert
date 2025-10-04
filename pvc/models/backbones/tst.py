# pvc/models/backbones/tst_backbone.py
import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SinusoidalPositionalEmbedding(nn.Module):
    """Classic sin/cos PE that can serve any length on the fly."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)                     # [max_len, d]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: [1, max_len, d] for easy broadcasting
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        # returns [1, length, d_model]
        if length > self.pe.size(1):
            # lazy-grow if caller requests longer sequences
            self._extend(length)
        return self.pe[:, :length]

    def _extend(self, new_max: int):
        d_model = self.d_model
        pe = torch.zeros(new_max, d_model, device=self.pe.device)
        position = torch.arange(0, new_max, dtype=torch.float, device=self.pe.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.pe.device).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, new_max, d]


class TSTBackbone(nn.Module):
    """
    Time Series Transformer with a CLS token, mask-aware encoder, and
    sinusoidal positional embeddings for variable-length beats.

    Inputs
    ------
    x    : [B, 247, L]
    mask : [B, L] bool, True for valid timesteps, False for padding (optional)

    Output
    ------
    z    : [B, d_model]  (CLS embedding)
    """
    def __init__(self, d_model=64, nhead=4, nlayers=4, dropout=0.1, pe_max_len: int = 4096):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.proj = nn.Linear(247, d_model)  # lead mixing (per time step)
        enc_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,     # <— important: we use [B, S, d]
            norm_first=True
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb   = SinusoidalPositionalEmbedding(d_model, max_len=pe_max_len)
        self.out_channels = d_model
        self.norm_out = nn.LayerNorm(d_model)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    [B, 247, L] (variable L)
        mask: [B, L] bool, True = valid (optional)
        """
        B, C, L = x.shape
        # 1) time-major + linear projection of leads -> d_model
        x = self.proj(x.transpose(1, 2))              # [B, L, d]

        # 2) prepend CLS and add positional encodings
        cls = self.cls_token.expand(B, 1, -1)         # [B, 1, d]
        x = torch.cat([cls, x], dim=1)                # [B, L+1, d]
        x = x + self.pos_emb(L + 1)                   # broadcast PE

        # 3) key padding mask for encoder (True = ignore)
        # our collate mask is True=valid, so invert for padding
        if mask is not None:
            assert mask.shape == (B, L)
            pad = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)  # CLS never padded
            kpm = torch.cat([pad, ~mask], dim=1)        # [B, L+1]
        else:
            kpm = None

        # 4) encode & take CLS
        h = self.encoder(x, src_key_padding_mask=kpm)  # [B, L+1, d]
        z = h[:, 0, :]                                 # [B, d]
        return self.norm_out(z)
