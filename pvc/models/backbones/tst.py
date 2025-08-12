# pvc/models/backbones/tst.py
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TSTBackbone(nn.Module):
    def __init__(self, d_model=64, nhead=4, nlayers=4, dropout=0.1):
        super().__init__()
        self.proj   = nn.Linear(247, d_model)          # lead mixing
        enc_layer   = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = TransformerEncoder(enc_layer, nlayers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb   = nn.Parameter(torch.randn(1, 431, d_model))  # 430+CLS
        self.out_channels = d_model

    def forward(self, x):                              # x: [B,247,430]
        x = self.proj(x.transpose(1, 2))               # [B,430,d]
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_emb[:, :x.size(1)+1]
        z = self.encoder(x)[:, 0]                      # CLS output  [B,d]
        return z
