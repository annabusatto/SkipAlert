import torch
import torch.nn as nn
import torch.nn.functional as F

class _AttnPool(nn.Module):
    """Additive attention over time axis for sequences [B, T, D]."""
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D]
        logits = self.score(x).squeeze(-1)  # [B, T]
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        w = torch.softmax(logits, dim=1)    # [B, T]
        return (w.unsqueeze(-1) * x).sum(dim=1)  # [B, D]


class BiLSTMAttnBackbone(nn.Module):
    """
    Bidirectional LSTM + attention backbone.

    Expects input x: [B, 247, L] (L typically 430). Returns features: [B, 2*hidden].
    """
    out_channels: int

    def __init__(self,
                 hidden: int = 128,
                 n_layers: int = 2,
                 dropout_lstm: float = 0.30,
                 dropout_head: float = 0.20,
                 bidirectional: bool = True):
        super().__init__()
        self.hidden = hidden
        self.bidirectional = bidirectional

        d_in = 247
        d_lstm = hidden
        num_dirs = 2 if bidirectional else 1
        d_out = d_lstm * num_dirs

        self.lstm = nn.LSTM(
            input_size=d_in,
            hidden_size=d_lstm,
            num_layers=n_layers,
            dropout=dropout_lstm if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,   # we will feed [B, T, D]
        )
        self.pool = _AttnPool(d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout_head)

        self.out_channels = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 247, L]  -> permute to [B, L, 247] for LSTM over time.
        Returns: [B, out_channels]
        """
        # to sequence-first for LSTM
        x = x.transpose(1, 2)  # [B, L, 247]
        # If you ever pass variable lengths, add a mask here; with fixed L we use all-True.
        y, _ = self.lstm(x)    # [B, L, 2H]
        # optional mask = torch.ones(y.size(0), y.size(1), dtype=torch.bool, device=y.device)
        z = self.pool(y, mask=None)   # [B, 2H]
        z = self.norm(z)
        z = self.dropout(z)
        return z
