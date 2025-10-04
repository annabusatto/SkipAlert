import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class _AttnPool(nn.Module):
    """Additive attention over time axis for sequences [B, T, D]."""
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T] (True = valid)
        logits = self.score(x).squeeze(-1)        # [B, T]
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        w = torch.softmax(logits, dim=1)          # [B, T]
        return (w.unsqueeze(-1) * x).sum(dim=1)   # [B, D]


class BiLSTMAttnBackbone(nn.Module):
    """
    Bidirectional LSTM + attention backbone (variable-length aware).

    Expects:
      x:    [B, 247, L]
      mask: [B, L] bool (True = valid), optional

    Returns:
      z: [B, 2*hidden]
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
            batch_first=True,   # we feed [B, T, D]
        )
        self.pool = _AttnPool(d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout_head)

        self.out_channels = d_out

    @staticmethod
    def _lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
        # mask: [B, T] (True=valid) → lengths on CPU int64
        return mask.sum(dim=1).to(dtype=torch.long).cpu()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, 247, L]  -> LSTM expects [B, T, D] = [B, L, 247]
        mask: [B, L] (optional). If None, uses all timesteps.
        """
        # To sequence-first for LSTM
        x = x.transpose(1, 2)  # [B, L, 247]
        B, T, _ = x.shape

        if mask is not None:
            # Efficiently skip padded tail inside the LSTM
            lengths = self._lengths_from_mask(mask)
            # pack -> LSTM -> unpack (keeps batch_first)
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            y, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)  # [B, T, 2H]
            # Attention with the same mask (True = valid)
            z = self.pool(y, mask=mask)
        else:
            # Fixed-length (or already-trimmed) path
            y, _ = self.lstm(x)                 # [B, L, 2H]
            # All timesteps valid
            z = self.pool(y, mask=None)

        z = self.norm(z)
        z = self.dropout(z)
        return z
