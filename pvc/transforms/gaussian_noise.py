import torch
import torch.nn as nn

class GaussianNoise(nn.Module):
    """Additive Gaussian noise at a target SNR (dB)."""
    def __init__(self, snr_db: float = 20.0):
        super().__init__()
        self.snr = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_signal = x.pow(2).mean()
        p_noise  = p_signal / (10 ** (self.snr / 10))
        noise = torch.randn_like(x) * p_noise.sqrt()
        return x + noise
