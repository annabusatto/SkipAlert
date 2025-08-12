import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Union

class GraphLaplacianBlur(nn.Module):
    """
    Laplacian-based spatial smoothing over the 247-electrode graph.

    x: [B, C=247, L]  ->  y = (I - alpha * L)^K x
      - L can be unnormalized (D - A) or normalized (I - D^{-1/2} A D^{-1/2})
      - alpha can be a float or a (lo, hi) range sampled per call
      - K = steps (small integer)

    Args:
      adj_path: path to 247x247 adjacency (.npy or .npz -> key 'adj' or array)
      mode: 'normalized' | 'unnormalized'
      alpha: float or (lo, hi) tuple; strength of smoothing
      steps: number of diffusion steps (K)
      p: probability to apply the augmentation
      clip: if True, mild clamp to prevent numeric drift
    """

    def __init__(self,
                 adj_path: Union[str, Path],
                 mode: str = "normalized",
                 alpha: Union[float, Tuple[float, float]] = (0.03, 0.12),
                 steps: int = 1,
                 p: float = 0.5,
                 clip: bool = True):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.steps = steps
        self.p = float(p)
        self.clip = clip

        A = self._load_adj(adj_path)                  # [247, 247] float tensor (CPU)
        A = 0.5 * (A + A.T)                           # symmetrize
        A.fill_diagonal_(0)

        d = A.sum(dim=1)
        if mode == "normalized":
            eps = 1e-8
            d_inv_sqrt = torch.pow(d + eps, -0.5)
            Dm12 = torch.diag(d_inv_sqrt)
            L = torch.eye(A.size(0)) - Dm12 @ A @ Dm12
        elif mode == "unnormalized":
            D = torch.diag(d)
            L = D - A
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.register_buffer("L", L.float(), persistent=False)

    def _load_adj(self, path: Union[str, Path]) -> torch.Tensor:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Adjacency not found at {p}")

        if p.suffix == ".pt":
            obj = torch.load(p, map_location="cpu")
            if isinstance(obj, torch.Tensor):
                A = obj
            elif isinstance(obj, dict):
                # common keys
                for k in ("adj", "A", "adjacency", "matrix"):
                    if k in obj and isinstance(obj[k], torch.Tensor):
                        A = obj[k]
                        break
                else:
                    raise ValueError(f"PT file at {p} didn't contain a tensor under keys "
                                     "['adj','A','adjacency','matrix']")
            else:
                raise ValueError(f"Unsupported .pt content type: {type(obj)}")
        elif p.suffix in (".npy", ".npz"):
            if p.suffix == ".npz":
                npz = np.load(p)
                for k in ("adj", "A", "adjacency"):
                    if k in npz:
                        A = npz[k]
                        break
                else:
                    A = npz[npz.files[0]]
            else:
                A = np.load(p)
            A = torch.from_numpy(np.asarray(A, dtype=np.float32))
        else:
            raise ValueError(f"Unsupported adjacency format: {p.suffix}")

        if A.ndim != 2 or A.shape[0] != 247 or A.shape[1] != 247:
            raise ValueError(f"Expected adjacency (247,247), got {tuple(A.shape)}")
        return A.float()

    def _sample_alpha(self) -> float:
        if isinstance(self.alpha, (tuple, list)):
            lo, hi = float(self.alpha[0]), float(self.alpha[1])
            return float(torch.empty(()).uniform_(lo, hi).item())
        return float(self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 247, L]
        if not self.training or torch.rand(()) > self.p:
            return x
        L = self.L.to(dtype=x.dtype, device=x.device)
        alpha = self._sample_alpha()
        y = x
        for _ in range(max(1, self.steps)):
            Ly = torch.einsum("ij,bjl->bil", L, y)  # graph diff along channels
            y = y - alpha * Ly
        if self.clip:
            y = torch.clamp(y, min=(x.min() - 5.0), max=(x.max() + 5.0))
        return y
