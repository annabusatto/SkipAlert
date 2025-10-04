# pvc/datasets/beat_sequence_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from pvc.transforms import default_augment 

def _load_sequence(path: Path) -> Dict:
    """
    Load one sequence from .npy. The sequence files store potvals, time_to_pvc,
    label, and sequence_length inside the same file.
    """
    arr = np.load(path, allow_pickle=True)
    if hasattr(arr, "item"):
        obj = arr.item()
        x = np.asarray(obj["potvals"], dtype=np.float32)    # [247, N_total]
        ttp = float(obj["time_to_pvc"])
        label = obj.get("label", "unknown")
        seq_len = int(obj.get("sequence_length", 1))
        return {"potvals": x, "time_to_pvc": ttp, "label": label, "sequence_length": seq_len}
    raise ValueError(f"Unsupported sequence file format: {path}")


class BeatSequenceDataset(Dataset):
    """
    Dataset for pre-created beat sequences.
    
    Each sequence file contains multiple beats concatenated together, with
    the time_to_pvc target being from the middle beat of the sequence.

    Parameters
    ----------
    root : str | Path
        Base directory that contains sequence files referenced by 'file_list'.
    file_list : list[str]
        Relative paths to sequence files for this split (train/val/test).
    augment : callable | torch.nn.Module | None
        Optional augmentation applied to the full sequence tensor [247, L].
    max_len : int | None
        Optional max crop length along time (center crop).
    """

    def __init__(
        self,
        root: str | Path,
        file_list: List[str],
        augment=True,
        adj_path: str | None = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.file_list = file_list
        self.augment = default_augment(adj_path=adj_path) if augment else None

    def __len__(self) -> int:
        return len(self.file_list)

    def _load_and_maybe_crop(self, rel: str) -> Tuple[torch.Tensor, float, str]:
        obj = _load_sequence(self.root / rel)
        x = obj["potvals"]  # np array [247, N_total]
        y = obj["time_to_pvc"] / 1000.0  # convert ms to seconds
        label = obj["label"]

        x = x.astype(np.float32)
        x -= np.median(x, axis=1, keepdims=True)
        x /= np.std(x, axis=1, keepdims=True) + 1e-6

        xt = torch.from_numpy(x).to(torch.float32)   # [247, N]
        if self.augment:
            xt = self.augment(xt)
        return xt, float(y), label

    def __getitem__(self, idx: int):
        rel = self.file_list[idx]
        x, y, label = self._load_and_maybe_crop(rel)
        
        return {
            "potvals": x,                              # [247, L_total]
            "time_to_pvc": torch.tensor(y, dtype=torch.float32),
            "label": label,                            # sequence label
            "seq_file": rel,                          # for logging if needed
        }
