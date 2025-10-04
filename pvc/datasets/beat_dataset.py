# pvc/datasets/beat_dataset.py
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset
from pvc.transforms import default_augment 

def load_npy(path):
    """
    Load a single .npy file containing a beat's data.
    Returns potvals, time_to_pvc, arr_len, and label.
    """
    data = np.load(path, allow_pickle=True).item()
    potvals = data['potvals']
    time_to_pvc = data['time_to_pvc']
    label = data['label']
    arr_len = potvals.shape[0]  # Length of the beat sequence
    return potvals, time_to_pvc, arr_len, label
    
class BeatDataset(Dataset):
    def __init__(self, root, file_list, augment=True, target_unit="s", adj_path: str | None = None, max_len: int | None = None):
        """
        target_unit: "s" | "ms" | "samples"
        """
        self.root = Path(root)
        self.files = file_list
        self.augment = default_augment(adj_path=adj_path) if augment else None
        self.max_len = max_len  # optional hard cap on length

        if target_unit == "s":
            self._y_scale = 1.0/1000.0  # Convert ms to seconds
        elif target_unit == "ms":
            self._y_scale = 1.0
    
        else:
            raise ValueError(f"unknown target_unit: {target_unit}")

    # --------------------------------------------------------------
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rel = Path(self.files[idx])
        path = self.root / rel

        # --------------- 1. LOAD -------------------------------
        x, y, _, _ = load_npy(path)
        y = y * self._y_scale        

        # --------------- 2. ORIENT & NORMALISE -----------------
        if x.shape[0] != 247:                  # time × leads → leads × time
            x = x.T
        assert x.shape[0] == 247, f"bad lead count in {path}"

        x = x.astype(np.float32)
        x -= np.median(x, axis=1, keepdims=True)
        x /= np.std(x, axis=1, keepdims=True) + 1e-6
        x = torch.from_numpy(x)
        if self.max_len is not None and x.shape[1] > self.max_len:
            # --------------- 3. RESAMPLE to max_len --------------------
            x = x.unsqueeze(0)            # [1,247,N]
            x = F.interpolate(x, size=self.max_len, mode="linear", align_corners=False)
            x = x.squeeze(0)                                     # [247,max_len]

        # --------------- 4. AUGMENT ----------------------------
        if self.augment:
            x = self.augment(x.unsqueeze(0)).squeeze(0)

        return {"potvals": x, "time_to_pvc": torch.tensor(y, dtype=torch.float32), "arr_len": torch.tensor(x.shape[1])}

