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
    def __init__(self, root, file_list, augment=True, target_unit="s", sample_rate=None, adj_path: str | None = None):
        """
        target_unit: "s" | "ms" | "samples"
        sample_rate: Hz (required if target_unit == "samples")
        """
        self.root = Path(root)
        self.files = file_list
        self.augment = default_augment(adj_path=adj_path) if augment else None

        if target_unit == "s":
            self._y_scale = 1.0/1000.0  # Convert ms to seconds
        elif target_unit == "ms":
            self._y_scale = 1.0
        elif target_unit == "samples":
            assert sample_rate, "sample_rate must be set when target_unit='samples'"
            self._y_scale = 1.0 / float(sample_rate)
        else:
            raise ValueError(f"unknown target_unit: {target_unit}")

    # --------------------------------------------------------------
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rel = Path(self.files[idx])
        path = self.root / rel

        # --------------- 1. LOAD -------------------------------
        signal, y, _, _ = load_npy(path)
        y = y * self._y_scale        

        # --------------- 2. ORIENT & NORMALISE -----------------
        if signal.shape[0] != 247:                  # time × leads → leads × time
            signal = signal.T
        assert signal.shape[0] == 247, f"bad lead count in {path}"

        signal = signal.astype(np.float32)
        signal -= np.median(signal, axis=1, keepdims=True)
        signal /= np.std(signal, axis=1, keepdims=True) + 1e-6

        # --------------- 3. RESAMPLE to 430 --------------------
        x = torch.from_numpy(signal).unsqueeze(0)            # [1,247,N]
        x = F.interpolate(x, size=430, mode="linear", align_corners=False)
        x = x.squeeze(0)                                     # [247,430]

        # --------------- 4. AUGMENT ----------------------------
        if self.augment:
            x = self.augment(x.unsqueeze(0)).squeeze(0)

        return x, torch.tensor(y, dtype=torch.float32)

