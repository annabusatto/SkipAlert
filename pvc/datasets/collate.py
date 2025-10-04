# pvc/datasets/collate.py
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any

def pad_collate(batch, pad_value: float = 0.0, dtype: torch.dtype = torch.float32):
    """
    batch: list of dicts with keys:
      'potvals': [C,L_i] (np.array or torch.Tensor)
      'time_to_pvc': scalar (np/torch)
      optional: 'label'/'id' etc.
    Returns:
      dict with:
        'potvals': [B,C,Lmax] torch.float32
        'time_to_pvc': [B] torch.float32
        'arr_len': [B] torch.long
        'mask': [B,Lmax] bool  (True=valid)
        (optional) 'labels': list[str] if present in items
    """
    B = len(batch)

    # ---- normalize types ----
    xs = []
    ys = []
    labels: Optional[list] = None
    for i, b in enumerate(batch):
        x = b["potvals"]
        # coerce to torch tensor (ensures torch dtype, not numpy)
        x = torch.as_tensor(x, dtype=dtype)          # [C, L_i]
        xs.append(x)

        y = torch.as_tensor(b["time_to_pvc"], dtype=torch.float32)
        ys.append(y)

        # carry along string labels if present (optional)
        if "label" in b or "id" in b:
            if labels is None:
                labels = []
            labels.append(b.get("label", b.get("id", "")))

    C = xs[0].shape[0]
    lengths = torch.tensor([x.shape[1] for x in xs], dtype=torch.long)  # [B]
    Lmax = int(lengths.max())

    # ---- pad to max length in batch ----
    xpad = torch.full((B, C, Lmax), fill_value=float(pad_value), dtype=dtype)
    for i, x in enumerate(xs):
        Li = x.shape[1]
        xpad[i, :, :Li] = x

    mask = torch.arange(Lmax).unsqueeze(0) < lengths.unsqueeze(1)  # [B,Lmax], bool
    y = torch.stack(ys, dim=0)                                     # [B]

    out = {"potvals": xpad, "time_to_pvc": y, "arr_len": lengths, "mask": mask}
    if labels is not None:
        out["labels"] = labels
    return out



def pad_collate_seq(batch: List[Dict[str, Any]], pad_value: float = 0.0):
    """
    Collate for BeatSequenceDataset with pack='stack'.
    Input items have:
      - beats: list[Tensor [247, Lj]] with length S
      - time_to_pvc: float tensor scalar
    Output:
      - potvals: [B, S, 247, Lmax]
      - mask:    [B, S, Lmax]  (True=valid)
      - arr_len: [B, S]        (per-beat lengths)
      - time_to_pvc: [B]
      - seq_files: list[list[str]] (optional, for logging)
    """
    B = len(batch)
    S = len(batch[0]["beats"])
    C = batch[0]["beats"][0].shape[0]  # 247
    lengths = torch.tensor([[b.shape[1] for b in item["beats"]] for item in batch], dtype=torch.long)  # [B,S]
    Lmax = int(lengths.max().item())

    x = torch.full((B, S, C, Lmax), float(pad_value), dtype=torch.float32)
    mask = torch.zeros((B, S, Lmax), dtype=torch.bool)
    for i, item in enumerate(batch):
        for s, beat in enumerate(item["beats"]):
            Li = beat.shape[1]
            x[i, s, :, :Li] = beat
            mask[i, s, :Li] = True

    y = torch.stack([item["time_to_pvc"] for item in batch], dim=0).float()
    out = {"potvals": x, "mask": mask, "arr_len": lengths, "time_to_pvc": y}
    if "seq_files" in batch[0]:
        out["seq_files"] = [item["seq_files"] for item in batch]
    return out


def pad_collate_seq_concat(batch: List[Dict[str, Any]], pad_value: float = 0.0):
    """
    Collate for BeatSequenceDataset with pack='concat'.
    Input items have:
      - potvals: Tensor [247, L_total]
      - seg_lens: [S]
      - time_to_pvc: scalar
    Output:
      - potvals: [B, 247, Lmax]
      - mask:    [B, Lmax]  (True=valid)
      - arr_len: [B]        (total lengths)
      - seg_lens: list[Tensor [S]] (kept for optional use)
    """
    B = len(batch)
    C = batch[0]["potvals"].shape[0]
    lengths = torch.tensor([item["potvals"].shape[1] for item in batch], dtype=torch.long)  # [B]
    Lmax = int(lengths.max().item())

    x = torch.full((B, C, Lmax), float(pad_value), dtype=torch.float32)
    mask = torch.zeros((B, Lmax), dtype=torch.bool)
    for i, item in enumerate(batch):
        Li = item["potvals"].shape[1]
        x[i, :, :Li] = item["potvals"]
        mask[i, :Li] = True

    y = torch.stack([item["time_to_pvc"] for item in batch], dim=0).float()
    out = {"potvals": x, "mask": mask, "arr_len": lengths, "time_to_pvc": y}
    if "seg_lens" in batch[0]:
        out["seg_lens"] = [item["seg_lens"] for item in batch]
    if "seq_files" in batch[0]:
        out["seq_files"] = [item["seq_files"] for item in batch]
    return out

