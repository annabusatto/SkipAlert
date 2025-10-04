# pvc/datamodules/beat_sequence_datamodule.py
from __future__ import annotations

import json
from pathlib import Path
from functools import partial
import lightning as L
from torch.utils.data import DataLoader

from pvc.datasets.beat_sequence_dataset import BeatSequenceDataset
from pvc.datasets.collate import pad_collate_seq_concat


class BeatSequenceDataModule(L.LightningDataModule):
    """
    LightningDataModule for pre-created beat sequences.

    Each sequence file contains concatenated beats with the time_to_pvc 
    target from the middle beat of the sequence.
    """

    def __init__(
        self,
        data_dir: str,
        split_dir: str,
        train_file: str = "train.json",
        val_file: str = "val.json",
        test_file: str | None = None,
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        # variable-length controls
        pad_value: float = 0.0,
        # augment hook (per-sequence)
        use_augment: bool = True,
        augment_train = None,          # callable or nn.Module applied per sequence
        augment_eval  = None,          # callable or nn.Module for val/test
        **_ignore,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # use concat collate function since sequences are already concatenated
        self.collate_fn = partial(pad_collate_seq_concat, pad_value=pad_value)

        # (optional) keep augment handles
        self._aug_train = augment_train if use_augment else None
        self._aug_eval  = augment_eval  if use_augment else None

        self.train_ds = None
        self.val_ds   = None
        self.test_ds  = None

    # ───────── helpers ─────────
    def _load_json_list(self, name: str) -> list[str]:
        path = Path(self.hparams.split_dir) / name
        with path.open() as fp:
            return json.load(fp)

    def _make_ds(self, files: list[str], augment):
        return BeatSequenceDataset(
            root=self.hparams.data_dir,
            file_list=files,
            augment=augment,
        )

    # ───────── Lightning API ─────────
    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            tr_files = self._load_json_list(self.hparams.train_file)
            va_files = self._load_json_list(self.hparams.val_file)
            self.train_ds = self._make_ds(tr_files, augment=self._aug_train)
            self.val_ds   = self._make_ds(va_files,   augment=self._aug_eval)

        if stage in (None, "test", "predict") and self.hparams.test_file:
            te_files = self._load_json_list(self.hparams.test_file)
            self.test_ds = self._make_ds(te_files, augment=self._aug_eval)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return self.val_dataloader()
