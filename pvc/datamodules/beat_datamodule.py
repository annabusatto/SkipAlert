# pvc/datamodules/beat_datamodule.py
import json
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader
from pvc.datasets.beat_dataset import BeatDataset


class BeatDataModule(L.LightningDataModule):
    def __init__(self, data_dir, split_dir, train_file="train.json",
                 val_file="val.json", test_file=None, batch_size=256,
                 num_workers=8, pin_memory=True, **_ignore):
        super().__init__()
        self.save_hyperparameters(logger=False)


    # ───────────────── helpers ─────────────────
    def _load_json_list(self, name):
        f = Path(self.hparams.split_dir) / name
        with open(f) as fp:
            return json.load(fp)                    # list[str]

    def _make_ds(self, file_list, augment):
        return BeatDataset(
            self.hparams.data_dir,
            file_list,
            augment=augment,
            target_unit=self.hparams.get("target_unit", "s"),
            sample_rate=self.hparams.get("sample_rate", None),
            adj_path=self.hparams.get("adj_path", None),
        )

    # ───────────────── Lightning API ─────────────────
    def setup(self, stage=None):
        if stage in (None, "fit"):
            tr_files = self._load_json_list(self.hparams.train_file)
            va_files = self._load_json_list(self.hparams.val_file)
            self.train_ds = self._make_ds(tr_files, augment=True)
            self.val_ds   = self._make_ds(va_files, augment=False)

        if stage in (None, "test", "predict") and self.hparams.test_file:
            te_files = self._load_json_list(self.hparams.test_file)
            self.test_ds = self._make_ds(te_files, augment=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.num_workers > 0)

    def test_dataloader(self):
        if hasattr(self, "test_ds"):
            return DataLoader(self.test_ds,
                              batch_size=self.hparams.batch_size,
                              shuffle=False,
                              num_workers=self.hparams.num_workers,
                              pin_memory=self.hparams.pin_memory)
