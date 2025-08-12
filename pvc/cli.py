# pvc/cli.py
# ---------------------------------------------------------------------------
# Entry-point that wires Hydra → Lightning → wandb.
#
# Run examples
# ------------
# 1) default (ResNet, fixed 80/10/10 split)
#       python -m pvc.cli
#
# 2) TCN, k-fold CV (fold_2)
#       python -m pvc.cli \
#           model=tcn \
#           data.split_dir=data/splits/k5 \
#           data.train_file=fold_2_train.json \
#           data.val_file=fold_2_val.json
#
# 3) disable wandb, enable fast_dev_run
#       python -m pvc.cli wandb=offline trainer.fast_dev_run=true
# ---------------------------------------------------------------------------
import os
import sys
import hydra
import lightning as L
import torch
import json
from torch.utils.data import DataLoader
from pvc.datasets.beat_dataset import BeatDataset
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# Local imports
from pvc.datamodules.beat_datamodule import BeatDataModule
from pvc.models.pvc_module import build_model

torch.set_float32_matmul_precision("high")  # or "medium" if you prefer

# ─────────────────────────────────────────────────────────────────────────────
def _build_callbacks(cfg):
    cb_list = []

    if "callbacks" in cfg and cfg.callbacks is not None:
        cbs = cfg.callbacks

        # Single callback: DictConfig
        if isinstance(cbs, DictConfig):
            cb_list.append(instantiate(cbs, _recursive_=False))

        # Multiple callbacks: ListConfig
        elif isinstance(cbs, ListConfig):
            for cb in cbs:
                cb_list.append(instantiate(cb, _recursive_=False))

        else:
            raise TypeError(f"callbacks must be DictConfig or ListConfig, got {type(cbs)}")
    ckpt = ModelCheckpoint(
        monitor="val_mae", mode="min", save_top_k=3, save_last=True,
        dirpath=Path(cfg.paths.ckpt_dir).as_posix(),
        filename=cfg.paths.ckpt_name,
        auto_insert_metric_name=False,  # we already put {val_mae} in filename if we want
    )
    # Always add checkpoint + LR monitor (optional)
    cb_list += [ckpt,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    return cb_list


def _build_logger(cfg: DictConfig):
    """Return a Lightning logger (wandb by default)."""
    if cfg.wandb.mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
    return WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity", None),
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        save_code=cfg.wandb.get("save_code", True),
        log_model=cfg.wandb.get("log_model", False),
    )


# ─────────────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # Pretty-print config
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Reproducibility
    L.seed_everything(cfg.seed, workers=True)

    # DATA
    dm = BeatDataModule(
    data_dir   = cfg.split.data_dir,
    split_dir  = cfg.split.split_dir,
    train_file = cfg.split.train_file,
    val_file   = cfg.split.val_file,
    test_file  = cfg.split.get("test_file"),
    batch_size = cfg.data.batch_size,
    num_workers= cfg.data.num_workers,
    pin_memory = cfg.data.pin_memory,
    # optional label scaling, if you added it to the dataset
    target_unit= cfg.data.get("target_unit", "s"),
    sample_rate= cfg.data.get("sample_rate", None),
    )

    # MODEL
    model = build_model(cfg.model.name, **cfg.model.params, T_max=cfg.trainer.epochs)

    # LOGGER & CALLBACKS
    logger = _build_logger(cfg)
    callbacks = _build_callbacks(cfg)

    # TRAINER
    trainer = L.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.trainer.gpus,
        precision=cfg.trainer.get("precision", 32),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 0.0),
        fast_dev_run=cfg.trainer.get("fast_dev_run", False),
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=cfg.trainer.get("enable_progress_bar", False),
    )

    # FIT
    trainer.fit(model, dm)

    def _best_ckpt_path(trainer):
        cb = getattr(trainer, "checkpoint_callback", None)
        if cb is not None and getattr(cb, "best_model_path", ""):
            return cb.best_model_path
        return None

    best_ckpt = _best_ckpt_path(trainer)


    # ─── Adapt on held-out 10% if present ───────────────────────────────────────
    adapt_file = getattr(cfg.split, "adapt_file", None)
    if adapt_file:
        adapt_list = json.load(open(Path(cfg.split.split_dir) / adapt_file))
        adapt_ds = BeatDataset(cfg.split.data_dir, adapt_list, augment=False)
        adapt_loader = DataLoader(
            adapt_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )

        # lower LR for quick adaptation (e.g., 10× smaller)
        for g in trainer.optimizers[0].param_groups:
            g["lr"] = float(cfg.model.params.lr) * 0.1

        adapt_trainer = L.Trainer(
            max_epochs=20,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=cfg.trainer.gpus,
            default_root_dir=cfg.paths.run_dir,
            enable_checkpointing=False,   # no checkpointing during adapt
            logger=trainer.logger,
            enable_progress_bar=cfg.trainer.get("enable_progress_bar", False),
        )

        # If we have a path to the best ckpt from the main training, resume from it.
        if best_ckpt:
            trainer.test(model, datamodule=dm, ckpt_path=best_ckpt)  # pre-adapt
            adapt_trainer.fit(model, train_dataloaders=adapt_loader, ckpt_path=best_ckpt)
        else:
            # Fall back to continuing from in-memory weights
            adapt_trainer.fit(model, train_dataloaders=adapt_loader)

    # ─── Final test on the CURRENT (possibly adapted) weights ───────────────────
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    sys.exit(main())
