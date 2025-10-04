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
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
from pvc.datasets.beat_dataset import BeatDataset
from pvc.datasets.beat_sequence_dataset import BeatSequenceDataset
from pvc.datasets.collate import pad_collate, pad_collate_seq_concat
from pathlib import Path
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# Local imports
from pvc.datamodules.beat_datamodule import BeatDataModule
from pvc.datamodules.beat_sequence_datamodule import BeatSequenceDataModule
from pvc.models.pvc_module import build_model

torch.set_float32_matmul_precision("high")  # or "medium" if you prefer

# ─────────────────────────────────────────────────────────────────────────────
def _build_callbacks(cfg):
    cb_list = []
    monitor="val/mae" if cfg.model.params.task == "regression" else "val/acc"
    mode= "min" if cfg.model.params.task == "regression" else "max"

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
        monitor=monitor, mode=mode, save_top_k=3, save_last=True,
        dirpath=Path(cfg.paths.ckpt_dir).as_posix(),
        filename=cfg.paths.ckpt_name,
        auto_insert_metric_name=False,  # we already put {val_mae} in filename if we want
    )
    # Always add checkpoint + LR monitor (optional)
    cb_list += [ckpt,
        LearningRateMonitor(logging_interval="epoch"),
    ]
    return cb_list


def _find_latest_checkpoint(ckpt_dir: str, target_epochs: int) -> tuple:
    """Find the latest checkpoint in the checkpoint directory.
    Returns (checkpoint_path, is_training_complete)
    """
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None, False
    
    # Look for checkpoint files (.ckpt extension)
    ckpt_files = list(ckpt_path.glob("*.ckpt"))
    if not ckpt_files:
        return None, False
    
    # Sort by modification time and get the latest
    latest_ckpt = max(ckpt_files, key=lambda f: f.stat().st_mtime)
    
    # Load the checkpoint to check the actual epoch number
    try:
        checkpoint = torch.load(latest_ckpt, map_location="cpu")
        current_epoch = checkpoint.get("epoch", 0)
        
        # Training is complete if we've reached the target epochs
        training_complete = current_epoch >= target_epochs
        
        if training_complete:
            print(f"Found checkpoint at epoch {current_epoch}/{target_epochs} - training complete: {latest_ckpt}")
        else:
            print(f"Found checkpoint at epoch {current_epoch}/{target_epochs} - will resume training: {latest_ckpt}")
            
        return str(latest_ckpt), training_complete
        
    except Exception as e:
        print(f"Error reading checkpoint {latest_ckpt}: {e}")
        print(f"Will use checkpoint anyway: {latest_ckpt}")
        return str(latest_ckpt), False


def _check_if_results_exist(cfg: DictConfig) -> bool:
    """Check if test results already exist for this run."""
    project_root = Path.cwd()
    out_dir = project_root / "results" / cfg.wandb.project / cfg.run_name
    out_csv = out_dir / f"{cfg.run_name}_test_predictions.csv"
    metrics_path = out_dir / f"{cfg.run_name}_test_metrics.json"
    
    if out_csv.exists() and metrics_path.exists():
        print(f"Results already exist: {out_csv}")
        return True
    return False


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
    kind = cfg.data.get("kind", "single")   # "single" or "sequence"

    if kind == "single":
        dm = BeatDataModule(
            data_dir       = cfg.split.data_dir,
            split_dir      = cfg.split.split_dir,
            train_file     = cfg.split.train_file,
            val_file       = cfg.split.val_file,
            test_file      = cfg.split.get("test_file"),
            batch_size     = cfg.data.batch_size,
            num_workers    = cfg.data.num_workers,
            pin_memory     = cfg.data.pin_memory,
            target_unit    = cfg.data.get("target_unit", "s"),
            variable_length= cfg.data.get("variable_length", True),
            pad_value      = cfg.data.get("pad_value", 0.0),
            adj_path       = cfg.data.get("adj_path", None),
        )
        adapt_collate = pad_collate if cfg.data.get("variable_length", True) else None

    elif kind == "sequence":
        dm = BeatSequenceDataModule(
            data_dir       = cfg.split.data_dir,
            split_dir      = cfg.split.split_dir,
            train_file     = cfg.split.train_file,
            val_file       = cfg.split.val_file,
            test_file      = cfg.split.get("test_file"),
            batch_size     = cfg.data.batch_size,
            num_workers    = cfg.data.num_workers,
            pin_memory     = cfg.data.pin_memory,
            pad_value      = cfg.data.get("pad_value", 0.0),
            use_augment    = cfg.data.get("use_augment", True),
            augment_train  = None,   # plug your per-sequence pipeline if desired
            augment_eval   = None,
        )
        # sequences are pre-concatenated, so use concat collate
        adapt_collate = partial(pad_collate_seq_concat, pad_value=cfg.data.get("pad_value", 0.0))
    else:
        raise ValueError("cfg.data.kind must be 'single' or 'sequence'")


    # Check if results already exist
    if _check_if_results_exist(cfg):
        print("Results already exist for this run. Skipping training and testing.")
        return

    # Debug: Print checkpoint directory
    print(f"Looking for checkpoints in: {cfg.paths.ckpt_dir}")
    
    # Check for existing checkpoint to resume from
    resume_ckpt_path, training_complete = _find_latest_checkpoint(cfg.paths.ckpt_dir, cfg.trainer.epochs)
    print(f"Checkpoint analysis: path={resume_ckpt_path}, complete={training_complete}")
    
    if training_complete:
        print(f"Training is complete ({cfg.trainer.epochs} epochs). Skipping to testing phase.")
        skip_training = True
    else:
        skip_training = False
        if resume_ckpt_path:
            print(f"Resuming training from checkpoint to reach {cfg.trainer.epochs} epochs")
        else:
            print(f"No existing checkpoint found. Starting training from scratch for {cfg.trainer.epochs} epochs.")

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

    # FIT - either resume training or skip if already complete
    if not skip_training:
        trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt_path)
    elif resume_ckpt_path:
        # Load the completed model for testing
        try:
            checkpoint = torch.load(resume_ckpt_path, map_location=model.device)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Successfully loaded completed model from {resume_ckpt_path}")
        except Exception as e:
            print(f"Error loading checkpoint {resume_ckpt_path}: {e}")
            print("Training will be restarted from scratch.")
            trainer.fit(model, datamodule=dm)

    def _best_ckpt_path(trainer):
        cb = getattr(trainer, "checkpoint_callback", None)
        if cb is not None and getattr(cb, "best_model_path", ""):
            return cb.best_model_path
        return None

    # Get best checkpoint path - either from training or from existing checkpoints
    if skip_training:
        best_ckpt = resume_ckpt_path  # Use the loaded checkpoint
    else:
        best_ckpt = _best_ckpt_path(trainer)


    # ─── Adapt on held-out 10% if present ───────────────────────────────────────
    adapt_file = getattr(cfg.split, "adapt_file", None)
    if adapt_file:
        adapt_list = json.load(open(Path(cfg.split.split_dir) / adapt_file))
        if kind == "single":
            adapt_ds = BeatDataset(
                cfg.split.data_dir, adapt_list,
                augment=False,
                variable_length=cfg.data.get("variable_length", True),
                adj_path=cfg.data.get("adj_path", None),
            )
        else:  # sequence
            adapt_ds = BeatSequenceDataset(
                root=cfg.split.data_dir,
                file_list=adapt_list,
                augment=None,
            )

        adapt_loader = DataLoader(
            adapt_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=adapt_collate,
        )

        # For adaptation, we'll create a new model instance with adapted learning rate
        lr_multiplier = cfg.adapt.get("lr_multiplier", 0.1)
        adapted_lr = float(cfg.model.params.lr) * lr_multiplier
        print(f"Using adapted learning rate: {adapted_lr}")
        
        # Create adaptation model with lower learning rate
        adapt_model = build_model(cfg.model.name, **{**cfg.model.params, "lr": adapted_lr}, T_max=cfg.adapt.get("epochs", 20))
        
        # Load the trained weights into the adaptation model
        if best_ckpt:
            checkpoint = torch.load(best_ckpt, map_location=adapt_model.device)
            # Load only the model weights, not the optimizer state (since we want new LR)
            adapt_model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded weights from {best_ckpt} for adaptation")
        else:
            # Copy weights from the current model
            adapt_model.load_state_dict(model.state_dict())
            print("Copied current model weights for adaptation")

        adapt_epochs = cfg.adapt.get("epochs", 20)
        adapt_trainer = L.Trainer(
            max_epochs=adapt_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=cfg.trainer.gpus,
            default_root_dir=cfg.paths.run_dir,
            enable_checkpointing=False,   # no checkpointing during adapt
            logger=trainer.logger,
            enable_progress_bar=cfg.trainer.get("enable_progress_bar", False),
        )

        # Run pre-adaptation test if we have a best checkpoint
        if best_ckpt:
            trainer.test(model, datamodule=dm, ckpt_path=best_ckpt)  # pre-adapt test
        
        # Freeze backbone parameters for adaptation - only train the head
        freeze_backbone = cfg.adapt.get("freeze_backbone", True)
        if freeze_backbone:
            print("Freezing backbone parameters for adaptation...")
            for param in adapt_model.backbone.parameters():
                param.requires_grad = False
            # Ensure head parameters remain trainable
            for param in adapt_model.head.parameters():
                param.requires_grad = True
        
        # Fit the adaptation model
        adapt_trainer.fit(adapt_model, train_dataloaders=adapt_loader)
        
        # Copy the adapted weights back to the main model for final testing
        model.load_state_dict(adapt_model.state_dict())

    # ─── Final test on the CURRENT (possibly adapted) weights ───────────────────
    print("=" * 50)
    print("Starting final test phase...")
    print("=" * 50)
    
    # Use a fresh trainer for testing to avoid any state issues
    test_trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.trainer.gpus,
        logger=False,  # No logging needed for final test
        enable_checkpointing=False,
        enable_progress_bar=cfg.trainer.get("enable_progress_bar", False),
    )
    
    test_trainer.test(model, datamodule=dm)
    print("=" * 50)
    print("Final test phase completed.")
    print("=" * 50)

    # Save test predictions and metrics to CSV
    # Gather predictions and targets
    if hasattr(model, '_gather_epoch_preds'):
        y_true, y_pred = model._gather_epoch_preds("test")
        y_true = y_true.detach().cpu().numpy().flatten()
        y_pred = y_pred.detach().cpu().numpy().flatten()


        # Compute metrics
        mae = np.mean(np.abs(y_true - y_pred))
        # Mean relative error: mean(|y_true - y_pred| / |y_true|)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err = np.abs(y_true - y_pred) / np.abs(y_true)
            rel_err = rel_err[~np.isnan(rel_err)]  # Remove NaNs from division by zero
            mre = np.mean(rel_err)

        # Prepare DataFrame
        df = pd.DataFrame({
            'true_time_to_pvc': y_true,
            'predicted_time_to_pvc': y_pred
        })
        df['abs_error'] = np.abs(df['true_time_to_pvc'] - df['predicted_time_to_pvc'])
        df['rel_error'] = np.abs(df['true_time_to_pvc'] - df['predicted_time_to_pvc']) / np.abs(df['true_time_to_pvc'])

        # Output directory - save to results/{project}/{run_name}/ folder
        project_root = Path.cwd()
        out_dir = project_root / "results" / cfg.wandb.project / cfg.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{cfg.run_name}_test_predictions.csv"
        df.to_csv(out_csv, index=False)

        # Save metrics as a separate file
        metrics = {
            'mean_absolute_error': float(mae),
            'mean_relative_error': float(mre)
        }
        metrics_path = out_dir / f"{cfg.run_name}_test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Test predictions saved to: {out_csv}")
        print(f"Test metrics saved to: {metrics_path}")
    else:
        print("Model does not implement _gather_epoch_preds; skipping CSV export.")

if __name__ == "__main__":
    sys.exit(main())
