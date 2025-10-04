"""
Custom callback to save wandb run ID for resumption.
"""
import os
from pathlib import Path
from lightning.pytorch.callbacks import Callback


class WandbRunIDSaver(Callback):
    """Callback to save wandb run ID to checkpoint directory for resumption."""
    
    def __init__(self, ckpt_dir: str):
        super().__init__()
        self.ckpt_dir = Path(ckpt_dir)
        self.run_id_saved = False
    
    def on_train_start(self, trainer, pl_module):
        """Save wandb run ID when training starts."""
        if self.run_id_saved:
            return
            
        logger = trainer.logger
        if hasattr(logger, 'experiment') and logger.experiment:
            try:
                run_id = logger.experiment.id
                run_id_file = self.ckpt_dir / "wandb_run_id.txt"
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                run_id_file.write_text(run_id)
                print(f"Saved wandb run ID {run_id} to {run_id_file}")
                self.run_id_saved = True
            except Exception as e:
                print(f"Warning: Could not save wandb run ID: {e}")
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Fallback: try to save run ID at the start of first epoch if not saved yet."""
        if not self.run_id_saved:
            self.on_train_start(trainer, pl_module)