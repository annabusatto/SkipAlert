#!/usr/bin/env python3
"""
Test script to validate wandb resumption logic.
Usage: python scripts/test_resumption.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pvc.cli import _find_wandb_run_id, _find_latest_checkpoint

def test_resumption_logic():
    """Test the checkpoint and wandb resumption logic."""
    print("Testing resumption logic...")
    
    # Test checkpoint directory
    test_ckpt_dir = "resources/checkpoints/test_run"
    
    # Test find latest checkpoint
    ckpt_path, training_complete = _find_latest_checkpoint(test_ckpt_dir)
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Training complete: {training_complete}")
    
    # Test find wandb run ID
    run_id = _find_wandb_run_id(test_ckpt_dir, "test_run")
    print(f"Wandb run ID: {run_id}")
    
    # Look for actual checkpoint directories
    ckpt_base = Path("resources/checkpoints")
    if ckpt_base.exists():
        print("\nActual checkpoint directories:")
        for d in ckpt_base.iterdir():
            if d.is_dir():
                ckpt_path, training_complete = _find_latest_checkpoint(str(d))
                run_id = _find_wandb_run_id(str(d), d.name)
                print(f"  {d.name}:")
                print(f"    Checkpoint: {'✓' if ckpt_path else '✗'}")
                print(f"    Complete: {'✓' if training_complete else '✗'}")
                print(f"    Wandb ID: {'✓' if run_id else '✗'}")
    else:
        print("No checkpoint directory found")

if __name__ == "__main__":
    test_resumption_logic()