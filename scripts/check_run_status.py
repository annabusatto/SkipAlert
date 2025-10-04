#!/usr/bin/env python3
"""
Script to check the status of cross-validation runs and their checkpoints.
Usage: python scripts/check_run_status.py [--clean-incomplete]
"""
import argparse
import json
from pathlib import Path
import shutil

def check_run_status(base_dir="resources/checkpoints", results_dir="results", project="New_lstm_attn_seq_loo_adapt_5_adapt20_15percent"):
    """Check status of all runs and their checkpoints."""
    base_path = Path(base_dir)
    results_path = Path(results_dir)
    
    if not base_path.exists():
        print(f"Checkpoint directory {base_path} does not exist.")
        return
    
    print("=" * 80)
    print("RUN STATUS REPORT")
    print("=" * 80)
    
    # Find all run directories
    run_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    run_dirs.sort()
    
    incomplete_runs = []
    complete_runs = []
    runs_with_results = []
    
    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"\nRun: {run_name}")
        print("-" * 60)
        
        # Check for checkpoints
        ckpt_files = list(run_dir.glob("*.ckpt"))
        last_ckpt = run_dir / "last.ckpt"
        wandb_run_id_file = run_dir / "wandb_run_id.txt"
        
        # Check for wandb run ID
        wandb_run_id = None
        if wandb_run_id_file.exists():
            try:
                wandb_run_id = wandb_run_id_file.read_text().strip()
            except Exception:
                pass
        
        if last_ckpt.exists():
            print("  ✓ Training complete (last.ckpt exists)")
            complete_runs.append(run_name)
        elif ckpt_files:
            latest_ckpt = max(ckpt_files, key=lambda f: f.stat().st_mtime)
            print(f"  ⚠ Training incomplete (latest: {latest_ckpt.name})")
            incomplete_runs.append(run_name)
        else:
            print("  ✗ No checkpoints found")
            incomplete_runs.append(run_name)
        
        if wandb_run_id:
            print(f"  📊 Wandb run ID: {wandb_run_id}")
        
        # Check for results
        result_dir = results_path / project / run_name
        result_csv = result_dir / f"{run_name}_test_predictions.csv"
        result_metrics = result_dir / f"{run_name}_test_metrics.json"
        
        if result_csv.exists() and result_metrics.exists():
            print("  ✓ Results exist")
            runs_with_results.append(run_name)
            
            # Show metrics if available
            try:
                with open(result_metrics) as f:
                    metrics = json.load(f)
                mae = metrics.get('mean_absolute_error', 'N/A')
                mre = metrics.get('mean_relative_error', 'N/A')
                print(f"    MAE: {mae:.4f}, MRE: {mre:.4f}" if mae != 'N/A' else f"    MAE: {mae}, MRE: {mre}")
            except Exception:
                print("    (Could not read metrics)")
        else:
            print("  ✗ No results found")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total runs: {len(run_dirs)}")
    print(f"Complete training: {len(complete_runs)}")
    print(f"Incomplete training: {len(incomplete_runs)}")
    print(f"Runs with results: {len(runs_with_results)}")
    
    if incomplete_runs:
        print("\nIncomplete runs that can be resumed:")
        for run in incomplete_runs:
            print(f"  - {run}")
    
    return incomplete_runs, complete_runs, runs_with_results

def clean_incomplete_runs(incomplete_runs, base_dir="resources/checkpoints"):
    """Remove checkpoint directories for incomplete runs."""
    if not incomplete_runs:
        print("No incomplete runs to clean.")
        return
    
    print(f"\nCleaning {len(incomplete_runs)} incomplete run directories...")
    base_path = Path(base_dir)
    
    for run_name in incomplete_runs:
        run_dir = base_path / run_name
        if run_dir.exists():
            shutil.rmtree(run_dir)
            print(f"  Removed: {run_dir}")
    
    print("Cleanup complete!")

def main():
    parser = argparse.ArgumentParser(description="Check status of cross-validation runs")
    parser.add_argument("--clean-incomplete", action="store_true", 
                       help="Remove checkpoint directories for incomplete runs")
    parser.add_argument("--project", default="New_lstm_attn_seq_loo_adapt_5_adapt20_15percent",
                       help="Project name for results directory")
    parser.add_argument("--base-dir", default="resources/checkpoints",
                       help="Base directory for checkpoints")
    parser.add_argument("--results-dir", default="results",
                       help="Results directory")
    
    args = parser.parse_args()
    
    incomplete, complete, with_results = check_run_status(
        args.base_dir, args.results_dir, args.project
    )
    
    if args.clean_incomplete:
        response = input(f"\nAre you sure you want to delete {len(incomplete)} incomplete run directories? (y/N): ")
        if response.lower().startswith('y'):
            clean_incomplete_runs(incomplete, args.base_dir)
        else:
            print("Cleanup cancelled.")

if __name__ == "__main__":
    main()