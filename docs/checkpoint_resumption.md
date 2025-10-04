# Checkpoint Resumption Guide

## Overview
The cross-validation script now supports automatic checkpoint resumption. If a job times out before completing all epochs, you can simply resubmit the same job and it will pick up where it left off.

## How It Works

### Automatic Resumption
1. **Deterministic Paths**: Each run creates a checkpoint directory based on the run parameters (model, split, fold, etc.), not timestamps
2. **Smart Detection**: The system automatically detects:
   - Existing checkpoints to resume from
   - Completed training (when `last.ckpt` exists)
   - Existing results (to avoid recomputation)

### Run States
- **New Run**: No checkpoints exist, starts from scratch
- **Resumable**: Partial checkpoints exist, resumes from latest
- **Training Complete**: `last.ckpt` exists, skips to testing/adaptation
- **Fully Complete**: Results already exist, skips entirely

## Usage

### Resuming a Timed-Out Job
Simply resubmit the same SBATCH array job:
```bash
sbatch scripts/cross_validation.sbatch
```

The script will automatically:
1. Check for existing checkpoints
2. Resume from the latest checkpoint if found
3. Continue training to completion
4. Perform testing and adaptation

### Checking Run Status
Use the status checker to see which runs are complete/incomplete:
```bash
python scripts/check_run_status.py
```

Options:
- `--clean-incomplete`: Remove incomplete run directories
- `--project PROJECT_NAME`: Specify different project name
- `--base-dir DIR`: Specify different checkpoint base directory

### Manual Intervention

#### Force Restart a Run
To restart a run from scratch, delete its checkpoint directory:
```bash
rm -rf resources/checkpoints/RUN_NAME
```

#### Clean All Incomplete Runs
```bash
python scripts/check_run_status.py --clean-incomplete
```

## File Structure

```
resources/checkpoints/
├── lstm_attn_seq_loo_adapt_f0_E100_B256_adapt20_15pct/
│   ├── epoch-042-val_mae-123.456.ckpt    # Latest checkpoint
│   └── last.ckpt                         # Final checkpoint (if complete)
└── lstm_attn_seq_loo_adapt_f1_E100_B256_adapt20_15pct/
    └── ...

results/PROJECT_NAME/
├── RUN_NAME/
│   ├── RUN_NAME_test_predictions.csv
│   └── RUN_NAME_test_metrics.json
└── ...
```

## Benefits

1. **No Lost Work**: Timeout jobs can be resumed without losing progress
2. **Efficient Resource Usage**: Skip completed runs automatically
3. **Robust**: Handles edge cases and corrupted checkpoints gracefully
4. **Transparent**: Clear logging of what's happening

## Notes

- The checkpoint directory is based on the `run_name` parameter, which includes all key parameters
- Results are saved in a structured directory under `results/PROJECT_NAME/RUN_NAME/`
- The system respects Lightning's checkpoint format and training state
- Adaptation phase also benefits from the main training resumption