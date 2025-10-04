#!/usr/bin/env bash
#SBATCH --job-name=cross_validation
#SBATCH --array=0-20  #0-20            # 7 folds → indices 0..6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16 # 16, 8
#SBATCH --mem=150G   # 90G, 150 cannot request more than 188 in CEG, 250
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=CEG-gpu  # CEG-gpu, CIBC-gpu, spartacus
#SBATCH --output=logs/%A/%x-%A_%a.out  # one log per fold
#SBATCH --error=logs/%A/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anna.busatto@utah.edu

# NOTE: This script now supports automatic checkpoint resumption!
# - If a job times out, simply resubmit the same array job
# - The script will automatically find and resume from the latest checkpoint
# - If training is already complete, it will skip to testing
# - If results already exist, it will skip the entire run

set -euo pipefail

# --- activate env & cd to repo root ---
cd /uufs/sci.utah.edu/projects/CEG/ActiveProjects/IschemiaPVCPrediction
export PYTHONPATH=$PWD
source .venv/bin/activate
mkdir -p logs/${SLURM_JOB_ID}

# --- hygiene ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_CACHE_DIR=/uufs/sci.utah.edu/projects/CEG/ActiveProjects/IschemiaPVCPrediction/.cache/wandb/artifacts

# --- choose fold from array index ---
FOLD=${SLURM_ARRAY_TASK_ID}
MODEL=lstm_attn    # change to lstm_attn/resnet/tcn/wave_cnn/tst if you like
SPLIT=seq_loo_adapt  # change to kfold_within/loo_adapt if you like
EPOCHS=100          # adjust as needed
BATCH_SIZE=256      # adjust as needed
BEATS=5
FINETUNE=15 # percentage of the held out experiment used for fine-tuning
ADAPT_EPOCHS=20 

# --- quick import sanity (prevents Hydra callback import error) ---
python - <<'PY'
import os, importlib
print("CWD:", os.getcwd())
m = importlib.import_module("pvc.callbacks.confmat_logger")
print("ConfusionMatrixLogger?", hasattr(m, "ConfusionMatrixLogger"))
PY

# --- build args safely (prevents line-continuation bugs) ---
args=(
  split=${SPLIT}
  split.fold=${FOLD}
  split.beats=${BEATS}
  split.fine_tune=${FINETUNE}
  adapt.freeze_backbone=false
  adapt.epochs=${ADAPT_EPOCHS}
  data=sequence
  # data.pack=concat        # concat or stack (stack requires SequenceWrapper in model)
  model=${MODEL}
  # model.params.loss_type=huber
  model.params.lr=3e-4
  model.params.backbone.n_layers=6 # for lstm
  # model.params.backbone.n_blocks=6 # for resnet
  model.params.task=regression   # or classification
  trainer.epochs=${EPOCHS}
  callbacks=scatter       # add 'scatter' too with callbacks='[scatter,confmat]'
  data.batch_size=${BATCH_SIZE}
  data.num_workers=16 # 8
  run_name=${MODEL}_${SPLIT}_f${FOLD}_E${EPOCHS}_B${BATCH_SIZE}_adapt${ADAPT_EPOCHS}_${FINETUNE}pct
  # wandb.project=${MODEL}_${SPLIT}_${BEATS}
  wandb.project=${MODEL}_${SPLIT}_${BEATS}_adapt${ADAPT_EPOCHS}_${FINETUNE}percent
)

# (optional) offline logging during debugging:
# export WANDB_MODE=offline

set -x
srun python -m pvc.cli "${args[@]}"
set +x

# python -m pvc.cli model=resnet model.params.task=classification model.params.cls_window_s=180
