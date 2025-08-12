#!/usr/bin/env bash
#SBATCH --job-name=cross_validation
#SBATCH --array=9-20              # 7 folds → indices 0..6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=CIBC-gpu
#SBATCH --output=logs/%A/%x-%A_%a.out  # one log per fold
#SBATCH --error=logs/%A/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anna.busatto@utah.edu

set -euo pipefail

# --- activate env & cd to repo root ---
cd /uu/sci.utah.edu/projects/CEG/ActiveProjects/IschemiaPVCPrediction
source .venv/bin/activate
mkdir -p logs/${SLURM_JOB_ID}

# --- paths (env fallbacks used by your Hydra configs) ---
# export DATA_DIR=/uu/sci.utah.edu/projects/CEG/ActiveProjects/IschemiaPVCPrediction/data/beats
# export SPLIT_DIR=/uu/sci.utah.edu/projects/CEG/ActiveProjects/IschemiaPVCPrediction/data/splits/k7_90_10
# export RUNS_DIR=/uu/sci.utah.edu/projects/CEG/ActiveProjects/IschemiaPVCPrediction/resources

# --- hygiene ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# --- choose fold from array index ---
FOLD=${SLURM_ARRAY_TASK_ID}
MODEL=resnet    # change to resnet/tcn/wave_cnn/tst if you like
SPLIT=loo_adapt15  # change to kfold_within/loo_adapt if you like
EPOCHS=100          # adjust as needed
BATCH_SIZE=512      # adjust as needed

# --- build args safely (prevents line-continuation bugs) ---
args=(
  split=${SPLIT}
  split.fold=${FOLD}
  model=${MODEL}
  model.params.lr=3e-4
  trainer.epochs=${EPOCHS}
  data.batch_size=${BATCH_SIZE}
  data.num_workers=14
  run_name=${MODEL}_${SPLIT}_f${FOLD}_E${EPOCHS}_B${BATCH_SIZE}
  wandb.project=loo_adapt_15
)

# (optional) offline logging during debugging:
# export WANDB_MODE=offline

set -x
srun python -m pvc.cli "${args[@]}"
set +x
