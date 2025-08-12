#!/usr/bin/env bash
#SBATCH --job-name=training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --partition=CEG-gpu
#SBATCH --output=logs/%A/%x-%A.out  # one log per fold
#SBATCH --error=logs/%A/%x-%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anna.busatto@utah.edu

set -euo pipefail

# --- activate env & cd to repo root ---
cd /uu/sci.utah.edu/projects/CEG/ActiveProjects/IschemiaPVCPrediction
source .venv/bin/activate
mkdir -p logs/${SLURM_JOB_ID}

# --- hygiene ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# --- choose fold from array index ---
MODEL=wave_cnn    # change to resnet/tcn/wave_cnn/tst if you like
EPOCHS=300          # adjust as needed
BATCH_SIZE=1024      # adjust as needed

# --- build args safely (prevents line-continuation bugs) ---
args=(
  split=fixed
  model=${MODEL}
  model.params.lr=3e-4
  trainer.epochs=${EPOCHS}
  data.batch_size=${BATCH_SIZE}
  data.num_workers=14
  run_name=${MODEL}_E${EPOCHS}_B${BATCH_SIZE}
  wandb.project=80-10-10_split
)

# (optional) offline logging during debugging:
# export WANDB_MODE=offline

set -x
srun python -m pvc.cli "${args[@]}"
set +x
