#!/bin/bash
#SBATCH --job-name=data_prep_sequence
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anna.busatto@utah.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  # Adjusted for efficiency; test with 16 or 32
#SBATCH --mem=256G           # Adjusted based on expected usage
#SBATCH --time=1-00:00:00     # Adjusted; tune based on test runs
#SBATCH --partition=CIBC-cpu
#SBATCH --output=logs/%A/%x-%A.out  # one log per fold
#SBATCH --error=logs/%A/%x-%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anna.busatto@utah.edu


cd /uufs/sci.utah.edu/projects/CEG/ActiveProjects/IschemiaPVCPrediction
export PYTHONPATH=$PWD
source .venv/bin/activate
mkdir -p logs/${SLURM_JOB_ID}
set -x
srun python scripts/data_prep_sequence.py
set +x
