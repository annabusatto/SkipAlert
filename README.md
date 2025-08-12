# PVC Prediction — Deep-Learning Pipeline for Predicting Time-to-PVC from 247-Lead Sock Electrograms 🐷⚡️

A reproducible, Slurm-friendly codebase that turns **~87 000** individual beats (247 leads × variable length) into GPU-trained models that predict the **time (s) until the next premature ventricular contraction (PVC)**.  
Everything is driven by **Hydra configs**, logged to **Weights & Biases**, and ready for single-GPU or multi-node jobs on an HPC cluster.

---

## 1  Directory layout  ▶ _what lives where_

```text
IschemiaPVCPrediction/
├── configs/            # Hydra YAMLs (models, data, trainer, sweeps…)
├── data/               # raw beats + generated splits
│   ├── beats/          # expID_beat####.npy  (+ .json with label)
│   └── splits/         # train.json / val.json / test.json OR fold_* files
├── pvc/            # importable Python package
│   ├── datasets/       # BeatDataset → [247,430] + augmentations
│   ├── datamodules/    # LightningDataModule wrapper
│   ├── models/         # common.py, backbones/, pvc_module.py factory
│   ├── transforms/     # GaussianNoise, LeadDropout, TimeWarp, …
│   ├── callbacks/      # ScatterLogger (targets vs predictions)
│   └── cli.py          # Hydra → Lightning → wandb entry-point
├── scripts/
│   ├── make_splits.py  # fixed / loo / kfold split generator
│   ├── train_job.sbatch
│   ├── sweep.yaml      # wandb sweep spec
│   └── sweep_agents.sbatch
└── README.md
```

---

## 2  Data format & loader behaviour

| File                     | Contents                                                                  |
|--------------------------|---------------------------------------------------------------------------|
| `expID_beat####.npy`     | `float32` array **(247, N)** or **(N, 247)** (auto-transposed if needed)  |
| `expID_beat####.json`    | `{"time_to_pvc": <number>}` — ground-truth label in **seconds**           |

`BeatDataset` steps performed **per beat**  
1. **Median-remove** & **per-lead z-score**  
2. **Resample** → **430 samples** along time axis  
3. Optional **augmentations** (noise, wander, warp, lead-drop)

Returned tensor shape to the model: **`[247, 430]`**

---

## 3  Model zoo

| Hydra `model=` | Backbone                                 | RF-size      | Params | Highlights                     |
|----------------|------------------------------------------|--------------|--------|--------------------------------|
| `resnet`       | 1-D ResNet + global pool                 | 63 samples   | 0.6 M  | strong & fast baseline         |
| `tcn`          | Dilated causal CNN                       | full beat    | 0.5 M  | real-time capable              |
| `inception`    | InceptionTime (multi-scale CNN)          | multi-scale  | 2–3 M  | UCR/UEA SOTA                   |
| `tst`          | Time-Series Transformer                  | global       | 0.9 M  | native variable length         |
| `wave_cnn`     | WaveNet-style gated causal CNN           | full beat    | 0.6 M  | causal, GRU-free               |

_All share the same Lightning-module wrapper with heteroscedastic regression heads and ScatterLogger support._

---

## 4  Run it locally (single GPU)

```bash
git clone <repo> pig-pvc && cd pig-pvc
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export WANDB_API_KEY=<your-key>
```
```bash
# 1) generate per-experiment 80/10/10 split
python scripts/make_splits.py data/beats data/splits/fixed --mode fixed
# 2) train baseline ResNet
python -m pig_pvc.cli \
       data.data_dir=data/beats \
       data.split_dir=data/splits/fixed
```       
## 5 Generate other splits

```bash
# Leave-one-experiment-out  (21 folds)
python scripts/make_splits.py data/beats data/splits/loo --mode loo

# 5-fold CV (whole experiments)
python scripts/make_splits.py data/beats data/splits/k5 --mode kfold -k 5
```
Hydra overrides example (train fold 3 of k-fold):

```bash
python -m pig_pvc.cli \
  data.split_dir=data/splits/k5 \
  data.train_file=fold_3_train.json \
  data.val_file=fold_3_val.json \
  model=tcn
```

## 6 Data prep summary table

| Goal                                | Command                                                                      | Split files produced                      |
| ----------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------- |
| **80/10/10 inside each experiment** | `python scripts/make_splits.py data/beats data/splits/fixed --mode fixed --seed 123` | `train.json`, `val.json`, `test.json`     |
| **Leave-one-experiment-out CV**     | `python scripts/make_splits.py data/beats data/splits/loo --mode loo`                | `fold_0_train.json`, `fold_0_val.json`, … |
| **5-fold CV over experiments**      | `python scripts/make_splits.py data/beats data/splits/k7 --mode kfold -k 7 --seed 7` | `fold_0_train.json`, `fold_0_val.json`, … |

## 7 Cluster usage (Slurm)
### 7.1 Single training job
```bash
sbatch scripts/train_job.sbatch \
    --export=ALL,DATA_DIR=/scratch/data/beats,SPLIT_DIR=/scratch/data/splits/fixed
```

### 7.2 Weights & Biases sweep
```bash
# submit first agent (also creates sweep)
sbatch scripts/sweep_agents.sbatch
# add more agents later (reuse SWEEP_ID printed by the first job)
sbatch --export=SWEEP_ID=<ID> scripts/sweep_agents.sbatch
```

Each agent grabs one GPU and runs --count 5 random configs from scripts/sweep.yaml.

## 8 Troubleshooting / FAQ
Issue / error	Likely fix
Shape assertion not (247, 430)	Beat saved with wrong orientation or missing leads.
OOM on tst backbone	Reduce batch_size or d_model, or switch to ResNet/TCN.
No GPU visible in Slurm job	Ensure script uses srun python … and #SBATCH --gres=gpu:1.
Want offline logging	Add wandb=offline override when launching.

## 9 Citation
If this code helps your work, please cite:

Your Name. PVC: Deep learning pipeline for PVC timing prediction from high-density sock electrodes. 2025. GitHub repository: https://github.com/<user>/pvc

## 10 License & contact
Released under the MIT License — free to use, study, modify and publish.
Questions or pull requests welcome! ✉ anna.busatto@utah.edu