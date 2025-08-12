#!/usr/bin/env python
from pathlib import Path
from collections import defaultdict
import json, argparse, random, re
from math import ceil
from sklearn.model_selection import KFold

# ---------- helpers ----------
def collect_by_experiment(root: Path):
    """exp_id is the parent folder under root, e.g. '18-03-20'."""
    groups = defaultdict(list)
    for f in root.rglob("*.npy"):
        rel = f.relative_to(root)
        if len(rel.parts) >= 2:
            exp_id = rel.parts[0]
        else:
            m = re.search(r"\d{2}-\d{2}-\d{2}", f.stem)
            exp_id = m.group(0) if m else f.stem.split("_")[0]
        groups[exp_id].append(str(rel))
    return groups

def write_json(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f)
    print(f"{path}  ←  {len(rows):6d} beats")

def per_experiment_train_val(groups_subset, seed, p_val=0.10):
    rnd = random.Random(seed)
    train, val = [], []
    for exp_id, beats in sorted(groups_subset.items()):
        b = beats[:]
        rnd.shuffle(b)
        n_val = max(1, ceil(len(b) * p_val)) if b else 0
        val.extend(b[:n_val])
        train.extend(b[n_val:])
    return train, val

def split_within_exp(groups, seed, p_train=0.80, p_val=0.10):
    rnd = random.Random(seed)
    train, val, test = [], [], []
    for exp_id, beats in sorted(groups.items()):
        b = beats[:]
        rnd.shuffle(b)
        n = len(b)
        n_train = ceil(n * p_train)
        n_val   = ceil(n * p_val)
        train.extend(b[:n_train])
        val.extend(b[n_train:n_train+n_val])
        test.extend(b[n_train+n_val:])
    return train, val, test

# ---------- strategies ----------
def make_loo(groups, out_dir: Path, seed: int, val_pct: float):
    exps = sorted(groups)
    for k, held in enumerate(exps):
        train_groups = {e: groups[e] for e in exps if e != held}
        tr, va = per_experiment_train_val(train_groups, seed=hash((seed, k)), p_val=val_pct)
        te = groups[held]
        write_json(out_dir / f"fold_{k}_train.json", tr)
        write_json(out_dir / f"fold_{k}_val.json",   va)
        write_json(out_dir / f"fold_{k}_test.json",  te)

def make_kfold(groups, out_dir: Path, k: int, seed: int, val_mode: str, val_pct: float):
    exps = sorted(groups)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [[exps[i] for i in te] for _, te in kf.split(exps)]
    for f in range(k):
        test_exps = set(folds[f])
        if val_mode == "nextfold" and k >= 3:
            val_exps = set(folds[(f+1) % k])
            train_exps = [e for e in exps if e not in (test_exps | val_exps)]
            tr = [b for e in train_exps for b in groups[e]]
            va = [b for e in val_exps   for b in groups[e]]
            te = [b for e in test_exps  for b in groups[e]]
        else:
            non_test = {e: groups[e] for e in exps if e not in test_exps}
            tr, va = per_experiment_train_val(non_test, seed=hash((seed, f)), p_val=val_pct)
            te = [b for e in test_exps for b in groups[e]]
        write_json(out_dir / f"fold_{f}_train.json", tr)
        write_json(out_dir / f"fold_{f}_val.json",   va)
        write_json(out_dir / f"fold_{f}_test.json",  te)

def make_loo_adapt(groups, out_dir: Path, seed: int, val_pct: float, adapt_pct: float):
    """
    For each held-out experiment:
      - train/val from all held-in experiments (per-exp val_pct to val)
      - adapt = adapt_pct from held-out experiment
      - test  = remaining from held-out experiment
    """
    exps = sorted(groups)
    for k, held in enumerate(exps):
        # held-in train/val
        train_groups = {e: groups[e] for e in exps if e != held}
        tr, va = per_experiment_train_val(train_groups, seed=hash((seed, k)), p_val=val_pct)
        # held-out adapt/test
        rnd = random.Random(hash((seed, "adapt", k)))
        b = groups[held][:]
        rnd.shuffle(b)
        n_adapt = max(1, ceil(len(b) * adapt_pct)) if b else 0
        adapt = b[:n_adapt]
        test  = b[n_adapt:]
        write_json(out_dir / f"fold_{k}_train.json", tr)
        write_json(out_dir / f"fold_{k}_val.json",   va)
        write_json(out_dir / f"fold_{k}_adapt.json", adapt)
        write_json(out_dir / f"fold_{k}_test.json",  test)

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=Path)
    ap.add_argument("out_dir",   type=Path)
    ap.add_argument("--mode", choices=("loo", "kfold", "fixed", "loo_adapt"), required=True)
    ap.add_argument("-k", type=int, default=7, help="k for kfold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-pct", type=float, default=0.10, help="per-exp validation fraction")
    ap.add_argument("--val-mode", choices=("within", "nextfold"), default="within",
                    help="kfold validation strategy")
    ap.add_argument("--adapt-pct", type=float, default=0.10, help="held-out fraction for adaptation (loo_adapt)")
    args = ap.parse_args()

    groups = collect_by_experiment(args.data_root)

    if args.mode == "fixed":
        tr, va, te = split_within_exp(groups, args.seed)
        write_json(args.out_dir / "train.json", tr)
        write_json(args.out_dir / "val.json",   va)
        write_json(args.out_dir / "test.json",  te)
    elif args.mode == "loo":
        make_loo(groups, args.out_dir, args.seed, args.val_pct)
    elif args.mode == "kfold":
        make_kfold(groups, args.out_dir, args.k, args.seed, args.val_mode, args.val_pct)
    else:  # loo_adapt
        make_loo_adapt(groups, args.out_dir, args.seed, args.val_pct, args.adapt_pct)
