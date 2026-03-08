# Merlin7 HPC — nnU-Net Training Guide

Quick reference for running nnU-Net v2 training on the Merlin7 cluster at PSI.

---

## 1. Transfer Project to Merlin7

From your local Windows machine, use `scp` (or WinSCP/rsync):

```bash
# Transfer scripts (small files — go to home directory)
scp -r D:\M_thesis\Scripts  <username>@merlin7.psi.ch:~/M_thesis/Scripts
scp    D:\M_thesis\requirements.txt <username>@merlin7.psi.ch:~/M_thesis/

# Transfer the converted dataset (the nnU-Net format .tif files)
scp -r D:\M_thesis\nnUNet_data\nnUNet_raw\Dataset501_MickeyScroll \
       <username>@merlin7.psi.ch:~/M_thesis/nnUNet_data/nnUNet_raw/

# If already preprocessed locally, also transfer preprocessed data (saves time):
scp -r D:\M_thesis\nnUNet_data\nnUNet_preprocessed\Dataset501_MickeyScroll \
       <username>@merlin7.psi.ch:~/M_thesis/nnUNet_data/nnUNet_preprocessed/
```

> **Tip:** Data lives under `/data/user/<username>/` on Merlin7. `~` points there.

---

## 2. First-Time Setup

SSH into Merlin7 and run the setup script once:

```bash
ssh <username>@merlin7.psi.ch
cd ~/M_thesis
bash Scripts/setup_merlin7.sh
```

This creates the `nnunet` conda environment with Python 3.12, PyTorch (CUDA 12.4), and nnU-Net v2.

---

## 3. Preprocessing (if not transferred)

If you didn't transfer preprocessed data, run preprocessing on an interactive GPU node:

```bash
# Request interactive A100 session (up to 12h)
salloc --cluster=gmerlin7 --partition=a100-interactive --gres=gpu:1 --time=01:00:00

# Inside the allocation:
conda activate nnunet
export nnUNet_raw=~/M_thesis/nnUNet_data/nnUNet_raw
export nnUNet_preprocessed=~/M_thesis/nnUNet_data/nnUNet_preprocessed
export nnUNet_results=~/M_thesis/nnUNet_data/nnUNet_results

python -c "from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry; plan_and_preprocess_entry()" \
    -d 501 --verify_dataset_integrity -c 2d --clean

exit  # Release the interactive node
```

---

## 4. Submit Training Jobs

### Train a single fold (recommended to start):

```bash
cd ~/M_thesis
sbatch Scripts/slurm_train.sh 0       # fold 0
```

### Train all 5 folds:

```bash
sbatch Scripts/slurm_train.sh          # all folds sequentially
```

### Continue a interrupted training:

```bash
sbatch Scripts/slurm_train.sh 0 --c    # resume fold 0 from checkpoint
```

### Submit each fold as a separate job (parallel, uses 5 GPUs):

```bash
for fold in 0 1 2 3 4; do
    sbatch Scripts/slurm_train.sh $fold
done
```

---

## 5. Monitor Jobs

```bash
# Check your jobs (CPU cluster is default, so specify GPU cluster)
squeue --cluster=gmerlin7 -u $USER

# Detailed job info
scontrol --cluster=gmerlin7 show job <JOBID>

# Watch training output in real-time
tail -f logs/nnunet_<JOBID>.out

# Check for errors
cat logs/nnunet_<JOBID>.err

# Cancel a job
scancel --cluster=gmerlin7 <JOBID>
```

---

## 6. SLURM Partitions Reference

| Partition | Max Time | Priority | Best For |
|-----------|----------|----------|----------|
| `a100-hourly` | 1 hour | High | Quick tests |
| `a100-daily` | 24 hours | Medium | Single fold training |
| `a100-general` | 7 days | Low | All 5 folds in one job |
| `a100-interactive` | 12 hours | Very High | Preprocessing, debugging |

**Default:** The SLURM script uses `a100-daily` (23h). To change:

```bash
# Override partition at submit time
sbatch --partition=a100-general --time=5-00:00:00 Scripts/slurm_train.sh
```

---

## 7. Training Time Estimates

With 50 training cases (3063×3062), 250 epochs, single A100 80GB:

| Scenario | Estimated Time |
|----------|---------------|
| 1 fold | ~2–4 hours |
| 5 folds (sequential) | ~10–20 hours |
| 5 folds (parallel jobs) | ~2–4 hours (5 GPUs) |

> These are rough estimates. Actual time depends on augmentation pipeline and batch size chosen by nnU-Net's auto-configuration.

---

## 8. Check Results

After training completes:

```bash
# Results are stored here:
ls ~/M_thesis/nnUNet_data/nnUNet_results/Dataset501_MickeyScroll/

# Download results to local machine:
scp -r <username>@merlin7.psi.ch:~/M_thesis/nnUNet_data/nnUNet_results \
       D:\M_thesis\nnUNet_data\
```

---

## 9. Troubleshooting

| Issue | Solution |
|-------|----------|
| `conda: command not found` | Run setup script, it handles conda sourcing |
| `CUDA not available` | Ensure `--cluster=gmerlin7` and `--gres=gpu:1` are set |
| Job stuck in `PENDING` | Check `squeue --cluster=gmerlin7`; try `a100-hourly` for quick tests |
| Out of memory (GPU) | Shouldn't happen on A100 80GB; check `logs/*.err` |
| Out of time | Use `a100-general` (7 day limit) or submit folds separately |
| `ModuleNotFoundError: nnunetv2` | Run `conda activate nnunet` — env not activated |

---

## 10. File Structure on Merlin7

```
~/M_thesis/
├── Scripts/
│   ├── train_nnunet.py          # Local training script
│   ├── custom_trainer.py        # Custom trainer (tqdm + checkpoints)
│   ├── slurm_train.sh           # SLURM job script
│   ├── setup_merlin7.sh         # One-time setup
│   └── MERLIN7_GUIDE.md         # This file
├── nnUNet_data/
│   ├── nnUNet_raw/              # Raw dataset (.tif images + labels)
│   ├── nnUNet_preprocessed/     # Preprocessed data (auto-generated)
│   └── nnUNet_results/          # Training outputs + checkpoints
├── logs/                        # SLURM job logs
├── requirements.txt
└── .gitignore
```
