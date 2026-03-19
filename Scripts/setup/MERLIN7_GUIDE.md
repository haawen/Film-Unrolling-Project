# Merlin7 HPC Guide

Quick reference for running training and evaluation on the Merlin7 cluster at PSI.

---

## 1. Transfer Project to Merlin7

```bash
# Transfer scripts and requirements
scp -r C:\Users\li_k1\M_thesis\Scripts li_k1@login002.merlin7.psi.ch:~/M_thesis/
scp C:\Users\li_k1\M_thesis\requirements.txt li_k1@login002.merlin7.psi.ch:~/M_thesis/
```

### Transfer Datasets

```bash
# Dataset501 — 2D TIF (50 images from HDF5 subset)
scp -r C:\Users\li_k1\M_thesis\nnUNet_data\nnUNet_raw\Dataset501_MickeyScroll \
       li_k1@login002.merlin7.psi.ch:~/M_thesis/nnUNet_data/nnUNet_raw/

# Dataset502 — 3D NIfTI (25 volumes, ~20 slices each, 3063x3062 XY)
scp -r C:\Users\li_k1\M_thesis\nnUNet_data\nnUNet_raw\Dataset502_MickeyScroll3D \
       li_k1@login002.merlin7.psi.ch:~/M_thesis/nnUNet_data/nnUNet_raw/

# Dataset503 is auto-generated from Dataset502 by create_2d_from_3d.py
```

> **Tip:** Data lives under `/data/user/<username>/` on Merlin7. `~` points there.

---

## 2. First-Time Setup

```bash
ssh li_k1@login002.merlin7.psi.ch
cd ~/M_thesis
bash Scripts/setup/setup_merlin7.sh
bash Scripts/setup/setup_model_dirs.sh
```

---

## 3. Datasets

| ID  | Name | Type | Source | Used By |
|-----|------|------|--------|---------|
| 501 | Dataset501_MickeyScroll | 2D TIF (50 images) | HDF5 subset | nnU-Net 2D (original) |
| 502 | Dataset502_MickeyScroll3D | 3D NIfTI (25 volumes) | HDF5 3D volumes | nnU-Net 3D, UNet3D, SwinUNETR |
| 503 | Dataset503_MickeyScroll2Dfrom3D | 2D TIF (500 slices from 3D) | Extracted from Dataset502 | nnU-Net 2D (matched) |

---

## 4. Training Jobs

### nnU-Net 2D (Dataset501)
```bash
sbatch Scripts/slurm/slurm_train.sh 0          # fold 0
sbatch Scripts/slurm/slurm_train.sh             # all folds
sbatch Scripts/slurm/slurm_train.sh 0 --c       # resume fold 0
```

### nnU-Net 3D (Dataset502)
```bash
sbatch Scripts/slurm/slurm_train_3d.sh 0        # fold 0
sbatch Scripts/slurm/slurm_train_3d.sh           # all folds
```

### nnU-Net 2D from 3D (Dataset503)
```bash
sbatch Scripts/slurm/slurm_train_2d_from_3d.sh   # creates Dataset503 + trains
```

### MONAI UNet3D (Dataset502)
```bash
sbatch Scripts/slurm/slurm_train_3dunet.sh       # fold 0
```

### MONAI SwinUNETR (Dataset502)
```bash
sbatch Scripts/slurm/slurm_train_swinunetr.sh    # fold 0
```

---

## 5. Evaluation

### Fair Comparison (all models on same 3D validation data)
```bash
sbatch Scripts/slurm/slurm_fair_compare.sh
```

This runs: extract 2D slices -> nnU-Net 2D inference -> MONAI predictions -> reassemble & compare.

Results: `fair_comparison/fair_comparison_results.json`
Plots: `visualizations/fair_*.png`

---

## 6. Monitor Jobs

```bash
squeue --cluster=gmerlin7 -u $USER
scontrol --cluster=gmerlin7 show job <JOBID>
tail -f logs/<prefix>_<JOBID>.out
scancel --cluster=gmerlin7 <JOBID>
```

---

## 7. SLURM Partitions

| Partition | Max Time | Best For |
|-----------|----------|----------|
| `a100-hourly` | 1 hour | Tests, preprocessing, fair comparison |
| `a100-daily` | 24 hours | Single fold training |
| `a100-general` | 7 days | Multi-fold training |
| `a100-interactive` | 12 hours | Debugging |

---

## 8. File Structure

```
~/M_thesis/
├── Scripts/
│   ├── train_nnunet.py           # nnU-Net 2D training (HDF5 -> TIF conversion + training)
│   ├── train_monai.py            # MONAI training (UNet3D, SwinUNETR)
│   ├── predict_monai.py          # MONAI inference
│   ├── fair_compare.py           # Fair comparison across all models
│   ├── compare_results.py        # Results summary and plots
│   ├── convert_3d_data.py        # HDF5 -> NIfTI conversion for Dataset502
│   ├── create_2d_from_3d.py      # Dataset502 -> Dataset503 (2D slices)
│   ├── custom_trainer.py         # nnUNetTrainerProgress (250 epochs, tqdm)
│   ├── visualize.py              # 2D prediction visualization
│   ├── visualize_nnunet3d.py     # 3D prediction visualization (high-quality)
│   ├── slurm/                    # SLURM job scripts
│   │   ├── slurm_train.sh        # nnU-Net 2D (Dataset501)
│   │   ├── slurm_train_3d.sh     # nnU-Net 3D (Dataset502)
│   │   ├── slurm_train_2d_from_3d.sh  # nnU-Net 2D from 3D (Dataset503)
│   │   ├── slurm_train_3dunet.sh      # MONAI UNet3D
│   │   ├── slurm_train_swinunetr.sh   # MONAI SwinUNETR
│   │   ├── slurm_fair_compare.sh      # Fair comparison pipeline
│   │   ├── slurm_preprocess_3d.sh     # Dataset502 preprocessing
│   │   └── slurm_test.sh              # Smoke test (2 epochs)
│   └── setup/                    # One-time setup scripts
│       ├── setup_merlin7.sh      # Environment setup
│       ├── setup_model_dirs.sh   # Create output directories
│       └── MERLIN7_GUIDE.md      # This file
├── nnUNet_data/                  # nnU-Net raw/preprocessed/results
├── monai_results/                # MONAI checkpoints and predictions
├── fair_comparison/              # Fair comparison outputs
├── visualizations/               # Plots and overlay images
├── logs/                         # SLURM job logs
└── requirements.txt
```

---

## 9. Troubleshooting

| Issue | Solution |
|-------|----------|
| `conda: command not found` | Run setup script; it handles conda sourcing |
| `CUDA not available` | Ensure `--cluster=gmerlin7` and `--gres=gpu:1` |
| Job stuck in `PENDING` | Try `a100-hourly` for quick tests |
| `ModuleNotFoundError: nnunetv2` | `conda activate nnunet` |
| SwinUNETR low dice at inference | Check `SpatialPadd` uses `method="end"` |
