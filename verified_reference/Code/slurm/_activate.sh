# Sourced by every job script in Code/slurm/.  Edit the two lines marked EDIT
# if your cluster / environment differs from PSI Merlin7.
#
# Job scripts must be submitted FROM THE DELIVERABLE ROOT (they locate it via
# $SLURM_SUBMIT_DIR), e.g.
#     cd /path/to/M_thesis_deliverable && sbatch Code/slurm/seg_02_train_nnunet3d.sh

ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -f "$ROOT/Code/env.sh" ]; then
    echo "ERROR: submit from the deliverable root (folder with Code/env.sh); got $ROOT" >&2
    exit 1
fi

# EDIT: how to get `conda` into the job shell
CONDA_SH=/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
[ -f "$CONDA_SH" ] && source "$CONDA_SH"
# EDIT: name of the environment built from requirements.txt
conda activate "${CONDA_ENV:-nnunet}"

source "$ROOT/Code/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-12}"
mkdir -p "$WORK/logs"
cd "$ROOT"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
