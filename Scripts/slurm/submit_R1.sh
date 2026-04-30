#!/bin/bash
# =============================================================================
# Submit the 4 Round 1 variants in parallel.
#
#   sh Scripts/slurm/submit_R1.sh smoke   # a100-hourly, 1500 steps each
#   sh Scripts/slurm/submit_R1.sh full    # a100-daily, 10000 steps each
# =============================================================================

set -euo pipefail
MODE="${1:?usage: $0 <smoke|full>}"

case "${MODE}" in
  smoke) PART=a100-hourly; TIME="00:55:00" ;;
  full)  PART=a100-daily;  TIME="06:00:00" ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac

SCRIPT="Scripts/slurm/slurm_R1_variant.sh"

for v in R1a R1b R1c R1d; do
  echo "Submitting ${v} (${MODE}, ${PART}, ${TIME})"
  sbatch --partition="${PART}" --time="${TIME}" \
         --job-name="R1_${v}_${MODE}" \
         "${SCRIPT}" "${v}" "${MODE}"
done

echo ""
echo "Listing queued jobs:"
squeue -u "$USER" -M gmerlin7 || true
