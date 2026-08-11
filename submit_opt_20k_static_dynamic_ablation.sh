#!/usr/bin/env bash
set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs

PREP_JOB=$(sbatch --parsable run_opt_20k_mixed_prepare.slurm)
SWEEP_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" run_opt_20k_static_dynamic_ablation.slurm)

echo "Mixed-trajectory preparation job: ${PREP_JOB}"
echo "Static/dynamic ablation array: ${SWEEP_JOB}"
echo "The ablation starts only after the mixed cache is complete."
