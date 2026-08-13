#!/bin/bash
set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs

sbatch_args=(--parsable)
if [[ -n "${WM_SLURM_ACCOUNT:-}" ]]; then
  sbatch_args+=(--account "$WM_SLURM_ACCOUNT")
fi

ARRAY_JOB=$(sbatch "${sbatch_args[@]}" run_cross_rule_watermarkability.slurm)
MERGE_JOB=$(sbatch "${sbatch_args[@]}" \
  --dependency="afterok:${ARRAY_JOB}" \
  run_cross_rule_watermarkability_merge.slurm)

echo "cross-rule array job: $ARRAY_JOB"
echo "validation/archive job: $MERGE_JOB"
echo "monitor: squeue -j ${ARRAY_JOB},${MERGE_JOB}"
echo "status:  sacct -j ${ARRAY_JOB},${MERGE_JOB} --format=JobID,JobName,State,Elapsed,ExitCode"
