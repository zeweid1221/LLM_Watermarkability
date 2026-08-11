#!/bin/bash
set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs outputs

job_id=$(sbatch --parsable run_opt_atomic16_multirule.slurm)
printf '%s\n' "$job_id" > last_atomic16_multirule_job.txt

echo "multirule job: ${job_id}"
echo "monitor: squeue -j ${job_id}"
echo "log: tail -f logs/atom16_all_${job_id}.out"
