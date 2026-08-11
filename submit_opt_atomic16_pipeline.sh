#!/bin/bash
set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs

pipeline_job=$(sbatch --parsable run_opt_atomic16_pipeline.slurm)
printf '%s\n' "$pipeline_job" > last_atomic16_pipeline_job.txt

echo "pipeline job: ${pipeline_job}"
echo "monitor: squeue -j ${pipeline_job}"
echo "log: tail -f logs/atom16_pipeline_${pipeline_job}.out"
