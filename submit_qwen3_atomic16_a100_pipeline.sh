#!/bin/bash
set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs outputs assets/atomic16

partition_job=$(sbatch --parsable run_qwen3_atomic16_partition_a100.slurm)
experiment_job=$(
  sbatch --parsable \
    --dependency="afterok:${partition_job}" \
    run_qwen3_atomic16_7rules_a100.slurm
)

printf '%s\n' "$partition_job" > last_qwen3_partition_job.txt
printf '%s\n' "$experiment_job" > last_qwen3_7rules_job.txt

echo "partition job:  ${partition_job}"
echo "experiment job: ${experiment_job} (afterok:${partition_job})"
echo "monitor: squeue -j ${partition_job},${experiment_job}"
echo "partition log:  logs/qwen3_part_${partition_job}.out"
echo "experiment log: logs/qwen3_7rule_${experiment_job}.out"
