#!/bin/bash
set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs outputs

test -f data/lfqa_prompts_5756.jsonl
test -f assets/atomic16/opt-125m-lfqa5000-v1/watermark_compositions.json

sbatch_args=(--parsable)
if [[ -n "${WM_SLURM_ACCOUNT:-}" ]]; then
  sbatch_args+=(--account "$WM_SLURM_ACCOUNT")
fi

job_id=$(sbatch "${sbatch_args[@]}" \
  protocols/composition_pool_corrected/run_opt_composition_pool_corrected.slurm)

printf '%s\n' "$job_id" > last_opt_composition_pool_release_job.txt
echo "OPT release rerun job: $job_id"
echo "monitor: squeue -j $job_id"
