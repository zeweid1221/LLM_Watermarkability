#!/bin/bash
set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs outputs

test -f data/lfqa_prompts_5756.jsonl
test -f assets/atomic16/qwen3-8b-lfqa5000-v1/watermark_compositions.json

sbatch_args=(--parsable)
if [[ -n "${WM_SLURM_ACCOUNT:-}" ]]; then
  sbatch_args+=(--account "$WM_SLURM_ACCOUNT")
fi

job=$(sbatch "${sbatch_args[@]}" \
  protocols/composition_pool_corrected/run_qwen3_composition_pool_corrected_a100.slurm)
printf '%s\n' "$job" > last_qwen3_composition_pool_corrected_job.txt

echo "Qwen3 corrected composition-pool job: $job"
echo "monitor: squeue -j $job"
