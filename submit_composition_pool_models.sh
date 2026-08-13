#!/bin/bash
set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs outputs

test -f data/lfqa_prompts_5756.jsonl
test -f assets/atomic16/opt-125m-lfqa5000-v1/watermark_compositions.json
test -f assets/atomic16/qwen3-8b-lfqa5000-v1/watermark_compositions.json

sbatch_args=(--parsable)
if [[ -n "${WM_SLURM_ACCOUNT:-}" ]]; then
  sbatch_args+=(--account "$WM_SLURM_ACCOUNT")
fi

opt_job=$(sbatch "${sbatch_args[@]}" \
  protocols/composition_pool_corrected/run_opt_composition_pool_corrected.slurm)
qwen_job=$(sbatch "${sbatch_args[@]}" \
  protocols/composition_pool_corrected/run_qwen3_composition_pool_corrected_a100.slurm)

printf '%s\n' "$opt_job" > last_composition_pool_opt_job.txt
printf '%s\n' "$qwen_job" > last_composition_pool_qwen_job.txt

echo "OPT composition-pool job: $opt_job"
echo "Qwen3 composition-pool job: $qwen_job"
echo "monitor: squeue -j $opt_job,$qwen_job"
