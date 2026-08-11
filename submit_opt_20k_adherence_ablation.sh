#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
mkdir -p logs outputs

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "ERROR: run 'bash setup_env_msi.sh' first." >&2
  exit 1
fi
if [ ! -f "$ROOT/data/mixed_prompts_22256.jsonl" ]; then
  echo "ERROR: missing data/mixed_prompts_22256.jsonl; build it first." >&2
  exit 1
fi

PREP_JOB=$(sbatch --parsable \
  --export=ALL,WM_LORA_ROOT="$ROOT" \
  run_opt_20k_prepare.slurm)
ABLATION_JOB=$(sbatch --parsable \
  --dependency="afterok:${PREP_JOB}" \
  --export=ALL,WM_LORA_ROOT="$ROOT" \
  run_opt_20k_adherence_ablation.slurm)

echo "Preparation job: $PREP_JOB"
echo "Ablation array: $ABLATION_JOB (starts after successful preparation)"
