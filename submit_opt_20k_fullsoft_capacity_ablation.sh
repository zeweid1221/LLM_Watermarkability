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
  echo "ERROR: missing data/mixed_prompts_22256.jsonl." >&2
  exit 1
fi

CACHE="$ROOT/outputs/opt125m_lora_20k_shared/trajectory_cache.npz"
if [ -f "$CACHE" ]; then
  JOB=$(sbatch --parsable \
    --export=ALL,WM_LORA_ROOT="$ROOT" \
    run_opt_20k_fullsoft_capacity_ablation.slurm)
  echo "Reusing full-soft trajectory cache: $CACHE"
  echo "Full-soft 2x2 array: $JOB"
else
  PREP_JOB=$(sbatch --parsable \
    --export=ALL,WM_LORA_ROOT="$ROOT" \
    run_opt_20k_prepare.slurm)
  JOB=$(sbatch --parsable \
    --dependency="afterok:${PREP_JOB}" \
    --export=ALL,WM_LORA_ROOT="$ROOT" \
    run_opt_20k_fullsoft_capacity_ablation.slurm)
  echo "Preparation job: $PREP_JOB"
  echo "Full-soft 2x2 array: $JOB (starts after preparation)"
fi
