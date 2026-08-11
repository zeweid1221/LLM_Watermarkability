#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
mkdir -p logs outputs

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "ERROR: run 'bash setup_env_msi.sh' first." >&2
  exit 1
fi
if [ ! -f "$ROOT/data/mixed_prompts_5756.jsonl" ]; then
  echo "ERROR: packaged manifest is missing." >&2
  exit 1
fi

sbatch --export=ALL,WM_LORA_ROOT="$ROOT" run_opt_5k.slurm
