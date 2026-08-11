#!/bin/bash -l
#SBATCH -A liyanxie
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J wm_atom16_opt
#SBATCH -o /users/2/deng0379/watermarkability_lora_msi/logs/atom16_opt_%j.out
#SBATCH -e /users/2/deng0379/watermarkability_lora_msi/logs/atom16_opt_%j.err

set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}

cd "$ROOT"
mkdir -p logs assets/atomic16
source "$ROOT/.venv/bin/activate"

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

python build_atomic16_partition.py \
  --model-name facebook/opt-125m \
  --manifest data/lfqa_prompts_5756.jsonl \
  --source-partition-dir assets/source_opt_partition \
  --output-dir assets/atomic16/opt-125m-lfqa5000-v1 \
  --calibration-count 5000 \
  --validation-count 500 \
  --max-prompt-tokens 160 \
  --forward-batch-size 32 \
  --partition-steps 160 \
  --context-batch-size 256 \
  --rounding-context-count 256 \
  --validation-batch-size 32 \
  --semantic-lsh-bits 8 \
  --mass-strata-bins 256 \
  --device cuda \
  --local-files-only
