#!/bin/bash -l
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -J wm_qwen3_part
#SBATCH -o logs/qwen3_part_%j.out
#SBATCH -e logs/qwen3_part_%j.err

set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
OUTPUT_PARTITION=assets/atomic16/qwen3-8b-lfqa5000-v1

cd "$ROOT"
mkdir -p logs assets/atomic16
source "$ROOT/.venv/bin/activate"

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

test -f data/lfqa_prompts_5756.jsonl

python build_atomic16_partition.py \
  --model-name Qwen/Qwen3-8B \
  --manifest data/lfqa_prompts_5756.jsonl \
  --output-dir "$OUTPUT_PARTITION" \
  --calibration-count 5000 \
  --validation-count 500 \
  --max-prompt-tokens 160 \
  --forward-batch-size 8 \
  --partition-steps 160 \
  --context-batch-size 128 \
  --rounding-context-count 256 \
  --validation-batch-size 16 \
  --semantic-lsh-bits 8 \
  --mass-strata-bins 256 \
  --device cuda \
  --local-files-only

test -f "$OUTPUT_PARTITION/watermark_compositions.json"
test -f "$OUTPUT_PARTITION/atomic16_report.json"
