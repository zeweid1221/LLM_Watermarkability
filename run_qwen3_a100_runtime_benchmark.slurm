#!/bin/bash -l
#SBATCH -A liyanxie
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH -J wm_qwen3_bench
#SBATCH -o /users/2/deng0379/watermarkability_lora_msi/logs/qwen3_bench_%j.out
#SBATCH -e /users/2/deng0379/watermarkability_lora_msi/logs/qwen3_bench_%j.err

set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs
source "$ROOT/.venv/bin/activate"

export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

python benchmark_qwen3_runtime.py \
  --model-name Qwen/Qwen3-8B \
  --batch-sizes 2,4,8 \
  --generation-batch-size 8 \
  --prompt-length 160 \
  --generated-tokens 96 \
  --condition-top-k 128 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --target-adherence 0.99 \
  --timed-training-steps 2 \
  --local-files-only
