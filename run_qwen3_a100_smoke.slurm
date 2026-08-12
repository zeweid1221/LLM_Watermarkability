#!/bin/bash -l
#SBATCH -A liyanxie
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -J wm_qwen3_smoke
#SBATCH -o /users/2/deng0379/watermarkability_lora_msi/logs/qwen3_smoke_%j.out
#SBATCH -e /users/2/deng0379/watermarkability_lora_msi/logs/qwen3_smoke_%j.err

set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs
source "$ROOT/.venv/bin/activate"

export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

python smoke_qwen3_training.py \
  --model-name Qwen/Qwen3-8B \
  --sequence-length 255 \
  --target-positions 96 \
  --condition-top-k 128 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --target-adherence 0.99 \
  --local-files-only
