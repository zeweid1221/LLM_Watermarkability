#!/bin/bash -l
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH -J wm_optmixprep
#SBATCH -o logs/optmixprep_%j.out
#SBATCH -e logs/optmixprep_%j.err

set -euo pipefail
ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

python run_experiment.py \
  --training-protocol schedule_pool \
  --manifest data/mixed_prompts_22256.jsonl \
  --partition-dir assets/opt_hierarchy \
  --output-dir outputs/opt125m_lora_20k_mixed_kgw25_shared \
  --model-name facebook/opt-125m \
  --device cuda \
  --local-files-only \
  --trajectory-cache outputs/opt125m_lora_20k_mixed_kgw25_shared/trajectory_cache.npz \
  --prepare-only \
  --train-count 20000 \
  --validation-count 500 \
  --test-count 256 \
  --generated-tokens 96 \
  --max-prompt-tokens 160 \
  --trajectory-batch-size 32 \
  --kgw-train-fraction 0.25 \
  --kgw-seed-count 1 \
  --kgw-seed-offset 3779 \
  --temperature 0.75 \
  --seed 20260810
