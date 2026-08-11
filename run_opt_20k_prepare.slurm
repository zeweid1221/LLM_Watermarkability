#!/bin/bash -l
#SBATCH -A liyanxie
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH -J wm_opt20k_prep
#SBATCH -o logs/opt20k_prep_%j.out
#SBATCH -e logs/opt20k_prep_%j.err

set -euo pipefail
ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

python run_experiment.py \
  --manifest data/mixed_prompts_22256.jsonl \
  --partition-dir assets/opt_hierarchy \
  --output-dir outputs/opt125m_lora_20k_shared \
  --trajectory-cache outputs/opt125m_lora_20k_shared/trajectory_cache.npz \
  --model-name facebook/opt-125m \
  --device cuda \
  --local-files-only \
  --train-count 20000 \
  --validation-count 500 \
  --test-count 256 \
  --generated-tokens 96 \
  --max-prompt-tokens 160 \
  --trajectory-batch-size 32 \
  --temperature 0.75 \
  --seed 20260810 \
  --prepare-only
