#!/bin/bash -l
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -J wm_opt_lora5k
#SBATCH -o logs/opt_lora5k_%j.out
#SBATCH -e logs/opt_lora5k_%j.err

set -euo pipefail
ROOT=${WM_LORA_ROOT:-${SLURM_SUBMIT_DIR:-$HOME/watermarkability_lora_msi}}
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "This experiment requires a GPU allocation."
PY

python run_experiment.py \
  --training-protocol schedule_pool \
  --manifest data/mixed_prompts_5756.jsonl \
  --partition-dir assets/opt_hierarchy \
  --output-dir outputs/opt125m_lora_5k_seed20260810 \
  --model-name facebook/opt-125m \
  --device cuda \
  --local-files-only \
  --train-count 5000 \
  --validation-count 500 \
  --test-count 256 \
  --generated-tokens 96 \
  --max-prompt-tokens 160 \
  --trajectory-batch-size 32 \
  --train-batch-size 4 \
  --eval-batch-size 16 \
  --gradient-accumulation 4 \
  --epochs 3 \
  --learning-rate 2e-4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --target-adherence 0.99 \
  --seed 20260810
