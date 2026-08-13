#!/bin/bash -l
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH -J wm_opt20k_obj
#SBATCH -o logs/opt20k_obj_%A_%a.out
#SBATCH -e logs/opt20k_obj_%A_%a.err

set -euo pipefail
ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    LABEL=current_residual_0p02
    RESIDUAL_WEIGHT=0.02
    TARGET_WEIGHT=0.0
    ;;
  1)
    LABEL=strong_residual_0p10
    RESIDUAL_WEIGHT=0.10
    TARGET_WEIGHT=0.0
    ;;
  2)
    LABEL=direct_target_0p50
    RESIDUAL_WEIGHT=0.02
    TARGET_WEIGHT=0.50
    ;;
  *)
    echo "ERROR: unknown array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available(), "This experiment requires a GPU allocation."
PY

python run_experiment.py \
  --training-protocol schedule_pool \
  --manifest data/mixed_prompts_22256.jsonl \
  --partition-dir assets/opt_hierarchy \
  --output-dir "outputs/opt125m_lora_20k_${LABEL}_seed20260810" \
  --model-name facebook/opt-125m \
  --device cuda \
  --local-files-only \
  --trajectory-cache outputs/opt125m_lora_20k_shared/trajectory_cache.npz \
  --skip-prepare \
  --train-count 20000 \
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
  --residual-loss-weight "$RESIDUAL_WEIGHT" \
  --target-mass-loss-weight "$TARGET_WEIGHT" \
  --kl-loss-weight 0.02 \
  --target-adherence 0.99 \
  --top-k 40 \
  --seed 20260810
