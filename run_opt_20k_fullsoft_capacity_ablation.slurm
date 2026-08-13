#!/bin/bash -l
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH -J wm_opt20k_fs
#SBATCH -o logs/opt20k_fullsoft_%A_%a.out
#SBATCH -e logs/opt20k_fullsoft_%A_%a.err

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
    LABEL=rank8_epochs3
    LORA_RANK=8
    LORA_ALPHA=16
    EPOCHS=3
    ;;
  1)
    LABEL=rank8_epochs6
    LORA_RANK=8
    LORA_ALPHA=16
    EPOCHS=6
    ;;
  2)
    LABEL=rank16_epochs3
    LORA_RANK=16
    LORA_ALPHA=32
    EPOCHS=3
    ;;
  3)
    LABEL=rank16_epochs6
    LORA_RANK=16
    LORA_ALPHA=32
    EPOCHS=6
    ;;
  *)
    echo "ERROR: unknown array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

CACHE=outputs/opt125m_lora_20k_shared/trajectory_cache.npz
if [ ! -f "$CACHE" ]; then
  echo "ERROR: missing $CACHE; run run_opt_20k_prepare.slurm first." >&2
  exit 1
fi

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
  --output-dir "outputs/opt125m_lora_20k_fullsoft_${LABEL}_seed20260810" \
  --model-name facebook/opt-125m \
  --device cuda \
  --local-files-only \
  --trajectory-cache "$CACHE" \
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
  --epochs "$EPOCHS" \
  --learning-rate 2e-4 \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --residual-loss-weight 0.02 \
  --target-mass-loss-weight 0.0 \
  --kl-loss-weight 0.02 \
  --target-adherence 0.99 \
  --top-k 0 \
  --seed 20260810
