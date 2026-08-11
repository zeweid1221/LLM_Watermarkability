#!/bin/bash -l
#SBATCH -A liyanxie
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH -J wm_optsdyn
#SBATCH -o logs/optsdyn_%A_%a.out
#SBATCH -e logs/optsdyn_%A_%a.err

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
    LABEL=static_only_topk
    KGW_FRACTION=0.0
    CACHE=outputs/opt125m_lora_20k_shared/trajectory_cache.npz
    ;;
  1)
    LABEL=mixed_kgw25_topk
    KGW_FRACTION=0.25
    CACHE=outputs/opt125m_lora_20k_mixed_kgw25_shared/trajectory_cache.npz
    ;;
  *)
    echo "ERROR: unknown array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

python run_experiment.py \
  --manifest data/mixed_prompts_22256.jsonl \
  --partition-dir assets/opt_hierarchy \
  --output-dir "outputs/opt125m_lora_20k_${LABEL}_seed20260810" \
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
  --train-batch-size 4 \
  --eval-batch-size 16 \
  --gradient-accumulation 4 \
  --epochs 3 \
  --learning-rate 2e-4 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --residual-loss-weight 0.02 \
  --target-mass-loss-weight 0.0 \
  --kl-loss-weight 0.02 \
  --target-adherence 0.99 \
  --kgw-train-fraction "$KGW_FRACTION" \
  --eval-rules ecc,kgw \
  --condition-summary-mode topk_contrast \
  --condition-top-k 128 \
  --training-targets-per-sequence 12 \
  --kgw-seed-count 1 \
  --kgw-seed-offset 3779 \
  --temperature 0.75 \
  --top-k 40 \
  --seed 20260810
