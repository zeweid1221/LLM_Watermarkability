#!/bin/bash -l
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -J wm_atom16_ecc
#SBATCH -o logs/atom16_ecc_%j.out
#SBATCH -e logs/atom16_ecc_%j.err

set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
PARTITION_DIR=assets/atomic16/opt-125m-lfqa5000-v1
OUTPUT_DIR=outputs/opt125m_atomic16_ecc_lfqa5000_rank16_epochs6_seed20260810

cd "$ROOT"
mkdir -p logs outputs
source "$ROOT/.venv/bin/activate"

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

test -f "$PARTITION_DIR/watermark_compositions.json"

python run_experiment.py \
  --training-protocol schedule_pool \
  --manifest data/lfqa_prompts_5756.jsonl \
  --partition-dir "$PARTITION_DIR" \
  --output-dir "$OUTPUT_DIR" \
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
  --epochs 6 \
  --learning-rate 2e-4 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --residual-loss-weight 0.02 \
  --target-mass-loss-weight 0.0 \
  --kl-loss-weight 0.02 \
  --target-adherence 0.99 \
  --train-key-count 1 \
  --eval-key-count 1 \
  --eval-rules ecc \
  --top-k 0 \
  --seed 20260810

python validate_atomic16_pipeline.py \
  --partition-dir "$PARTITION_DIR" \
  --experiment-dir "$OUTPUT_DIR"
