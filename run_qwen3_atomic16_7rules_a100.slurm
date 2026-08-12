#!/bin/bash -l
#SBATCH -A liyanxie
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -J wm_qwen3_7rule
#SBATCH -o /users/2/deng0379/watermarkability_lora_msi/logs/qwen3_7rule_%j.out
#SBATCH -e /users/2/deng0379/watermarkability_lora_msi/logs/qwen3_7rule_%j.err

set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
PARTITION_DIR=assets/atomic16/qwen3-8b-lfqa5000-v1
OUTPUT_DIR=outputs/qwen3_8b_atomic16_universal_7rules_lfqa5000_each_rank16_epochs6_seed20260811
RULES=ecc,binary,cw_abcd,unigram,atomic_kgw,atomic_kgw25,entropy_kgw

cd "$ROOT"
mkdir -p logs outputs
source "$ROOT/.venv/bin/activate"

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

test -f "$PARTITION_DIR/watermark_compositions.json"

python run_experiment.py \
  --manifest data/lfqa_prompts_5756.jsonl \
  --partition-dir "$PARTITION_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-name Qwen/Qwen3-8B \
  --device cuda \
  --local-files-only \
  --train-count 5000 \
  --validation-count 500 \
  --test-count 256 \
  --generated-tokens 96 \
  --max-prompt-tokens 160 \
  --trajectory-batch-size 8 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --gradient-accumulation 2 \
  --epochs 6 \
  --learning-rate 2e-4 \
  --weight-decay 0.01 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --residual-loss-weight 0.02 \
  --target-mass-loss-weight 0.0 \
  --kl-loss-weight 0.02 \
  --target-adherence 0.99 \
  --training-targets-per-sequence 0 \
  --train-key-count 1 \
  --eval-key-count 1 \
  --train-rules "$RULES" \
  --eval-rules "$RULES" \
  --condition-summary-mode topk_contrast \
  --condition-top-k 128 \
  --entropy-gate-threshold 2.0 \
  --temperature 0.75 \
  --top-k 0 \
  --seed 20260810

python validate_atomic16_pipeline.py \
  --partition-dir "$PARTITION_DIR" \
  --experiment-dir "$OUTPUT_DIR" \
  --expected-rules "$RULES"
