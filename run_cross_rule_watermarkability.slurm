#!/bin/bash -l
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH -J wm_crossrule
#SBATCH -o logs/crossrule_%A_%a.out
#SBATCH -e logs/crossrule_%A_%a.err

set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs outputs/cross_rule_watermarkability_20260813
source "$ROOT/.venv/bin/activate"

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RULES=ecc,binary,cw_abcd,unigram,atomic_kgw,atomic_kgw25
COMMON=(
  --manifest data/lfqa_prompts_5756.jsonl
  --rules "$RULES"
  --test-count 256
  --generated-tokens 96
  --max-prompt-tokens 160
  --temperature 0.75
  --top-k 0
  --target-mass 0.99
  --fixed-bias 20
  --train-key-count 1
  --eval-key-count 1
  --kgw-seed-count 1
  --kgw-seed-offset 3779
  --seed 20260810
  --device cuda
  --local-files-only
  --require-reference-match
)

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  MODEL=facebook/opt-125m
  PARTITION=assets/atomic16/opt-125m-lfqa5000-v1
  REFERENCE=outputs/opt125m_composition_pool_corrected_lfqa5000x4_rank16_epochs6_seed20260812
  OUTPUT=outputs/cross_rule_watermarkability_20260813/opt-125m
  BATCH=16
else
  MODEL=Qwen/Qwen3-8B
  PARTITION=assets/atomic16/qwen3-8b-lfqa5000-v1
  REFERENCE=outputs/qwen3_8b_composition_pool_corrected_lfqa5000x4_rank16_epochs6_seed20260812
  OUTPUT=outputs/cross_rule_watermarkability_20260813/qwen3-8b
  BATCH=8
fi

test -s "$PARTITION/watermark_compositions.json"
test -s "$REFERENCE/evaluation_details.csv"
test -s "$REFERENCE/evaluation_summary.csv"

python analyze_cross_rule_watermarkability.py \
  "${COMMON[@]}" \
  --model-name "$MODEL" \
  --partition-dir "$PARTITION" \
  --reference-details "$REFERENCE/evaluation_details.csv" \
  --reference-summary "$REFERENCE/evaluation_summary.csv" \
  --output-dir "$OUTPUT" \
  --batch-size "$BATCH"
