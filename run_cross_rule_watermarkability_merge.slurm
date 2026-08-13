#!/bin/bash -l
#SBATCH -p agsmall
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -J wm_crossmerge
#SBATCH -o logs/crossmerge_%j.out
#SBATCH -e logs/crossmerge_%j.err

set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
RESULT_ROOT=outputs/cross_rule_watermarkability_20260813
cd "$ROOT"
mkdir -p logs "$RESULT_ROOT/combined"
source "$ROOT/.venv/bin/activate"

python validate_cross_rule_watermarkability.py \
  --result-dir "$RESULT_ROOT/opt-125m" \
  --result-dir "$RESULT_ROOT/qwen3-8b" \
  --output-dir "$RESULT_ROOT/combined"

tar -czf cross_rule_watermarkability_20260813.tar.gz "$RESULT_ROOT"
echo "archive: $ROOT/cross_rule_watermarkability_20260813.tar.gz"
