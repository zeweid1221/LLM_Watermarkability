#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
source "$ROOT/.venv/bin/activate"

python extend_manifest.py \
  --base data/mixed_prompts_5756.jsonl \
  --output data/mixed_prompts_22256.jsonl \
  --train-count 20000 \
  --train-source-weights ultrachat:0.40,lfqa:0.30,oasst1:0.10,fineweb_edu:0.20 \
  --seed 20260810
