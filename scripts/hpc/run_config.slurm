#!/bin/bash -l
# Generic example for a single-GPU Slurm cluster. Scientific settings come
# exclusively from the JSON config passed as the first script argument.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -J watermarkability
#SBATCH -o logs/watermarkability_%j.out
#SBATCH -e logs/watermarkability_%j.err

set -euo pipefail

CONFIG=${1:?"usage: sbatch scripts/hpc/run_config.slurm CONFIG.json"}
ROOT=${WM_LORA_ROOT:-$PWD}
VENV=${WM_VENV:-$ROOT/.venv}

cd "$ROOT"
mkdir -p logs outputs
source "$VENV/bin/activate"

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

python run_paper_protocol.py --config "$CONFIG"
