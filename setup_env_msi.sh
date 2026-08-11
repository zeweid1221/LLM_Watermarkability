#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load python/3.10.9_anaconda2023.03_libmamba
fi
if [ -x "$HOME/.conda/envs/watermark/bin/python" ]; then
  BASE_PYTHON="$HOME/.conda/envs/watermark/bin/python"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate watermark
  BASE_PYTHON=$(command -v python)
else
  echo "ERROR: the existing MSI watermark environment is unavailable." >&2
  exit 1
fi

if [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$ROOT/.venv"
fi
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements_msi.txt"

python - <<'PY'
import sys
import numpy, pandas, torch, transformers
print("python", sys.executable)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("numpy", numpy.__version__, "pandas", pandas.__version__)
PY

echo "Environment ready: $ROOT/.venv"
