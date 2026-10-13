#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/AB-SSViT/environment.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx ab_ssvit; then
  conda env create -f "$ENV_FILE"
fi

conda run -n ab_ssvit python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -n ab_ssvit python -m pip install -r "$ROOT/AB-SSViT/requirements.txt"
conda run -n ab_ssvit python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(torch.cuda.get_device_name(0))
print(torch.__version__)
PY

conda env config vars set -n ab_ssvit PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Ready: conda activate ab_ssvit"
