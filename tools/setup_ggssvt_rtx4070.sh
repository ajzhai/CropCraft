#!/usr/bin/env bash
set -euo pipefail

# Create the modern GG-SSVT environment without disturbing Nerfstudio's
# separate cropcraft environment.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/ggssvt/environment.yml"
REQ_FILE="$REPO_ROOT/ggssvt/requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; install Miniconda or Anaconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx ggssvt; then
  echo "Conda environment ggssvt already exists; updating dependencies."
else
  conda env create -f "$ENV_FILE"
fi

conda run -n ggssvt python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -n ggssvt python -m pip install -r "$REQ_FILE"
conda run -n ggssvt python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the ggssvt environment")
print(torch.cuda.get_device_name(0))
print(torch.__version__)
PY

conda env config vars set -n ggssvt PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Environment ready. Run: conda activate ggssvt"
