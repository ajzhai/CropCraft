#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
WORK="$ROOT/work_dirs/ab_ssvit"
CACHE="$ROOT/work_dirs/ggssvt/cache"
TSDF_CACHE="$ROOT/work_dirs/ggssvt/cache_tsdf"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$WORK" "$TSDF_CACHE"

if [ ! -f "$CACHE/quality.json" ]; then
  "$PYTHON" -m ggssvt.cli preprocess --cache-dir "$CACHE"
fi

"$PYTHON" -m ggssvt.cli fuse \
  --cache-dir "$CACHE" \
  --write-cache "$TSDF_CACHE" \
  --out "$ROOT/work_dirs/ggssvt/reports/fusion.json"
cp "$ROOT/work_dirs/ggssvt/reports/fusion.json" "$WORK/fusion.json"

"$PYTHON" -m ggssvt.cli baselines \
  --cache-dir "$CACHE" \
  > "$WORK/baselines.txt"

"$PYTHON" -m ggssvt.cli dino-probe \
  --cache-dir "$CACHE" \
  --backbones dinov2 \
  --variant base \
  --components 8 \
  --alphas 0.1 1.0 10.0 \
  --out "$WORK/dino_probe.json"

"$PYTHON" -m ggssvt.eval.ab_ssvit \
  --cache-dir "$CACHE" \
  --variant base \
  --alpha 1.0 \
  --out "$WORK/ab_ssvit.json"

printf '\nAB-SSViT complete. Results are in %s\n' "$WORK"
