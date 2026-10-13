#!/usr/bin/env bash
set -euo pipefail

# Run GG-SSVT on an RTX 4070 with conservative memory settings.
# Usage: tools/run_campaigns_rtx4070.sh [smoke|core|full]
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAN="${1:-core}"
PYTHON="${PYTHON:-python}"
CHUNK="${GGSSVT_QUERY_CHUNK:-128}"
WORKERS="${GGSSVT_WORKERS:-0}"
BATCH_SIZE="${GGSSVT_BATCH_SIZE:-1}"

case "$PLAN" in
  smoke|core|full) ;;
  *) echo "Usage: $0 [smoke|core|full]" >&2; exit 2 ;;
esac

cd "$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GGSSVT_QUERY_CHUNK="$CHUNK"

exec "$PYTHON" "$REPO_ROOT/tools/run_campaign_wrapper.py" \
  --plan "$PLAN" \
  --device cuda \
  --workers "$WORKERS" \
  --batch-size "$BATCH_SIZE"
