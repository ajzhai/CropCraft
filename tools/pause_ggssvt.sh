#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SNAPSHOT="$REPO_ROOT/work_dirs/ggssvt_snapshot_${STAMP}.tar.zst"

systemctl --user stop ggssvt-orchestrator.service
tar --zstd -cf "$SNAPSHOT" -C "$REPO_ROOT" \
  work_dirs/ggssvt/campaign \
  work_dirs/ggssvt/campaign_core.log \
  work_dirs/ggssvt/campaign_full.log \
  work_dirs/ggssvt/campaign_smoke.log \
  tools/run_campaigns.log
sha256sum "$SNAPSHOT" > "${SNAPSHOT}.sha256"
printf 'GGSSVT paused. Snapshot: %s\n' "$SNAPSHOT"
