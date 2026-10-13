#!/usr/bin/env bash
set -euo pipefail

# Orchestrator to run ggssvt smoke -> core -> full plans sequentially.
# - Backs up existing campaign dirs
# - Runs each plan through tools/run_campaign_wrapper.py so query_chunk can be adjusted
# - Retries with smaller query_chunk if OOM detected
# - Logs actions to tools/run_campaigns.log

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="/home/titan/anaconda3/envs/ggssvt/bin/python"
WRAPPER="$REPO_ROOT/tools/run_campaign_wrapper.py"
NOTIFY_PY="$REPO_ROOT/tools/notify_email.py"
NOTIFY_WEBHOOK_PY="$REPO_ROOT/tools/notify_webhook.py"
LOG="$REPO_ROOT/tools/run_campaigns.log"
mkdir -p "$(dirname "$LOG")"

# Helper to send email and webhook notifications when configured
notify() {
  subject="$1"
  body="$2"

  # Email notify (if configured)
  if [ "${GGSSVT_DISABLE_EMAIL:-0}" != "1" ] && [ -x "$PYTHON" ] && [ -f "$NOTIFY_PY" ] && [ -f "$HOME/.config/ggssvt/notify.conf" ]; then
    tmpf=$(mktemp)
    printf "%s\n" "$body" > "$tmpf"
    "$PYTHON" "$NOTIFY_PY" --subject "$subject" --body-file "$tmpf" || echo "Warning: notify email script failed" >> "$LOG"
    rm -f "$tmpf"
  else
    echo "Notify email skipped (missing config or script)" >> "$LOG"
  fi

  # Webhook notify (if configured)
  if [ -x "$PYTHON" ] && [ -f "$NOTIFY_WEBHOOK_PY" ] && [ -f "$HOME/.config/ggssvt/webhook.conf" ]; then
    tmpf=$(mktemp)
    # Simple JSON payload
    printf '{"subject": "%s", "message": "%s", "host": "%s", "time": "%s"}' \
      "$(echo "$subject" | sed 's/"/\\"/g')" \
      "$(echo "$body" | sed 's/"/\\"/g')" \
      "$(hostname)" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$tmpf"
    "$PYTHON" "$NOTIFY_WEBHOOK_PY" --body-file "$tmpf" || echo "Warning: notify webhook script failed" >> "$LOG"
    rm -f "$tmpf"
  else
    echo "Notify webhook skipped (missing config or script)" >> "$LOG"
  fi
}

# Start a background progress pinger that sends a short summary every hour while
# the orchestrator is running. The pinger writes its PID to a file so it can be
# cleaned up by the main script's EXIT trap.
start_progress_pinger() {
  # Only run if at least one notifier is configured
  if [ ! -f "$HOME/.config/ggssvt/webhook.conf" ] && [ "${GGSSVT_DISABLE_EMAIL:-0}" = "1" ]; then
    echo "No notification config found; progress pinger disabled" >> "$LOG"
    return 0
  fi
  (
    echo "Progress pinger started at $(date -u)" >> "$LOG"
    while true; do
      # Sleep one hour
      sleep 3600
      # Build a short summary: last 50 lines of master log
      if [ -f "$LOG" ]; then
        summary=$(tail -n 50 "$LOG" | sed "s/'/\\'/g")
      else
        summary="(no log yet)"
      fi
      notify "GGSSVT progress on $(hostname)" "$summary"
    done
  ) &
  PINGER_PID=$!
  echo "$PINGER_PID" > /home/titan/ggssvt_progress.pid
  echo "Started progress pinger pid $PINGER_PID" >> "$LOG"
}

# Ensure the progress pinger is killed when the orchestrator exits
trap 'if [ -n "${PINGER_PID-}" ]; then kill "${PINGER_PID}" 2>/dev/null || true; fi' EXIT


echo "=== Run started at $(date -u) ===" >> "$LOG"
# Start the hourly progress pinger if notification is configured
start_progress_pinger || true

# Backup existing campaign dirs if any
WORK_DIR="$REPO_ROOT/work_dirs/ggssvt"
CAMPAIGN_DIR="$WORK_DIR/campaign"
CAMPAIGN_SMOKE_DIR="$WORK_DIR/campaign_smoke"
if [ -d "$CAMPAIGN_SMOKE_DIR" ]; then
  ts=$(date +"%Y%m%d_%H%M%S")
  bk="$WORK_DIR/backup_before_run_$ts"
  echo "Backing up existing campaign directories to $bk" | tee -a "$LOG"
  mkdir -p "$bk"
  if [ -d "$CAMPAIGN_SMOKE_DIR" ]; then mv "$CAMPAIGN_SMOKE_DIR" "$bk/"; fi
fi

# Helper to run a plan with given chunk/workers/batch
run_plan() {
  plan="$1"
  chunk="$2"
  workers="$3"
  batch="$4"

  out_log="$WORK_DIR/campaign_${plan}.log"
  pidfile="/home/titan/ggssvt_campaign_${plan}.pid"

  echo "\n--- Starting plan $plan (chunk=$chunk, workers=$workers, batch=$batch) at $(date) ---" | tee -a "$LOG" "$out_log"

  # Start the plan detached using nohup so it survives logout
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True GGSSVT_QUERY_CHUNK="$chunk" \
    "$PYTHON" "$WRAPPER" --plan "$plan" --device cuda --workers "$workers" --batch-size "$batch" \
    > "$out_log" 2>&1 &
  pid=$!
  echo "$pid" > "$pidfile"
  echo "Started pid $pid for plan $plan; log: $out_log" | tee -a "$LOG"

  # Wait loop: poll process, watch log for OOM
  oom_detected=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    # tail last lines to the master log for tracking
    tail -n 50 "$out_log" >> "$LOG" || true
    if grep -q "OutOfMemoryError" "$out_log" 2>/dev/null; then
      echo "OOM detected in plan $plan (pid $pid) at $(date)" | tee -a "$LOG"
      oom_detected=1
      break
    fi
  done

  # If process exited normally, capture final lines
  if kill -0 "$pid" 2>/dev/null; then
    # still running (shouldn't happen because loop exits when no pid), but handle
    kill "$pid" || true
  fi

  wait_exit=0
  if [ -f "$pidfile" ]; then
    # wait for process to actually exit and reap
    wait "$pid" 2>/dev/null || true
  fi

  tail -n 200 "$out_log" >> "$LOG" || true

  if [ "$oom_detected" -eq 1 ]; then
    echo "Plan $plan failed with OOM. Will attempt mitigation." | tee -a "$LOG"
    # Notify via email if configured
    notify "GGSSVT OOM: plan $plan" "OutOfMemoryError detected for plan $plan. Check log: $out_log"
    return 2
  fi

  # Check for generic failure indicators
  if grep -q "failed" "$out_log" 2>/dev/null; then
    echo "Plan $plan log contains 'failed' lines; inspect $out_log" | tee -a "$LOG"
    notify "GGSSVT failure: plan $plan" "Plan $plan log contains 'failed' lines. Check log: $out_log"
  fi

  echo "Plan $plan completed (no OOM detected) at $(date)" | tee -a "$LOG"
  # Optional per-plan completion notification (commented out to reduce noise)
  # notify "GGSSVT completed: plan $plan" "Plan $plan completed successfully. Log: $out_log"
  return 0
}

# Sequence with simple retry policy for OOM: try chunk 16384 -> 8192 -> 4096
plans=("core" "full")
for plan in "${plans[@]}"; do
  success=0
  # Try combinations of chunk sizes and workers progressively to reduce memory pressure
  for chunk in 512 256 128; do
    for workers in 4 2 0; do
      batch=1
      ret=0
      echo "Attempting plan $plan with chunk=$chunk, workers=$workers, batch=$batch" | tee -a "$LOG"
      run_plan "$plan" "$chunk" "$workers" "$batch" || ret=$?
      if [ "$ret" -eq 0 ]; then
        success=1
        break 2
      fi
      if [ "$ret" -eq 2 ]; then
        # OOM detected: try next worker setting (fewer workers) or smaller chunk
        echo "OOM detected for plan $plan with chunk=$chunk, workers=$workers; will try next settings" | tee -a "$LOG"
        # short cool-down before retrying
        sleep 10
        continue
      fi
      # Non-OOM failure: stop retrying this plan
      echo "Plan $plan failed with non-OOM error (ret=$ret); see log $WORK_DIR/campaign_${plan}.log" | tee -a "$LOG"
      break 2
    done
  done
  if [ "$success" -eq 1 ]; then
    echo "Plan $plan completed successfully" | tee -a "$LOG"
  else
    echo "Plan $plan exhausted attempts and did not complete" | tee -a "$LOG"
  fi
  echo "Finished attempts for plan $plan" | tee -a "$LOG"
done

echo "=== Run finished at $(date -u) ===" >> "$LOG"

summary=$(tail -n 200 "$LOG" | sed -n '1,200p')
# Send final notification (if configured)
notify "GGSSVT orchestration finished on $(hostname)" "$summary\nFull log at: $LOG\nWork dir: $WORK_DIR"

echo "Orchestration complete. Master log: $LOG" 
