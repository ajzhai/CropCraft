# resume_experiments.ps1
# Resumes the GG-SSVT experiment pipeline after an interruption (e.g. power trip).
# Safe to run at any time: run_all skips every step whose artefact is already
# current, and this script refuses to start a second run if one is already active
# (two campaigns writing the same work_dir would race and corrupt results).
#
# Manual use:   powershell -ExecutionPolicy Bypass -File C:\Users\u25737806\CropCraft\resume_experiments.ps1
# Registered as scheduled task "GGSSVT-Resume" (at logon) by Claude on 2026-09-04.

$ErrorActionPreference = 'Stop'
$repo  = 'C:\Users\u25737806\CropCraft'
$conda = 'C:\Users\u25737806\AppData\Local\anaconda3\Scripts\conda.exe'
$logDir = Join-Path $repo 'work_dirs\ggssvt\resume_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log   = Join-Path $logDir "resume_$stamp.log"

function Write-Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    $line | Tee-Object -FilePath $log -Append
}

# --- Guard: is a run already active? (prevents double-launch racing the campaign)
$active = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'ggssvt\.(run_all|campaign)' }
if ($active) {
    Write-Log "A GG-SSVT run is already active (PID $($active.ProcessId -join ',')). Nothing to do."
    exit 0
}

# --- Already fully complete? Then nothing to resume.
$summary = Join-Path $repo 'work_dirs\ggssvt\campaign\summary.txt'
if (Test-Path $summary) {
    Write-Log "Campaign already complete ($summary exists). Nothing to resume."
    # Self-disable the logon task now that the work is done (ignore if not present).
    try { schtasks /Delete /TN 'GGSSVT-Resume' /F | Out-Null; Write-Log "Removed GGSSVT-Resume logon task." } catch {}
    exit 0
}

Write-Log "Resuming pipeline. Log: $log"
Set-Location $repo
# run_all is idempotent: skips current artefacts, restarts only the unfinished campaign.
& $conda run -n ggssvt --no-capture-output python -m ggssvt.run_all --device cuda --batch-size 1 2>&1 |
    Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
Write-Log "run_all exited with code $code"
exit $code
