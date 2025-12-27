# run-ui-stop-background.ps1 - stop the production UI server started by run-ui-start-background.ps1
# Usage: .\run-ui-stop-background.ps1 [ -ForceAllNode ]

Param(
  [switch]$ForceAllNode
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$uiDir = Join-Path $ScriptDir 'ui'

# First attempt: targeted stop of node processes that reference the UI folder in their command line
try {
  $uiRelatedNodes = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'node' -and ($_.CommandLine -and ($_.CommandLine -match [regex]::Escape($uiDir))) }
  if ($ForceAllNode) {
    Write-Output "Force flag set: killing all node.exe processes"
    Get-Process -Name node -ErrorAction SilentlyContinue | ForEach-Object { Write-Output "Killing node PID $($_.Id)"; Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
    # Clean up PID file if present
    $pidFile = Join-Path $uiDir 'ui_start.pid'
    if (Test-Path $pidFile) { Remove-Item $pidFile -ErrorAction SilentlyContinue }
    Write-Output "Force-kill completed"
    exit 0
  } elseif ($uiRelatedNodes -and $uiRelatedNodes.Count -gt 0) {
    foreach ($p in $uiRelatedNodes) {
      try {
        Write-Output "Stopping UI-related node process $($p.ProcessId) (cmdline contains UI path)"
        Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
      } catch {
        Write-Output "Failed to stop UI-related process $($p.ProcessId): $($_.Exception.Message)"
      }
    }
    # Remove PID file if present
    $pidFile = Join-Path $uiDir 'ui_start.pid'
    if (Test-Path $pidFile) { Remove-Item $pidFile -ErrorAction SilentlyContinue }
    Write-Output "Stopped all UI-related node processes"
    exit 0
  }
} catch {
  # ignore errors and fall back to PID-file based stop
  Write-Output "Targeted stop attempt failed: $($_.Exception.Message)"
}

$pidFile = Join-Path $uiDir 'ui_start.pid'
if (-Not (Test-Path $pidFile)) {
  Write-Output "PID file not found: $pidFile"
  exit 1
}
# Read and validate PID content
$raw = Get-Content $pidFile | Select-Object -First 1
$raw = $raw -as [string]
$pidText = $raw.Trim()
[int]$pidInt = 0
if (-not [int]::TryParse($pidText,[ref]$pidInt)) {
  Write-Output "Invalid PID in file: $pidFile (contents: '$pidText')"
  # Remove corrupted PID file to avoid repeated failures
  Remove-Item $pidFile -ErrorAction SilentlyContinue
  exit 1
}
try {
  Stop-Process -Id $pidInt -ErrorAction Stop
  Remove-Item $pidFile -ErrorAction SilentlyContinue
  Write-Output "Stopped UI process ${pidInt} and removed PID file"
} catch {
  # If the process is already gone, treat that as success and remove the stale PID file.
  $msg = $_.Exception.Message
  if ($msg -and $msg -match 'Cannot find a process') {
    Write-Output "Process ${pidInt} not running; removing stale PID file."
    Remove-Item $pidFile -ErrorAction SilentlyContinue
    exit 0
  } else {
    Write-Output "Failed to stop process ${pidInt}: $msg"
    exit 1
  }
}
