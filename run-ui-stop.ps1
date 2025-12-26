# run-ui-stop.ps1 - stop the production UI server started by run-ui-start.ps1
# Usage: .\run-ui-stop.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $ScriptDir 'ui' 'ui_start.pid'
if (-Not (Test-Path $pidFile)) {
  Write-Output "PID file not found: $pidFile"
  exit 1
}
$pid = Get-Content $pidFile | Select-Object -First 1
try {
  Stop-Process -Id $pid -ErrorAction Stop
  Remove-Item $pidFile -ErrorAction SilentlyContinue
  Write-Output "Stopped UI process $pid and removed PID file"
} catch {
  Write-Output "Failed to stop process $pid: $_"
  exit 1
}
