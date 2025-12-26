# run-ui-start.ps1 - start the production UI server in the background (agent-safe)
# Usage: .\run-ui-start.ps1
# Creates a PID file at ./ui/ui_start.pid

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir

# Ensure dependencies are installed and DB is up to date non-interactively
npm --prefix ui install --silent
npm --prefix ui run update_db --silent
npm --prefix ui run build --silent

# Start the production server in background and write PID
$proc = Start-Process -FilePath npm -ArgumentList '--prefix','ui','run','start' -NoNewWindow -WindowStyle Hidden -PassThru
# Wait a short moment to ensure the process started
Start-Sleep -Seconds 1
$pidFile = Join-Path $ScriptDir 'ui' 'ui_start.pid'
$proc.Id | Out-File -FilePath $pidFile -Encoding ascii
Write-Output "Started UI server (pid=$($proc.Id)). PID written to $pidFile"

Pop-Location
