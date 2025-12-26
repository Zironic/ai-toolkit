# run-ui-start.ps1 - start the production UI server in the background (agent-safe)
# Usage: .\run-ui-start.ps1
# Creates a PID file at ./ui/ui_start.pid

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir

# Ensure dependencies are installed and DB is up to date non-interactively
npm --prefix ui install --silent
npm --prefix ui run update_db --silent
npm --prefix ui run build --silent

# Start the production server detached, redirect logs, and write PID
$uiDir = Join-Path $ScriptDir 'ui'
$logOut = Join-Path $uiDir 'ui_start.log'
$logErr = Join-Path $uiDir 'ui_start.err'
# Start-Process without -NoNewWindow creates a detached new process; redirect output so agent can inspect logs
$proc = Start-Process -FilePath 'npm' -ArgumentList '--prefix','ui','run','start' -WorkingDirectory $uiDir -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru
Start-Sleep -Seconds 1
$pidFile = Join-Path $uiDir 'ui_start.pid'
if ($proc -and $proc.Id) {
    $proc.Id | Out-File -FilePath $pidFile -Encoding ascii
    Write-Output "Started UI server (pid=$($proc.Id)). Logs: $logOut, $logErr. PID written to $pidFile"
} else {
    Write-Output "Failed to start UI server"
    exit 1
}

Pop-Location
