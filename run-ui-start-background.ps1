# run-ui-start-background.ps1 - start the production UI server in the background (agent-safe)
# Usage: .\run-ui-start-background.ps1
# Builds the UI by default (runs `npm run build`) and then starts the server.
# PID file creation is disabled by default to avoid stale PID issues.
# For usage examples (flags, env vars like NO_UI_BUILD), see AGENTS.md in the repository root.

Param(
    [switch]$NoBuild
)

# Honor env var NO_UI_BUILD=1 or 'true' to skip the build
if ($env:NO_UI_BUILD -and $env:NO_UI_BUILD -ne '') {
    if ($env:NO_UI_BUILD -eq '1' -or $env:NO_UI_BUILD.ToLower() -eq 'true') { $NoBuild = $true }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir

# Use working dir based npm invocation to perform build_and_start
$uiDir = Join-Path $ScriptDir 'ui'
$logOut = Join-Path $uiDir 'ui_start.log'
$logErr = Join-Path $uiDir 'ui_start.err'

# Prefer 'npm.cmd' on Windows
$npmExe = 'npm.cmd'
if (-Not (Get-Command $npmExe -ErrorAction SilentlyContinue)) { $npmExe = 'npm' }

if ($NoBuild) {
    Write-Output "NoBuild set or NO_UI_BUILD=1 - skipping build. Starting UI server (npm run start)..."
    $proc = Start-Process -FilePath $npmExe -ArgumentList 'run','start' -WorkingDirectory $uiDir -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru
} else {
    # Always build first so local code changes are picked up, then start the server.
    Write-Output "Running 'npm run build' in $uiDir (logs: $logOut, $logErr)..."
    $buildProc = Start-Process -FilePath $npmExe -ArgumentList 'run','build' -WorkingDirectory $uiDir -NoNewWindow -RedirectStandardOutput $logOut -RedirectStandardError $logErr -Wait -PassThru

    if ($buildProc.ExitCode -ne 0) {
        Write-Output "Build failed (exit=$($buildProc.ExitCode)). Check logs: $logOut, $logErr."
        Pop-Location
        exit 1
    }

    Write-Output "Build succeeded, starting UI server (npm run start)..."
    $proc = Start-Process -FilePath $npmExe -ArgumentList 'run','start' -WorkingDirectory $uiDir -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr -PassThru
}
Start-Sleep -Seconds 1
$pidFile = Join-Path $uiDir 'ui_start.pid'

# Try to discover the actual server process listening on port 8675 (the 'next start' process)
$serverPid = $null
$maxAttempts = 10
for ($i = 0; $i -lt $maxAttempts; $i++) {
    try {
        # Prefer Get-NetTCPConnection when available
        $conn = Get-NetTCPConnection -LocalPort 8675 -ErrorAction Stop | Where-Object { $_.State -eq 'Listen' } | Select-Object -First 1
        if ($conn -and $conn.OwningProcess) { $serverPid = $conn.OwningProcess; break }
    } catch {
        # Fall back to parsing netstat output (works on systems without Get-NetTCPConnection)
        try {
            $ns = netstat -ano | Select-String ':8675'
            if ($ns) {
                # take the first match and parse the PID (last whitespace token)
                $first = $ns[0].ToString()
                $parts = $first -split '\s+' | Where-Object { $_ -ne '' }
                $maybePid = $parts[-1]
                [int]$parsed = 0
                if ([int]::TryParse($maybePid, [ref]$parsed)) { $serverPid = $parsed; break }
            }
        } catch {
            # ignore parsing errors and retry
        }
    }
    Start-Sleep -Seconds 1
}

if ($serverPid -ne $null) {
    Write-Output "Started UI server (pid=$serverPid). Logs: $logOut, $logErr. (PID file creation disabled)"
} elseif ($proc -and $proc.Id) {
    # Fallback: we were unable to detect the final server PID but the helper launched.
    Write-Output "Started UI helper (pid=$($proc.Id)), server PID not yet detected on port 8675. Logs: $logOut, $logErr. (PID file creation disabled)"
} else {
    Write-Output "Failed to start UI server"
    exit 1
}

Pop-Location
