param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)
# run-ui.ps1 - forward to UI npm scripts
# Usage: .\run-ui.ps1 [npm-script] [args]
# Example: .\run-ui.ps1 (defaults to build_and_start)
# Example: .\run-ui.ps1 test:caption --silent

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location (Join-Path $scriptDir 'ui')

if ($Args.Count -eq 0) {
    cmd /c "npm run build_and_start"
    Pop-Location
    exit $LASTEXITCODE
}

$script = $Args[0]
$rest = @()
if ($Args.Count -gt 1) { $rest = $Args[1..($Args.Count - 1)] }
if ($rest.Count -eq 0) {
    cmd /c "npm run $script"
} else {
    $restArg = $rest -join ' '
    cmd /c "npm run $script -- $restArg"
}
Pop-Location
