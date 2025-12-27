Param(
    [string]$Body,
    [string]$File,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

if (-not $Body -and -not $File) {
    Write-Error "Usage: .\scripts\run_python.ps1 -Body '<python code>' [-Args 'arg1','arg2'] OR -File '<path>' [-Args ...]"
    exit 2
}

# Create temporary script if Body provided
if ($Body) {
    $tmp = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), ([System.Guid]::NewGuid().ToString() + '.py'))
    Set-Content -Path $tmp -Value $Body -Encoding UTF8
    $script = $tmp
} else {
    $script = $File
}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "python executable not found in PATH"
    if ($Body) { Remove-Item $tmp -Force -ErrorAction SilentlyContinue }
    exit 1
}

# Run the python script and capture exit code and output
try {
    $procOutput = & $pythonCmd.Source $script @Args 2>&1
    $exitCode = $LASTEXITCODE
    if ($procOutput) { Write-Output $procOutput }
} catch {
    Write-Error $_
    $exitCode = 1
}

# Clean up temporary file if created
if ($Body) {
    try { Remove-Item $tmp -Force -ErrorAction SilentlyContinue } catch { }
}

exit $exitCode
