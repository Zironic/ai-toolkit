# Check Playwright setup (PowerShell)
# Usage: .\scripts\check_playwright_setup.ps1

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path -Path (Join-Path $scriptDir '..')
$uiPath = Join-Path $repoRoot 'ui'

Write-Host "Repo root: $repoRoot"
Write-Host "UI folder: $uiPath"

if (-not (Test-Path $uiPath)) {
  Write-Error "UI folder not found at $uiPath"
  exit 2
}

# Run Playwright commands using npm --prefix to avoid cwd issues
Write-Host "Checking Playwright version (using local ui package)..."
& npm --prefix $uiPath exec -- playwright --version
$ver = $LASTEXITCODE
if ($ver -ne 0) { Write-Warning "Playwright version command finished with code $ver" }

Write-Host "Ensuring Playwright browsers are installed..."
& npm --prefix $uiPath exec -- playwright install --with-deps
$code = $LASTEXITCODE
if ($code -ne 0) {
  Write-Error "Playwright install failed with code $code"
  exit $code
}

Write-Host "Playwright setup looks OK."