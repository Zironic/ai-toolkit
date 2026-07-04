$ROOT = Split-Path $PSScriptRoot -Parent
$PYTHON = "$ROOT\venv\Scripts\python.exe"
$SCRIPT = "$ROOT\scripts\score_krea_turbo_candidates.py"
$VRAM_LIMIT_MB = 3000

$vramUsed = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
if ($vramUsed -gt $VRAM_LIMIT_MB) {
    Write-Host "ABORT: VRAM at ${vramUsed} MB (limit ${VRAM_LIMIT_MB} MB)"
    exit 1
}
Write-Host "VRAM check OK: ${vramUsed} MB used"

$COMMON = @(
    "--features", "$ROOT\loras\krea_vector_explore\txtfusion_probe\cache\features.pt",
    "--subset-json", "$ROOT\loras\krea_vector_explore\txtfusion_probe\captures\krea_txtfusion_projector_subset_search.json",
    "--top-k", "5",
    "--steps", "4",
    "--layer-offloading",
    "--layer-offloading-transformer-percent", "0.5"
)

Write-Host ""
Write-Host "=== Strength 0.03 ==="
& $PYTHON $SCRIPT @COMMON `
    --strength 0.03 `
    --output-dir "$ROOT\loras\krea_vector_explore\txtfusion_probe\turbo_score_s003"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Strength 0.05 ==="
& $PYTHON $SCRIPT @COMMON `
    --strength 0.05 `
    --output-dir "$ROOT\loras\krea_vector_explore\txtfusion_probe\turbo_score_s005"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Done. Check turbo_score_s003/ and turbo_score_s005/ ==="
