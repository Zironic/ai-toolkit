$ROOT = Split-Path $PSScriptRoot -Parent
$PYTHON = "$ROOT\venv\Scripts\python.exe"
$SCRIPT = "$ROOT\scripts\capture_krea_txtfusion_forward_batch.py"
$VAE = "D:\.cache\huggingface\hub\models--Qwen--Qwen-Image\snapshots\75e0b4be04f60ec59a75f475837eced720f823b6"
$VRAM_LIMIT_MB = 3000

New-Item -ItemType Directory -Force "$ROOT\logs" | Out-Null

$vramUsed = [int](& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
if ($vramUsed -gt $VRAM_LIMIT_MB) {
    Write-Host "ABORT: VRAM at ${vramUsed} MB (limit ${VRAM_LIMIT_MB} MB) - another model process may be running"
    exit 1
}
Write-Host "VRAM check OK: ${vramUsed} MB used"

Write-Host "=== Run 1: testpng t050 image noise=0.5 ==="
& $PYTHON $SCRIPT `
    --output-dir "$ROOT\loras\krea_vector_explore\txtfusion_probe\captures\multicapture_testpng_t050_float8" `
    --latent-mode image `
    --noise-t 0.5 `
    --vae-path $VAE `
    --skip-existing `
    --width 512 --height 512 `
    --qtype float8 `
    --layer-offloading `
    --layer-offloading-transformer-percent 0.5
if ($LASTEXITCODE -ne 0) { Write-Host "Run 1 FAILED (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Run 2: random t100 noise=1.0 ==="
& $PYTHON $SCRIPT `
    --output-dir "$ROOT\loras\krea_vector_explore\txtfusion_probe\captures\multicapture_random_t100_float8" `
    --latent-mode random `
    --noise-t 1.0 `
    --skip-vae `
    --skip-existing `
    --width 512 --height 512 `
    --qtype float8 `
    --layer-offloading `
    --layer-offloading-transformer-percent 0.5
if ($LASTEXITCODE -ne 0) { Write-Host "Run 2 FAILED (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Both captures complete ==="
