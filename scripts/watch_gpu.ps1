param([int]$IntervalSec = 3)
while ($true) {
    $ts = Get-Date -Format "HH:mm:ss"
    $out = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>$null
    if ($out) {
        $parts = $out -split ','
        $used = $parts[0].Trim()
        $total = $parts[1].Trim()
        $gpu3d = $parts[2].Trim()
        $memBw = $parts[3].Trim()
        Write-Host "$ts  VRAM ${used}/${total} MB   3D ${gpu3d}%   MemBW ${memBw}%"
    }
    Start-Sleep -Seconds $IntervalSec
}
