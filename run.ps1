param(
    [int]$Port = 8000,
    [ValidateSet('auto', 'cpu', 'cuda:0')][string]$Device = 'auto'
)
. "$PSScriptRoot\scripts\project-env.ps1"
if (-not (Test-Path -LiteralPath $ProjectPython)) { throw 'Environment belum tersedia. Jalankan setup.ps1 terlebih dahulu.' }
$env:PCB_DEVICE = $Device
Push-Location $ProjectRoot
try {
    & $ProjectPython -B -m uvicorn pcb_inspector.api:app --host 127.0.0.1 --port $Port --workers 1
    if ($LASTEXITCODE -ne 0) { throw 'Server berhenti dengan error.' }
} finally {
    Pop-Location
}
