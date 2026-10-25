param([string]$BasePython, [string]$UvPath)
. "$PSScriptRoot\scripts\project-env.ps1"
if (-not $BasePython) {
    $BasePython = Join-Path (Split-Path -Parent $ProjectRoot) 'tabular_ml\.local\python\cpython-3.12.13-windows-x86_64-none\python.exe'
}
if (-not $UvPath) {
    $UvPath = Join-Path (Split-Path -Parent $ProjectRoot) 'tabular_ml\.local\tools\uv.exe'
}
if (-not (Test-Path -LiteralPath $BasePython)) { throw 'Base Python tidak ditemukan. Berikan -BasePython dengan path Python di D:.' }
if (-not (Test-Path -LiteralPath $UvPath)) { throw 'uv tidak ditemukan. Berikan -UvPath dengan path uv.exe yang tersedia.' }
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    & $UvPath venv --python $BasePython (Join-Path $ProjectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Pembuatan venv gagal.' }
}
& $UvPath pip install --python $ProjectPython --index-url 'https://download.pytorch.org/whl/cu121' 'torch==2.5.1' 'torchvision==0.20.1'
if ($LASTEXITCODE -ne 0) { throw 'Instalasi PyTorch CUDA gagal.' }
$DependencyFile = if (Test-Path -LiteralPath "$ProjectRoot\requirements.txt") { "$ProjectRoot\requirements.txt" } else { "$ProjectRoot\requirements.in" }
& $UvPath pip install --python $ProjectPython --index-url 'https://pypi.org/simple' -r $DependencyFile
if ($LASTEXITCODE -ne 0) { throw 'Instalasi dependensi gagal.' }
& $ProjectPython -B -c 'import torch; print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")'
if ($LASTEXITCODE -ne 0) { throw 'Verifikasi environment gagal.' }
