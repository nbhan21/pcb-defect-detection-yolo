param([string]$BasePython, [string]$UvPath)
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$LocalRoot = Join-Path $ProjectRoot '.local'
foreach ($Path in @('tmp', 'cache\uv', 'cache\pip', 'cache\torch', 'cache\xdg', 'config\ultralytics', 'matplotlib', 'state')) {
    New-Item -ItemType Directory -Path (Join-Path $LocalRoot $Path) -Force | Out-Null
}
$env:TEMP = Join-Path $LocalRoot 'tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:UV_CACHE_DIR = Join-Path $LocalRoot 'cache\uv'
$env:PIP_CACHE_DIR = Join-Path $LocalRoot 'cache\pip'
$env:TORCH_HOME = Join-Path $LocalRoot 'cache\torch'
$env:XDG_CACHE_HOME = Join-Path $LocalRoot 'cache\xdg'
$env:XDG_STATE_HOME = Join-Path $LocalRoot 'state'
$env:YOLO_CONFIG_DIR = Join-Path $LocalRoot 'config\ultralytics'
$env:MPLCONFIGDIR = Join-Path $LocalRoot 'matplotlib'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:YOLO_AUTOINSTALL = 'false'
if (-not $UvPath) { throw 'Berikan -UvPath ke uv.exe di D:.' }
if (-not (Test-Path -LiteralPath $UvPath)) { throw "uv tidak ditemukan: $UvPath" }
$ProjectDrive = [IO.Path]::GetPathRoot($ProjectRoot)
if ([IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $UvPath).Path) -ne $ProjectDrive) {
    throw "uv.exe harus berada pada drive proyek ($ProjectDrive)."
}
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    if (-not $BasePython) { throw 'Berikan -BasePython ke Python 3.12 di D: untuk membuat .venv.' }
    if (-not (Test-Path -LiteralPath $BasePython)) { throw "Base Python tidak ditemukan: $BasePython" }
    if ([IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $BasePython).Path) -ne $ProjectDrive) {
        throw "Base Python harus berada pada drive proyek ($ProjectDrive)."
    }
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
