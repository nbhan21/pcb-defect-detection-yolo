$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LocalRoot = Join-Path $ProjectRoot '.local'
$ProjectDirs = @('tmp', 'cache\uv', 'cache\pip', 'cache\torch', 'cache\xdg', 'config\ultralytics', 'matplotlib', 'state')
foreach ($ProjectDir in $ProjectDirs) {
    New-Item -ItemType Directory -Path (Join-Path $LocalRoot $ProjectDir) -Force | Out-Null
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
$env:PYTHONPATH = $ProjectRoot
$ProjectPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
