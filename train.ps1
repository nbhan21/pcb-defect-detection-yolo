. "$PSScriptRoot\scripts\project-env.ps1"
Push-Location $ProjectRoot
try {
    & $ProjectPython -B scripts/train.py @args
    if ($LASTEXITCODE -ne 0) { throw 'Training gagal; periksa log sebelum mengulang.' }
} finally { Pop-Location }
