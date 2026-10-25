. "$PSScriptRoot\scripts\project-env.ps1"
Push-Location $ProjectRoot
try {
    & $ProjectPython -B scripts/prepare_data.py @args
    if ($LASTEXITCODE -ne 0) { throw 'Persiapan data gagal; lihat diagnosis di atas.' }
} finally { Pop-Location }
