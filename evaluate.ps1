. "$PSScriptRoot\scripts\project-env.ps1"
Push-Location $ProjectRoot
try {
    & $ProjectPython -B scripts/evaluate.py @args
    if ($LASTEXITCODE -ne 0) { throw 'Evaluasi gagal; jangan mempromosikan hasil yang belum terverifikasi.' }
} finally { Pop-Location }
