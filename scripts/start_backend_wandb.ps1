Set-StrictMode -Version Latest
# start_backend_wandb.ps1
# Sets WANDB env vars and launches uvicorn under the project's venv.

# Determine script directory and project root robustly
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Path $MyInvocation.MyCommand.Definition -Parent }
$projectRoot = Resolve-Path -Path (Join-Path $scriptDir '..')
Push-Location $projectRoot

# --- Configure these values if needed ---
$env:WANDB_API_KEY = '34d37fb4f02cafe74a2f9678ef11de119acae4cd'
$env:WANDB_ENTITY  = 'raja-ramiz-mukhtar6-szabist'
$env:WANDB_PROJECT = 'MLOPSPROJECT2'

Write-Host "Starting backend with WANDB_ENTITY=$($env:WANDB_ENTITY)"

# Ensure venv python exists
# Path to the venv python executable (projectRoot + .venv) 
$python = Join-Path -Path $projectRoot -ChildPath '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "Python executable not found at $python"
    Exit 1
}

# Build command to run uvicorn and redirect stdout/stderr to files using cmd.exe
$uvicornCmd = "`"$python`" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
$cmdline = "/c $uvicornCmd 1> backend.log 2> backend.err"

Write-Host "Launching uvicorn (logs -> backend.log/backend.err)"
Start-Process -FilePath cmd.exe -ArgumentList $cmdline -WindowStyle Hidden -WorkingDirectory $projectRoot

Pop-Location
