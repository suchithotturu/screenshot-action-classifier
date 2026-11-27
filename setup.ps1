$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    throw 'This Windows setup supports 64-bit Intel/AMD Windows (x64).'
}
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($uvCommand) { $uv = $uvCommand.Source }
elseif (Test-Path "$env:USERPROFILE\.local\bin\uv.exe") { $uv = "$env:USERPROFILE\.local\bin\uv.exe" }
else { throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ and reopen PowerShell.' }
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.python'
$env:CLASSIFIER_OCR = 'rapidocr'
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    & $uv venv --python 3.12.16
    if ($LASTEXITCODE -ne 0) { throw 'Python environment setup failed.' }
}
& $uv pip sync --python .venv\Scripts\python.exe requirements-windows.lock
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .\.venv\Scripts\python.exe -m classifier.setup_runtime
if ($LASTEXITCODE -ne 0) { throw 'Model setup failed.' }
Write-Host 'Ready. Double-click Launch-Windows.cmd, or run:'
Write-Host '.\.venv\Scripts\python.exe -m classifier app --mode classify'
