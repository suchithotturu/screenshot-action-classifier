$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MODELS = Join-Path $PSScriptRoot '.models'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
function Test-LocalOllama {
    try {
        $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 2
        return $true
    } catch { return $false }
}
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    $candidate = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (Test-Path $candidate) { $ollamaPath = $candidate }
    else { throw 'Optional annotation: install Ollama from https://ollama.com/download/windows first.' }
} else { $ollamaPath = $ollama.Source }
if (-not (Test-LocalOllama)) {
    New-Item -ItemType Directory -Force data | Out-Null
    $null = Start-Process -FilePath $ollamaPath -ArgumentList 'serve' -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $PSScriptRoot 'data\ollama.log') `
        -RedirectStandardError (Join-Path $PSScriptRoot 'data\ollama-error.log')
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        if (Test-LocalOllama) { break }
        Start-Sleep -Milliseconds 250
    }
}
if (-not (Test-LocalOllama)) { throw 'Ollama did not start. See data\ollama-error.log.' }
Write-Host 'Local Qwen service ready.'
