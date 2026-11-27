$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& .\service.ps1
$command = Get-Command ollama -ErrorAction SilentlyContinue
if ($command) { $ollamaPath = $command.Source }
else { $ollamaPath = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' }
$modelName = if ($env:VISION_MODEL) { $env:VISION_MODEL } else { 'qwen3-vl:4b-instruct' }
& $ollamaPath pull $modelName
if ($LASTEXITCODE -ne 0) { throw 'Qwen download failed.' }
Write-Host 'Optional local annotation model ready.'
