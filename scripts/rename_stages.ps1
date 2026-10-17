# Разовая перенумерация папок результатов без пропуска номера 7:
#   results\stage_8 (тандемы, метрики)       -> results\stage_7
#   results\stage_9 (вторая выборка)         -> results\stage_8
#   scripts\26_stage7_context_features.py    -> удаляется (заменён scripts\26_context_features.py)
# Запускать один раз из любой папки:  powershell -ExecutionPolicy Bypass -File "D:\Default Project\scripts\rename_stages.ps1"
$ErrorActionPreference = "Stop"
Set-Location "D:\Default Project"
if (Test-Path "results\stage_7") { throw "results\stage_7 уже существует: перенумерация либо выполнена, либо папку нужно проверить вручную." }
if (-not (Test-Path "results\stage_8")) { throw "Нет results\stage_8." }
Move-Item "results\stage_8" "results\stage_7"
if (Test-Path "results\stage_9") { Move-Item "results\stage_9" "results\stage_8" }
if ((Test-Path "scripts\26_context_features.py") -and (Test-Path "scripts\26_stage7_context_features.py")) { Remove-Item "scripts\26_stage7_context_features.py" }
Get-ChildItem results -Directory | Select-Object -ExpandProperty Name
Write-Host "Готово."
