# Weekly US->TW observation report launcher (called by Task Scheduler)
$ProjectDir = $PSScriptRoot
$PythonExe  = "C:\Users\danie\AppData\Local\Programs\Python\Python311\python.exe"
$LogDir     = Join-Path $ProjectDir "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

Set-Location $ProjectDir
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Date = Get-Date -Format "yyyy-MM-dd"
$Log  = Join-Path $LogDir "${Date}_observe.log"
"[$Date $(Get-Date -Format HHmm)] start us_tw_observe" | Out-File -FilePath $Log -Append -Encoding utf8
& $PythonExe "us_tw_observe.py" *>&1 | Out-File -FilePath $Log -Append -Encoding utf8
"[$Date $(Get-Date -Format HHmm)] done exit=$LASTEXITCODE" | Out-File -FilePath $Log -Append -Encoding utf8
