# 月度重訓啟動腳本（路徑從 $PSScriptRoot 取得，避免中文路徑編碼問題）
$ProjectDir = $PSScriptRoot
$PythonExe  = "C:\Users\danie\AppData\Local\Programs\Python\Python311\python.exe"
$LogDir     = Join-Path $ProjectDir "logs"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

$Month   = Get-Date -Format "yyyy-MM"
$LogFile = Join-Path $LogDir "retrain_$Month.log"

Set-Location $ProjectDir

# 讓 Python 輸出 UTF-8，避免 log 中文亂碼
$env:PYTHONIOENCODING = "utf-8"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$EnvFile = Join-Path $ProjectDir ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
}

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] start retrain" | Out-File -FilePath $LogFile -Append -Encoding utf8
& $PythonExe "run_retrain.py" "--model" "lstm" *>&1 | Out-File -FilePath $LogFile -Append -Encoding utf8
& $PythonExe "run_retrain.py" "--model" "transformer" *>&1 | Out-File -FilePath $LogFile -Append -Encoding utf8
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] done retrain" | Out-File -FilePath $LogFile -Append -Encoding utf8
