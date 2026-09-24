# 台股交易系統每日啟動腳本
# 由 Windows 工作排程器呼叫，參數：pre / intraday / post
# 路徑一律從 $PSScriptRoot 取得，避免中文路徑在非互動模式被編碼打亂
param(
    [string]$Session = "auto"
)

$ProjectDir = $PSScriptRoot
$PythonExe  = "C:\Users\danie\AppData\Local\Programs\Python\Python311\python.exe"
$LogDir     = Join-Path $ProjectDir "logs"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

$Date    = Get-Date -Format "yyyy-MM-dd"
$Time    = Get-Date -Format "HHmm"
$LogFile = Join-Path $LogDir "${Date}_${Session}.log"

# 非交易日（週末/國定假日）不執行 pre/intraday/post，避免產生無意義記錄
$dow = (Get-Date).DayOfWeek
$Holidays = @("2026-01-01","2026-02-16","2026-02-17","2026-02-18","2026-02-19","2026-02-20",
              "2026-02-27","2026-02-28","2026-04-03","2026-04-06","2026-05-01","2026-06-19",
              "2026-09-25","2026-10-09","2026-10-26","2026-12-25")
if ($dow -eq "Saturday" -or $dow -eq "Sunday" -or ($Holidays -contains $Date)) {
    "[$Date $Time] 非交易日，跳過 session=$Session" | Out-File -FilePath $LogFile -Append -Encoding utf8
    exit 0
}

Set-Location $ProjectDir

# 讓 Python 輸出 UTF-8，避免 log 中文亂碼
$env:PYTHONIOENCODING = "utf-8"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 載入 .env
$EnvFile = Join-Path $ProjectDir ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
}

"[$Date $Time] start session=$Session" | Out-File -FilePath $LogFile -Append -Encoding utf8
& $PythonExe "run_daily.py" $Session *>&1 | Out-File -FilePath $LogFile -Append -Encoding utf8
$SessionExit = $LASTEXITCODE
if ($Session -eq "post") {
    & $PythonExe "prediction_audit.py" *>&1 | Out-File -FilePath $LogFile -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) { $SessionExit = $LASTEXITCODE }
    & $PythonExe "daily_retrain.py" "train" *>&1 | Out-File -FilePath $LogFile -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) { $SessionExit = $LASTEXITCODE }
}
"[$Date $(Get-Date -Format HHmm)] done exit=$SessionExit" | Out-File -FilePath $LogFile -Append -Encoding utf8

exit $SessionExit
