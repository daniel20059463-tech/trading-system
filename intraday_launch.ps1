# intraday_launch.ps1 - 盤中風險監控啟動器（排程每10分鐘呼叫，跑一輪檢查）
# 須存成 UTF-8 BOM。
$ProjectDir = $PSScriptRoot
$PythonExe  = "C:\Users\danie\AppData\Local\Programs\Python\Python311\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 非交易日（週末/國定假日）不跑
$dow = (Get-Date).DayOfWeek
$Today = Get-Date -Format "yyyy-MM-dd"
$Holidays = @("2026-01-01","2026-02-16","2026-02-17","2026-02-18","2026-02-19","2026-02-20",
              "2026-02-27","2026-02-28","2026-04-03","2026-04-06","2026-05-01","2026-06-19",
              "2026-09-25","2026-10-09","2026-10-26","2026-12-25")
if ($dow -eq "Saturday" -or $dow -eq "Sunday" -or ($Holidays -contains $Today)) { exit 0 }

Set-Location $ProjectDir

# 盤前漏跑自動補跑：若今天沒有 wiki 記錄（電腦 08:30 在睡導致盤前沒跑），
# 背景補跑盤前一次（每天只補一次，用旗標防重複）。不阻塞盤中監控。
$wikiToday   = Join-Path "D:\trading-wiki" "$Today.md"
$catchupFlag = Join-Path $ProjectDir "logs\${Today}_precatchup.flag"
if (-not (Test-Path $wikiToday) -and -not (Test-Path $catchupFlag)) {
    New-Item -ItemType File -Path $catchupFlag -Force | Out-Null
    Start-Process powershell -WindowStyle Hidden -ArgumentList @(
        "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", "`"$ProjectDir\launch.ps1`"", "-Session", "pre"
    )
}

$LogFile = Join-Path $ProjectDir "logs\${Today}_intraday_monitor.log"
& $PythonExe "intraday_monitor.py" *>&1 | Out-File -FilePath $LogFile -Append -Encoding utf8
