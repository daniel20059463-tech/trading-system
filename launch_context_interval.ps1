# 08:30 盤前主流程完成後，用當日美股快照產生第二組區間。
$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$PythonExe = "C:\Users\danie\AppData\Local\Programs\Python\Python311\python.exe"
$LogDir = Join-Path $ProjectDir "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }
$Date = Get-Date -Format "yyyy-MM-dd"
$LogFile = Join-Path $LogDir "${Date}_context_interval.log"
$dow = (Get-Date).DayOfWeek
$Holidays = @("2026-01-01","2026-02-16","2026-02-17","2026-02-18","2026-02-19","2026-02-20",
              "2026-02-27","2026-02-28","2026-04-03","2026-04-06","2026-05-01","2026-06-19",
              "2026-09-25","2026-10-09","2026-10-26","2026-12-25")
if ($dow -eq "Saturday" -or $dow -eq "Sunday" -or ($Holidays -contains $Date)) { exit 0 }
Set-Location $ProjectDir
$env:PYTHONIOENCODING = "utf-8"
$Result = 1
for ($Attempt = 1; $Attempt -le 10; $Attempt++) {
    & $PythonExe "premarket_context_interval.py" *>&1 | Out-File -FilePath $LogFile -Append -Encoding utf8
    $Result = $LASTEXITCODE
    if ($Result -eq 0) { break }
    if ((Get-Date).TimeOfDay -ge [TimeSpan]::FromHours(8.83)) { break }
    Start-Sleep -Seconds 30
}
exit $Result
